"""Guarded release rehearsal; opt-in Linux test uses a substitute dependency parent."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import uuid

import pytest

from tests.integration import test_zero_interval_release as previous_tests
from tests.integration.test_restored_availability_release import repair as availability

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("restored_turn_release", ROOT / "scripts/deploy-restored-turn-repair.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)
PREDECESSOR = "f70eeb0812ef4166a2067e90b4515e03c738bae3"
IMAGE = "sha256:6643b086c7ae831260a747e0a3391c28697f24f9803a2be9ce496d01a81b2e67"


def baseline_source():
    return availability.patch_orchestrator(subprocess.check_output([
        "git", "show", availability.RESTORE + ":services/conversation/orchestrator.py"], cwd=ROOT))


def make_harness(tmp_path, monkeypatch, fault=None):
    monkeypatch.setattr(previous_tests, "repair", repair)
    monkeypatch.setattr(previous_tests, "TEST_IMAGE", IMAGE)
    monkeypatch.setattr(repair, "LIVE_CALENDAR_PROBE", object(), raising=False)
    harness = previous_tests.Harness(tmp_path, monkeypatch, fault)
    harness.predecessor = PREDECESSOR
    original_inspect, original_run = harness.inspect_once, harness.run
    def inspect(name):
        return original_inspect(name.replace("ai-phone-restored-turn-repair", "ai-phone-zero-interval-repair"))
    def run(*args, **kwargs):
        if args[:4] == ("docker", "exec", "app", "cat"):
            harness.events.append(args)
            return SimpleNamespace(stdout=baseline_source())
        result = original_run(*args, **kwargs)
        if result.stdout == b"ZERO_INTERVAL_OFFLINE_OK":
            result.stdout = b"RESTORED_TURN_OFFLINE_OK"
        return result
    harness.inspect_once, harness.run = inspect, run
    return harness


def test_success_preserves_settings_calendar_and_schema(tmp_path, monkeypatch):
    harness = make_harness(tmp_path, monkeypatch)
    repair.execute(harness, harness.helper)
    expected = copy.deepcopy(harness.original)
    expected["services"]["app"]["image"] = "candidate"
    assert json.loads((tmp_path / "docker-compose.override.yml").read_bytes()) == expected
    assert ("mark", "deployed") in harness.events
    assert set(p.name for p in (harness.directory / "build").iterdir()) == {
        "Dockerfile", "orchestrator.py", "language_policy.py", "restored_turn_admission.py"}


@pytest.mark.parametrize("fault", ["source", "build", "sdk", "compose", "parent"])
def test_precutover_failures_leave_current_app(tmp_path, monkeypatch, fault):
    harness = make_harness(tmp_path, monkeypatch, fault)
    with pytest.raises(RuntimeError):
        repair.execute(harness, harness.helper)
    assert not harness.cutover
    assert harness.current == IMAGE


def test_failed_readiness_restores_exact_predecessor(tmp_path, monkeypatch):
    harness = make_harness(tmp_path, monkeypatch, "ready")
    with pytest.raises(RuntimeError):
        repair.execute(harness, harness.helper)
    assert harness.current == IMAGE
    assert ("mark", "recovered") in harness.events
    assert json.loads((tmp_path / "docker-compose.override.yml").read_bytes()) == harness.original


def test_unexpected_image_stops_before_build(tmp_path, monkeypatch):
    harness = make_harness(tmp_path, monkeypatch)
    harness.current = "sha256:" + "c" * 64
    override = copy.deepcopy(harness.original)
    override["services"]["app"]["image"] = harness.current
    (tmp_path / "docker-compose.override.yml").write_text(json.dumps(override))
    with pytest.raises(RuntimeError, match="running image differs"):
        repair.execute(harness, harness.helper)
    assert not any(event[:2] == ("docker", "build") for event in harness.events)


@pytest.mark.skipif(os.environ.get("RESTORED_TURN_RUN_DOCKER_TESTS") != "1", reason="opt-in local Linux image")
def test_actual_linux_candidate_builder_and_installed_probe(tmp_path):
    def run(*args, timeout=120, check=True):
        try:
            result = subprocess.run(args, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise AssertionError((error.stderr or b"no diagnostic stderr").decode(errors="replace")[-6000:]) from None
        if check and result.returncode:
            raise AssertionError((result.stdout + result.stderr).decode(errors="replace"))
        return result
    def inspect(name):
        return json.loads(run("docker", "inspect", name).stdout)[0]
    from scripts.build_restored_turn_repair import patch_orchestrator
    parent = inspect("ai-phone-t049a-candidate:20260913")["Id"]
    directory = tmp_path / ("turn-review-" + uuid.uuid4().hex[:10])
    baseline = directory / "baseline"
    baseline.mkdir(parents=True)
    manifest = json.loads((ROOT / repair.MANIFEST).read_bytes())
    paths = set(repair.SUPPORTED_BASELINES[PREDECESSOR][1])
    for section in ("runtime", "configuration", "migrations", "protected_runtime"):
        paths.update(manifest[section])
    source = baseline_source()
    for path in paths:
        target = baseline / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        content = source if path == "services/conversation/orchestrator.py" else run(
            "git", "-C", str(ROOT), "show", PREDECESSOR + ":" + path).stdout
        target.write_bytes(content)
    parent_tag = "ai-phone-turn-review-parent:" + directory.name
    prepared_tag = "ai-phone-turn-review-prepared:" + directory.name
    candidate_tag = "ai-phone-restored-turn-repair:" + directory.name
    temporary_parent = "ai-phone-restored-turn-repair-parent:" + directory.name
    container = "turn-review-" + directory.name
    try:
        run("docker", "tag", parent, parent_tag)
        (baseline / "Dockerfile").write_text("FROM " + parent_tag + "\nCOPY source/ /app/\n", encoding="utf-8")
        run("docker", "build", "--network", "none", "--pull=false", "-t", prepared_tag, str(baseline), timeout=180)
        release = SimpleNamespace(directory=directory, commit="b" * 40, run=run, inspect_once=inspect)
        sources = {path: (patch_orchestrator(source) if path == "services/conversation/orchestrator.py"
                         else (ROOT / path).read_bytes()) for path in repair.SOURCES}
        assert {p: repair.digest(data) for p, data in sources.items()} == repair.SOURCES
        candidate = repair.build_candidate(release, sources, inspect(prepared_tag)["Id"])
        assert run("docker", "inspect", temporary_parent, check=False).returncode != 0
        result = run("docker", "run", "--rm", "--network", "none", "--no-healthcheck", "--name", container,
                     "--entrypoint", "python", candidate, "-B", "-c",
                     (ROOT / repair.PROBE_PATH).read_text(encoding="utf-8"), timeout=45)
        assert result.stdout.strip() == b"RESTORED_TURN_OFFLINE_OK"
        # Preserve the real restored availability/confirmation guards too.
        result = run("docker", "run", "--rm", "--network", "none", "--name", container,
                     "--entrypoint", "python", candidate, "-B", "-c", (ROOT / "scripts/probe-zero-interval-repair.py").read_text(encoding="utf-8"))
        assert result.stdout.strip() == b"ZERO_INTERVAL_OFFLINE_OK"
    finally:
        run("docker", "rm", "-f", container, check=False)
        for tag in (candidate_tag, temporary_parent, prepared_tag, parent_tag):
            run("docker", "image", "rm", tag, check=False)
            assert run("docker", "inspect", tag, check=False).returncode != 0
        assert run("docker", "inspect", container, check=False).returncode != 0
