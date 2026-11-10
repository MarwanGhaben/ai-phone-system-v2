"""Offline release boundary checks for the pause/reply repair; no Docker or provider access."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repair = load("reply_pause_release", "scripts/deploy-reply-pause-repair.py")
voice = load("voice_model_release_for_reply_pause", "scripts/deploy-voice-model.py")
availability = load("restored_availability_for_reply_pause", "scripts/deploy-restored-availability.py")
turn_builder = load("restored_turn_builder_for_reply_pause", "scripts/build_restored_turn_repair.py")
reply_builder = load("restored_reply_builder_for_reply_pause", "scripts/build_restored_reply_repair.py")

OLD_IMAGE = repair.ACCEPTED_IMAGE
REVISION = repair.ACCEPTED_REVISION
VOICE_REVISION = "3d70fd1d461b809aec9ecbf8033aa4d85df13d49"
SETTINGS = {"elevenlabs_model": "eleven_v4_turbo", "elevenlabs_voice_id": "SomeVoice0123456789ab"}


def digest(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def deployed_baseline():
    original = subprocess.check_output(
        ["git", "-c", "safe.directory=*", "show",
         availability.RESTORE + ":services/conversation/orchestrator.py"], cwd=ROOT)
    return turn_builder.patch_orchestrator(availability.patch_orchestrator(original))


BASELINE = deployed_baseline()


class Harness:
    def __init__(self, tmp, monkeypatch, fault=None):
        self.directory = tmp / "private"
        self.directory.mkdir()
        self.root = tmp
        self.commit = "b" * 40
        self.app_container = "app"
        self.project_name = "ai-phone-system-v2"
        self.fault = fault
        self.events = []
        self.current = OLD_IMAGE
        self.cutover = False
        self.variant = 1
        self.values = {"verified_phone_booking_enabled": False, "openai_model": "gpt-6-luna",
                       "secret": "synthetic", **SETTINGS}
        self.original = {"services": {"app": {"image": self.current,
            "environment": {"private": "synthetic", "OPENAI_MODEL": "gpt-6-luna"},
            "volumes": ["keep:/app/private"]}, "migrate": {"image": self.current}}}
        (tmp / "docker-compose.override.yml").write_text(json.dumps(self.original))
        (tmp / "docker-compose.override.yml").chmod(0o600)
        (tmp / "docker-compose.yml").write_text("synthetic")
        (tmp / "nginx").mkdir()
        (tmp / "nginx/nginx.conf").write_text("synthetic")
        monkeypatch.setattr(repair, "ROOT", tmp)
        # Windows does not expose POSIX uid/mode semantics; emulate this one preflight.
        original_stat = Path.stat
        target = tmp / "docker-compose.override.yml"
        def stat(path, *args, **kwargs):
            value = original_stat(path, *args, **kwargs)
            if path == target:
                return SimpleNamespace(st_uid=0, st_mode=0o100600)
            return value
        monkeypatch.setattr(Path, "stat", stat)
        self.helper = SimpleNamespace(strict_json=json.loads,
            typed_mapping_equal=lambda a, b: a == b,
            source_probe_code=lambda hashes, raw: json.dumps(
                {"hashes": {**hashes, **raw}, "settings": self.settings()}))

    def old(self):
        return self.current == OLD_IMAGE

    def inspect_once(self, name):
        if name.startswith("ai-phone-reply-pause-parent:"):
            return {"Id": "wrong" if self.fault == "parent" else OLD_IMAGE}
        if name.startswith("ai-phone-reply-pause:"):
            return {"Id": "candidate", "Config": {"Labels": {
                "org.opencontainers.image.revision": self.commit}}}
        return {"Image": self.current, "Config": {"Env": ["A=1", "OPENAI_MODEL=gpt-6-luna"],
            "Labels": {"org.opencontainers.image.revision": REVISION if self.old() else self.commit,
                       "com.docker.compose.project": self.project_name}},
            "Mounts": [{"Destination": "/app/private", "Source": "keep"}],
            "NetworkSettings": {"Networks": {"net": {}}}, "HostConfig": {"PortBindings": {}}}

    def settings(self):
        return dict(self.values)

    def wait_ready(self):
        if self.fault == "ready" and self.current == "candidate":
            raise RuntimeError("synthetic readiness failure")

    def public_health(self):
        pass

    def git_bytes(self, path):
        return (ROOT / path).read_bytes() + (b"# unreviewed" if self.fault == "source" else b"")

    def run(self, *args, **kwargs):
        self.events.append(args)
        if args[:2] == ("docker", "build") and self.fault == "build":
            raise RuntimeError("synthetic build failure")
        if args[0] == "git":
            value = subprocess.check_output(
                ["git", "-c", "safe.directory=*", "show", args[-1]], cwd=ROOT)
        elif args[:2] == ("docker", "exec"):
            if args[-1] == repair.FUTURE_OPERATIONS_PROBE:
                value = b'{"unresolved_future_operations":0}'
            elif "import hashlib,json; from pathlib" in args[-1]:
                value = json.dumps(repair.SUPPORTED_BASELINES[REVISION][self.variant]).encode()
            elif args[-2:] == ("cat", "/app/services/conversation/orchestrator.py"):
                value = BASELINE + (b"# unexpected" if self.fault == "baseline" else b"")
            else:
                value = args[-1].encode()
        elif args[:2] == ("docker", "run"):
            if self.fault == "offline":
                raise RuntimeError("synthetic offline probe failure")
            value = b'{"status":"REPLY_PAUSE_OK","mode":"offline"}'
        else:
            value = b""
        return SimpleNamespace(stdout=value)

    def private_write(self, name, data):
        path = self.directory / name
        path.write_bytes(data)
        return path

    def render_compose(self, path):
        value = json.loads(path.read_bytes())
        if self.fault == "compose" and path.name == "candidate-override.json":
            value["services"]["app"]["environment"]["private"] = "changed"
        return value

    def compose(self, path, *args, **kwargs):
        self.events.append(("compose", *args))
        if args[0] == "run":
            if args[-1] == "--live":
                failed = self.fault == "live"
                return SimpleNamespace(returncode=int(failed), stdout=json.dumps(
                    {"status": "REPLY_PAUSE_FAILED", "mode": "live", "reason": "invalid_request"}
                    if failed else {"status": "REPLY_PAUSE_OK", "mode": "live"}).encode())
            if args[-1] == repair.FUTURE_OPERATIONS_PROBE:
                return SimpleNamespace(stdout=b'{"unresolved_future_operations":0}')
            return SimpleNamespace(stdout=args[-1].encode())
        self.current = json.loads(path.read_bytes())["services"]["app"]["image"]

    def mark(self, name, *args):
        self.events.append(("mark", name))
        if name == "cutover-started":
            self.cutover = True

    def stop_writers(self, strict):
        self.events.append(("stop",))

    def install_override(self, path):
        (self.root / "docker-compose.override.yml").write_bytes(path.read_bytes())

    def start_ingress(self):
        self.events.append(("start",))

    def force_close_ingress(self):
        self.events.append(("close",))


def no_build_or_cutover(h):
    return not h.cutover and not any(e[:2] == ("docker", "build") for e in h.events)


def test_success_changes_only_the_app_image(tmp_path, monkeypatch, capsys):
    h = Harness(tmp_path, monkeypatch)
    repair.execute(h, h.helper)
    expected = copy.deepcopy(h.original)
    expected["services"]["app"]["image"] = "candidate"
    assert json.loads((tmp_path / "docker-compose.override.yml").read_bytes()) == expected
    assert set(p.name for p in (h.directory / "build").iterdir()) == {
        "Dockerfile", "elevenlabs_stt_service.py", "orchestrator.py"}
    assert ("mark", "deployed") in h.events
    assert h.settings() == h.values
    out = capsys.readouterr().out
    assert "REPLY_PAUSE_DEPLOYED_READY_HTTPS_OK" in out
    assert out.index("DEPLOYED_COMMIT=") < out.index("DEPLOYED_IMAGE=")
    assert all(line.startswith(("REPLY_PAUSE_", "DEPLOYED_")) for line in out.splitlines())


def test_offline_probe_and_live_check_both_run_before_cutover(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    repair.execute(h, h.helper)
    cutover = h.events.index(("mark", "cutover-started"))
    offline = next(i for i, e in enumerate(h.events)
                   if e[:2] == ("docker", "run") and "--network" in e and "none" in e)
    live = next(i for i, e in enumerate(h.events) if e[0] == "compose" and e[-1] == "--live")
    assert offline < cutover and live < cutover


def test_build_uses_verified_local_tag_not_bare_image_id(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    repair.execute(h, h.helper)
    dockerfile = (h.directory / "build/Dockerfile").read_text()
    assert not dockerfile.startswith("FROM sha256:")
    parent = dockerfile.splitlines()[0].removeprefix("FROM ")
    assert ("docker", "tag", OLD_IMAGE, parent) in h.events
    assert ("docker", "image", "rm", parent) in h.events
    assert "COPY orchestrator.py /app/services/conversation/orchestrator.py" in dockerfile
    assert "COPY elevenlabs_stt_service.py /app/services/stt/elevenlabs_stt_service.py" in dockerfile


def test_built_orchestrator_is_the_reviewed_reply_repair(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    repair.execute(h, h.helper)
    built = (h.directory / "build/orchestrator.py").read_bytes()
    assert built == reply_builder.patch_orchestrator(BASELINE)
    assert digest(built) == repair.SOURCES["services/conversation/orchestrator.py"]


@pytest.mark.parametrize("fault", ["source", "baseline", "offline", "compose", "parent", "build", "live"])
def test_preparation_failure_does_not_stop_app(tmp_path, monkeypatch, fault):
    h = Harness(tmp_path, monkeypatch, fault)
    before = (tmp_path / "docker-compose.override.yml").read_bytes()
    with pytest.raises(RuntimeError):
        repair.execute(h, h.helper)
    assert not h.cutover
    assert ("stop",) not in h.events
    assert h.current == OLD_IMAGE
    assert (tmp_path / "docker-compose.override.yml").read_bytes() == before


def test_failed_live_check_prints_only_a_short_reason_code(tmp_path, monkeypatch, capsys):
    h = Harness(tmp_path, monkeypatch, "live")
    with pytest.raises(RuntimeError):
        repair.execute(h, h.helper)
    out = capsys.readouterr().out
    assert "REPLY_PAUSE_LIVE_CHECK_FAILED_REASON=invalid_request" in out
    assert "REPLY_PAUSE_LIVE_SESSION_ACCEPTED_OK" not in out
    assert "REPLY_PAUSE_DEPLOYED" not in out


def test_unsafe_live_reason_is_not_printed(tmp_path, monkeypatch, capsys):
    h = Harness(tmp_path, monkeypatch, "live")
    compose = h.compose
    def unsafe(path, *args, **kwargs):
        if args[-1] == "--live":
            return SimpleNamespace(returncode=1, stdout=json.dumps(
                {"status": "REPLY_PAUSE_FAILED", "reason": "key sk-secret 123"}).encode())
        return compose(path, *args, **kwargs)
    h.compose = unsafe
    with pytest.raises(RuntimeError):
        repair.execute(h, h.helper)
    out = capsys.readouterr().out
    assert "REPLY_PAUSE_LIVE_CHECK_FAILED_REASON=none" in out
    assert "secret" not in out


def test_failed_candidate_restores_exact_previous_override_and_image(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, "ready")
    before = (tmp_path / "docker-compose.override.yml").read_bytes()
    with pytest.raises(RuntimeError):
        repair.execute(h, h.helper)
    assert h.current == OLD_IMAGE
    assert (tmp_path / "docker-compose.override.yml").read_bytes() == before
    assert ("mark", "recovered") in h.events
    assert ("mark", "deployed") not in h.events
    assert h.events[-2:] == [("start",), ("mark", "recovered")]


@pytest.mark.parametrize("variant", [0, 1])
def test_each_accepted_source_variant_deploys(tmp_path, monkeypatch, variant):
    h = Harness(tmp_path, monkeypatch)
    h.variant = variant
    repair.execute(h, h.helper)
    assert ("mark", "deployed") in h.events


def test_other_running_image_cannot_start_build_or_cutover(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    other = "sha256:" + "c" * 64
    h.current = other
    h.old = lambda: True
    h.original["services"]["app"]["image"] = other
    (tmp_path / "docker-compose.override.yml").write_text(json.dumps(h.original))
    with pytest.raises(RuntimeError, match="accepted running image"):
        repair.execute(h, h.helper)
    assert no_build_or_cutover(h)


@pytest.mark.parametrize("revision", ["unknown", VOICE_REVISION])
def test_unknown_predecessor_revision_cannot_start_build_or_cutover(tmp_path, monkeypatch, revision):
    h = Harness(tmp_path, monkeypatch)
    inspect = h.inspect_once
    def wrong_revision(name):
        value = inspect(name)
        if name == h.app_container:
            value["Config"]["Labels"]["org.opencontainers.image.revision"] = revision
        return value
    h.inspect_once = wrong_revision
    with pytest.raises(RuntimeError, match="predecessor image"):
        repair.execute(h, h.helper)
    assert no_build_or_cutover(h)


def test_wrong_effective_predecessor_source_cannot_start_cutover(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    h.helper.source_probe_code = lambda hashes, raw: json.dumps({"hashes": {}, "settings": h.values})
    with pytest.raises(RuntimeError, match="effective source"):
        repair.execute(h, h.helper)
    assert no_build_or_cutover(h)


@pytest.mark.parametrize("key,value,message", [
    ("elevenlabs_model", "eleven_multilingual_v2", "voice model"),
    ("openai_model", "gpt-4o", "Luna"),
    ("verified_phone_booking_enabled", True, "restored booking flow"),
    ("elevenlabs_model", None, "voice model"),
])
def test_wrong_settings_cannot_start_build_or_cutover(tmp_path, monkeypatch, key, value, message):
    h = Harness(tmp_path, monkeypatch)
    h.values[key] = value
    with pytest.raises(RuntimeError, match=message):
        repair.execute(h, h.helper)
    assert no_build_or_cutover(h)


def test_future_unresolved_operation_blocks_before_build(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    run = h.run
    def pending(*args, **kwargs):
        if args[-1] == repair.FUTURE_OPERATIONS_PROBE:
            return SimpleNamespace(stdout=b'{"unresolved_future_operations":1}')
        return run(*args, **kwargs)
    h.run = pending
    with pytest.raises(RuntimeError, match="unresolved future"):
        repair.execute(h, h.helper)
    assert no_build_or_cutover(h)


def test_new_unresolved_operation_after_writer_stop_restores_previous_app(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    compose = h.compose
    def late(path, *args, **kwargs):
        if args[-1] == repair.FUTURE_OPERATIONS_PROBE:
            return SimpleNamespace(stdout=b'{"unresolved_future_operations":1}')
        return compose(path, *args, **kwargs)
    h.compose = late
    before = (tmp_path / "docker-compose.override.yml").read_bytes()
    with pytest.raises(RuntimeError, match="unresolved future"):
        repair.execute(h, h.helper)
    assert h.current == OLD_IMAGE
    assert (tmp_path / "docker-compose.override.yml").read_bytes() == before
    assert ("mark", "recovered") in h.events
    assert ("mark", "deployed") not in h.events


def test_controlled_override_changes_only_the_app_image():
    original = {"services": {"app": {"image": "old", "environment": {"KEEP": "value"}},
                             "redis": {"image": "same"}}}
    changed = repair.changed_override(original, "candidate")
    assert original["services"]["app"]["image"] == "old"
    assert changed == {"services": {"app": {"image": "candidate", "environment": {"KEEP": "value"}},
                                    "redis": {"image": "same"}}}


def test_pinned_hashes_match_the_files_in_the_working_tree():
    assert digest((ROOT / repair.STT_PATH).read_bytes()) == repair.SOURCES[repair.STT_PATH]
    assert digest(reply_builder.patch_orchestrator(BASELINE)) == repair.SOURCES[repair.ORCHESTRATOR]
    assert digest((ROOT / repair.PROBE_PATH).read_bytes()) == repair.PROBE_SHA
    assert digest((ROOT / repair.BUILDER_PATH).read_bytes()) == repair.BUILDER_SHA
    assert set(repair.SOURCES) == {repair.STT_PATH, repair.ORCHESTRATOR}
    assert digest(BASELINE) == reply_builder.BASELINE_SHA


def test_supported_baselines_match_the_voice_model_release():
    expected = [{**variant, **voice.SOURCES} for variant in voice.SUPPORTED_BASELINES[VOICE_REVISION]]
    assert repair.SUPPORTED_BASELINES == {REVISION: expected}
    assert len(expected) == 2
    for variant in expected:
        assert variant["services/tts/elevenlabs_service.py"]
        assert variant["services/conversation/orchestrator.py"] == reply_builder.BASELINE_SHA


def test_accepted_image_and_predecessor_are_the_deployed_release():
    assert repair.ACCEPTED_REVISION == "9dda99b498f3823c8430ca73cc4a38c3ad00b9a1"
    assert repair.ACCEPTED_IMAGE == "sha256:20fb0101ad476726ef367180fb449ed09b403d74e7766c1ddf691391d0cf6273"
    assert repair.BASE == "6b3babf0eee0ed611c05d7031caacebe0e9dd652"


# ---- probe behaviour (installed-image checks, run here against this source tree) ----

@pytest.fixture
def probe(monkeypatch):
    module = load("reply_pause_probe", "scripts/probe-reply-pause-repair.py")
    monkeypatch.delenv(module.ENV_NAME, raising=False)  # restored after each test
    return module


def test_probe_threshold_check_passes_for_this_source_tree(probe):
    probe.check_threshold()
    assert probe.ENV_NAME not in os.environ


def test_probe_reply_cap_check_accepts_the_repaired_source_and_rejects_the_old(probe, tmp_path, monkeypatch):
    from services.conversation import orchestrator
    repaired = tmp_path / "repaired.py"
    repaired.write_bytes(reply_builder.patch_orchestrator(BASELINE))
    monkeypatch.setattr(orchestrator, "__file__", str(repaired))
    probe.check_reply_caps()
    old = tmp_path / "old.py"
    old.write_bytes(BASELINE)
    monkeypatch.setattr(orchestrator, "__file__", str(old))
    with pytest.raises(probe.Failure) as error:
        probe.check_reply_caps()
    assert error.value.reason == "old_reply_cap"


def test_probe_main_prints_one_status_line_without_secrets(probe, monkeypatch, capsys):
    async def failing():
        raise probe.Failure("connect")
    monkeypatch.setattr(probe, "offline", failing)
    monkeypatch.setattr(sys, "argv", ["probe"])
    with pytest.raises(SystemExit) as exit_info:
        probe.main()
    assert exit_info.value.code == 1
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ['{"status": "REPLY_PAUSE_FAILED", "mode": "offline", "reason": "connect"}']
