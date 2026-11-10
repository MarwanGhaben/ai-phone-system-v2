"""Offline release boundary checks for the turn timing release; no Docker or provider access."""
import ast
import asyncio
import copy
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = load("turn_timing_release", "scripts/deploy-turn-timing.py")
lock = load("language_lock_release_for_timing", "scripts/deploy-language-lock.py")
availability = load("restored_availability_for_timing", "scripts/deploy-restored-availability.py")
turn_builder = load("restored_turn_builder_for_timing", "scripts/build_restored_turn_repair.py")
reply_builder = load("restored_reply_builder_for_timing", "scripts/build_restored_reply_repair.py")
timing_builder = load("restored_timing_builder_for_release", "scripts/build_restored_timing.py")

OLD_IMAGE = "sha256:" + "a" * 64
REVISION = release.ACCEPTED_REVISION
LOCK_ENV = "ELEVENLABS_STT_LANGUAGE_LOCK"
SETTINGS = {"elevenlabs_model": "eleven_v4_turbo", "elevenlabs_voice_id": "SomeVoice0123456789ab"}
LOCK_VALUES = ["on", "off", None]


def digest(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def deployed_baseline():
    original = subprocess.check_output(
        ["git", "-c", "safe.directory=*", "show",
         availability.RESTORE + ":services/conversation/orchestrator.py"], cwd=ROOT)
    return reply_builder.patch_orchestrator(
        turn_builder.patch_orchestrator(availability.patch_orchestrator(original)))


BASELINE = deployed_baseline()


class Harness:
    def __init__(self, tmp, monkeypatch, fault=None, lock_value="on", image=OLD_IMAGE, record_image=True):
        self.directory = tmp / "private"
        self.directory.mkdir()
        self.root = tmp
        self.commit = "b" * 40
        self.app_container = "app"
        self.project_name = "ai-phone-system-v2"
        self.fault = fault
        self.events = []
        self.cutover = False
        self.variant = 1
        self.current = image
        self.revisions = {image: REVISION, "candidate": self.commit}
        self.original_image = image
        self.values = {"verified_phone_booking_enabled": False, "openai_model": "gpt-6-luna",
                       "secret": "synthetic", **SETTINGS}
        environment = {"private": "synthetic", "OPENAI_MODEL": "gpt-6-luna",
                       "ELEVENLABS_STT_VAD_SILENCE_SECS": "0.7"}
        if lock_value is not None:
            environment[LOCK_ENV] = lock_value
        self.original = {"services": {"app": {"image": image, "environment": environment,
            "volumes": ["keep:/app/private"]}, "migrate": {"image": image}}}
        self.override = tmp / "docker-compose.override.yml"
        self.override.write_text(json.dumps(self.original))
        self.override.chmod(0o600)
        self.original_bytes = self.override.read_bytes()
        (tmp / "docker-compose.yml").write_text("synthetic")
        (tmp / "nginx").mkdir()
        (tmp / "nginx/nginx.conf").write_text("synthetic")
        monkeypatch.setattr(release, "ROOT", tmp)
        if record_image:
            monkeypatch.setattr(release, "ACCEPTED_IMAGE", OLD_IMAGE)
        # Windows does not expose POSIX uid/mode semantics; emulate this one preflight.
        original_stat = Path.stat
        target = self.override

        def stat(path_, *args, **kwargs):
            value = original_stat(path_, *args, **kwargs)
            if path_ == target:
                return SimpleNamespace(st_uid=0, st_mode=0o100600)
            return value

        monkeypatch.setattr(Path, "stat", stat)
        self.helper = SimpleNamespace(strict_json=json.loads,
            typed_mapping_equal=lambda a, b: a == b,
            source_probe_code=lambda hashes, raw: json.dumps(
                {"hashes": {**hashes, **raw}, "settings": self.settings()}))
        self.container_env = self.env_entries(environment)

    @staticmethod
    def env_entries(environment):
        return ["A=1"] + [key + "=" + value for key, value in environment.items()]

    def inspect_once(self, name):
        if name.startswith("ai-phone-turn-timing-parent:"):
            return {"Id": "wrong" if self.fault == "parent" else self.original_image}
        if name.startswith("ai-phone-turn-timing:"):
            return {"Id": "candidate", "Config": {"Labels": {
                "org.opencontainers.image.revision": self.commit}}}
        return {"Image": self.current, "Config": {"Env": list(self.container_env),
            "Labels": {"org.opencontainers.image.revision": self.revisions[self.current],
                       "com.docker.compose.project": self.project_name}},
            "Mounts": [{"Destination": "/app/private", "Source": "keep"}],
            "NetworkSettings": {"Networks": {"net": {}}}, "HostConfig": {"PortBindings": {}}}

    def settings(self):
        return dict(self.values)

    def wait_ready(self):
        if self.fault == "ready" and self.override.read_bytes() != self.original_bytes:
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
            if args[-1] == release.FUTURE_OPERATIONS_PROBE:
                value = b'{"unresolved_future_operations":0}'
            elif "import hashlib,json; from pathlib" in args[-1]:
                sources = release.SUPPORTED_BASELINES[REVISION][self.variant]
                if self.fault == "variant":
                    sources = {**sources, "services/stt/elevenlabs_stt_service.py": "0" * 64}
                value = json.dumps(sources).encode()
            elif args[-2:] == ("cat", "/app/services/conversation/orchestrator.py"):
                value = BASELINE + (b"# unexpected" if self.fault == "baseline" else b"")
            else:
                value = args[-1].encode()
        elif args[:2] == ("docker", "run"):
            if self.fault == "offline":
                raise RuntimeError("synthetic offline probe failure")
            value = b'{"status":"TURN_TIMING_OK","mode":"offline"}'
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
            if args[-1] == release.FUTURE_OPERATIONS_PROBE:
                return SimpleNamespace(stdout=b'{"unresolved_future_operations":0}')
            return SimpleNamespace(stdout=args[-1].encode())
        document = json.loads(path.read_bytes())["services"]["app"]
        self.current = document["image"]
        self.container_env = self.env_entries(document.get("environment", {}))

    def mark(self, name, *args):
        self.events.append(("mark", name))
        if name == "cutover-started":
            self.cutover = True

    def stop_writers(self, strict):
        self.events.append(("stop",))

    def install_override(self, path):
        self.override.write_bytes(path.read_bytes())

    def start_ingress(self):
        self.events.append(("start",))

    def force_close_ingress(self):
        self.events.append(("close",))


def no_build_or_cutover(h):
    return not h.cutover and not any(e[:2] in (("docker", "build"), ("docker", "tag")) for e in h.events)


# ---- success ----

@pytest.mark.parametrize("lock_value", LOCK_VALUES)
def test_success_changes_only_the_app_image_and_keeps_every_environment_value(
        tmp_path, monkeypatch, capsys, lock_value):
    h = Harness(tmp_path, monkeypatch, lock_value=lock_value)
    before_env = list(h.container_env)
    release.execute(h, h.helper)
    expected = copy.deepcopy(h.original)
    expected["services"]["app"]["image"] = "candidate"
    assert json.loads(h.override.read_bytes()) == expected
    environment = json.loads(h.override.read_bytes())["services"]["app"]["environment"]
    assert environment.get(LOCK_ENV) == lock_value
    assert h.container_env == before_env  # the running container's environment is identical
    assert set(p.name for p in (h.directory / "build").iterdir()) == {
        "Dockerfile", "turn_timing.py", "orchestrator.py"}
    assert ("mark", "deployed") in h.events
    assert h.settings() == h.values
    out = capsys.readouterr().out
    assert "TURN_TIMING_DEPLOYED_READY_HTTPS_OK" in out
    assert out.index("DEPLOYED_COMMIT=") < out.index("DEPLOYED_IMAGE=")
    assert out.index("TURN_TIMING_DEPLOYED_READY_HTTPS_OK") < out.index("DEPLOYED_COMMIT=")
    assert all(line.startswith(("TURN_TIMING_", "DEPLOYED_")) for line in out.splitlines())


def test_offline_probe_runs_before_cutover_and_no_live_provider_check_exists(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    release.execute(h, h.helper)
    cutover = h.events.index(("mark", "cutover-started"))
    offline = next(i for i, e in enumerate(h.events)
                   if e[:2] == ("docker", "run") and "--network" in e and "none" in e)
    assert offline < cutover
    assert not any(e[0] == "compose" and "--live" in e for e in h.events)
    assert not hasattr(release, "check_live_session")


def test_build_uses_verified_local_tag_not_bare_image_id(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    release.execute(h, h.helper)
    dockerfile = (h.directory / "build/Dockerfile").read_text()
    assert not dockerfile.startswith("FROM sha256:")
    parent = dockerfile.splitlines()[0].removeprefix("FROM ")
    assert ("docker", "tag", OLD_IMAGE, parent) in h.events
    assert ("docker", "image", "rm", parent) in h.events
    copies = sorted(line for line in dockerfile.splitlines() if line.startswith("COPY "))
    assert copies == [
        "COPY orchestrator.py /app/services/conversation/orchestrator.py",
        "COPY turn_timing.py /app/services/conversation/turn_timing.py"]
    assert "network" in " ".join(next(e for e in h.events if e[:2] == ("docker", "build")))


def test_built_files_are_the_reviewed_sources(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    release.execute(h, h.helper)
    built = (h.directory / "build/orchestrator.py").read_bytes()
    assert built == timing_builder.patch_orchestrator(BASELINE)
    assert digest(built) == release.SOURCES[release.ORCHESTRATOR]
    helper_file = (h.directory / "build/turn_timing.py").read_bytes()
    assert helper_file == (ROOT / release.TIMING_PATH).read_bytes()
    assert digest(helper_file) == release.SOURCES[release.TIMING_PATH]


# ---- preparation faults leave the app untouched ----

@pytest.mark.parametrize("lock_value", ["on", "off"])
@pytest.mark.parametrize("fault", ["source", "baseline", "offline", "compose", "parent", "build", "variant"])
def test_preparation_failure_does_not_stop_app(tmp_path, monkeypatch, fault, lock_value):
    h = Harness(tmp_path, monkeypatch, fault, lock_value=lock_value)
    with pytest.raises(RuntimeError):
        release.execute(h, h.helper)
    assert not h.cutover
    assert ("stop",) not in h.events
    assert h.current == OLD_IMAGE
    assert h.override.read_bytes() == h.original_bytes


@pytest.mark.parametrize("lock_value", ["on", "off"])
def test_failed_candidate_restores_exact_previous_override_and_image(tmp_path, monkeypatch, lock_value):
    h = Harness(tmp_path, monkeypatch, "ready", lock_value=lock_value)
    before_env = list(h.container_env)
    with pytest.raises(RuntimeError):
        release.execute(h, h.helper)
    assert h.current == OLD_IMAGE
    assert h.override.read_bytes() == h.original_bytes
    assert h.container_env == before_env
    assert ("mark", "recovered") in h.events
    assert ("mark", "deployed") not in h.events
    assert h.events[-2:] == [("start",), ("mark", "recovered")]


def test_failed_recovery_force_closes_ingress(tmp_path, monkeypatch, capsys):
    h = Harness(tmp_path, monkeypatch, "ready")
    stop = h.stop_writers
    calls = []

    def stop_then_fail(strict):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("synthetic recovery failure")
        stop(strict)

    h.stop_writers = stop_then_fail
    with pytest.raises(RuntimeError):
        release.execute(h, h.helper)
    assert ("close",) in h.events
    out = capsys.readouterr().out
    assert "TURN_TIMING_RECOVERY_UNVERIFIED" in out
    assert "TURN_TIMING_PREVIOUS_APP_RESTORED" not in out


@pytest.mark.parametrize("variant", [0, 1])
def test_each_accepted_source_variant_deploys(tmp_path, monkeypatch, variant):
    h = Harness(tmp_path, monkeypatch)
    h.variant = variant
    release.execute(h, h.helper)
    assert ("mark", "deployed") in h.events


# ---- refusals ----

def test_script_cannot_run_while_the_accepted_image_is_only_a_placeholder(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, record_image=False)
    monkeypatch.setattr(release, "ACCEPTED_IMAGE", release.PLACEHOLDER_IMAGE)
    with pytest.raises(RuntimeError, match="not been recorded"):
        release.execute(h, h.helper)
    assert h.events == [] and no_build_or_cutover(h)
    # Even a container that happens to carry the placeholder value is not accepted.
    second = tmp_path / "second"
    second.mkdir()
    zero = Harness(second, monkeypatch, record_image=False, image=release.PLACEHOLDER_IMAGE)
    monkeypatch.setattr(release, "ACCEPTED_IMAGE", release.PLACEHOLDER_IMAGE)
    with pytest.raises(RuntimeError, match="not been recorded"):
        release.execute(zero, zero.helper)
    assert no_build_or_cutover(zero)


def test_script_as_committed_is_either_unfilled_or_a_real_image_id():
    value = release.ACCEPTED_IMAGE
    assert value == release.PLACEHOLDER_IMAGE or re.fullmatch(r"sha256:[0-9a-f]{64}", value)
    assert release.PLACEHOLDER_IMAGE == "sha256:" + "0" * 64


def test_the_image_check_is_an_exact_comparison_in_the_source():
    tree = ast.parse((ROOT / "scripts/deploy-turn-timing.py").read_text(encoding="utf-8"))
    execute = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "execute")
    exact = [n for n in ast.walk(execute) if isinstance(n, ast.Compare)
             and isinstance(n.left, ast.Name) and n.left.id == "old_image"
             and len(n.ops) == 1 and isinstance(n.ops[0], ast.NotEq)
             and isinstance(n.comparators[0], ast.Name) and n.comparators[0].id == "ACCEPTED_IMAGE"]
    assert len(exact) == 1


@pytest.mark.parametrize("other", [
    "sha256:" + "c" * 64,
    "sha256:" + "a" * 63 + "b",
    "sha256:" + "A" * 64,
    "sha256:" + "a" * 64 + "0",
    "sha256:" + "a" * 63,
])
def test_other_running_image_cannot_start_build_or_cutover(tmp_path, monkeypatch, other):
    h = Harness(tmp_path, monkeypatch, image=other)
    monkeypatch.setattr(release, "ACCEPTED_IMAGE", OLD_IMAGE)
    with pytest.raises(RuntimeError):
        release.execute(h, h.helper)
    assert no_build_or_cutover(h)


@pytest.mark.parametrize("revision", ["unknown", "84a9f975eda54c2003e41b8b55e74dff5d8d7aba",
                                      "3d70fd1d461b809aec9ecbf8033aa4d85df13d49", "b" * 40])
def test_unknown_predecessor_revision_cannot_start_build_or_cutover(tmp_path, monkeypatch, revision):
    h = Harness(tmp_path, monkeypatch)
    h.revisions[OLD_IMAGE] = revision
    with pytest.raises(RuntimeError, match="predecessor image"):
        release.execute(h, h.helper)
    assert no_build_or_cutover(h)


def test_override_image_must_match_the_running_image(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    changed = copy.deepcopy(h.original)
    changed["services"]["app"]["image"] = "sha256:" + "e" * 64
    h.override.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="predecessor override"):
        release.execute(h, h.helper)
    assert no_build_or_cutover(h)


def test_wrong_effective_predecessor_source_cannot_start_cutover(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    h.helper.source_probe_code = lambda hashes, raw: json.dumps({"hashes": {}, "settings": h.values})
    with pytest.raises(RuntimeError, match="effective source"):
        release.execute(h, h.helper)
    assert no_build_or_cutover(h)


def test_a_reply_pause_only_speech_service_is_not_an_accepted_predecessor(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    older = dict(release.SUPPORTED_BASELINES[REVISION][1])
    older["services/stt/elevenlabs_stt_service.py"] = lock.SUPPORTED_BASELINES[
        lock.PREDECESSOR_REVISION][1]["services/stt/elevenlabs_stt_service.py"]
    h.variant = 0
    run = h.run

    def old_stt(*args, **kwargs):
        if args[:2] == ("docker", "exec") and "import hashlib,json; from pathlib" in args[-1]:
            return SimpleNamespace(stdout=json.dumps(older).encode())
        return run(*args, **kwargs)

    h.run = old_stt
    with pytest.raises(RuntimeError, match="predecessor source"):
        release.execute(h, h.helper)
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
        release.execute(h, h.helper)
    assert no_build_or_cutover(h)


def test_settings_changing_during_preparation_blocks_cutover(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    calls = []
    original = h.settings

    def drifting():
        calls.append(1)
        value = original()
        if len(calls) >= 3:
            value["secret"] = "changed"
        return value

    h.settings = drifting
    with pytest.raises(RuntimeError):
        release.execute(h, h.helper)
    assert not h.cutover


def test_future_unresolved_operation_blocks_before_build(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    run = h.run

    def pending(*args, **kwargs):
        if args[-1] == release.FUTURE_OPERATIONS_PROBE:
            return SimpleNamespace(stdout=b'{"unresolved_future_operations":1}')
        return run(*args, **kwargs)

    h.run = pending
    with pytest.raises(RuntimeError, match="unresolved future"):
        release.execute(h, h.helper)
    assert no_build_or_cutover(h)
    assert not any(e[:2] == ("docker", "run") for e in h.events)
    assert not (h.directory / "previous-override.json").exists()


@pytest.mark.parametrize("answer", [b'{"unresolved_future_operations":null}', b'{"status":"read_failed"}',
                                    b'{"unresolved_future_operations":true}', b"[]"])
def test_unreadable_future_operation_answer_blocks_before_build(tmp_path, monkeypatch, answer):
    h = Harness(tmp_path, monkeypatch)
    run = h.run

    def odd(*args, **kwargs):
        if args[-1] == release.FUTURE_OPERATIONS_PROBE:
            return SimpleNamespace(stdout=answer)
        return run(*args, **kwargs)

    h.run = odd
    with pytest.raises(RuntimeError):
        release.execute(h, h.helper)
    assert no_build_or_cutover(h)


def test_new_unresolved_operation_after_writer_stop_restores_previous_app(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    compose = h.compose

    def late(path_, *args, **kwargs):
        if args[-1] == release.FUTURE_OPERATIONS_PROBE:
            return SimpleNamespace(stdout=b'{"unresolved_future_operations":1}')
        return compose(path_, *args, **kwargs)

    h.compose = late
    with pytest.raises(RuntimeError, match="unresolved future"):
        release.execute(h, h.helper)
    assert h.current == OLD_IMAGE
    assert h.override.read_bytes() == h.original_bytes
    assert ("mark", "recovered") in h.events
    assert ("mark", "deployed") not in h.events


def test_the_future_operation_check_runs_before_build_and_again_after_writers_stop(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    release.execute(h, h.helper)
    checks = [i for i, e in enumerate(h.events)
              if e and (e[-1] == release.FUTURE_OPERATIONS_PROBE)]
    build = next(i for i, e in enumerate(h.events) if e[:2] == ("docker", "build"))
    stop = h.events.index(("stop",))
    assert len(checks) == 2 and checks[0] < build and checks[1] > stop


# ---- override and contract ----

def test_controlled_override_changes_only_the_app_image():
    original = {"services": {"app": {"image": "old", "environment": {"KEEP": "value", LOCK_ENV: "off"}},
                             "redis": {"image": "same"}}}
    changed = release.changed_override(original, "candidate")
    assert original["services"]["app"]["image"] == "old"
    assert changed == {"services": {"app": {"image": "candidate",
                                             "environment": {"KEEP": "value", LOCK_ENV: "off"}},
                                    "redis": {"image": "same"}}}


# ---- pinned hashes ----

def test_pinned_hashes_match_the_files_in_the_working_tree():
    assert set(release.SOURCES) == {release.TIMING_PATH, release.ORCHESTRATOR}
    assert digest((ROOT / release.TIMING_PATH).read_bytes()) == release.SOURCES[release.TIMING_PATH]
    assert digest(timing_builder.patch_orchestrator(BASELINE)) == release.SOURCES[release.ORCHESTRATOR]
    assert release.SOURCES[release.ORCHESTRATOR] == timing_builder.PATCHED_SHA
    assert digest((ROOT / release.PROBE_PATH).read_bytes()) == release.PROBE_SHA
    assert digest((ROOT / release.BUILDER_PATH).read_bytes()) == release.BUILDER_SHA
    assert digest(BASELINE) == timing_builder.BASELINE_SHA


def test_pinned_predecessor_is_the_deployed_language_lock_release():
    assert release.ACCEPTED_REVISION == "8629becc623db91a0b2bff56dd48d1efa6314650"
    assert release.BASE == lock.BASE
    assert release.MANIFEST == lock.MANIFEST and release.MANIFEST_SHA == lock.MANIFEST_SHA
    assert release.LLM_MODEL == lock.LLM_MODEL and release.VOICE_MODEL == lock.VOICE_MODEL


def test_supported_baselines_are_recomputed_from_the_language_lock_release():
    expected = {REVISION: [{**variant, **lock.SOURCES}
                           for variant in lock.SUPPORTED_BASELINES[lock.PREDECESSOR_REVISION]]}
    assert release.SUPPORTED_BASELINES == expected
    assert len(expected[REVISION]) == 2
    for variant in expected[REVISION]:
        assert variant["services/stt/elevenlabs_stt_service.py"] == lock.SOURCES[lock.STT_PATH]
        assert variant[release.ORCHESTRATOR] == timing_builder.BASELINE_SHA
        assert release.TIMING_PATH not in variant


def test_release_files_use_unix_line_endings():
    for path in ("scripts/deploy-turn-timing.py", "scripts/probe-turn-timing.py",
                 "scripts/read-turn-timing.py", "scripts/build_restored_timing.py",
                 "services/conversation/turn_timing.py", "tests/conversation/test_turn_timing.py",
                 "tests/conversation/test_restored_timing.py",
                 "tests/integration/test_turn_timing_release.py",
                 "tests/integration/test_turn_timing_reader.py"):
        assert b"\r" not in (ROOT / path).read_bytes(), path


def test_release_script_carries_no_unreviewed_placeholders():
    text = (ROOT / "scripts/deploy-turn-timing.py").read_text(encoding="utf-8")
    assert "@@" not in text
    assert "PENDING" not in text
    # The owner-reported running image of the language-lock release (2026-11-10).
    assert release.ACCEPTED_IMAGE == (
        "sha256:0383f37031c3ba6e1dbea7129242f6771e7d65c66889d9ff4454f403836ac7d8")
    assert release.ACCEPTED_IMAGE != release.PLACEHOLDER_IMAGE


# ---- the offline probe, run against the patched restored orchestrator ----

@pytest.fixture
def probe():
    module = load("turn_timing_probe", "scripts/probe-turn-timing.py")
    yield module
    logging.disable(logging.NOTSET)


@pytest.fixture
def instrumented(tmp_path):
    source = timing_builder.patch_orchestrator(BASELINE)
    module = types.ModuleType("restored_timing_probe_runtime")
    sys.modules[module.__name__] = module
    path = tmp_path / "orchestrator.py"
    path.write_bytes(source)
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module, path


def test_probe_passes_for_the_instrumented_orchestrator(probe, instrumented):
    from services.conversation import turn_timing
    module, _path = instrumented
    probe.check_installed_source(module)
    probe.check_line(turn_timing)
    asyncio.run(probe.check_behaviour(module, turn_timing))


def test_probe_rejects_the_uninstrumented_orchestrator(probe, tmp_path):
    module = types.ModuleType("plain_probe_runtime")
    path = tmp_path / "orchestrator.py"
    path.write_bytes(BASELINE)
    module.__file__ = str(path)
    with pytest.raises(probe.Failure) as error:
        probe.check_installed_source(module)
    assert error.value.reason == "instrumentation_missing"


def test_probe_rejects_a_line_that_leaks_free_text(probe, monkeypatch):
    from services.conversation import turn_timing
    original = turn_timing.TurnTiming.line
    monkeypatch.setattr(turn_timing.TurnTiming, "line",
                        lambda self: original(self) + " note=Canary")
    with pytest.raises(probe.Failure) as error:
        probe.check_line(turn_timing)
    assert error.value.reason == "line_shape"


def test_probe_detects_instrumentation_that_changes_behaviour(probe, instrumented, monkeypatch):
    from services.conversation import turn_timing
    module, _path = instrumented
    original = turn_timing.arm

    def arm_and_interfere(context):
        original(context)
        context.turn_count += 1  # an instrumentation side effect on the call

    monkeypatch.setattr(turn_timing, "arm", arm_and_interfere)
    with pytest.raises(probe.Failure) as error:
        asyncio.run(probe.check_behaviour(module, turn_timing))
    assert error.value.reason in ("instrumentation_changed_behaviour", "no_single_timing_line")


def test_probe_main_prints_one_status_line_without_secrets(probe, monkeypatch, capsys):
    async def failing():
        raise probe.Failure("instrumentation_missing")

    monkeypatch.setattr(probe, "offline", failing)
    monkeypatch.setattr(sys, "argv", ["probe"])
    with pytest.raises(SystemExit) as exit_info:
        probe.main()
    assert exit_info.value.code == 1
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ['{"status": "TURN_TIMING_FAILED", "mode": "offline", "reason": "instrumentation_missing"}']


def test_probe_main_success_exit_code(probe, monkeypatch, capsys):
    async def passing():
        return {"status": "TURN_TIMING_OK", "mode": "offline"}

    monkeypatch.setattr(probe, "offline", passing)
    monkeypatch.setattr(sys, "argv", ["probe"])
    with pytest.raises(SystemExit) as exit_info:
        probe.main()
    assert exit_info.value.code == 0
    assert json.loads(capsys.readouterr().out) == {"status": "TURN_TIMING_OK", "mode": "offline"}


def test_script_entry_point_refuses_bad_arguments_with_the_stop_marker():
    for arguments in ([], ["a" * 40, "extra"], ["not-a-commit"]):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/deploy-turn-timing.py"), *arguments],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 1
        assert "TURN_TIMING_STOPPED_KEEP_PROTECTED_FILES" in result.stdout
