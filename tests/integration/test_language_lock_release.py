"""Offline release boundary checks for the language lock release; no Docker or provider access."""
import asyncio
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


lock = load("language_lock_release", "scripts/deploy-language-lock.py")
repair = load("reply_pause_release_for_language_lock", "scripts/deploy-reply-pause-repair.py")

OLD_IMAGE = lock.PREDECESSOR_IMAGE
PREVIOUS = lock.PREDECESSOR_REVISION
TOGGLE_IMAGE = "sha256:" + "d" * 64
LOCK_ENV = "ELEVENLABS_STT_LANGUAGE_LOCK"
SETTINGS = {"elevenlabs_model": "eleven_v4_turbo", "elevenlabs_voice_id": "SomeVoice0123456789ab"}
PATHS = ["install", "toggle"]


def digest(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


class Harness:
    def __init__(self, tmp, monkeypatch, fault=None, path="install", lock_value=None,
                 lock_key=LOCK_ENV):
        self.path = path
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
        if path == "install":
            self.current = OLD_IMAGE
            self.revisions = {OLD_IMAGE: PREVIOUS, "candidate": self.commit}
        else:
            self.current = TOGGLE_IMAGE
            self.revisions = {TOGGLE_IMAGE: self.commit}
        self.original_image = self.current
        self.values = {"verified_phone_booking_enabled": False, "openai_model": "gpt-6-luna",
                       "secret": "synthetic", **SETTINGS}
        environment = {"private": "synthetic", "OPENAI_MODEL": "gpt-6-luna"}
        if lock_value is None and path == "toggle":
            lock_value = "on"
        if lock_value is not None:
            environment[lock_key] = lock_value
        self.original = {"services": {"app": {"image": self.current, "environment": environment,
            "volumes": ["keep:/app/private"]}, "migrate": {"image": self.current}}}
        self.override = tmp / "docker-compose.override.yml"
        self.override.write_text(json.dumps(self.original))
        self.override.chmod(0o600)
        self.original_bytes = self.override.read_bytes()
        (tmp / "docker-compose.yml").write_text("synthetic")
        (tmp / "nginx").mkdir()
        (tmp / "nginx/nginx.conf").write_text("synthetic")
        monkeypatch.setattr(lock, "ROOT", tmp)
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
        self.container_lock = self.lock_entry(self.original["services"]["app"]["environment"])

    @staticmethod
    def lock_entry(environment):
        return [key + "=" + value for key, value in environment.items() if key.upper() == LOCK_ENV]

    def running_sources(self):
        variant = lock.SUPPORTED_BASELINES[PREVIOUS][self.variant]
        if self.path == "toggle":
            return {**variant, **lock.SOURCES}
        return variant

    def inspect_once(self, name):
        if name.startswith("ai-phone-language-lock-parent:"):
            return {"Id": "wrong" if self.fault == "parent" else OLD_IMAGE}
        if name.startswith("ai-phone-language-lock:"):
            return {"Id": "candidate", "Config": {"Labels": {
                "org.opencontainers.image.revision": self.commit}}}
        return {"Image": self.current, "Config": {
            "Env": ["A=1", "OPENAI_MODEL=gpt-6-luna", *self.container_lock],
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
            if args[-1] == lock.FUTURE_OPERATIONS_PROBE:
                value = b'{"unresolved_future_operations":0}'
            elif "import hashlib,json; from pathlib" in args[-1]:
                sources = self.running_sources()
                if self.fault == "variant":
                    sources = {**sources, lock.STT_PATH: "0" * 64}
                value = json.dumps(sources).encode()
            else:
                value = args[-1].encode()
        elif args[:2] == ("docker", "run"):
            if self.fault == "offline":
                raise RuntimeError("synthetic offline probe failure")
            value = b'{"status":"LANGUAGE_LOCK_OK","mode":"offline"}'
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
                environment = json.loads(path.read_bytes())["services"]["app"]["environment"]
                mode = "unlocked" if environment.get(LOCK_ENV) == "off" else "locked"
                if self.fault == "lock_mismatch":
                    mode = "locked" if mode == "unlocked" else "unlocked"
                failed = self.fault == "live"
                return SimpleNamespace(returncode=int(failed), stdout=json.dumps(
                    {"status": "LANGUAGE_LOCK_FAILED", "mode": "live", "reason": "invalid_request"}
                    if failed else {"status": "LANGUAGE_LOCK_OK", "mode": "live", "lock": mode}).encode())
            if args[-1] == lock.FUTURE_OPERATIONS_PROBE:
                return SimpleNamespace(stdout=b'{"unresolved_future_operations":0}')
            return SimpleNamespace(stdout=args[-1].encode())
        document = json.loads(path.read_bytes())["services"]["app"]
        self.current = document["image"]
        self.container_lock = self.lock_entry(document.get("environment", {}))

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


def expected_override(h, mode, image=None):
    result = copy.deepcopy(h.original)
    if image is not None:
        result["services"]["app"]["image"] = image
    environment = result["services"]["app"]["environment"]
    for key in [key for key in environment if key.upper() == LOCK_ENV]:
        del environment[key]
    environment[LOCK_ENV] = {"locked": "on", "unlocked": "off"}[mode]
    return result


def installed(h):
    return json.loads(h.override.read_bytes())


def has_compose_up(h):
    return any(e[:2] == ("compose", "up") for e in h.events)


# ---- first install ----

@pytest.mark.parametrize("mode", ["locked", "unlocked"])
def test_first_install_changes_only_image_and_lock_value(tmp_path, monkeypatch, capsys, mode):
    h = Harness(tmp_path, monkeypatch)
    lock.execute(h, h.helper, mode)
    assert installed(h) == expected_override(h, mode, "candidate")
    assert set(p.name for p in (h.directory / "build").iterdir()) == {
        "Dockerfile", "elevenlabs_stt_service.py"}
    dockerfile = (h.directory / "build/Dockerfile").read_text()
    assert not dockerfile.startswith("FROM sha256:")
    parent = dockerfile.splitlines()[0].removeprefix("FROM ")
    assert ("docker", "tag", OLD_IMAGE, parent) in h.events
    assert ("docker", "image", "rm", parent) in h.events
    assert dockerfile.count("COPY ") == 1
    assert "COPY elevenlabs_stt_service.py /app/services/stt/elevenlabs_stt_service.py" in dockerfile
    assert ("mark", "deployed") in h.events
    assert h.settings() == h.values
    lines = capsys.readouterr().out.splitlines()
    assert all(line.startswith(("LANGUAGE_LOCK_", "DEPLOYED_")) for line in lines)
    assert lines[-4:] == ["LANGUAGE_LOCK_DEPLOYED_READY_HTTPS_OK", "DEPLOYED_COMMIT=" + "b" * 40,
                          "DEPLOYED_IMAGE=candidate", "LANGUAGE_LOCK_MODE=" + mode]


def test_first_install_removes_other_cased_duplicate_of_the_key(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, lock_value="off", lock_key="elevenlabs_stt_language_lock")
    lock.execute(h, h.helper, "locked")
    environment = installed(h)["services"]["app"]["environment"]
    assert [k for k in environment if k.upper() == LOCK_ENV] == [LOCK_ENV]
    assert environment[LOCK_ENV] == "on"
    assert installed(h) == expected_override(h, "locked", "candidate")


def test_first_install_probes_run_before_cutover(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    lock.execute(h, h.helper, "unlocked")
    cutover = h.events.index(("mark", "cutover-started"))
    offline = next(i for i, e in enumerate(h.events)
                   if e[:2] == ("docker", "run") and "--network" in e and "none" in e)
    live = next(i for i, e in enumerate(h.events) if e[0] == "compose" and e[-1] == "--live")
    assert offline < cutover and live < cutover


# ---- toggle ----

@pytest.mark.parametrize("start,mode", [("on", "unlocked"), ("off", "locked"), (None, "unlocked"),
                                         ("junk", "locked")])
def test_toggle_changes_only_the_lock_value(tmp_path, monkeypatch, capsys, start, mode):
    h = Harness(tmp_path, monkeypatch, path="toggle", lock_value=start)
    if start is None:
        del h.original["services"]["app"]["environment"][LOCK_ENV]
        h.override.write_text(json.dumps(h.original))
        h.original_bytes = h.override.read_bytes()
        h.container_lock = []
    lock.execute(h, h.helper, mode)
    assert installed(h) == expected_override(h, mode)
    assert installed(h)["services"]["app"]["image"] == TOGGLE_IMAGE
    assert h.current == TOGGLE_IMAGE
    assert not any(e[:2] in (("docker", "build"), ("docker", "tag"), ("docker", "image"))
                   for e in h.events)
    assert not (h.directory / "build").exists()
    assert ("mark", "deployed") in h.events
    lines = capsys.readouterr().out.splitlines()
    assert all(line.startswith(("LANGUAGE_LOCK_", "DEPLOYED_")) for line in lines)
    assert lines[-4:] == ["LANGUAGE_LOCK_DEPLOYED_READY_HTTPS_OK", "DEPLOYED_COMMIT=" + "b" * 40,
                          "DEPLOYED_IMAGE=" + TOGGLE_IMAGE, "LANGUAGE_LOCK_MODE=" + mode]


def test_toggle_candidate_override_differs_from_previous_only_by_the_key(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, path="toggle", lock_value="on")
    lock.execute(h, h.helper, "unlocked")
    candidate = json.loads((h.directory / "candidate-override.json").read_bytes())
    previous = json.loads((h.directory / "previous-override.json").read_bytes())
    assert previous == h.original
    changed = copy.deepcopy(previous)
    changed["services"]["app"]["environment"][LOCK_ENV] = "off"
    assert candidate == changed


@pytest.mark.parametrize("value,mode,marker", [("on", "locked", "LANGUAGE_LOCK_ALREADY_LOCKED"),
                                               ("off", "unlocked", "LANGUAGE_LOCK_ALREADY_UNLOCKED")])
def test_toggle_to_the_active_mode_changes_nothing(tmp_path, monkeypatch, capsys, value, mode, marker):
    h = Harness(tmp_path, monkeypatch, path="toggle", lock_value=value)
    lock.execute(h, h.helper, mode)
    out = capsys.readouterr().out
    assert marker in out.splitlines()
    assert "LANGUAGE_LOCK_MODE=" + mode in out
    assert "DEPLOYED" not in out
    assert ("stop",) not in h.events
    assert not has_compose_up(h)
    assert not any(e[0] == "mark" for e in h.events)
    assert not any(e[:2] in (("docker", "build"), ("docker", "tag")) for e in h.events)
    assert not any(e[0] == "compose" and e[-1] == "--live" for e in h.events)
    assert h.override.read_bytes() == h.original_bytes
    assert not (h.directory / "candidate-override.json").exists()


def test_already_active_check_does_not_apply_to_first_install(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, lock_value="on")
    lock.execute(h, h.helper, "locked")
    assert ("mark", "deployed") in h.events
    assert installed(h)["services"]["app"]["image"] == "candidate"


@pytest.mark.parametrize("key", ["elevenlabs_stt_language_lock"])
def test_toggle_with_other_cased_key_is_not_treated_as_already_active(tmp_path, monkeypatch, key):
    h = Harness(tmp_path, monkeypatch, path="toggle", lock_value="on", lock_key=key)
    lock.execute(h, h.helper, "locked")
    assert ("mark", "deployed") in h.events
    assert installed(h) == expected_override(h, "locked")


# ---- live check ----

@pytest.mark.parametrize("path", PATHS)
def test_live_probe_reporting_the_other_mode_stops_before_cutover(tmp_path, monkeypatch, capsys, path):
    h = Harness(tmp_path, monkeypatch, "lock_mismatch", path=path)
    with pytest.raises(RuntimeError, match="did not reach"):
        lock.execute(h, h.helper, "unlocked")
    out = capsys.readouterr().out
    assert "LANGUAGE_LOCK_LIVE_CHECK_FAILED_REASON=lock_mismatch" in out
    assert "LANGUAGE_LOCK_LIVE_SESSION_ACCEPTED_OK" not in out
    assert not h.cutover and ("stop",) not in h.events
    assert h.override.read_bytes() == h.original_bytes


@pytest.mark.parametrize("path", PATHS)
def test_failed_live_check_prints_only_a_short_reason_code(tmp_path, monkeypatch, capsys, path):
    h = Harness(tmp_path, monkeypatch, "live", path=path)
    with pytest.raises(RuntimeError):
        lock.execute(h, h.helper, "unlocked")
    out = capsys.readouterr().out
    assert "LANGUAGE_LOCK_LIVE_CHECK_FAILED_REASON=invalid_request" in out
    assert "LANGUAGE_LOCK_DEPLOYED" not in out


def test_unsafe_live_reason_is_not_printed(tmp_path, monkeypatch, capsys):
    h = Harness(tmp_path, monkeypatch)
    compose = h.compose
    def unsafe(path, *args, **kwargs):
        if args[-1] == "--live":
            return SimpleNamespace(returncode=1, stdout=json.dumps(
                {"status": "LANGUAGE_LOCK_FAILED", "reason": "key sk-secret 123"}).encode())
        return compose(path, *args, **kwargs)
    h.compose = unsafe
    with pytest.raises(RuntimeError):
        lock.execute(h, h.helper, "locked")
    out = capsys.readouterr().out
    assert "LANGUAGE_LOCK_LIVE_CHECK_FAILED_REASON=none" in out
    assert "secret" not in out


@pytest.mark.parametrize("status", [
    {"status": "LANGUAGE_LOCK_OK", "mode": "offline", "lock": "locked"},
    {"status": "LANGUAGE_LOCK_OK", "mode": "live"},
    {"status": "LANGUAGE_LOCK_OK", "mode": "live", "lock": "maybe"},
    {"status": "REPLY_PAUSE_OK", "mode": "live", "lock": "locked"},
])
def test_incomplete_live_status_is_refused(tmp_path, monkeypatch, status):
    h = Harness(tmp_path, monkeypatch)
    compose = h.compose
    def odd(path, *args, **kwargs):
        if args[-1] == "--live":
            return SimpleNamespace(returncode=0, stdout=json.dumps(status).encode())
        return compose(path, *args, **kwargs)
    h.compose = odd
    with pytest.raises(RuntimeError):
        lock.execute(h, h.helper, "locked")
    assert not h.cutover


# ---- preparation faults ----

@pytest.mark.parametrize("path,fault", [
    ("install", "source"), ("install", "offline"), ("install", "compose"), ("install", "parent"),
    ("install", "build"), ("install", "live"), ("install", "lock_mismatch"),
    ("toggle", "source"), ("toggle", "offline"), ("toggle", "compose"),
    ("toggle", "live"), ("toggle", "lock_mismatch"),
])
def test_preparation_failure_does_not_stop_app(tmp_path, monkeypatch, path, fault):
    h = Harness(tmp_path, monkeypatch, fault, path=path)
    with pytest.raises(RuntimeError):
        lock.execute(h, h.helper, "unlocked")
    assert not h.cutover
    assert ("stop",) not in h.events
    assert not has_compose_up(h)
    assert h.current == h.original_image
    assert h.override.read_bytes() == h.original_bytes


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("mode", ["locked", "unlocked"])
def test_failed_candidate_restores_exact_previous_override_and_image(tmp_path, monkeypatch, path, mode):
    h = Harness(tmp_path, monkeypatch, "ready", path=path, lock_value="off" if mode == "locked" else "on")
    with pytest.raises(RuntimeError):
        lock.execute(h, h.helper, mode)
    assert h.current == h.original_image
    assert h.override.read_bytes() == h.original_bytes
    assert ("mark", "recovered") in h.events
    assert ("mark", "deployed") not in h.events
    assert h.events[-2:] == [("start",), ("mark", "recovered")]


@pytest.mark.parametrize("path", PATHS)
def test_failed_recovery_force_closes_ingress(tmp_path, monkeypatch, capsys, path):
    h = Harness(tmp_path, monkeypatch, "ready", path=path)
    stop = h.stop_writers
    calls = []
    def stop_then_fail(strict):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("synthetic recovery failure")
        stop(strict)
    h.stop_writers = stop_then_fail
    with pytest.raises(RuntimeError):
        lock.execute(h, h.helper, "unlocked")
    assert ("close",) in h.events
    out = capsys.readouterr().out
    assert "LANGUAGE_LOCK_RECOVERY_UNVERIFIED" in out
    assert "LANGUAGE_LOCK_PREVIOUS_APP_RESTORED" not in out


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("variant", [0, 1])
def test_each_accepted_source_variant_deploys(tmp_path, monkeypatch, path, variant):
    h = Harness(tmp_path, monkeypatch, path=path)
    h.variant = variant
    lock.execute(h, h.helper, "unlocked")
    assert ("mark", "deployed") in h.events


# ---- refusals ----

def test_other_running_image_cannot_start_build_or_cutover(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    other = "sha256:" + "c" * 64
    h.revisions[other] = PREVIOUS
    h.current = other
    h.original["services"]["app"]["image"] = other
    h.override.write_text(json.dumps(h.original))
    with pytest.raises(RuntimeError, match="unexpected predecessor image"):
        lock.execute(h, h.helper, "locked")
    assert no_build_or_cutover(h)
    assert ("stop",) not in h.events


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("revision", ["unknown", "3d70fd1d461b809aec9ecbf8033aa4d85df13d49",
                                      "9dda99b498f3823c8430ca73cc4a38c3ad00b9a1", "", "c" * 40])
def test_unknown_predecessor_revision_cannot_start_build_or_cutover(tmp_path, monkeypatch, path, revision):
    h = Harness(tmp_path, monkeypatch, path=path)
    h.revisions[h.current] = revision
    with pytest.raises(RuntimeError, match="unexpected predecessor image"):
        lock.execute(h, h.helper, "locked")
    assert no_build_or_cutover(h)
    assert ("stop",) not in h.events


def test_install_refuses_the_exact_predecessor_commit_given_as_the_release(tmp_path, monkeypatch):
    # Passing the predecessor itself as the release would otherwise look like a toggle.
    h = Harness(tmp_path, monkeypatch)
    h.commit = PREVIOUS
    with pytest.raises(RuntimeError):
        lock.execute(h, h.helper, "locked")
    assert no_build_or_cutover(h)


@pytest.mark.parametrize("path", PATHS)
def test_override_image_must_match_the_running_image(tmp_path, monkeypatch, path):
    h = Harness(tmp_path, monkeypatch, path=path)
    h.original["services"]["app"]["image"] = "sha256:" + "e" * 64
    h.override.write_text(json.dumps(h.original))
    with pytest.raises(RuntimeError, match="predecessor override"):
        lock.execute(h, h.helper, "locked")
    assert no_build_or_cutover(h)


@pytest.mark.parametrize("path", PATHS)
def test_wrong_effective_predecessor_source_cannot_start_cutover(tmp_path, monkeypatch, path):
    h = Harness(tmp_path, monkeypatch, "variant", path=path)
    with pytest.raises(RuntimeError, match="predecessor source differs"):
        lock.execute(h, h.helper, "locked")
    assert no_build_or_cutover(h)


@pytest.mark.parametrize("path", PATHS)
def test_each_path_accepts_only_its_own_running_sources(tmp_path, monkeypatch, path):
    h = Harness(tmp_path, monkeypatch, path=path)
    wrong = h.running_sources
    other = {**wrong(), **lock.SOURCES} if path == "install" else lock.SUPPORTED_BASELINES[PREVIOUS][1]
    h.running_sources = lambda: other
    with pytest.raises(RuntimeError, match="predecessor source differs"):
        lock.execute(h, h.helper, "locked")
    assert no_build_or_cutover(h)


@pytest.mark.parametrize("path", PATHS)
def test_wrong_effective_source_in_the_probe_cannot_start_cutover(tmp_path, monkeypatch, path):
    h = Harness(tmp_path, monkeypatch, path=path)
    h.helper.source_probe_code = lambda hashes, raw: json.dumps({"hashes": {}, "settings": h.values})
    with pytest.raises(RuntimeError, match="effective source"):
        lock.execute(h, h.helper, "locked")
    assert no_build_or_cutover(h)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("key,value,message", [
    ("elevenlabs_model", "eleven_multilingual_v2", "voice model"),
    ("openai_model", "gpt-4o", "Luna"),
    ("verified_phone_booking_enabled", True, "restored booking flow"),
    ("elevenlabs_model", None, "voice model"),
])
def test_wrong_settings_cannot_start_build_or_cutover(tmp_path, monkeypatch, path, key, value, message):
    h = Harness(tmp_path, monkeypatch, path=path)
    h.values[key] = value
    with pytest.raises(RuntimeError, match=message):
        lock.execute(h, h.helper, "locked")
    assert no_build_or_cutover(h)


@pytest.mark.parametrize("path", PATHS)
def test_settings_changing_during_preparation_blocks_cutover(tmp_path, monkeypatch, path):
    h = Harness(tmp_path, monkeypatch, path=path)
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
        lock.execute(h, h.helper, "unlocked")
    assert not h.cutover


@pytest.mark.parametrize("path", PATHS)
def test_future_unresolved_operation_blocks_before_preparation(tmp_path, monkeypatch, path):
    h = Harness(tmp_path, monkeypatch, path=path)
    run = h.run
    def pending(*args, **kwargs):
        if args[-1] == lock.FUTURE_OPERATIONS_PROBE:
            return SimpleNamespace(stdout=b'{"unresolved_future_operations":1}')
        return run(*args, **kwargs)
    h.run = pending
    with pytest.raises(RuntimeError, match="unresolved future"):
        lock.execute(h, h.helper, "unlocked")
    assert no_build_or_cutover(h)
    assert not any(e[:2] == ("docker", "run") for e in h.events)
    assert not (h.directory / "previous-override.json").exists()


@pytest.mark.parametrize("path", PATHS)
def test_new_unresolved_operation_after_writer_stop_restores_previous_app(tmp_path, monkeypatch, path):
    h = Harness(tmp_path, monkeypatch, path=path)
    compose = h.compose
    def late(path_, *args, **kwargs):
        if args[-1] == lock.FUTURE_OPERATIONS_PROBE:
            return SimpleNamespace(stdout=b'{"unresolved_future_operations":1}')
        return compose(path_, *args, **kwargs)
    h.compose = late
    with pytest.raises(RuntimeError, match="unresolved future"):
        lock.execute(h, h.helper, "unlocked")
    assert h.current == h.original_image
    assert h.override.read_bytes() == h.original_bytes
    assert ("mark", "recovered") in h.events
    assert ("mark", "deployed") not in h.events


# ---- arguments ----

BAD_MODES = ["", "Locked", "UNLOCKED", "lock", "unlock", "on", "off", " locked", "unlocked ",
             "locked\n", "true", "none", None, 1]


@pytest.mark.parametrize("mode", BAD_MODES)
def test_invalid_mode_is_refused_before_anything_runs(tmp_path, monkeypatch, mode):
    h = Harness(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        lock.execute(h, h.helper, mode)
    assert h.events == []
    assert h.override.read_bytes() == h.original_bytes


@pytest.mark.parametrize("mode", BAD_MODES)
def test_invalid_mode_is_refused_before_main_touches_the_server(monkeypatch, mode):
    def forbidden(*args, **kwargs):
        raise AssertionError("main must refuse the mode first")
    monkeypatch.setattr(lock, "load_helper", forbidden)
    monkeypatch.setattr(lock.tempfile, "mkdtemp", forbidden)
    monkeypatch.setattr(lock.os, "open", forbidden)
    with pytest.raises(ValueError):
        lock.main("a" * 40, mode)


@pytest.mark.parametrize("arguments", [["a" * 40, "bogus"], ["a" * 40], ["a" * 40, "locked", "x"], []])
def test_script_entry_point_refuses_bad_arguments_with_the_stop_marker(arguments):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/deploy-language-lock.py"), *arguments],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert result.stdout.splitlines() == ["LANGUAGE_LOCK_STOPPED_KEEP_PROTECTED_FILES"]


@pytest.mark.parametrize("mode", ["locked", "unlocked"])
def test_valid_modes_map_to_exactly_one_environment_value(mode):
    assert lock.lock_environment(mode) == {LOCK_ENV: {"locked": "on", "unlocked": "off"}[mode]}


# ---- override and contract helpers ----

def test_controlled_override_changes_only_the_image_and_the_lock_value():
    original = {"services": {"app": {"image": "old", "environment": {
        "KEEP": "value", "elevenlabs_stt_language_lock": "off"}}, "redis": {"image": "same"}}}
    changed = lock.changed_override(original, "candidate", "locked")
    assert original["services"]["app"]["image"] == "old"
    assert changed == {"services": {"app": {"image": "candidate", "environment": {
        "KEEP": "value", LOCK_ENV: "on"}}, "redis": {"image": "same"}}}
    assert lock.changed_override(original, None, "unlocked")["services"]["app"]["image"] == "old"


def test_override_environment_must_be_a_mapping():
    with pytest.raises(RuntimeError):
        lock.changed_override({"services": {"app": {"image": "x", "environment": ["A=1"]}}}, None, "locked")


def test_expected_container_contract_replaces_only_the_lock_entry():
    previous = {"Config": {"Env": ["B=2", "A=1", "elevenlabs_stt_language_lock=off"], "Cmd": ["x"]},
                "Mounts": [{"Destination": "/d"}], "NetworkSettings": {"Networks": {"n": {}}},
                "HostConfig": {"PortBindings": {}}}
    base = lock.contract(previous)
    changed = lock.lock_contract(previous, "locked")
    assert changed["environment"] == ["A=1", "B=2", LOCK_ENV + "=on"]
    assert {k: v for k, v in changed.items() if k != "environment"} == {
        k: v for k, v in base.items() if k != "environment"}
    assert lock.lock_contract(previous, "unlocked")["environment"][-1] == LOCK_ENV + "=off"


# ---- pinned hashes ----

def test_pinned_hashes_match_the_files_in_the_working_tree():
    assert set(lock.SOURCES) == {lock.STT_PATH}
    assert digest((ROOT / lock.STT_PATH).read_bytes()) == lock.SOURCES[lock.STT_PATH]
    assert digest((ROOT / lock.PROBE_PATH).read_bytes()) == lock.PROBE_SHA
    assert lock.SOURCES[lock.STT_PATH] != repair.SOURCES[lock.STT_PATH]


def test_pinned_predecessor_is_the_deployed_reply_pause_release():
    assert lock.PREDECESSOR_REVISION == "84a9f975eda54c2003e41b8b55e74dff5d8d7aba"
    assert lock.PREDECESSOR_IMAGE == "sha256:9c8fb06735b311f083f20873ef0a825a43da9ef1c34561976f6adbb568d17d35"
    assert lock.BASE == repair.BASE
    assert lock.MANIFEST_SHA == repair.MANIFEST_SHA


def test_supported_baselines_are_recomputed_from_the_reply_pause_release():
    expected = {PREVIOUS: [{**variant, **repair.SOURCES}
                           for variant in repair.SUPPORTED_BASELINES[repair.ACCEPTED_REVISION]]}
    assert lock.SUPPORTED_BASELINES == expected
    assert len(expected[PREVIOUS]) == 2
    for variant in expected[PREVIOUS]:
        assert variant[repair.STT_PATH] == repair.SOURCES[repair.STT_PATH]
        assert variant[repair.ORCHESTRATOR] == repair.SOURCES[repair.ORCHESTRATOR]


def test_release_files_use_unix_line_endings():
    for path in ("scripts/deploy-language-lock.py", "scripts/probe-language-lock.py",
                 "tests/integration/test_language_lock_release.py", "tests/stt/test_language_lock.py"):
        assert b"\r" not in (ROOT / path).read_bytes(), path


# ---- probe behaviour (installed-image checks, run here against this source tree) ----

@pytest.fixture
def probe(monkeypatch):
    module = load("language_lock_probe", "scripts/probe-language-lock.py")
    monkeypatch.delenv(module.ENV_NAME, raising=False)
    for key, value in module.SYNTHETIC_ENV.items():
        monkeypatch.setenv(key, value)
    return module


def test_offline_probe_passes_for_this_source_tree_and_restores_the_environment(probe, monkeypatch):
    monkeypatch.setenv(probe.ENV_NAME, "unlocked")
    result = asyncio.run(probe.offline())
    assert result == {"status": "LANGUAGE_LOCK_OK", "mode": "offline"}
    assert os.environ[probe.ENV_NAME] == "unlocked"


def test_offline_probe_rejects_a_service_that_always_sends_the_language(probe, monkeypatch):
    from services.stt.elevenlabs_stt_service import ElevenLabsSTT
    original = ElevenLabsSTT._connection_url
    def always(self):
        url = original(self)
        return url if "language_code" in url or not self.language else url + "&language_code=" + self.language
    monkeypatch.setattr(ElevenLabsSTT, "_connection_url", always)
    with pytest.raises(probe.Failure) as error:
        asyncio.run(probe.offline())
    assert error.value.reason == "url_language"


def test_offline_probe_rejects_a_service_that_reconnects_while_unlocked(probe, monkeypatch):
    from services.stt.elevenlabs_stt_service import ElevenLabsSTT
    async def always_replace(self, language_code):
        self.language = language_code
        await self._replace_transport_locked(self._terminal_revision)
        return True
    monkeypatch.setattr(ElevenLabsSTT, "reconnect_with_language", always_replace)
    with pytest.raises(probe.Failure) as error:
        asyncio.run(probe.offline())
    assert error.value.reason in ("transport_replacement", "inactive_session")


def test_probe_main_prints_one_status_line_without_secrets(probe, monkeypatch, capsys):
    async def failing():
        raise probe.Failure("connect")
    monkeypatch.setattr(probe, "offline", failing)
    monkeypatch.setattr(sys, "argv", ["probe"])
    with pytest.raises(SystemExit) as exit_info:
        probe.main()
    assert exit_info.value.code == 1
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ['{"status": "LANGUAGE_LOCK_FAILED", "mode": "offline", "reason": "connect"}']


def test_probe_main_success_exit_code(probe, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["probe"])
    with pytest.raises(SystemExit) as exit_info:
        probe.main()
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == '{"status": "LANGUAGE_LOCK_OK", "mode": "offline"}'


@pytest.fixture
def live_probe(probe, monkeypatch):
    """The live probe with a fake provider session; the real service builds the URL."""
    import services.stt as stt_package
    from services.stt.elevenlabs_stt_service import ElevenLabsSTT, STTTransportOutcome
    try:
        import config.settings as settings_module
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip("settings unavailable: " + type(error).__name__)
    log = SimpleNamespace(disconnects=0, languages=[], connects=0, outcome=STTTransportOutcome.SESSION_STARTED)

    class Fake(ElevenLabsSTT):
        async def connect(self):
            log.connects += 1
            return True

        @property
        def transport_outcome(self):
            return log.outcome

        async def disconnect(self):
            log.disconnects += 1

    def create(provider, config):
        log.languages.append(config["elevenlabs_stt_language"])
        return Fake(api_key="synthetic", language=config["elevenlabs_stt_language"])

    monkeypatch.setattr(settings_module, "settings", SimpleNamespace(
        stt_provider="elevenlabs", model_dump=lambda: {}))
    monkeypatch.setattr(stt_package, "create_stt_service", create)
    return probe, log, Fake


@pytest.mark.parametrize("value,lock_name", [(None, "locked"), ("on", "locked"), ("off", "unlocked")])
def test_live_probe_reports_the_mode_read_from_the_environment(live_probe, monkeypatch, value, lock_name):
    probe, log, _ = live_probe
    if value is not None:
        monkeypatch.setenv(probe.ENV_NAME, value)
    result = asyncio.run(probe.live())
    assert result == {"status": "LANGUAGE_LOCK_OK", "mode": "live", "lock": lock_name}
    assert log.languages == ["en"]
    assert log.connects == 1 and log.disconnects == 1


def test_live_probe_fails_when_the_url_disagrees_with_the_mode_and_still_disconnects(live_probe, monkeypatch):
    probe, log, Fake = live_probe
    monkeypatch.setenv(probe.ENV_NAME, "off")
    monkeypatch.setattr(Fake, "_connection_url", lambda self: "wss://example.invalid/?language_code=en")
    with pytest.raises(probe.Failure) as error:
        asyncio.run(probe.live())
    assert error.value.reason == "url_language"
    assert log.connects == 0 and log.disconnects == 1


def test_live_probe_fails_when_locked_url_lacks_the_language(live_probe, monkeypatch):
    probe, log, Fake = live_probe
    monkeypatch.setenv(probe.ENV_NAME, "on")
    monkeypatch.setattr(Fake, "_connection_url", lambda self: "wss://example.invalid/?model_id=x")
    with pytest.raises(probe.Failure) as error:
        asyncio.run(probe.live())
    assert error.value.reason == "url_language"


def test_live_probe_reports_a_session_that_never_starts(live_probe, monkeypatch):
    from services.stt.elevenlabs_stt_service import STTTransportOutcome
    probe, log, _ = live_probe
    log.outcome = STTTransportOutcome.REMOTE_CLOSED
    with pytest.raises(probe.Failure) as error:
        asyncio.run(probe.live())
    assert error.value.reason == "rejected"
    assert log.disconnects == 1


@pytest.mark.parametrize("container_env", [[], [LOCK_ENV + "=on"], ["elevenlabs_stt_language_lock=off"]])
def test_override_alone_does_not_count_as_already_active(tmp_path, monkeypatch, capsys, container_env):
    """The file says unlocked but the running container does not: recreate it."""
    h = Harness(tmp_path, monkeypatch, path="toggle", lock_value="off")
    h.container_lock = list(container_env)
    assert not lock.container_lock_matches(h.inspect_once(h.app_container), "unlocked")
    try:
        lock.execute(h, h.helper, "unlocked")
    except RuntimeError:
        pass  # the synthetic harness may refuse later; what matters is no early exit
    assert "LANGUAGE_LOCK_ALREADY_UNLOCKED" not in capsys.readouterr().out.splitlines()
