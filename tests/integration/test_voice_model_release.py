"""Offline release boundary checks for the voice switch; no Docker or provider access."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("voice_model", ROOT / "scripts/deploy-voice-model.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)

OLD_IMAGE = repair.ACCEPTED_IMAGE
REVISION = "3d70fd1d461b809aec9ecbf8033aa4d85df13d49"
VOICE = "NewVoice0123456789ab"
OLD_ENV = {"elevenlabs_model": "eleven_multilingual_v2", "elevenlabs_voice_id": "OldVoice0123456789ab"}
NEW_ENV = {"elevenlabs_model": "eleven_v4_turbo", "elevenlabs_voice_id": VOICE}


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
                       "secret": "synthetic"}
        self.original = {"services": {"app": {"image": self.current,
            "environment": {"private": "synthetic", "OPENAI_MODEL": "gpt-6-luna",
                            "ELEVENLABS_VOICE_ID": OLD_ENV["elevenlabs_voice_id"]},
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
        if name.startswith("ai-phone-voice-model-parent:"):
            return {"Id": "wrong" if self.fault == "parent" else OLD_IMAGE}
        if name.startswith("ai-phone-voice-model:"):
            return {"Id": "candidate", "Config": {"Labels": {
                "org.opencontainers.image.revision": self.commit}}}
        environment = ["A=1", "OPENAI_MODEL=gpt-6-luna"]
        if self.old():
            environment.append("ELEVENLABS_VOICE_ID=" + OLD_ENV["elevenlabs_voice_id"])
        else:
            environment += ["ELEVENLABS_MODEL=eleven_v4_turbo", "ELEVENLABS_VOICE_ID=" + VOICE]
        return {"Image": self.current, "Config": {"Env": environment,
            "Labels": {"org.opencontainers.image.revision": REVISION if self.old() else self.commit,
                       "com.docker.compose.project": self.project_name}},
            "Mounts": [{"Destination": "/app/private", "Source": "keep"}],
            "NetworkSettings": {"Networks": {"net": {}}}, "HostConfig": {"PortBindings": {}}}

    def settings(self):
        return {**self.values, **(OLD_ENV if self.old() else NEW_ENV)}

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
            value = (b'{"unresolved_future_operations":0}' if args[-1] == repair.FUTURE_OPERATIONS_PROBE
                     else json.dumps(repair.SUPPORTED_BASELINES[REVISION][self.variant]).encode()
                     if "import hashlib,json; from pathlib" in args[-1]
                     else args[-1].encode())
        elif args[:2] == ("docker", "run"):
            if self.fault == "request":
                raise RuntimeError("synthetic request probe failure")
            value = b'{"status":"VOICE_MODEL_OK","mode":"offline"}'
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
                failed = self.fault == "live-voice"
                return SimpleNamespace(returncode=int(failed), stdout=json.dumps(
                    {"status": "VOICE_MODEL_FAILED", "mode": "live", "reason": "request",
                     "http_status": 400} if failed else
                    {"status": "VOICE_MODEL_OK", "mode": "live", "audio_seconds": 4.0}).encode())
            if args[-1] == repair.FUTURE_OPERATIONS_PROBE:
                return SimpleNamespace(stdout=b'{"unresolved_future_operations":0}')
            value = json.loads(args[-1])
            candidate = path.name == "candidate-override.json"
            if self.fault == "settings" and candidate:
                candidate = False  # the new environment did not reach the settings
            value["settings"].update(NEW_ENV if candidate else OLD_ENV)
            return SimpleNamespace(stdout=json.dumps(value).encode())
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


def test_success_changes_only_image_model_and_voice(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    repair.execute(h, h.helper, VOICE)
    expected = copy.deepcopy(h.original)
    expected["services"]["app"]["image"] = "candidate"
    expected["services"]["app"]["environment"]["ELEVENLABS_MODEL"] = "eleven_v4_turbo"
    expected["services"]["app"]["environment"]["ELEVENLABS_VOICE_ID"] = VOICE
    assert json.loads((tmp_path / "docker-compose.override.yml").read_bytes()) == expected
    assert set(p.name for p in (h.directory / "build").iterdir()) == {
        "Dockerfile", "elevenlabs_service.py"}
    assert ("mark", "deployed") in h.events
    live_index = next(i for i, e in enumerate(h.events) if e[0] == "compose" and e[-1] == "--live")
    assert live_index < h.events.index(("mark", "cutover-started"))


def test_build_uses_verified_local_tag_not_bare_image_id(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    repair.execute(h, h.helper, VOICE)
    dockerfile = (h.directory / "build/Dockerfile").read_text()
    assert not dockerfile.startswith("FROM sha256:")
    parent = dockerfile.splitlines()[0].removeprefix("FROM ")
    assert ("docker", "tag", OLD_IMAGE, parent) in h.events
    assert ("docker", "image", "rm", parent) in h.events


@pytest.mark.parametrize("fault", ["source", "request", "compose", "parent", "build",
                                   "settings", "live-voice"])
def test_preparation_failure_does_not_stop_app(tmp_path, monkeypatch, fault):
    h = Harness(tmp_path, monkeypatch, fault)
    before = (tmp_path / "docker-compose.override.yml").read_bytes()
    with pytest.raises(RuntimeError):
        repair.execute(h, h.helper, VOICE)
    assert not h.cutover
    assert ("stop",) not in h.events
    assert h.current == OLD_IMAGE
    assert (tmp_path / "docker-compose.override.yml").read_bytes() == before


def test_failed_candidate_restores_exact_previous_override_and_image(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, "ready")
    before = (tmp_path / "docker-compose.override.yml").read_bytes()
    with pytest.raises(RuntimeError):
        repair.execute(h, h.helper, VOICE)
    assert h.current == OLD_IMAGE
    assert (tmp_path / "docker-compose.override.yml").read_bytes() == before
    assert ("mark", "recovered") in h.events
    assert ("mark", "deployed") not in h.events
    assert h.events[-2:] == [("start",), ("mark", "recovered")]


@pytest.mark.parametrize("variant", [0, 1])
def test_each_accepted_source_variant_deploys(tmp_path, monkeypatch, variant):
    h = Harness(tmp_path, monkeypatch)
    h.variant = variant
    repair.execute(h, h.helper, VOICE)
    assert ("mark", "deployed") in h.events


def test_other_running_image_cannot_start_build_or_cutover(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    other = "sha256:" + "c" * 64
    h.current = other
    h.old = lambda: True
    h.original["services"]["app"]["image"] = other
    (tmp_path / "docker-compose.override.yml").write_text(json.dumps(h.original))
    with pytest.raises(RuntimeError, match="accepted running image"):
        repair.execute(h, h.helper, VOICE)
    assert not h.cutover and not any(e[:2] == ("docker", "build") for e in h.events)


def test_unknown_predecessor_revision_cannot_start_build_or_cutover(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    inspect = h.inspect_once
    def wrong_revision(name):
        value = inspect(name)
        if name == h.app_container:
            value["Config"]["Labels"]["org.opencontainers.image.revision"] = "unknown"
        return value
    h.inspect_once = wrong_revision
    with pytest.raises(RuntimeError, match="predecessor image"):
        repair.execute(h, h.helper, VOICE)
    assert not h.cutover and not any(e[:2] == ("docker", "build") for e in h.events)


def test_wrong_effective_predecessor_source_cannot_start_cutover(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    h.helper.source_probe_code = lambda hashes, raw: json.dumps({"hashes": {}, "settings": h.values})
    with pytest.raises(RuntimeError, match="effective source"):
        repair.execute(h, h.helper, VOICE)
    assert not h.cutover and not any(e[:2] == ("docker", "build") for e in h.events)


def test_future_unresolved_operation_blocks_before_build(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    run = h.run
    def pending(*args, **kwargs):
        if args[-1] == repair.FUTURE_OPERATIONS_PROBE:
            return SimpleNamespace(stdout=b'{"unresolved_future_operations":1}')
        return run(*args, **kwargs)
    h.run = pending
    with pytest.raises(RuntimeError, match="unresolved future"):
        repair.execute(h, h.helper, VOICE)
    assert not h.cutover
    assert not any(e[:2] == ("docker", "build") for e in h.events)


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
        repair.execute(h, h.helper, VOICE)
    assert h.current == OLD_IMAGE
    assert (tmp_path / "docker-compose.override.yml").read_bytes() == before
    assert ("mark", "recovered") in h.events
    assert ("mark", "deployed") not in h.events


@pytest.mark.parametrize("voice", ["", "short", "has space 0123456789", "x" * 41,
                                   "abc;rm -rf 0123456789", None])
def test_invalid_voice_id_is_refused_before_anything_runs(tmp_path, monkeypatch, voice):
    h = Harness(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="voice ID"):
        repair.execute(h, h.helper, voice)
    assert h.events == []


def test_controlled_override_changes_only_image_model_and_voice():
    original = {"services": {"app": {"image": "old", "environment": {
        "KEEP": "value", "elevenlabs_model": "eleven_multilingual_v2",
        "ELEVENLABS_VOICE_ID": "old"}}, "redis": {"image": "same"}}}
    changed = repair.changed_override(original, "candidate", VOICE)
    assert original["services"]["app"]["environment"]["elevenlabs_model"] == "eleven_multilingual_v2"
    assert changed == {"services": {"app": {"image": "candidate", "environment": {
        "KEEP": "value", "ELEVENLABS_MODEL": "eleven_v4_turbo",
        "ELEVENLABS_VOICE_ID": VOICE}}, "redis": {"image": "same"}}}


def test_expected_container_environment_replaces_only_the_two_voice_values():
    previous = {"Config": {"Env": ["A=1", "ELEVENLABS_VOICE_ID=old", "elevenlabs_model=x"]},
                "Mounts": [], "NetworkSettings": {"Networks": {}}, "HostConfig": {}}
    assert repair.voice_contract(previous, VOICE)["environment"] == sorted(
        ["A=1", "ELEVENLABS_MODEL=eleven_v4_turbo", "ELEVENLABS_VOICE_ID=" + VOICE])


def test_pinned_sources_match_the_reviewed_files():
    for path, expected in {**repair.SOURCES, repair.PROBE_PATH: repair.PROBE_SHA}.items():
        assert repair.digest((ROOT / path).read_bytes()) == expected
    for variant in repair.SUPPORTED_BASELINES[REVISION]:
        assert variant["services/conversation/restored_turn_admission.py"]
        assert "services/tts/elevenlabs_service.py" not in variant
