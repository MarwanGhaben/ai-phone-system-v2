"""Offline release boundary checks; no Docker or provider access."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("llm_repair", ROOT / "scripts/deploy-verified-llm-repair.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


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
        self.current = repair.OLD_IMAGE
        self.cutover = False
        self.source = (ROOT / repair.SOURCE).read_bytes()
        self.baseline = self.source.replace(b"\r\n", b"\n").replace(
            b'                    # The pinned 1.10 SDK predates this named parameter, but\n'
            b'                    # extra_body forwards it unchanged to the API request.\n'
            b'                    kwargs["extra_body"] = {"parallel_tool_calls": False}\n',
            b'                    kwargs["parallel_tool_calls"] = False\n')
        self.values = {"verified_phone_booking_enabled": True, "secret": "synthetic"}
        self.original = {"services": {"app": {"image": self.current,
            "environment": {"private": "synthetic"}, "volumes": ["keep:/app/private"]},
            "migrate": {"image": self.current}}}
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
            source_probe_code=lambda hashes, raw: json.dumps({"hashes": {**hashes, **raw}, "settings": self.values}))

    def inspect_once(self, name):
        if name.startswith("ai-phone-llm-repair:"):
            return {"Id": "candidate"}
        return {"Image": self.current, "Config": {"Env": ["A=1"],
            "Labels": {"org.opencontainers.image.revision": repair.BASE if self.current == repair.OLD_IMAGE else self.commit,
                       "com.docker.compose.project": self.project_name}},
            "Mounts": [{"Destination": "/app/private", "Source": "keep"}],
            "NetworkSettings": {"Networks": {"net": {}}}, "HostConfig": {"PortBindings": {}}}

    def settings(self):
        return self.values

    def wait_ready(self):
        if self.fault == "ready" and self.current == "candidate":
            raise RuntimeError("synthetic readiness failure")

    def public_health(self):
        pass

    def git_bytes(self, path):
        return self.source + (b"# unreviewed" if self.fault == "source" else b"")

    def run(self, *args, **kwargs):
        self.events.append(args)
        if args[0] == "git":
            value = self.baseline if args[-1].endswith(repair.SOURCE) else (ROOT / repair.MANIFEST).read_bytes()
        elif args[:2] == ("docker", "exec"):
            value = args[-1].encode()
        elif args[:2] == ("docker", "run"):
            if self.fault == "sdk":
                raise RuntimeError("synthetic SDK failure")
            value = b"SDK_OFFLINE_TRANSPORT_OK"
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


def test_success_changes_only_app_image_and_build_has_no_private_data(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    repair.execute(h, h.helper)
    expected = copy.deepcopy(h.original)
    expected["services"]["app"]["image"] = "candidate"
    assert json.loads((tmp_path / "docker-compose.override.yml").read_bytes()) == expected
    assert set(p.name for p in (h.directory / "build").iterdir()) == {"Dockerfile", "openai_service.py"}
    assert ("mark", "deployed") in h.events
    assert not any("--prepare-operations" in e for e in h.events)


@pytest.mark.parametrize("fault", ["source", "sdk", "compose"])
def test_preparation_failure_does_not_stop_app(tmp_path, monkeypatch, fault):
    h = Harness(tmp_path, monkeypatch, fault)
    with pytest.raises(RuntimeError):
        repair.execute(h, h.helper)
    assert not h.cutover
    assert ("stop",) not in h.events
    assert h.current == repair.OLD_IMAGE


def test_failed_candidate_restores_exact_previous_override_and_image(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, "ready")
    before = (tmp_path / "docker-compose.override.yml").read_bytes()
    with pytest.raises(RuntimeError):
        repair.execute(h, h.helper)
    assert h.current == repair.OLD_IMAGE
    assert (tmp_path / "docker-compose.override.yml").read_bytes() == before
    assert ("mark", "recovered") in h.events
    assert ("mark", "deployed") not in h.events
    assert h.events[-2:] == [("start",), ("mark", "recovered")]


def test_contract_compares_mount_contents_not_order():
    value = {"Config": {"Env": ["B=2", "A=1"]}, "Mounts": [{"a": 1}, {"b": 2}],
             "NetworkSettings": {"Networks": {"one": {}, "two": {}}}, "HostConfig": {}}
    changed = copy.deepcopy(value)
    changed["Mounts"].reverse()
    changed["Config"]["Env"].reverse()
    assert repair.contract(value) == repair.contract(changed)
    changed["Mounts"][0]["b"] = 3
    assert repair.contract(value) != repair.contract(changed)
