"""Offline release boundary checks; no Docker or provider access."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("restored_availability", ROOT / "scripts/deploy-zero-interval-repair.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


TEST_IMAGE = "sha256:" + "a" * 64


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
        self.current = TEST_IMAGE
        self.cutover = False
        self.predecessor = 'e3f96437fcb1b4af57ef82c9be8752371ea983d9'
        self.variant = 1
        self.values = {"verified_phone_booking_enabled": False, "openai_model": "gpt-6-luna", "secret": "synthetic"}
        self.original = {"services": {"app": {"image": self.current,
            "environment": {"private": "synthetic", "VERIFIED_PHONE_BOOKING_ENABLED": "false", "OPENAI_MODEL": "gpt-6-luna"}, "volumes": ["keep:/app/private"]},
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
            source_probe_code=lambda hashes, raw: json.dumps({"hashes": {**hashes, **raw}, "settings": self.settings()}))

    def inspect_once(self, name):
        if name.startswith("ai-phone-zero-interval-repair-parent:"):
            return {"Id": "wrong" if self.fault == "parent" else TEST_IMAGE}
        if name.startswith("ai-phone-zero-interval-repair:"):
            return {"Id": "candidate", "Config": {"Labels": {
                "org.opencontainers.image.revision": self.commit}}}
        return {"Image": self.current, "Config": {"Env": ["A=1", "VERIFIED_PHONE_BOOKING_ENABLED=false", "OPENAI_MODEL=" + ("gpt-6-luna" if self.current == TEST_IMAGE else "gpt-6-luna")],
            "Labels": {"org.opencontainers.image.revision": self.predecessor if self.current == TEST_IMAGE else self.commit,
                       "com.docker.compose.project": self.project_name}},
            "Mounts": [{"Destination": "/app/private", "Source": "keep"}],
            "NetworkSettings": {"Networks": {"net": {}}}, "HostConfig": {"PortBindings": {}}}

    def settings(self):
        return {**self.values, "openai_model": "gpt-6-luna" if self.current == TEST_IMAGE else "gpt-6-luna"}

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
            value = subprocess.check_output(["git", "show", args[-1]], cwd=ROOT)
            if self.fault == "source" and args[-1].startswith(repair.RESTORE + ":"):
                value += b"# unreviewed"
        elif args[:2] == ("docker", "exec"):
            value = (b'{"unresolved_future_operations":0}' if args[-1] == repair.FUTURE_OPERATIONS_PROBE
                     else json.dumps(repair.SUPPORTED_BASELINES[self.predecessor][self.variant]).encode() if "import hashlib,json; from pathlib" in args[-1]
                     else args[-1].encode())
        elif args[:2] == ("docker", "run"):
            if self.fault == "sdk":
                raise RuntimeError("synthetic SDK failure")
            value = b'ZERO_INTERVAL_OFFLINE_OK'
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
            if args[-1] == repair.LIVE_CALENDAR_PROBE:
                failed = self.fault == 'live-api'
                return SimpleNamespace(returncode=int(failed), stdout=json.dumps({
                    'status':'LIVE_CALENDAR_READS_FAILED' if failed else 'LIVE_CALENDAR_READS_OK',
                    'checks':[{'consultant':'Rami','slot_count':4},{'consultant':'Abdul','slot_count':6}]}).encode())
            if args[-1] == repair.FUTURE_OPERATIONS_PROBE:
                return SimpleNamespace(stdout=b'{"unresolved_future_operations":0}')
            value = json.loads(args[-1])
            value["settings"]["openai_model"] = "gpt-6-luna" if path.name == "candidate-override.json" else "gpt-6-luna"
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


def test_success_changes_only_app_image_and_build_has_no_private_data(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    repair.execute(h, h.helper)
    expected = copy.deepcopy(h.original)
    expected["services"]["app"]["image"] = "candidate"
    expected["services"]["app"]["environment"]["OPENAI_MODEL"] = "gpt-6-luna"
    assert json.loads((tmp_path / "docker-compose.override.yml").read_bytes()) == expected
    assert set(p.name for p in (h.directory / "build").iterdir()) == {
        "Dockerfile", "legacy_availability.py"}
    assert ("mark", "deployed") in h.events
    assert not any("--prepare-operations" in e for e in h.events)


def test_build_uses_verified_local_tag_not_bare_image_id(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    repair.execute(h, h.helper)
    dockerfile = (h.directory / "build/Dockerfile").read_text()
    assert not dockerfile.startswith("FROM sha256:")
    parent = dockerfile.splitlines()[0].removeprefix("FROM ")
    assert ("docker", "tag", TEST_IMAGE, parent) in h.events
    assert ("docker", "image", "rm", parent) in h.events


@pytest.mark.parametrize("fault", ["source", "sdk", "compose", "parent", "build", "live-api"])
def test_preparation_failure_does_not_stop_app(tmp_path, monkeypatch, fault):
    h = Harness(tmp_path, monkeypatch, fault)
    with pytest.raises(RuntimeError):
        repair.execute(h, h.helper)
    assert not h.cutover
    assert ("stop",) not in h.events
    assert h.current == TEST_IMAGE
    if fault in ("parent", "build"):
        assert ("docker", "image", "rm", "ai-phone-zero-interval-repair-parent:" + h.directory.name) in h.events
    if fault == "parent":
        assert not any(e[:2] == ("docker", "build") for e in h.events)


def test_failed_candidate_restores_exact_previous_override_and_image(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, "ready")
    before = (tmp_path / "docker-compose.override.yml").read_bytes()
    with pytest.raises(RuntimeError):
        repair.execute(h, h.helper)
    assert h.current == TEST_IMAGE
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
        repair.execute(h, h.helper)
    assert not h.cutover and not any(e[:2] == ("docker", "build") for e in h.events)


def test_wrong_effective_predecessor_source_cannot_start_cutover(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    h.helper.source_probe_code = lambda hashes, raw: json.dumps({"hashes": {}, "settings": h.values})
    with pytest.raises(RuntimeError, match="effective source"):
        repair.execute(h, h.helper)
    assert not h.cutover and not any(e[:2] == ("docker", "build") for e in h.events)


@pytest.mark.parametrize("revision", tuple(repair.SUPPORTED_BASELINES))
@pytest.mark.parametrize("variant", [0,1])
@pytest.mark.parametrize("fault", [None, "ready"])
def test_each_supported_predecessor_is_preserved_or_recovered_exactly(tmp_path, monkeypatch, revision, variant, fault):
    h = Harness(tmp_path, monkeypatch, fault)
    inspect = h.inspect_once
    def baseline(name):
        value = inspect(name)
        if name == h.app_container and h.current == TEST_IMAGE:
            value["Config"]["Labels"]["org.opencontainers.image.revision"] = revision
        return value
    h.inspect_once = baseline
    h.variant = variant
    h.predecessor = revision
    if fault:
        with pytest.raises(RuntimeError):
            repair.execute(h, h.helper)
        assert h.current == TEST_IMAGE and ("mark", "recovered") in h.events
        assert h.inspect_once(h.app_container)["Config"]["Labels"]["org.opencontainers.image.revision"] == revision
    else:
        repair.execute(h, h.helper)
        assert ("mark", "deployed") in h.events



@pytest.mark.skipif(os.environ.get("ZERO_INTERVAL_RUN_DOCKER_TESTS") != "1",
                   reason="opt-in offline Linux image build and restored booking flow")
def test_actual_build_and_parser_probe_in_linux(tmp_path, monkeypatch):
    def run(*args, timeout=90, check=True):
        result = subprocess.run(args, capture_output=True, timeout=timeout)
        if check and result.returncode:
            raise AssertionError((result.stdout + result.stderr).decode(errors="replace"))
        return result
    def inspect(name):
        return json.loads(run("docker", "inspect", name).stdout)[0]
    parent_name = os.environ.get("ZERO_INTERVAL_TEST_PARENT", "ai-phone-t049a-candidate:20260913")
    base_id = inspect(parent_name)["Id"]
    directory = tmp_path / ("graph-build-" + uuid.uuid4().hex[:10])
    directory.mkdir()
    parent_context = directory / "baseline"
    parent_context.mkdir()
    manifest = json.loads((ROOT / repair.MANIFEST).read_bytes())
    texts = {}
    for section in ("runtime", "configuration", "migrations", "protected_runtime"):
        texts.update(manifest[section])
    # The exact production image is remote. Prepare the accepted Git source on
    # the available dependency parent, then exercise the real candidate builder.
    for path in set(texts) | set(repair.SUPPORTED_BASELINES['e3f96437fcb1b4af57ef82c9be8752371ea983d9'][1]):
        target = parent_context / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if path == 'services/conversation/orchestrator.py':
            from tests.integration.test_restored_availability_release import repair as availability
            content = availability.patch_orchestrator(run('git', '-C', str(ROOT), 'show', availability.RESTORE + ':' + path).stdout)
        else:
            content = run('git', '-C', str(ROOT), 'show', 'e3f96437fcb1b4af57ef82c9be8752371ea983d9:' + path).stdout
        target.write_bytes(content)
    base_tag = "ai-phone-restore-call-test-base:" + directory.name
    prepared_tag = "ai-phone-restore-call-test-prepared:" + directory.name
    candidate_tag = "ai-phone-zero-interval-repair:" + directory.name
    parent_tag = "ai-phone-zero-interval-repair-parent:" + directory.name
    name = "graph-probe-" + directory.name
    try:
        run("docker", "tag", base_id, base_tag)
        assert inspect(base_tag)["Id"] == base_id
        (parent_context / "Dockerfile").write_text(
            "FROM " + base_tag + "\nCOPY source/ /app/\n", encoding="utf-8")
        run("docker", "build", "--network", "none", "--pull=false", "-t", prepared_tag,
            str(parent_context), timeout=180)
        release = SimpleNamespace(directory=directory, commit="b" * 40, run=run, inspect_once=inspect)
        candidate = repair.build_candidate(release, {path:(ROOT / path).read_bytes() for path in repair.SOURCES}, inspect(prepared_tag)['Id'])
        assert run("docker", "inspect", parent_tag, check=False).returncode != 0
        probe_source = (ROOT / repair.PROBE_PATH).read_text(encoding="utf-8")
        broken = run("docker", "run", "--rm", "--network", "none", "--name", name,
            "--entrypoint", "python", inspect(prepared_tag)['Id'], "-B", "-c", probe_source, check=False)
        assert broken.returncode != 0
        assert b'AssertionError: AVAILABILITY_UNVERIFIED' in broken.stderr
        result = run("docker", "run", "--rm", "--network", "none", "--name", name,
            "--entrypoint", "python", candidate, "-B", "-c", probe_source)
        assert result.stdout.strip() == b'ZERO_INTERVAL_OFFLINE_OK'
        program = "import hashlib,json; from pathlib import Path; print(json.dumps({p:hashlib.sha256(Path('/app',p).read_bytes().replace(b'\\r\\n',b'\\n')).hexdigest() for p in " + repr(repair.SOURCES) + "}))"
        result = run("docker", "run", "--rm", "--network", "none", "--name", name,
            "--entrypoint", "python", candidate, "-B", "-c", program)
        assert json.loads(result.stdout) == repair.SOURCES

        # The same restored English/Arabic caller handlers still execute unchanged.
        restored_probe = (ROOT / 'scripts/probe-restored-availability.py').read_text(encoding='utf-8')
        result=run('docker','run','--rm','--network','none','--name',name,
            '--entrypoint','python',candidate,'-B','-c',restored_probe)
        assert b'RESTORED_AVAILABILITY_OFFLINE_OK' in result.stdout
    finally:
        run("docker", "rm", "-f", name, check=False)
        for tag in (candidate_tag, parent_tag, prepared_tag, base_tag):
            run("docker", "image", "rm", tag, check=False)
            assert run("docker", "inspect", tag, check=False).returncode != 0
        assert run("docker", "inspect", name, check=False).returncode != 0


def test_future_unresolved_operation_blocks_before_build(tmp_path, monkeypatch):
    h=Harness(tmp_path,monkeypatch)
    run=h.run
    def pending(*args, **kwargs):
        if args[-1] == repair.FUTURE_OPERATIONS_PROBE:
            return SimpleNamespace(stdout=b'{"unresolved_future_operations":1}')
        return run(*args, **kwargs)
    h.run=pending
    with pytest.raises(RuntimeError, match="unresolved future"):
        repair.execute(h,h.helper)
    assert not h.cutover
    assert not any(e[:2] == ("docker", "build") for e in h.events)


def test_controlled_override_changes_only_image():
    original={'services':{'app':{'image':'old','environment':{'KEEP':'value','VERIFIED_PHONE_BOOKING_ENABLED':'false', 'openai_model':'gpt-6-luna'}},'redis':{'image':'same'}}}
    changed=repair.changed_override(original,'candidate')
    assert original['services']['app']['environment']['openai_model'] == 'gpt-6-luna'
    expected=copy.deepcopy(original)
    expected['services']['app']['image']='candidate'
    assert changed == expected


def test_new_unresolved_operation_after_writer_stop_restores_previous_app(tmp_path, monkeypatch):
    h=Harness(tmp_path,monkeypatch)
    compose=h.compose
    def late(path,*args,**kwargs):
        if args[-1] == repair.FUTURE_OPERATIONS_PROBE:
            return SimpleNamespace(stdout=b'{"unresolved_future_operations":1}')
        return compose(path,*args,**kwargs)
    h.compose=late
    before=(tmp_path / "docker-compose.override.yml").read_bytes()
    with pytest.raises(RuntimeError,match="unresolved future"):
        repair.execute(h,h.helper)
    assert h.current == TEST_IMAGE
    assert (tmp_path / "docker-compose.override.yml").read_bytes() == before
    assert ("mark","recovered") in h.events
    assert ("mark","deployed") not in h.events


@pytest.mark.parametrize('fault', [None, 'reversed', 'http', 'second-read'])
def test_live_probe_uses_real_calendar_reader_without_mutations(monkeypatch, capsys, fault):
    import httpx
    from services.calendar.ms_bookings_service import MSBookingsService
    requests = []
    async def transport(request):
        requests.append((request.method, request.url.path))
        if request.url.path.endswith('/token'):
            return httpx.Response(200, json={'access_token': 'PRIVATE_CANARY', 'expires_in': 3600})
        if request.url.path.endswith('/staffMembers'):
            return httpx.Response(200, json={'value': []})
        assert request.method == 'POST' and request.url.path.endswith('/getStaffAvailability')
        staff = json.loads(request.content)['staffIds'][0]
        if fault == 'http' or (fault == 'second-read' and staff == 'Abdul'):
            return httpx.Response(403, json={'error': 'PRIVATE_CANARY'})
        return httpx.Response(200, json={'value': [{'staffId': staff, 'availabilityItems': [
            {'status': 'available', 'startDateTime': {'dateTime': '2030-01-08T10:00:00'},
             'endDateTime': {'dateTime': '2030-01-08T09:00:00' if fault == 'reversed' else '2030-01-08T11:00:00'}},
            {'status': 'outOfOffice', 'startDateTime': {'dateTime': '2030-01-08T10:00:00'},
             'endDateTime': {'dateTime': '2030-01-08T10:00:00'}},
        ]}]})
    calendar = MSBookingsService({'tenant_id': 'synthetic', 'client_id': 'synthetic',
        'client_secret': 'synthetic', 'business_id': 'synthetic'})
    calendar._http_client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    monkeypatch.setattr('services.calendar.ms_bookings_service.get_calendar_service', lambda: calendar)
    monkeypatch.setattr('services.config.accountants_service.get_accountants_service', lambda:
        SimpleNamespace(get_accountant_by_name=lambda name: {'staff_id': name, 'service_id': 'service'}))
    if fault:
        with pytest.raises(SystemExit) as error:
            exec(compile(repair.LIVE_CALENDAR_PROBE, 'live-probe', 'exec'), {})
        assert error.value.code == 1
    else:
        exec(compile(repair.LIVE_CALENDAR_PROBE, 'live-probe', 'exec'), {})
    output = capsys.readouterr().out
    assert 'PRIVATE_CANARY' not in output
    value = json.loads(output)
    assert value['status'] == ('LIVE_CALENDAR_READS_FAILED' if fault else 'LIVE_CALENDAR_READS_OK')
    if not fault:
        assert value['checks'] == [{'consultant': 'Rami', 'slot_count': 2}, {'consultant': 'Abdul', 'slot_count': 2}]
    assert all(path.endswith(('/token', '/staffMembers', '/getStaffAvailability')) for _, path in requests)
    assert calendar._http_client is None
