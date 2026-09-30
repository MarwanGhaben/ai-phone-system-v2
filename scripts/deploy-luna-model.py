"""Owner-run Luna adapter and model switch; retain the exact restored caller flow."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types

BASE = '6b3babf0eee0ed611c05d7031caacebe0e9dd652'
MODEL = 'gpt-6-luna'
ROOT = Path("/opt/ai-phone-system-v2")
SOURCES = {'services/llm/openai_service.py': '87924271d2ddaffc65f61a39800aa07a51e0e179df1ae6368dc23d0975497f40'}
SUPPORTED_BASELINES = {'ca46f311e05e4a0c2f3093ed7e3781077c57f7ed': [{'services/calendar/booking_mutations.py': '6d58c7a5208d1269d59afe9958c9d2c26d7eee467c6c1bc6ff1fa76f07091a1b', 'services/calendar/contracts.py': 'aefb4279d7c07429fe497abd92318e1e6daa9104bcc8ed775a96e3eebb88e639', 'services/calendar/service_facts.py': '3e1c3a6977e9d008495e6bb312713b5544ed3fd03108597e61d520b3d53a6822', 'services/conversation/booking_dialogue.py': 'bc1afb8a0d3a3b72bd0c4e5070b0519877e3ed53b9c0f3956565ad225bc6e01a', 'services/conversation/booking_session.py': '99e5644ecaec817055816f07941bce43ee051d0d60d507872ecbe1a92ffe0387', 'services/conversation/language_policy.py': 'a50e8ec58737dd896686dccb4c52c427f94477d4898c2a17a9af74fdf44f8b9b', 'services/conversation/orchestrator.py': '3c91e5f042c440ed974c14efc8356db8b891dbdfd5f81be5342eaf06ef4a1fcb', 'services/llm/openai_service.py': '333acc9e5e022d2341fcd9b07df46927eec34bd195c86abe3711073523468d61', 'services/llm/tool_protocol.py': '82054d4cab596aec27d81b7e1278737c10ea4fdf0cd4bfe4b23dfba6da8b06fd', 'services/scheduling/booking_service.py': '5d8a7a062daf7946d9848e41b07b93e49c499bd4a556396f47c5414b11462f9d', 'services/scheduling/proposals.py': 'df52bb23abd8f7855b6edb7e2b38a79c7cf13aae273c860cdc9f3369800c103b', 'services/scheduling/spoken_arabic.py': 'a11ad2a8bbd6e8c57e964fe717c37f7037b8d206c1b4e3d4d8c189ca3275d130', 'services/telephony/twilio_service.py': '28e6a75851620271d1e0244f74fb5dc3b3ce125cae54a3f3c482ddfde1399b3e'}, {'services/calendar/booking_mutations.py': '6d58c7a5208d1269d59afe9958c9d2c26d7eee467c6c1bc6ff1fa76f07091a1b', 'services/calendar/contracts.py': 'aefb4279d7c07429fe497abd92318e1e6daa9104bcc8ed775a96e3eebb88e639', 'services/calendar/service_facts.py': '3e1c3a6977e9d008495e6bb312713b5544ed3fd03108597e61d520b3d53a6822', 'services/conversation/booking_dialogue.py': 'bc1afb8a0d3a3b72bd0c4e5070b0519877e3ed53b9c0f3956565ad225bc6e01a', 'services/conversation/booking_session.py': 'ca20ac533d8bcaa99d4a7250295cb04d351864e599d7b3e8d1cd0e93cb170bbd', 'services/conversation/language_policy.py': 'a50e8ec58737dd896686dccb4c52c427f94477d4898c2a17a9af74fdf44f8b9b', 'services/conversation/orchestrator.py': '3c91e5f042c440ed974c14efc8356db8b891dbdfd5f81be5342eaf06ef4a1fcb', 'services/llm/openai_service.py': '333acc9e5e022d2341fcd9b07df46927eec34bd195c86abe3711073523468d61', 'services/llm/tool_protocol.py': '82054d4cab596aec27d81b7e1278737c10ea4fdf0cd4bfe4b23dfba6da8b06fd', 'services/scheduling/booking_service.py': '5d8a7a062daf7946d9848e41b07b93e49c499bd4a556396f47c5414b11462f9d', 'services/scheduling/proposals.py': 'df52bb23abd8f7855b6edb7e2b38a79c7cf13aae273c860cdc9f3369800c103b', 'services/scheduling/spoken_arabic.py': 'a11ad2a8bbd6e8c57e964fe717c37f7037b8d206c1b4e3d4d8c189ca3275d130', 'services/telephony/twilio_service.py': 'f3e097c35fd6cd2dd27b6732c11436953ef7fff934c2cb5f3be8393ce28e3cfc'}], '02c3995f196b74a84e8c71ecd1c167d546955530': [{'services/calendar/booking_mutations.py': '6d58c7a5208d1269d59afe9958c9d2c26d7eee467c6c1bc6ff1fa76f07091a1b', 'services/calendar/contracts.py': 'aefb4279d7c07429fe497abd92318e1e6daa9104bcc8ed775a96e3eebb88e639', 'services/calendar/service_facts.py': '3e1c3a6977e9d008495e6bb312713b5544ed3fd03108597e61d520b3d53a6822', 'services/conversation/booking_dialogue.py': 'bc1afb8a0d3a3b72bd0c4e5070b0519877e3ed53b9c0f3956565ad225bc6e01a', 'services/conversation/booking_session.py': '99e5644ecaec817055816f07941bce43ee051d0d60d507872ecbe1a92ffe0387', 'services/conversation/language_policy.py': 'a50e8ec58737dd896686dccb4c52c427f94477d4898c2a17a9af74fdf44f8b9b', 'services/conversation/orchestrator.py': 'bc8afe8c245b552a7e40cedaef5809497440912948ee2a8ae6e543b8b5082aac', 'services/llm/openai_service.py': '333acc9e5e022d2341fcd9b07df46927eec34bd195c86abe3711073523468d61', 'services/llm/tool_protocol.py': '82054d4cab596aec27d81b7e1278737c10ea4fdf0cd4bfe4b23dfba6da8b06fd', 'services/scheduling/booking_service.py': '5d8a7a062daf7946d9848e41b07b93e49c499bd4a556396f47c5414b11462f9d', 'services/scheduling/proposals.py': 'df52bb23abd8f7855b6edb7e2b38a79c7cf13aae273c860cdc9f3369800c103b', 'services/scheduling/spoken_arabic.py': 'a11ad2a8bbd6e8c57e964fe717c37f7037b8d206c1b4e3d4d8c189ca3275d130', 'services/telephony/twilio_service.py': '28e6a75851620271d1e0244f74fb5dc3b3ce125cae54a3f3c482ddfde1399b3e', 'services/calendar/ms_bookings_service.py': 'eaa33b77b88cd927137c4d9ce91fd75bb011758299c924b9892777cf3b76ca47', 'services/calendar/legacy_availability.py': 'aa16cadf112fb9da2cd50918d63a3aebdffaded7295708c9a7ae6277ae0be828', 'services/conversation/availability_reply.py': '752ba4430d6e151dbc5213fbd46974618ef836f22b7ac0e284bd8ac2c8d950b1'}, {'services/calendar/booking_mutations.py': '6d58c7a5208d1269d59afe9958c9d2c26d7eee467c6c1bc6ff1fa76f07091a1b', 'services/calendar/contracts.py': 'aefb4279d7c07429fe497abd92318e1e6daa9104bcc8ed775a96e3eebb88e639', 'services/calendar/service_facts.py': '3e1c3a6977e9d008495e6bb312713b5544ed3fd03108597e61d520b3d53a6822', 'services/conversation/booking_dialogue.py': 'bc1afb8a0d3a3b72bd0c4e5070b0519877e3ed53b9c0f3956565ad225bc6e01a', 'services/conversation/booking_session.py': 'ca20ac533d8bcaa99d4a7250295cb04d351864e599d7b3e8d1cd0e93cb170bbd', 'services/conversation/language_policy.py': 'a50e8ec58737dd896686dccb4c52c427f94477d4898c2a17a9af74fdf44f8b9b', 'services/conversation/orchestrator.py': 'bc8afe8c245b552a7e40cedaef5809497440912948ee2a8ae6e543b8b5082aac', 'services/llm/openai_service.py': '333acc9e5e022d2341fcd9b07df46927eec34bd195c86abe3711073523468d61', 'services/llm/tool_protocol.py': '82054d4cab596aec27d81b7e1278737c10ea4fdf0cd4bfe4b23dfba6da8b06fd', 'services/scheduling/booking_service.py': '5d8a7a062daf7946d9848e41b07b93e49c499bd4a556396f47c5414b11462f9d', 'services/scheduling/proposals.py': 'df52bb23abd8f7855b6edb7e2b38a79c7cf13aae273c860cdc9f3369800c103b', 'services/scheduling/spoken_arabic.py': 'a11ad2a8bbd6e8c57e964fe717c37f7037b8d206c1b4e3d4d8c189ca3275d130', 'services/telephony/twilio_service.py': 'f3e097c35fd6cd2dd27b6732c11436953ef7fff934c2cb5f3be8393ce28e3cfc', 'services/calendar/ms_bookings_service.py': 'eaa33b77b88cd927137c4d9ce91fd75bb011758299c924b9892777cf3b76ca47', 'services/calendar/legacy_availability.py': 'aa16cadf112fb9da2cd50918d63a3aebdffaded7295708c9a7ae6277ae0be828', 'services/conversation/availability_reply.py': '752ba4430d6e151dbc5213fbd46974618ef836f22b7ac0e284bd8ac2c8d950b1'}]}
PROBE_PATH = "scripts/probe-luna-compatibility.py"
PROBE_SHA = '9c88189d140f32708701e8a6825951adefc7fdbbdb843ec5c6abdaf40b667d74'
MANIFEST = "docs/delegation/T020-C-contention-hashes.json"
MANIFEST_SHA = "c3f89ece10d06f9edf14ea191fbad9f2b12c1eaa7cc8533b0ec528607a46de4a"




def digest(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def contract(value):
    config = value["Config"]
    return {
        "mounts": sorted(json.dumps(x, sort_keys=True) for x in value["Mounts"]),
        "environment": sorted(config.get("Env") or []),
        "networks": sorted(value["NetworkSettings"]["Networks"]),
        "ports": value["HostConfig"].get("PortBindings"),
        **{key: config.get(key) for key in ("Cmd", "Entrypoint", "User", "WorkingDir", "Healthcheck")},
    }


def changed_override(original, image):
    result = copy.deepcopy(original)
    result["services"]["app"]["image"] = image
    environment = result["services"]["app"].setdefault("environment", {})
    if not isinstance(environment, dict):
        raise RuntimeError("override environment must be a mapping")
    environment["OPENAI_MODEL"] = MODEL
    environment.pop("openai_model", None)
    return result


def restored_contract(previous):
    result = contract(previous)
    result["environment"] = sorted(
        [item for item in result["environment"]
         if item.split("=", 1)[0].lower() != "openai_model"]
        + ["OPENAI_MODEL=" + MODEL])
    return result


def load_helper():
    result = subprocess.run(["git", "show", BASE + ":scripts/deploy-verified-phone-booking.py"],
                            cwd=ROOT, capture_output=True, check=True, timeout=30)
    module = types.ModuleType("pinned_booking_release")
    exec(compile(result.stdout, "pinned_booking_release", "exec"), module.__dict__)
    return module


def build_candidate(release, sources, old_image):
    """Resolve the accepted image through a verified local tag for BuildKit."""
    build = release.directory / "build"
    build.mkdir(mode=0o700)
    for path, content in sources.items():
        (build / Path(path).name).write_bytes(content)
    parent = "ai-phone-luna-model-parent:" + release.directory.name
    tag = "ai-phone-luna-model:" + release.directory.name
    try:
        release.run("docker", "tag", old_image, parent)
        if release.inspect_once(parent)["Id"] != old_image:
            raise RuntimeError("local parent tag differs from accepted image")
        (build / "Dockerfile").write_bytes((
            "FROM " + parent + "\n" + "".join(
                "COPY " + Path(path).name + " /app/" + path + "\n" for path in sources)
            + '\nLABEL org.opencontainers.image.revision="' + release.commit + '"\n').encode())
        release.run("docker", "build", "--network", "none", "--pull=false", "-t", tag,
                    str(build), timeout=180)
        candidate = release.inspect_once(tag)
        if candidate["Config"]["Labels"].get("org.opencontainers.image.revision") != release.commit:
            raise RuntimeError("candidate revision differs")
        return candidate["Id"]
    finally:
        release.run("docker", "image", "rm", parent, check=False)


FUTURE_OPERATIONS_PROBE = r"""
import asyncio, json, logging
from loguru import logger
logger.remove()
logging.disable(logging.CRITICAL)
async def check():
    import asyncpg
    from config.settings import settings
    conn = await asyncpg.connect(settings.database_url, timeout=5, command_timeout=5)
    try:
        async with conn.transaction(readonly=True):
            await conn.execute("SET LOCAL statement_timeout='4s'; SET LOCAL lock_timeout='2s'")
            count = await conn.fetchval("SELECT count(*) FROM public.booking_operations WHERE state NOT IN ('applied','released') AND upper(claim_span) > now()")
        print(json.dumps({"unresolved_future_operations": count}))
    finally:
        await conn.close(timeout=3)
try:
    asyncio.run(asyncio.wait_for(check(), 15))
except BaseException:
    print('{"status":"read_failed"}')
    raise SystemExit(1) from None
"""


def check_future_operations(release, override=None):
    if override is None:
        raw = release.run("docker", "exec", release.app_container, "python", "-B", "-c",
                          FUTURE_OPERATIONS_PROBE, timeout=20).stdout
    else:
        name = "luna-claim-check-" + release.directory.name
        try:
            raw = release.compose(override, "run", "--rm", "--no-deps", "--pull", "never",
                                  "--name", name, "--entrypoint", "python", "app", "-B", "-c",
                                  FUTURE_OPERATIONS_PROBE, timeout=25).stdout
        finally:
            release.run("docker", "rm", "-f", name, check=False)
    result = json.loads(raw)
    if (type(result) is not dict or type(result.get("unresolved_future_operations")) is not int
            or result["unresolved_future_operations"] != 0):
        raise RuntimeError("unresolved future booking requires reconciliation before release")



def check_live_api(release, override, program):
    """Six synthetic requests, no Graph/Twilio/SMS calls; stop before cutover on failure."""
    name = 'luna-live-probe-' + release.directory.name
    try:
        result = release.compose(override, 'run', '--rm', '--no-deps', '--pull', 'never',
            '--name', name, '--entrypoint', 'python', 'app', '-B', '-c',
            program.decode('utf-8'), '--live', timeout=115, check=False)
        status = json.loads(result.stdout)
        if (result.returncode != 0 or status.get('status') != 'LUNA_COMPATIBILITY_OK'
                or status.get('mode') != 'live' or status.get('completed_requests') != 6):
            code = status.get('http_status')
            print('LUNA_API_CHECK_FAILED_HTTP=' + (str(code) if type(code) is int else 'none'), flush=True)
            raise RuntimeError('Luna live compatibility was not established')
        print('LUNA_SYNTHETIC_ENGLISH_ARABIC_TOOLS_STREAMING_OK', flush=True)
    finally:
        release.run('docker', 'rm', '-f', name, check=False)

def execute(release, helper):
    live = ROOT / "docker-compose.override.yml"
    if live.is_symlink() or live.stat().st_uid != 0 or live.stat().st_mode & 0o077:
        raise RuntimeError("override protection differs")
    previous_bytes = live.read_bytes()
    original = helper.strict_json(previous_bytes)
    previous = release.inspect_once(release.app_container)
    old_image = previous["Image"]
    if (not re.fullmatch(r"sha256:[0-9a-f]{64}", old_image)
            or original["services"]["app"]["image"] != old_image):
        raise RuntimeError("unexpected predecessor override")
    previous_revision = previous["Config"]["Labels"].get("org.opencontainers.image.revision")
    if (previous_revision not in SUPPORTED_BASELINES
            or previous["Config"]["Labels"].get("com.docker.compose.project") != release.project_name):
        raise RuntimeError("unexpected predecessor image")
    release.wait_ready()
    release.public_health()
    settings = release.settings()
    if settings.get("verified_phone_booking_enabled") is not False:
        raise RuntimeError("restored booking flow must remain active")
    target_settings = {**settings, "openai_model": MODEL}
    check_future_operations(release)
    sources = {path: release.git_bytes(path) for path in SOURCES}
    if {path: digest(content) for path, content in sources.items()} != SOURCES:
        raise RuntimeError("repair sources differ from reviewed hashes")
    probe_source = release.git_bytes(PROBE_PATH)
    if digest(probe_source) != PROBE_SHA:
        raise RuntimeError("offline probe differs")
    manifest_bytes = release.run("git", "show", BASE + ":" + MANIFEST).stdout
    if digest(manifest_bytes) != MANIFEST_SHA:
        raise RuntimeError("baseline manifest differs")
    manifest = json.loads(manifest_bytes)
    hashes = {}
    for section in ("runtime", "configuration", "migrations", "protected_runtime"):
        hashes.update(manifest[section])
    # Pin the published SDK/Graph repairs after the immutable schema-0005 manifest.
    # Restoration retained either accepted predecessor's parked code. Verify the
    # variant by effective source, then retain it exactly through this app-only repair.
    variant_probe = "import hashlib,json; from pathlib import Path; print(json.dumps({p:hashlib.sha256(Path('/app',p).read_bytes().replace(b'\\r\\n',b'\\n')).hexdigest() for p in " + repr(list(SUPPORTED_BASELINES[previous_revision][0])) + "}))"
    actual = helper.strict_json(release.run('docker', 'exec', release.app_container,
        'python', '-B', '-c', variant_probe).stdout)
    if actual not in SUPPORTED_BASELINES[previous_revision]:
        raise RuntimeError('restored predecessor source differs')
    hashes.update(actual)
    raw_hashes = manifest["model_assets"]
    def probe(image_hashes, override=None, expected_settings=None):
        program = helper.source_probe_code(image_hashes, raw_hashes)
        if override is None:
            data = release.run("docker", "exec", release.app_container,
                               "python", "-B", "-c", program).stdout
        else:
            name = "luna-source-probe-" + release.directory.name
            try:
                data = release.compose(override, "run", "--rm", "--no-deps", "--pull", "never",
                    "--name", name, "--entrypoint", "python", "app", "-B", "-c", program,
                    timeout=90).stdout
            finally:
                release.run("docker", "rm", "-f", name, check=False)
        value = helper.strict_json(data)
        if (value.get("hashes") != {**image_hashes, **raw_hashes}
                or not helper.typed_mapping_equal(value.get("settings"), expected_settings if expected_settings is not None else settings)):
            raise RuntimeError("effective source or settings differ")
    probe(hashes)
    previous_path = release.private_write("previous-override.json", previous_bytes)
    release.private_write("previous-app.json", json.dumps(previous).encode())
    release.private_write("previous-settings.json", json.dumps(settings).encode())
    files = {name: (ROOT / name).read_bytes()
             for name in ("docker-compose.yml", "nginx/nginx.conf")}
    rendered = release.render_compose(live)
    print("LUNA_MODEL_BASELINE_SOURCE_SETTINGS_READY_OK", flush=True)
    release.stage = "build"
    candidate = build_candidate(release, sources, old_image)
    print("LUNA_MODEL_LOCAL_PARENT_BUILD_OK", flush=True)
    release.stage = "confirmation-probe"
    name = "luna-offline-probe-" + release.directory.name
    try:
        result = release.run("docker", "run", "--rm", "--network", "none", "--name", name,
            "--entrypoint", "python", candidate, "-B", "-c", probe_source.decode('utf-8'), timeout=45)
        if helper.strict_json(result.stdout).get("status") != "LUNA_COMPATIBILITY_OK":
            raise RuntimeError("Luna SDK probe did not complete")
    finally:
        release.run("docker", "rm", "-f", name, check=False)
    candidate_path = release.private_write("candidate-override.json",
        json.dumps(changed_override(original, candidate), indent=2).encode())
    release.stage = "candidate-configuration"
    expected_rendered = changed_override(rendered, candidate)
    if release.render_compose(candidate_path) != expected_rendered:
        raise RuntimeError("rendered configuration changed beyond app image")
    candidate_hashes = {**hashes, **SOURCES}
    probe(candidate_hashes, candidate_path, target_settings)
    print("LUNA_MODEL_CANDIDATE_CONFIRMATION_AND_EFFECTIVE_SOURCE_OK", flush=True)
    release.stage = 'live-api-check'
    check_live_api(release, candidate_path, probe_source)
    release.stage = "pre-cutover"
    if (live.read_bytes() != previous_bytes
            or any((ROOT / name).read_bytes() != value for name, value in files.items())
            or release.inspect_once(release.app_container)["Image"] != old_image
            or contract(release.inspect_once(release.app_container)) != contract(previous)
            or not helper.typed_mapping_equal(release.settings(), settings)):
        raise RuntimeError("baseline changed during preparation")
    release.wait_ready()
    release.public_health()
    def replace(override, image, revision, expected_hashes, expected_settings, expected_contract):
        release.install_override(override)
        release.compose(live, "up", "-d", "--no-deps", "--no-build", "--pull", "never",
                        "--force-recreate", "app", timeout=120)
        release.wait_ready()
        current = release.inspect_once(release.app_container)
        if (current["Image"] != image or contract(current) != expected_contract
                or current["Config"]["Labels"].get("org.opencontainers.image.revision") != revision):
            raise RuntimeError("replacement contract differs")
        probe(expected_hashes, expected_settings=expected_settings)
        release.start_ingress()
    release.stage = "cutover"
    release.mark("cutover-started")
    try:
        release.stop_writers(strict=True)
        check_future_operations(release, live)
        replace(candidate_path, candidate, release.commit, candidate_hashes,
                target_settings, restored_contract(previous))
    except BaseException:
        try:
            release.stop_writers(strict=True)
            replace(previous_path, old_image, previous_revision, hashes, settings, contract(previous))
            release.mark("recovered")
            print("LUNA_MODEL_PREVIOUS_APP_RESTORED", flush=True)
        except BaseException:
            release.force_close_ingress()
            print("LUNA_MODEL_RECOVERY_UNVERIFIED", flush=True)
        raise
    release.mark("deployed", (release.commit + "\n" + candidate + "\n").encode())
    print("LUNA_MODEL_DEPLOYED_READY_HTTPS_OK", flush=True)
    print("DEPLOYED_COMMIT=" + release.commit, flush=True)
    print("DEPLOYED_IMAGE=" + candidate, flush=True)
    print("OPENAI_MODEL=" + MODEL, flush=True)


def main(commit):
    import fcntl
    if os.geteuid() != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("root and exact commit required")
    os.umask(0o077)
    fd = os.open("/run/ai-phone-deployment.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = Path(tempfile.mkdtemp(prefix="ai-phone-luna-model.", dir="/opt"))
        print("PROTECTED_RELEASE_DIRECTORY=" + str(directory), flush=True)
        helper = load_helper()
        release = helper.Release(directory, commit)
        try:
            execute(release, helper)
        except BaseException:
            print("LUNA_MODEL_STOPPED_STAGE=" + release.stage, flush=True)
            raise


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("exact commit required")
        main(sys.argv[1])
    except BaseException:
        print("LUNA_MODEL_STOPPED_KEEP_PROTECTED_FILES", flush=True)
        raise SystemExit(1) from None
