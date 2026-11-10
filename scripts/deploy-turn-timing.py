"""Owner-run release: per-reply timing measurement in the restored call flow.

Measurement only. Adds one numeric log line per spoken reply (TurnTiming ...) so the
owner can see where the delay between the end of a caller sentence and the start of
the agent's reply comes from. The spoken words, their timing, the booking logic, the
model, the voice, the speech recognition settings and every environment value (the
speech recognition language lock included) are kept. The override changes only the app
image. Usage: python3 - <40-hex commit>
"""
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
LLM_MODEL = 'gpt-6-luna'
VOICE_MODEL = 'eleven_v4_turbo'
ROOT = Path("/opt/ai-phone-system-v2")
ACCEPTED_REVISION = '8629becc623db91a0b2bff56dd48d1efa6314650'
PLACEHOLDER_IMAGE = "sha256:" + "0" * 64
ACCEPTED_IMAGE = "sha256:0383f37031c3ba6e1dbea7129242f6771e7d65c66889d9ff4454f403836ac7d8"  # owner-reported 2026-11-10
TIMING_PATH = "services/conversation/turn_timing.py"
ORCHESTRATOR = "services/conversation/orchestrator.py"
SOURCES = {
    "services/conversation/turn_timing.py": "23ed5b9ecd21810341e0bc823a7380b1a11ec119bd7b2bbd1094373f973a6e52",
    "services/conversation/orchestrator.py": "f8ee6d927ce287425d9ce46a72ec154322e9a6f983fe4381cb5ce23331703f43"
}
# The language lock release's two accepted variants: the reply-pause variants of the
# restored flow, each with that release's reviewed speech recognition service.
SUPPORTED_BASELINES = {
    "8629becc623db91a0b2bff56dd48d1efa6314650": [
        {
            "services/calendar/booking_mutations.py": "6d58c7a5208d1269d59afe9958c9d2c26d7eee467c6c1bc6ff1fa76f07091a1b",
            "services/calendar/contracts.py": "aefb4279d7c07429fe497abd92318e1e6daa9104bcc8ed775a96e3eebb88e639",
            "services/calendar/service_facts.py": "3e1c3a6977e9d008495e6bb312713b5544ed3fd03108597e61d520b3d53a6822",
            "services/conversation/booking_dialogue.py": "bc1afb8a0d3a3b72bd0c4e5070b0519877e3ed53b9c0f3956565ad225bc6e01a",
            "services/conversation/booking_session.py": "99e5644ecaec817055816f07941bce43ee051d0d60d507872ecbe1a92ffe0387",
            "services/conversation/language_policy.py": "225dd5a6411390b8206a2c790cbabccbfca542f12b295c55aae445e35f11b987",
            "services/conversation/orchestrator.py": "5ef96383a930d4fa38e8943bac7abd0afc47fd965b639ac42b6270b6410b7561",
            "services/llm/openai_service.py": "87924271d2ddaffc65f61a39800aa07a51e0e179df1ae6368dc23d0975497f40",
            "services/llm/tool_protocol.py": "82054d4cab596aec27d81b7e1278737c10ea4fdf0cd4bfe4b23dfba6da8b06fd",
            "services/scheduling/booking_service.py": "5d8a7a062daf7946d9848e41b07b93e49c499bd4a556396f47c5414b11462f9d",
            "services/scheduling/proposals.py": "df52bb23abd8f7855b6edb7e2b38a79c7cf13aae273c860cdc9f3369800c103b",
            "services/scheduling/spoken_arabic.py": "a11ad2a8bbd6e8c57e964fe717c37f7037b8d206c1b4e3d4d8c189ca3275d130",
            "services/telephony/twilio_service.py": "28e6a75851620271d1e0244f74fb5dc3b3ce125cae54a3f3c482ddfde1399b3e",
            "services/calendar/ms_bookings_service.py": "eaa33b77b88cd927137c4d9ce91fd75bb011758299c924b9892777cf3b76ca47",
            "services/calendar/legacy_availability.py": "ebc3dada46e9f7f7299fd3e279e133c39ec6d4e067c2bda0d4c6ca15f7d6d811",
            "services/conversation/availability_reply.py": "752ba4430d6e151dbc5213fbd46974618ef836f22b7ac0e284bd8ac2c8d950b1",
            "services/conversation/restored_turn_admission.py": "3f5dc87e9596f8b81686601065a3ec79e9d58d1925f4e1f3e017fb155d57b573",
            "services/tts/elevenlabs_service.py": "1a2332efafb2df3237671c1e508497350969041e144f1eed9a1fa27040cb6fe9",
            "services/stt/elevenlabs_stt_service.py": "b6af4b47c0bfeb96c44dda51f34a1042fd994083176c30d16dcd1e28ade2cafb"
        },
        {
            "services/calendar/booking_mutations.py": "6d58c7a5208d1269d59afe9958c9d2c26d7eee467c6c1bc6ff1fa76f07091a1b",
            "services/calendar/contracts.py": "aefb4279d7c07429fe497abd92318e1e6daa9104bcc8ed775a96e3eebb88e639",
            "services/calendar/service_facts.py": "3e1c3a6977e9d008495e6bb312713b5544ed3fd03108597e61d520b3d53a6822",
            "services/conversation/booking_dialogue.py": "bc1afb8a0d3a3b72bd0c4e5070b0519877e3ed53b9c0f3956565ad225bc6e01a",
            "services/conversation/booking_session.py": "ca20ac533d8bcaa99d4a7250295cb04d351864e599d7b3e8d1cd0e93cb170bbd",
            "services/conversation/language_policy.py": "225dd5a6411390b8206a2c790cbabccbfca542f12b295c55aae445e35f11b987",
            "services/conversation/orchestrator.py": "5ef96383a930d4fa38e8943bac7abd0afc47fd965b639ac42b6270b6410b7561",
            "services/llm/openai_service.py": "87924271d2ddaffc65f61a39800aa07a51e0e179df1ae6368dc23d0975497f40",
            "services/llm/tool_protocol.py": "82054d4cab596aec27d81b7e1278737c10ea4fdf0cd4bfe4b23dfba6da8b06fd",
            "services/scheduling/booking_service.py": "5d8a7a062daf7946d9848e41b07b93e49c499bd4a556396f47c5414b11462f9d",
            "services/scheduling/proposals.py": "df52bb23abd8f7855b6edb7e2b38a79c7cf13aae273c860cdc9f3369800c103b",
            "services/scheduling/spoken_arabic.py": "a11ad2a8bbd6e8c57e964fe717c37f7037b8d206c1b4e3d4d8c189ca3275d130",
            "services/telephony/twilio_service.py": "f3e097c35fd6cd2dd27b6732c11436953ef7fff934c2cb5f3be8393ce28e3cfc",
            "services/calendar/ms_bookings_service.py": "eaa33b77b88cd927137c4d9ce91fd75bb011758299c924b9892777cf3b76ca47",
            "services/calendar/legacy_availability.py": "ebc3dada46e9f7f7299fd3e279e133c39ec6d4e067c2bda0d4c6ca15f7d6d811",
            "services/conversation/availability_reply.py": "752ba4430d6e151dbc5213fbd46974618ef836f22b7ac0e284bd8ac2c8d950b1",
            "services/conversation/restored_turn_admission.py": "3f5dc87e9596f8b81686601065a3ec79e9d58d1925f4e1f3e017fb155d57b573",
            "services/tts/elevenlabs_service.py": "1a2332efafb2df3237671c1e508497350969041e144f1eed9a1fa27040cb6fe9",
            "services/stt/elevenlabs_stt_service.py": "b6af4b47c0bfeb96c44dda51f34a1042fd994083176c30d16dcd1e28ade2cafb"
        }
    ]
}
PROBE_PATH = "scripts/probe-turn-timing.py"
PROBE_SHA = 'ea57c09f8201540edb21c8ca6e5cd5cbd342709cefc6ccc1ee87e759c8732445'
BUILDER_PATH = "scripts/build_restored_timing.py"
BUILDER_SHA = '31a62e95d2757d8c165bafa7667db5fb43230f4b8c378fea73bb079a37feed29'
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
    parent = "ai-phone-turn-timing-parent:" + release.directory.name
    tag = "ai-phone-turn-timing:" + release.directory.name
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
        name = "turn-timing-claim-check-" + release.directory.name
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


def execute(release, helper):
    if ACCEPTED_IMAGE == PLACEHOLDER_IMAGE:
        raise RuntimeError("accepted running image has not been recorded yet")
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
    if old_image != ACCEPTED_IMAGE:
        raise RuntimeError("accepted running image differs")
    previous_revision = previous["Config"]["Labels"].get("org.opencontainers.image.revision")
    if (previous_revision != ACCEPTED_REVISION or previous_revision not in SUPPORTED_BASELINES
            or previous["Config"]["Labels"].get("com.docker.compose.project") != release.project_name):
        raise RuntimeError("unexpected predecessor image")
    release.wait_ready()
    release.public_health()
    settings = release.settings()
    if settings.get("verified_phone_booking_enabled") is not False:
        raise RuntimeError("restored booking flow must remain active")
    if settings.get("openai_model") != LLM_MODEL:
        raise RuntimeError("accepted Luna model differs")
    if settings.get("elevenlabs_model") != VOICE_MODEL:
        raise RuntimeError("accepted voice model differs")
    target_settings = settings
    check_future_operations(release)
    builder_bytes = release.git_bytes(BUILDER_PATH)
    if digest(builder_bytes) != BUILDER_SHA:
        raise RuntimeError("source builder differs")
    builder = types.ModuleType("reviewed_restored_timing_builder")
    exec(compile(builder_bytes, "reviewed_restored_timing_builder", "exec"), builder.__dict__)
    baseline = release.run("docker", "exec", release.app_container, "cat",
                           "/app/" + ORCHESTRATOR).stdout
    sources = {path: (builder.patch_orchestrator(baseline)
                     if path == ORCHESTRATOR else release.git_bytes(path)) for path in SOURCES}
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
    # The running restored flow is one of two accepted source variants. Verify it by
    # effective source, then retain it exactly through this app-only repair.
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
            name = "turn-timing-source-probe-" + release.directory.name
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
    print("TURN_TIMING_BASELINE_SOURCE_SETTINGS_READY_OK", flush=True)
    release.stage = "build"
    candidate = build_candidate(release, sources, old_image)
    print("TURN_TIMING_LOCAL_PARENT_BUILD_OK", flush=True)
    release.stage = "offline-probe"
    name = "turn-timing-offline-probe-" + release.directory.name
    try:
        result = release.run("docker", "run", "--rm", "--network", "none", "--no-healthcheck", "--name", name,
            "--entrypoint", "python", candidate, "-B", "-c", probe_source.decode('utf-8'), timeout=45)
        status = helper.strict_json(result.stdout)
        if status.get("status") != "TURN_TIMING_OK" or status.get("mode") != "offline":
            raise RuntimeError("turn timing probe did not complete")
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
    print("TURN_TIMING_CANDIDATE_SETTINGS_AND_EFFECTIVE_SOURCE_OK", flush=True)
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
                target_settings, contract(previous))
    except BaseException:
        try:
            release.stop_writers(strict=True)
            replace(previous_path, old_image, previous_revision, hashes, settings, contract(previous))
            release.mark("recovered")
            print("TURN_TIMING_PREVIOUS_APP_RESTORED", flush=True)
        except BaseException:
            release.force_close_ingress()
            print("TURN_TIMING_RECOVERY_UNVERIFIED", flush=True)
        raise
    release.mark("deployed", (release.commit + "\n" + candidate + "\n").encode())
    print("TURN_TIMING_DEPLOYED_READY_HTTPS_OK", flush=True)
    print("DEPLOYED_COMMIT=" + release.commit, flush=True)
    print("DEPLOYED_IMAGE=" + candidate, flush=True)


def main(commit):
    import fcntl
    if os.geteuid() != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("root and exact commit required")
    os.umask(0o077)
    fd = os.open("/run/ai-phone-deployment.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = Path(tempfile.mkdtemp(prefix="ai-phone-turn-timing.", dir="/opt"))
        print("PROTECTED_RELEASE_DIRECTORY=" + str(directory), flush=True)
        helper = load_helper()
        release = helper.Release(directory, commit)
        try:
            execute(release, helper)
        except BaseException:
            print("TURN_TIMING_STOPPED_STAGE=" + release.stage, flush=True)
            raise


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("exact commit required")
        main(sys.argv[1])
    except BaseException:
        print("TURN_TIMING_STOPPED_KEEP_PROTECTED_FILES", flush=True)
        raise SystemExit(1) from None
