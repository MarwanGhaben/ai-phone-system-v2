"""Owner-run app-only Graph parser and availability reply repair; never runs a migration."""
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

BASE = "ac02a494dd33614b606b325bd9753b90a62e5a67"
OLD_IMAGE = "sha256:cfd3e52a29e5518909a37073903d8a1bbaff025e031ce02694abf0e90add92ac"
ROOT = Path("/opt/ai-phone-system-v2")
SOURCES = {'services/calendar/contracts.py': '7b62912d0304849494597cad4323bab9c6b6303d41485a866762fd43b20d2a96', 'services/calendar/service_facts.py': '3e1c3a6977e9d008495e6bb312713b5544ed3fd03108597e61d520b3d53a6822', 'services/conversation/orchestrator.py': '9b60780d9b970251158ba8a19ddb52cc43eada3cf32f71d01a61be40ad774e0e'}
PROBE_PATH = "scripts/probe-graph-parser-repair.py"
PROBE_SHA = 'd9576daaea9ae78144da8aaa7d8bb3c70142002f0740f97d051a91640f076fe3'
SDK_SOURCE = "services/llm/openai_service.py"
SDK_SHA = '333acc9e5e022d2341fcd9b07df46927eec34bd195c86abe3711073523468d61'
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


def build_candidate(release, sources):
    """Resolve the accepted image through a verified local tag for BuildKit."""
    build = release.directory / "build"
    build.mkdir(mode=0o700)
    for path, content in sources.items():
        (build / Path(path).name).write_bytes(content)
    parent = "ai-phone-graph-repair-parent:" + release.directory.name
    tag = "ai-phone-graph-repair:" + release.directory.name
    try:
        release.run("docker", "tag", OLD_IMAGE, parent)
        if release.inspect_once(parent)["Id"] != OLD_IMAGE:
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


def execute(release, helper):
    live = ROOT / "docker-compose.override.yml"
    if live.is_symlink() or live.stat().st_uid != 0 or live.stat().st_mode & 0o077:
        raise RuntimeError("override protection differs")
    previous_bytes = live.read_bytes()
    original = helper.strict_json(previous_bytes)
    if original["services"]["app"]["image"] != OLD_IMAGE:
        raise RuntimeError("unexpected predecessor override")
    previous = release.inspect_once(release.app_container)
    if (previous["Image"] != OLD_IMAGE
            or previous["Config"]["Labels"].get("org.opencontainers.image.revision") != BASE
            or previous["Config"]["Labels"].get("com.docker.compose.project") != release.project_name):
        raise RuntimeError("unexpected predecessor image")
    release.wait_ready()
    release.public_health()
    settings = release.settings()
    if settings.get("verified_phone_booking_enabled") is not True:
        raise RuntimeError("verified booking must remain active")
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
    # The ac02 predecessor includes the SDK repair after this immutable manifest.
    if digest(release.run("git", "show", BASE + ":" + SDK_SOURCE).stdout) != SDK_SHA:
        raise RuntimeError("accepted SDK predecessor differs")
    hashes[SDK_SOURCE] = SDK_SHA
    raw_hashes = manifest["model_assets"]
    def probe(image_hashes, override=None):
        program = helper.source_probe_code(image_hashes, raw_hashes)
        if override is None:
            data = release.run("docker", "exec", release.app_container,
                               "python", "-B", "-c", program).stdout
        else:
            name = "graph-repair-probe-" + release.directory.name
            try:
                data = release.compose(override, "run", "--rm", "--no-deps", "--pull", "never",
                    "--name", name, "--entrypoint", "python", "app", "-B", "-c", program,
                    timeout=90).stdout
            finally:
                release.run("docker", "rm", "-f", name, check=False)
        value = helper.strict_json(data)
        if (value.get("hashes") != {**image_hashes, **raw_hashes}
                or not helper.typed_mapping_equal(value.get("settings"), settings)):
            raise RuntimeError("effective source or settings differ")
    probe(hashes)
    previous_path = release.private_write("previous-override.json", previous_bytes)
    release.private_write("previous-app.json", json.dumps(previous).encode())
    release.private_write("previous-settings.json", json.dumps(settings).encode())
    files = {name: (ROOT / name).read_bytes()
             for name in ("docker-compose.yml", "nginx/nginx.conf")}
    rendered = release.render_compose(live)
    print("GRAPH_REPAIR_BASELINE_SOURCE_SETTINGS_READY_OK", flush=True)
    release.stage = "build"
    candidate = build_candidate(release, sources)
    print("GRAPH_REPAIR_LOCAL_PARENT_BUILD_OK", flush=True)
    release.stage = "parser-probe"
    name = "graph-repair-parser-" + release.directory.name
    try:
        result = release.run("docker", "run", "--rm", "--network", "none", "--name", name,
            "--entrypoint", "python", candidate, "-B", "-c", probe_source.decode('utf-8'), timeout=45)
        if b"GRAPH_PARSER_OFFLINE_TRANSPORT_POLICY_OK" not in result.stdout:
            raise RuntimeError("Graph transport policy probe did not complete")
    finally:
        release.run("docker", "rm", "-f", name, check=False)
    candidate_path = release.private_write("candidate-override.json",
        json.dumps(changed_override(original, candidate), indent=2).encode())
    release.stage = "candidate-configuration"
    expected_rendered = changed_override(rendered, candidate)
    if release.render_compose(candidate_path) != expected_rendered:
        raise RuntimeError("rendered configuration changed beyond app image")
    candidate_hashes = {**hashes, **SOURCES}
    probe(candidate_hashes, candidate_path)
    print("GRAPH_REPAIR_CANDIDATE_PARSERS_AND_EFFECTIVE_SOURCE_OK", flush=True)
    release.stage = "pre-cutover"
    if (live.read_bytes() != previous_bytes
            or any((ROOT / name).read_bytes() != value for name, value in files.items())
            or release.inspect_once(release.app_container)["Image"] != OLD_IMAGE
            or contract(release.inspect_once(release.app_container)) != contract(previous)
            or not helper.typed_mapping_equal(release.settings(), settings)):
        raise RuntimeError("baseline changed during preparation")
    release.wait_ready()
    release.public_health()
    def replace(override, image, revision, expected_hashes):
        release.install_override(override)
        release.compose(live, "up", "-d", "--no-deps", "--no-build", "--pull", "never",
                        "--force-recreate", "app", timeout=120)
        release.wait_ready()
        current = release.inspect_once(release.app_container)
        if (current["Image"] != image or contract(current) != contract(previous)
                or current["Config"]["Labels"].get("org.opencontainers.image.revision") != revision):
            raise RuntimeError("replacement contract differs")
        probe(expected_hashes)
        release.start_ingress()
    release.stage = "cutover"
    release.mark("cutover-started")
    try:
        release.stop_writers(strict=True)
        replace(candidate_path, candidate, release.commit, candidate_hashes)
    except BaseException:
        try:
            release.stop_writers(strict=True)
            replace(previous_path, OLD_IMAGE, BASE, hashes)
            release.mark("recovered")
            print("GRAPH_REPAIR_PREVIOUS_APP_RESTORED", flush=True)
        except BaseException:
            release.force_close_ingress()
            print("GRAPH_REPAIR_RECOVERY_UNVERIFIED", flush=True)
        raise
    release.mark("deployed", (release.commit + "\n" + candidate + "\n").encode())
    print("GRAPH_PARSER_REPAIR_DEPLOYED_READY_HTTPS_OK", flush=True)
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
        directory = Path(tempfile.mkdtemp(prefix="ai-phone-graph-repair.", dir="/opt"))
        print("PROTECTED_RELEASE_DIRECTORY=" + str(directory), flush=True)
        helper = load_helper()
        release = helper.Release(directory, commit)
        try:
            execute(release, helper)
        except BaseException:
            print("GRAPH_REPAIR_STOPPED_STAGE=" + release.stage, flush=True)
            raise


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("exact commit required")
        main(sys.argv[1])
    except BaseException:
        print("GRAPH_REPAIR_STOPPED_KEEP_PROTECTED_FILES", flush=True)
        raise SystemExit(1) from None
