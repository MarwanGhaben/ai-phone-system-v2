"""Owner-run app-only SDK compatibility repair; never runs a migration."""
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

BASE = "af3873c91d0309ac545409d1926f22752594c65a"
OLD_IMAGE = "sha256:79de88be290df1c4895c9e2cc7af642737eb68133f0f495b1da93469c2e8c68a"
ROOT = Path("/opt/ai-phone-system-v2")
SOURCE = "services/llm/openai_service.py"
MANIFEST = "docs/delegation/T020-C-contention-hashes.json"
MANIFEST_SHA = "c3f89ece10d06f9edf14ea191fbad9f2b12c1eaa7cc8533b0ec528607a46de4a"

SDK_PROBE = r'''
import asyncio,json,httpx
from openai import AsyncOpenAI
from services.llm.openai_service import OpenAILLM
from services.llm.llm_base import LLMRequest,Message,LLMRole
async def main():
 seen=[]
 def respond(request):
  body=json.loads(request.content)
  assert body['parallel_tool_calls'] is False
  seen.append(body)
  return httpx.Response(200,json={'id':'chatcmpl_test','object':'chat.completion',
   'created':1,'model':'gpt-4o','choices':[{'index':0,'finish_reason':'stop',
   'message':{'role':'assistant','content':'Which day?'}}],
   'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}})
 async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
  async with AsyncOpenAI(api_key='synthetic',http_client=http,max_retries=0) as sdk:
   llm=OpenAILLM(api_key='synthetic'); llm._client=sdk
   for text in ["Check Hussam availability", "أريد معرفة مواعيد حسام"]:
    result=await llm.chat_with_tools(LLMRequest([Message(LLMRole.USER,text)],
     tools=[{'name':'check_appointment','parameters':{'type':'object','properties':{}}}],
     metadata={'single_tool_call':True}))
    assert result.content=='Which day?'
   assert len(seen)==2
 print('SDK_OFFLINE_TRANSPORT_OK')
asyncio.run(main())
'''


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
    source = release.git_bytes(SOURCE)
    prior_source = release.run("git", "show", BASE + ":" + SOURCE).stdout
    # Permit precisely the reviewed compatibility edit, not an arbitrary module replacement.
    expected = prior_source.replace(b"\r\n", b"\n").replace(
        b'                    kwargs["parallel_tool_calls"] = False\n',
        b'                    # The pinned 1.10 SDK predates this named parameter, but\n'
        b'                    # extra_body forwards it unchanged to the API request.\n'
        b'                    kwargs["extra_body"] = {"parallel_tool_calls": False}\n')
    if expected == prior_source.replace(b"\r\n", b"\n") or digest(source) != digest(expected):
        raise RuntimeError("repair source differs")
    manifest_bytes = release.run("git", "show", BASE + ":" + MANIFEST).stdout
    if digest(manifest_bytes) != MANIFEST_SHA:
        raise RuntimeError("baseline manifest differs")
    manifest = json.loads(manifest_bytes)
    hashes = {}
    for section in ("runtime", "configuration", "migrations", "protected_runtime"):
        hashes.update(manifest[section])
    raw_hashes = manifest["model_assets"]
    def probe(image_hashes, override=None):
        program = helper.source_probe_code(image_hashes, raw_hashes)
        if override is None:
            data = release.run("docker", "exec", release.app_container,
                               "python", "-B", "-c", program).stdout
        else:
            name = "llm-repair-probe-" + release.directory.name
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
    print("LLM_REPAIR_BASELINE_SOURCE_SETTINGS_READY_OK", flush=True)
    build = release.directory / "build"
    build.mkdir(mode=0o700)
    (build / "openai_service.py").write_bytes(source)
    (build / "Dockerfile").write_bytes((
        "FROM " + OLD_IMAGE + "\nCOPY openai_service.py /app/" + SOURCE
        + '\nLABEL org.opencontainers.image.revision="' + release.commit + '"\n').encode())
    tag = "ai-phone-llm-repair:" + release.directory.name
    release.run("docker", "build", "--network", "none", "--pull=false", "-t", tag,
                str(build), timeout=180)
    candidate = release.inspect_once(tag)["Id"]
    name = "llm-repair-sdk-" + release.directory.name
    try:
        result = release.run("docker", "run", "--rm", "--network", "none", "--name", name,
            "--entrypoint", "python", candidate, "-B", "-c", SDK_PROBE, timeout=45)
        if b"SDK_OFFLINE_TRANSPORT_OK" not in result.stdout:
            raise RuntimeError("SDK transport probe did not complete")
    finally:
        release.run("docker", "rm", "-f", name, check=False)
    candidate_path = release.private_write("candidate-override.json",
        json.dumps(changed_override(original, candidate), indent=2).encode())
    expected_rendered = changed_override(rendered, candidate)
    if release.render_compose(candidate_path) != expected_rendered:
        raise RuntimeError("rendered configuration changed beyond app image")
    candidate_hashes = {**hashes, SOURCE: digest(source)}
    probe(candidate_hashes, candidate_path)
    print("LLM_REPAIR_CANDIDATE_SDK_AND_EFFECTIVE_SOURCE_OK", flush=True)
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
    release.mark("cutover-started")
    try:
        release.stop_writers(strict=True)
        replace(candidate_path, candidate, release.commit, candidate_hashes)
    except BaseException:
        try:
            release.stop_writers(strict=True)
            replace(previous_path, OLD_IMAGE, BASE, hashes)
            release.mark("recovered")
            print("LLM_REPAIR_PREVIOUS_APP_RESTORED", flush=True)
        except BaseException:
            release.force_close_ingress()
            print("LLM_REPAIR_RECOVERY_UNVERIFIED", flush=True)
        raise
    release.mark("deployed", (release.commit + "\n" + candidate + "\n").encode())
    print("LLM_SDK_REPAIR_DEPLOYED_READY_HTTPS_OK", flush=True)
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
        directory = Path(tempfile.mkdtemp(prefix="ai-phone-llm-repair.", dir="/opt"))
        print("PROTECTED_RELEASE_DIRECTORY=" + str(directory), flush=True)
        helper = load_helper()
        execute(helper.Release(directory, commit), helper)


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("exact commit required")
        main(sys.argv[1])
    except BaseException:
        print("LLM_REPAIR_STOPPED_KEEP_PROTECTED_FILES", flush=True)
        raise SystemExit(1) from None
