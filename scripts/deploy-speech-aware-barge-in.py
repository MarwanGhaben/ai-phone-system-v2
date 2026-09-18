"""Owner-run T033-H app-only speech-aware barge-in beta rollout.

Importing this module is side-effect free.  The executable script is read from
the exact reviewed Git commit supplied on the command line.  Public output is
limited to fixed stage/type markers; command output and evidence remain inside
the root-protected release directory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from urllib.parse import unquote, urlsplit


ROOT = Path("/opt/ai-phone-system-v2")
BASE_CHECKOUT = "1480eeb0e36596e26d0dedde77f08beee91aec5b"
DEPLOYED_SOURCE = "4c135d883554496a250e135368e7e6eade90cc7a"
OLD_IMAGE = "sha256:089447b0a1acb36c1ffd9d5a14251dd3357f19e75bacd8ebed919df3ffae8913"
NGINX_HASH = "59ac9bdaa9a86c1022a573c9709faddf7e93a53015da8acfcc4330b3e288f3ab"
PROJECT = "ai-phone-system-v2"
MANIFEST_PATH = "docs/delegation/T033-H-accepted-hashes.json"
MANIFEST_SHA256 = "4d1aeb6f59b5291c88094493457edcd11806d49cddd56a4f1b7e129c489390a4"
REVIEWED_INPUT_HEAD = DEPLOYED_SOURCE
PUBLIC_HEALTH_URL = "https://aiagent.ghaben.ca:8443/health"
RELEASE_PATHS = (
    "scripts/deploy-speech-aware-barge-in.py",
    "tests/integration/test_speech_gate_release.py",
    "docs/operations/T033-H-speech-gate-rollout.md",
    "docs/delegation/T033-H-result.md",
)

RUNTIME_PATHS = (
    "config/settings.py",
    "services/conversation/orchestrator.py",
    "services/conversation/local_vad.py",
    "services/conversation/speech_activity_gate.py",
)
PROTECTED_RUNTIME_PATHS = (
    "services/conversation/events.py",
    "services/conversation/language_policy.py",
    "services/conversation/turn_controller.py",
    "services/stt/stt_base.py",
    "services/telephony/twilio_service.py",
    "services/tts/elevenlabs_service.py",
    "services/conversation/barge_in_diagnostics.py",
    "services/stt/elevenlabs_stt_service.py",
)
TEST_PATHS = (
    "tests/conftest.py",
    "tests/conversation/test_availability_acknowledgement.py",
    "tests/conversation/test_barge_in_diagnostics.py",
    "tests/conversation/test_booking_cancellation.py",
    "tests/conversation/test_booking_safety_guards.py",
    "tests/conversation/test_booking_time_validation.py",
    "tests/conversation/test_caller_language.py",
    "tests/conversation/test_language_policy.py",
    "tests/conversation/test_playback_ownership.py",
    "tests/conversation/test_transcript_identity.py",
    "tests/conversation/test_transfer_playback.py",
    "tests/conversation/test_turn_controller.py",
    "tests/stt/test_background_audio_pilot.py",
    "tests/stt/test_background_filter_contract.py",
    "tests/stt/test_elevenlabs_reset.py",
    "tests/stt/test_provider_compatibility_probe.py",
    "tests/stt/test_provider_failure_state.py",
    "tests/stt/test_reset_preservation.py",
    "tests/stt/test_utterance_events.py",
    "tests/stt/test_whisper_audio.py",
    "tests/telephony/test_playback_generations.py",
    "tests/telephony/test_twilio_stream.py",
    "tests/tts/test_call_isolation.py",
    "tests/conversation/test_speech_activity_gate.py",
    "tests/conversation/test_speech_aware_barge_in.py",
    "tests/evaluation/test_local_vad_audio.py",
)
TEST_SUPPORT_PATHS = (
    "docker-compose.yml",
    "scripts/check-stt-provider-compatibility.py",
    "scripts/evaluate-stt-background-audio.py",
)
BUILD_SUPPORT_PATHS = (
    "requirements.txt",
    ".gitattributes",
    "Dockerfile",
    "scripts/deploy-barge-in-diagnostics.py",
)
DOCUMENTATION_PATHS = (
    "docs/operations/speech-aware-barge-in.md",
    "docs/delegation/T033-G-result.md",
    "docs/delegation/T033-G-review.md",
)
MODEL_ASSET_PATHS = (
    "models/vad/silero_vad.onnx",
    "models/vad/LICENSE.silero-vad.txt",
    "models/vad/provenance.json",
)
ASSET_SECTIONS = (
    "runtime",
    "protected_runtime",
    "tests",
    "test_support",
    "build_support",
    "documentation",
    "model_assets",
)
SECTION_PATHS = {
    "runtime": RUNTIME_PATHS,
    "protected_runtime": PROTECTED_RUNTIME_PATHS,
    "tests": TEST_PATHS,
    "test_support": TEST_SUPPORT_PATHS,
    "build_support": BUILD_SUPPORT_PATHS,
    "documentation": DOCUMENTATION_PATHS,
    "model_assets": MODEL_ASSET_PATHS,
}
SOURCE_SCOPE = frozenset(
    {
        ".gitattributes",
        "config/settings.py",
        "docker-compose.yml",
        "requirements.txt",
        "services/conversation/orchestrator.py",
        "services/conversation/local_vad.py",
        "services/conversation/speech_activity_gate.py",
        "models/vad/silero_vad.onnx",
        "models/vad/LICENSE.silero-vad.txt",
        "models/vad/provenance.json",
    }
)
APPLICATION_SCOPE = (
    ".gitattributes",
    "api",
    "clients",
    "config",
    "migrations",
    "models",
    "services",
    "templates",
    "docker-compose.yml",
    "requirements.txt",
    "Dockerfile",
)

EXPECTED_HISTORY = (
    ("0001", "53c861ac91cae5ecaf9f794b15563c88ee58bc0dafa5ae64fa81c89d907aa3a9"),
    ("0002", "62addab700e047dff2a13973c881a874bfe6c7f88967d593f189987192782be6"),
    ("0003", "b3abaa09c340f89c32778b0b17b5c2a584845983b13eb95dbb9c366f3d6c5830"),
    ("0004", "a1c66c7705adbd49918966c7d41d8bbbc5e80d3d5d0534fe641fd15579731a9e"),
)
REQUIRED_SETTINGS = {
    "booking_observation_enabled": True,
    "booking_observation_interval_seconds": 60,
    "booking_observation_freshness_seconds": 180,
    "automatic_notifications_enabled": True,
    "automatic_notifications_interval_seconds": 60,
    "automatic_notification_workers_paused": False,
    "barge_in_diagnostics_enabled": True,
    "elevenlabs_stt_filter_background_audio": False,
}
GATE_ENV = {
    "SPEECH_AWARE_BARGE_IN_ENABLED": "true",
    "LOCAL_VAD_MODEL_PATH": "/app/models/vad/silero_vad.onnx",
    "LOCAL_VAD_PROBABILITY_THRESHOLD": "0.5",
    "LOCAL_VAD_SPEECH_DURATION_MS": "160",
    "LOCAL_VAD_MAX_INPUT_GAP_MS": "96",
    "LOCAL_VAD_MAX_INFERENCE_MS": "20.0",
}
GATE_SETTINGS = {
    "speech_aware_barge_in_enabled": True,
    "local_vad_model_path": "/app/models/vad/silero_vad.onnx",
    "local_vad_probability_threshold": 0.5,
    "local_vad_speech_duration_ms": 160,
    "local_vad_max_input_gap_ms": 96,
    "local_vad_max_inference_ms": 20.0,
}
SETTINGS_TARGET = "/app/config/settings.py"
MIN_FREE_DISK_BYTES = 1536 * 1024**2
MIN_PREFLIGHT_MEMORY_BYTES = 768 * 1024**2
POST_BUILD_RESERVE_BYTES = 384 * 1024**2

PACKAGE_VERSIONS = {
    "onnxruntime": "1.30.0",
    "flatbuffers": "25.12.19",
    "protobuf": "7.36.1",
}
WHEELS = {
    "onnxruntime": {
        "version": "1.30.0",
        "filename": "onnxruntime-1.30.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        "sha256": "fa688e7891a6aa206636fe7372e27ee75fd17713289f6b4fc7b190e0a7de9328",
        "url": "https://files.pythonhosted.org/packages/34/35/e7f862dbacbc99fadd9b14a614e49c99bf0f35fd9927a82f096e3de33531/onnxruntime-1.30.0-cp312-cp312-manylinux_2_28_x86_64.whl",
    },
    "flatbuffers": {
        "version": "25.12.19",
        "filename": "flatbuffers-25.12.19-py2.py3-none-any.whl",
        "sha256": "7634f50c427838bb021c2d66a3d1168e9d199b0607e6329399f04846d42e20b4",
        "url": "https://files.pythonhosted.org/packages/e8/2d/d2a548598be01649e2d46231d151a6c56d10b964d94043a335ae56ea2d92/flatbuffers-25.12.19-py2.py3-none-any.whl",
    },
    "protobuf": {
        "version": "7.36.1",
        "filename": "protobuf-7.36.1-cp310-abi3-manylinux2014_x86_64.whl",
        "sha256": "97198b77e369a0abd8e262b8f6c7266c55ddb796a3a12c76d7b8881188ed83aa",
        "url": "https://files.pythonhosted.org/packages/22/df/c799fe7a05ef16ba853a59db01f3a2c5f7d0676469589ccc4874f76a2a88/protobuf-7.36.1-cp310-abi3-manylinux2014_x86_64.whl",
    },
}

SETTINGS_CODE = (
    "import json; from config.settings import settings; "
    "print(json.dumps(settings.model_dump(mode='json'),sort_keys=True))"
)
PLATFORM_CODE = (
    "import json,platform,sys; from packaging.tags import sys_tags; "
    "from packaging.utils import parse_wheel_filename; "
    f"files={tuple(value['filename'] for value in WHEELS.values())!r}; "
    "supported=set(sys_tags()); "
    "print(json.dumps({'implementation':sys.implementation.name,"
    "'python':platform.python_version(),'machine':platform.machine(),"
    "'system':platform.system(),'libc_name':platform.libc_ver()[0],"
    "'libc_version':platform.libc_ver()[1],"
    "'compatible':{f:bool(parse_wheel_filename(f)[3]&supported) for f in files}},"
    "sort_keys=True))"
)
PACKAGES_CODE = (
    "import importlib.metadata as m,json; "
    "n=lambda s:s.lower().replace('_','-'); "
    "v={n(d.metadata['Name']):d.version for d in m.distributions() if d.metadata['Name']}; "
    "print(json.dumps(v,sort_keys=True))"
)
SCHEMA_0004_CODE = """import asyncio,json,os,asyncpg
from migrations.schema_contract import check_runtime_compatibility
EXPECTED = """ + repr(EXPECTED_HISTORY) + """
async def check():
 c=await asyncpg.connect(os.environ['MIGRATION_DATABASE_URL'],timeout=10,command_timeout=10)
 try:
  async with c.transaction(readonly=True):
   await check_runtime_compatibility(c,require_notification=True)
   rows=await c.fetch('SELECT version,checksum FROM public.schema_migrations ORDER BY version')
   history=tuple((r['version'],r['checksum']) for r in rows)
   if history != EXPECTED:
    raise RuntimeError('schema history differs')
   required=('booking_provider_observations','booking_provider_observation_control',
             'booking_notification_contract','booking_notification_reconciliation',
             'booking_notification_outbox')
   for table in required:
    if await c.fetchval("SELECT to_regclass('public.' || $1)",table) is None:
     raise RuntimeError('required schema object missing')
   print(json.dumps({'history':history,'read_only':True},sort_keys=True))
 finally:
  await c.close(timeout=5)
asyncio.run(check())
"""
SYNTHETIC_ENV = (
    "SECRET_KEY=synthetic-release-secret-key-000000000000",
    "DATABASE_URL=postgresql://synthetic:synthetic@invalid/test",
    "TWILIO_ACCOUNT_SID=ACsynthetic",
    "TWILIO_AUTH_TOKEN=synthetic",
    "TWILIO_PHONE_NUMBER=+12025550100",
    "DEEPGRAM_API_KEY=synthetic",
    "ELEVENLABS_API_KEY=synthetic",
    "OPENAI_API_KEY=synthetic",
    "MS_BOOKINGS_TENANT_ID=synthetic",
    "MS_BOOKINGS_CLIENT_ID=synthetic",
    "MS_BOOKINGS_CLIENT_SECRET=synthetic",
    "MS_BOOKINGS_BUSINESS_ID=synthetic",
    "TELNYX_API_KEY=synthetic",
    "TELNYX_PHONE_NUMBER=+12025550100",
    "ENVIRONMENT=test",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_text_bytes(data: bytes) -> bytes:
    """Apply the manifest's only permitted text transformation."""
    return data.replace(b"\r\n", b"\n")


def asset_digest(section: str, data: bytes) -> str:
    if section == "model_assets":
        return sha256(data)
    return sha256(normalized_text_bytes(data))


def strict_json(data: bytes) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise RuntimeError("JSON contains duplicate keys")
            value[key] = item
        return value

    try:
        value = json.loads(data, object_pairs_hook=pairs)
    except (TypeError, ValueError):
        raise RuntimeError("reviewed JSON is invalid") from None
    if not isinstance(value, dict):
        raise RuntimeError("reviewed JSON root is invalid")
    return value


def typed_mapping_equal(left: dict, right: dict) -> bool:
    return set(left) == set(right) and all(
        type(left[key]) is type(right[key]) and left[key] == right[key]
        for key in left
    )


def mount_fingerprint(mounts: list[dict]) -> list[str]:
    return sorted(json.dumps(mount, sort_keys=True) for mount in mounts)


def environment_fingerprint(environment: list[str]) -> list[str]:
    return sorted(environment)


def settings_for(previous: dict, *, candidate: bool) -> dict:
    expected = dict(previous)
    if candidate:
        expected.update(GATE_SETTINGS)
    return expected


def normalized_compose(
    configuration: dict,
    *,
    candidate: bool,
    settings_source: Path | None = None,
) -> dict:
    """Remove only the reviewed image, gate variables and settings overlay."""
    value = copy.deepcopy(configuration)
    app = value["services"]["app"]
    app["image"] = "<app-image>"
    environment = app.get("environment") or {}
    if not isinstance(environment, dict):
        raise RuntimeError("rendered app environment is not a mapping")
    present = {key: environment.get(key) for key in GATE_ENV if key in environment}
    if candidate:
        if present != GATE_ENV:
            raise RuntimeError("speech gate environment differs")
    elif present:
        raise RuntimeError("old speech gate environment unexpectedly exists")
    for key in GATE_ENV:
        environment.pop(key, None)
    app["environment"] = environment
    volumes = app.get("volumes", [])
    if not isinstance(volumes, list):
        raise RuntimeError("rendered application mounts are invalid")
    settings_mounts = [
        mount
        for mount in volumes
        if isinstance(mount, dict) and mount.get("target") == SETTINGS_TARGET
    ]
    if candidate:
        if (
            len(settings_mounts) != 1
            or settings_source is None
            or settings_mounts[0].get("source") != str(settings_source)
            or settings_mounts[0].get("type") != "bind"
            or settings_mounts[0].get("read_only") is not True
        ):
            raise RuntimeError("pinned settings mount differs")
    elif len(settings_mounts) != 1:
        raise RuntimeError("previous settings mount is not exact")
    app["volumes"] = sorted(
        [
            mount
            for mount in volumes
            if not (
                isinstance(mount, dict) and mount.get("target") == SETTINGS_TARGET
            )
        ],
        key=lambda item: json.dumps(item, sort_keys=True),
    )
    return value


def hidden_candidate_paths(mounts: list[dict]) -> tuple[str, ...]:
    hidden = []
    for source_path in (*RUNTIME_PATHS, *PROTECTED_RUNTIME_PATHS, *MODEL_ASSET_PATHS):
        if source_path == "config/settings.py":
            continue
        candidate = PurePosixPath("/app") / source_path
        for mount in mounts:
            destination = mount.get("Destination") or mount.get("target")
            if not isinstance(destination, str) or not destination.startswith("/"):
                continue
            mounted = PurePosixPath(destination)
            try:
                candidate.relative_to(mounted)
            except ValueError:
                continue
            hidden.append(source_path)
            break
    return tuple(sorted(hidden))


def container_contract(
    container: dict,
    *,
    candidate: bool,
    settings_source: Path | None = None,
) -> dict:
    config = container["Config"]
    host = container["HostConfig"]
    mounts = list(container.get("Mounts", []))
    settings_mounts = [
        mount for mount in mounts if mount.get("Destination") == SETTINGS_TARGET
    ]
    if candidate:
        if (
            len(settings_mounts) != 1
            or settings_source is None
            or settings_mounts[0].get("Source") != str(settings_source)
            or settings_mounts[0].get("Type") != "bind"
            or settings_mounts[0].get("RW") is not False
        ):
            raise RuntimeError("replacement pinned settings mount differs")
    elif len(settings_mounts) != 1:
        raise RuntimeError("previous settings mount is not exact")
    mounts = [
        mount for mount in mounts if mount.get("Destination") != SETTINGS_TARGET
    ]
    environment = []
    overrides: dict[str, list[str]] = {}
    for item in config.get("Env") or []:
        key, separator, value = item.partition("=")
        if key in GATE_ENV:
            overrides.setdefault(key, []).append(value if separator else "")
        else:
            environment.append(item)
    if candidate:
        if any(overrides.get(key) != [expected] for key, expected in GATE_ENV.items()):
            raise RuntimeError("replacement speech gate environment differs")
    elif overrides:
        raise RuntimeError("old speech gate environment unexpectedly exists")
    return {
        "mounts": mount_fingerprint(mounts),
        "environment": environment_fingerprint(environment),
        "command": config.get("Cmd"),
        "entrypoint": config.get("Entrypoint"),
        "ports": host.get("PortBindings"),
        "restart": host.get("RestartPolicy"),
        "networks": sorted(container["NetworkSettings"]["Networks"]),
    }


def select_wheels(platform: dict) -> dict:
    expected_keys = {
        "implementation",
        "python",
        "machine",
        "system",
        "libc_name",
        "libc_version",
        "compatible",
    }
    if set(platform) != expected_keys:
        raise RuntimeError("target image platform evidence differs")
    version = str(platform["python"]).split(".")
    try:
        libc = tuple(int(part) for part in str(platform["libc_version"]).split(".")[:2])
    except ValueError:
        raise RuntimeError("unsupported target image platform") from None
    expected_compatibility = {wheel["filename"]: True for wheel in WHEELS.values()}
    if (
        platform["implementation"] != "cpython"
        or version[:2] != ["3", "12"]
        or platform["machine"] != "x86_64"
        or platform["system"] != "Linux"
        or platform["libc_name"] != "glibc"
        or libc < (2, 28)
        or platform["compatible"] != expected_compatibility
    ):
        raise RuntimeError("unsupported target image platform")
    for wheel in WHEELS.values():
        if not wheel["url"].startswith("https://files.pythonhosted.org/packages/"):
            raise RuntimeError("wheel endpoint is not the official package host")
    return WHEELS


def source_probe_code(text_hashes: dict[str, str], model_hashes: dict[str, str]) -> str:
    modules = tuple(path[:-3].replace("/", ".") for path in RUNTIME_PATHS)
    return (
        "import hashlib,importlib,importlib.metadata as md,json; from pathlib import Path; "
        f"text={text_hashes!r}; raw={model_hashes!r}; modules={modules!r}; "
        "hashes={p:hashlib.sha256(Path('/app',p).read_bytes().replace(b'\\r\\n',b'\\n')).hexdigest() for p in text}; "
        "hashes.update({p:hashlib.sha256(Path('/app',p).read_bytes()).hexdigest() for p in raw}); "
        "[importlib.import_module(m) for m in modules]; "
        "from config.settings import settings; configured=settings.model_dump(mode='json'); "
        "from services.conversation.local_vad import LocalVadModel; "
        "model=LocalVadModel(configured['local_vad_model_path'],max_inference_ms=configured['local_vad_max_inference_ms']); "
        "norm=lambda s:s.lower().replace('_','-'); "
        "packages={norm(d.metadata['Name']):d.version for d in md.distributions() if d.metadata['Name']}; "
        "print(json.dumps({'hashes':hashes,'settings':configured,'packages':packages,'model_loaded':model is not None},sort_keys=True))"
    )


def validate_effective_probe(
    value: dict,
    *,
    hashes: dict[str, str],
    settings: dict,
    packages: dict[str, str],
) -> None:
    if (
        value.get("hashes") != hashes
        or not isinstance(value.get("settings"), dict)
        or not typed_mapping_equal(value["settings"], settings)
        or value.get("packages") != packages
        or value.get("model_loaded") is not True
    ):
        raise RuntimeError("effective candidate source, settings, packages or model differ")


def measurement_code() -> str:
    return """import json,statistics,time
from pathlib import Path
def rss():
 for line in Path('/proc/self/status').read_text().splitlines():
  if line.startswith('VmRSS:'):
   return int(line.split()[1])*1024
 raise RuntimeError('rss unavailable')
before=rss()
from services.conversation.local_vad import LocalVadModel
model=LocalVadModel('/app/models/vad/silero_vad.onnx',max_inference_ms=100.0)
state=model.create_state()
after=rss()
frame=[0]*256
for _ in range(20): model.classify(frame,state)
times=[]
for _ in range(500):
 started=time.perf_counter(); model.classify(frame,state); times.append((time.perf_counter()-started)*1000.0)
ordered=sorted(times)
print(json.dumps({'model_loaded':True,'frames':len(times),'rss_delta_bytes':max(0,after-before),
 'mean_ms':statistics.fmean(times),'p95_ms':ordered[int(len(ordered)*0.95)-1],
 'max_ms':max(times)},sort_keys=True))
"""


class Release:
    def __init__(self, directory: Path, commit: str):
        self.directory = directory
        self.commit = commit
        self.root = ROOT
        self.stage = "preflight"
        self.old: dict | None = None
        self.previous_settings: dict | None = None
        self.previous_rendered: dict | None = None
        self.previous_contract: dict | None = None
        self.network: str | None = None
        self.live_schema_env: Path | None = None
        self.manifest: dict | None = None
        self.runtime_hashes: dict[str, str] = {}
        self.protected_runtime_hashes: dict[str, str] = {}
        self.model_hashes: dict[str, str] = {}
        self.baseline_packages: dict[str, str] = {}
        self.candidate_packages: dict[str, str] = {}
        self.platform: dict | None = None
        self.assets = self.directory / "assets"
        self.wheelhouse = self.directory / "wheelhouse"
        self.settings_overlay = self.directory / "candidate-settings.py"

    def private_write(self, name: str, data: bytes) -> Path:
        path = self.directory / name
        if path.is_symlink():
            raise RuntimeError("private path is a symlink")
        with path.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        path.chmod(0o600)
        return path

    def asset_write(self, relative: str, data: bytes) -> Path:
        path = self.assets / relative
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.is_symlink() or path.exists():
            raise RuntimeError("staged asset path is not empty")
        with path.open("xb") as output:
            output.write(data)
        path.chmod(0o600)
        return path

    def run(self, *args, timeout=60, check=True):
        try:
            result = subprocess.run(
                args,
                cwd=self.root,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            with (self.directory / "private.log").open("ab") as log:
                log.write((error.stdout or b"") + (error.stderr or b""))
            raise RuntimeError("command timed out; protected log retained") from None
        except OSError:
            raise RuntimeError("command unavailable; protected log retained") from None
        with (self.directory / "private.log").open("ab") as log:
            log.write(result.stdout + result.stderr)
        if check and result.returncode:
            raise RuntimeError("command failed; protected log retained")
        return result

    def inspect(self, name: str) -> dict:
        return json.loads(self.run("docker", "inspect", name).stdout)[0]

    def compose(self, override: Path, *args, **kwargs):
        return self.run(
            "docker",
            "compose",
            "--project-directory",
            str(self.root),
            "-p",
            PROJECT,
            "-f",
            str(self.root / "docker-compose.yml"),
            "-f",
            str(override),
            *args,
            **kwargs,
        )

    def git_bytes(self, path: str) -> bytes:
        return self.run("git", "show", self.commit + ":" + path).stdout

    def settings(self, name: str = "ai-voice-app") -> dict:
        return strict_json(
            self.run("docker", "exec", name, "python", "-c", SETTINGS_CODE).stdout
        )

    def render_compose(self, override: Path) -> dict:
        return strict_json(self.compose(override, "config", "--format", "json").stdout)

    def public_health(self) -> None:
        for _ in range(12):
            result = self.run(
                "curl",
                "-fsS",
                "--max-time",
                "5",
                PUBLIC_HEALTH_URL,
                check=False,
            )
            if result.returncode == 0:
                try:
                    if json.loads(result.stdout).get("status") == "healthy":
                        return
                except ValueError:
                    pass
            time.sleep(1)
        raise RuntimeError("public certificate-validated HTTPS health failed")

    def wait_ready(self, name: str = "ai-voice-app") -> None:
        for _ in range(60):
            result = self.run(
                "docker",
                "exec",
                name,
                "curl",
                "-fsS",
                "--max-time",
                "3",
                "http://localhost:8000/ready",
                check=False,
            )
            if result.returncode == 0:
                try:
                    if json.loads(result.stdout).get("status") == "ready":
                        state = self.inspect(name)["State"]
                        if state["Running"] and state.get("Health", {}).get("Status") == "healthy":
                            return
                except ValueError:
                    pass
            time.sleep(2)
        raise RuntimeError("application readiness deadline exceeded")

    def remove_container(self, name: str) -> None:
        self.run("docker", "rm", "-f", name, check=False)
        listing = self.run("docker", "ps", "-a", "--format", "{{.Names}}")
        if name in listing.stdout.decode().splitlines():
            raise RuntimeError("tracked container cleanup unverified")

    def log_cleanup_failure(self, error: BaseException) -> None:
        with (self.directory / "private.log").open("ab") as log:
            log.write(("cleanup_failure_type=" + type(error).__name__ + "\n").encode())

    def one_shot(
        self,
        name: str,
        image: str,
        arguments: list[str],
        *,
        network: str = "none",
        env_file: Path | None = None,
        mounts: tuple[str, ...] = (),
        synthetic_env: bool = False,
        timeout: int = 120,
        check: bool = True,
        memory: str = "512m",
    ):
        command = [
            "docker",
            "run",
            "--name",
            name,
            "--pull",
            "never",
            "--network",
            network,
            "--memory",
            memory,
            "--cpus",
            "1",
            "--pids-limit",
            "128",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=96m",
        ]
        if env_file is not None:
            command.extend(("--env-file", str(env_file)))
        if synthetic_env:
            for value in SYNTHETIC_ENV:
                command.extend(("--env", value))
        for mount in mounts:
            command.extend(("--mount", mount))
        command.extend(("--entrypoint", "python", image, *arguments))
        try:
            result = self.run(*command, timeout=timeout, check=check)
        except BaseException:
            try:
                self.remove_container(name)
            except BaseException as cleanup_error:
                self.log_cleanup_failure(cleanup_error)
            raise
        else:
            self.remove_container(name)
            return result

    def available_memory(self) -> int:
        match = re.search(
            r"^MemAvailable:\s+(\d+) kB$",
            Path("/proc/meminfo").read_text(),
            re.MULTILINE,
        )
        if match is None:
            raise RuntimeError("release memory evidence unavailable")
        return int(match.group(1)) * 1024

    def capacity(
        self,
        *,
        available_memory: int | None = None,
        measured_rss_delta: int | None = None,
    ) -> None:
        if shutil.disk_usage(self.root).free < MIN_FREE_DISK_BYTES:
            raise RuntimeError("insufficient release disk capacity")
        memory = self.available_memory() if available_memory is None else available_memory
        if memory < MIN_PREFLIGHT_MEMORY_BYTES and measured_rss_delta is None:
            raise RuntimeError("insufficient release memory capacity")
        if measured_rss_delta is not None:
            if measured_rss_delta <= 0:
                raise RuntimeError("candidate model RSS measurement is invalid")
            required = POST_BUILD_RESERVE_BYTES + 4 * measured_rss_delta
            if memory < required:
                raise RuntimeError("insufficient four-worker candidate memory capacity")

    def verify_source(self) -> None:
        if self.run("git", "rev-parse", "HEAD").stdout.decode().strip() != BASE_CHECKOUT:
            raise RuntimeError("server checkout changed")
        for revision in (DEPLOYED_SOURCE, self.commit):
            actual = self.run("git", "rev-parse", revision + "^{commit}").stdout.decode().strip()
            if actual != revision:
                raise RuntimeError("release source unavailable")
            if self.run(
                "git",
                "merge-base",
                "--is-ancestor",
                BASE_CHECKOUT,
                revision,
                check=False,
            ).returncode:
                raise RuntimeError("release source is not descended from server baseline")
        if self.run(
            "git",
            "merge-base",
            "--is-ancestor",
            DEPLOYED_SOURCE,
            self.commit,
            check=False,
        ).returncode:
            raise RuntimeError("release source is not descended from deployed app")
        status = self.run("git", "status", "--porcelain", "--untracked-files=no").stdout.decode().strip()
        if status != "M nginx/nginx.conf":
            raise RuntimeError("unexpected server checkout modification")
        if sha256((self.root / "nginx/nginx.conf").read_bytes()) != NGINX_HASH:
            raise RuntimeError("nginx hotfix changed")
        changed = self.run(
            "git",
            "diff",
            "--name-only",
            DEPLOYED_SOURCE,
            self.commit,
            "--",
            *APPLICATION_SCOPE,
        ).stdout.decode().splitlines()
        if len(changed) != len(SOURCE_SCOPE) or set(changed) != SOURCE_SCOPE:
            raise RuntimeError("release source scope differs")
        self.run("git", "cat-file", "-e", self.commit + ":" + MANIFEST_PATH)
        for path in RELEASE_PATHS:
            self.run("git", "cat-file", "-e", self.commit + ":" + path)

    def stage_assets(self) -> None:
        manifest_source = self.git_bytes(MANIFEST_PATH)
        manifest_bytes = normalized_text_bytes(manifest_source)
        if sha256(manifest_bytes) != MANIFEST_SHA256:
            raise RuntimeError("accepted manifest hash differs")
        manifest = strict_json(manifest_bytes)
        if (
            manifest.get("schema") != 1
            or manifest.get("baseline_commit") != DEPLOYED_SOURCE
            or manifest.get("baseline_image") != OLD_IMAGE
            or manifest.get("branch") != "codex/phase-4-arabic-voice-quality"
            or manifest.get("reviewed_worktree_head") != REVIEWED_INPUT_HEAD
            or any(set(manifest.get(section, {})) != set(paths) for section, paths in SECTION_PATHS.items())
            or set(manifest.get("source_scope", ())) != SOURCE_SCOPE
            or manifest.get("candidate_setting_overrides") != GATE_SETTINGS
        ):
            raise RuntimeError("accepted manifest contract differs")
        if "model_assets" not in str(manifest.get("hash_convention", "")) or "RAW" not in str(
            manifest.get("hash_convention", "")
        ):
            raise RuntimeError("accepted manifest byte convention differs")
        if sum(len(manifest[section]) for section in ASSET_SECTIONS) != 51:
            raise RuntimeError("accepted manifest asset count differs")
        self.assets.mkdir(mode=0o700)
        self.asset_write("manifest.json", manifest_bytes)
        for section, paths in SECTION_PATHS.items():
            for path in paths:
                source = self.git_bytes(path)
                data = source if section == "model_assets" else normalized_text_bytes(source)
                if sha256(data) != manifest[section][path]:
                    raise RuntimeError("reviewed asset hash differs")
                self.asset_write(path, data)
        self.manifest = manifest
        self.runtime_hashes = dict(manifest["runtime"])
        self.protected_runtime_hashes = dict(manifest["protected_runtime"])
        self.model_hashes = dict(manifest["model_assets"])
        print("REVIEWED_SPEECH_GATE_ASSETS_STAGED_51", flush=True)

    def inspect_platform(self, image: str) -> dict:
        result = self.one_shot(
            "t033h-platform-" + self.directory.name,
            image,
            ["-c", PLATFORM_CODE],
            timeout=60,
        )
        value = strict_json(result.stdout)
        select_wheels(value)
        return value

    def package_inventory(self, image: str) -> dict[str, str]:
        result = self.one_shot(
            "t033h-packages-" + self.directory.name,
            image,
            ["-c", PACKAGES_CODE],
            timeout=60,
        )
        value = strict_json(result.stdout)
        if not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
            raise RuntimeError("package inventory is invalid")
        return value

    def verify_package_transition(self, baseline: dict, candidate: dict) -> None:
        if baseline.get("numpy") != "1.26.4":
            raise RuntimeError("baseline package numpy differs")
        if any(name in baseline for name in PACKAGE_VERSIONS):
            raise RuntimeError("baseline package additions unexpectedly exist")
        expected = {**baseline, **PACKAGE_VERSIONS}
        if candidate != expected:
            raise RuntimeError("candidate package inventory differs")

    def verify_wheelhouse(self, wheelhouse: Path) -> dict:
        expected_names = {wheel["filename"] for wheel in WHEELS.values()}
        actual_names = {path.name for path in wheelhouse.iterdir() if path.is_file()}
        if actual_names != expected_names:
            raise RuntimeError("downloaded wheel set differs")
        evidence = {}
        for package, wheel in WHEELS.items():
            path = wheelhouse / wheel["filename"]
            digest = sha256(path.read_bytes())
            if digest != wheel["sha256"]:
                raise RuntimeError("downloaded wheel integrity differs")
            evidence[package] = {
                "version": wheel["version"],
                "filename": wheel["filename"],
                "sha256": digest,
                "source": wheel["url"],
            }
        return evidence

    def prepare_wheels(self, image: str) -> None:
        if self.platform is None:
            self.platform = self.inspect_platform(image)
        selected = select_wheels(self.platform)
        if not self.baseline_packages:
            self.baseline_packages = self.package_inventory(image)
        if self.baseline_packages.get("numpy") != "1.26.4":
            raise RuntimeError("baseline package numpy differs")
        if any(name in self.baseline_packages for name in PACKAGE_VERSIONS):
            raise RuntimeError("baseline package additions unexpectedly exist")
        # ONNX 1.30 requires SymPy only for its unrequested symbolic extra.
        if "packaging" not in self.baseline_packages:
            raise RuntimeError("baseline ONNX dependency differs")
        self.wheelhouse.mkdir(mode=0o700)
        downloads = [(value["filename"], value["url"]) for value in selected.values()]
        code = (
            "import json,pathlib,urllib.parse,urllib.request; "
            f"items={downloads!r}; root=pathlib.Path('/wheelhouse'); "
            "[(lambda r,n:(r.geturl().startswith('https://files.pythonhosted.org/packages/') or (_ for _ in ()).throw(RuntimeError('redirect')), (root/n).write_bytes(r.read())))(urllib.request.urlopen(u,timeout=30),n) for n,u in items]"
        )
        self.one_shot(
            "t033h-wheels-" + self.directory.name,
            image,
            ["-c", code],
            network="bridge",
            mounts=(
                "type=bind,src=" + str(self.wheelhouse) + ",dst=/wheelhouse",
            ),
            timeout=180,
        )
        evidence = self.verify_wheelhouse(self.wheelhouse)
        self.private_write(
            "wheel-evidence.json",
            (
                json.dumps(
                    {
                        "index": "https://pypi.org/simple",
                        "package_host": "https://files.pythonhosted.org",
                        "platform": self.platform,
                        "wheels": evidence,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode(),
        )
        print("PINNED_TARGET_WHEELS_DOWNLOADED_AND_HASHED", flush=True)

    def _validate_settings(self, settings: dict, *, candidate: bool) -> None:
        if any(
            key not in settings
            or type(settings[key]) is not type(value)
            or settings[key] != value
            for key, value in REQUIRED_SETTINGS.items()
        ):
            raise RuntimeError("notification, diagnostics or provider filter settings differ")
        if candidate:
            expected = settings_for(self.previous_settings, candidate=True)
            if not typed_mapping_equal(settings, expected):
                raise RuntimeError("candidate effective settings differ")
        elif any(key in settings for key in GATE_SETTINGS):
            raise RuntimeError("old settings unexpectedly contain speech gate fields")

    def schema_matches(self, image: str = OLD_IMAGE) -> bool:
        result = self.one_shot(
            "t033h-schema-" + self.directory.name,
            image,
            ["-c", SCHEMA_0004_CODE],
            network=self.network,
            env_file=self.live_schema_env,
            timeout=60,
            check=False,
        )
        if result.returncode:
            return False
        try:
            value = json.loads(result.stdout)
        except ValueError:
            return False
        return value.get("read_only") is True and tuple(
            tuple(item) for item in value.get("history", ())
        ) == EXPECTED_HISTORY

    def preflight(self) -> dict:
        self.verify_source()
        self.capacity()
        override = self.root / "docker-compose.override.yml"
        metadata = override.stat()
        if (
            override.is_symlink()
            or metadata.st_uid != 0
            or metadata.st_mode & 0o077
            or not stat.S_ISREG(metadata.st_mode)
        ):
            raise RuntimeError("private override is not root protected")
        override_bytes = override.read_bytes()
        original = strict_json(override_bytes)
        if original["services"]["app"]["image"] != OLD_IMAGE:
            raise RuntimeError("private override does not pin running image")
        old = self.inspect("ai-voice-app")
        if (
            old["Image"] != OLD_IMAGE
            or not old["State"]["Running"]
            or old["State"].get("Health", {}).get("Status") != "healthy"
            or old["Config"]["Labels"].get("org.opencontainers.image.revision") != DEPLOYED_SOURCE
            or old["Config"]["Labels"].get("com.docker.compose.project") != PROJECT
        ):
            raise RuntimeError("running app does not match accepted source and image")
        if hidden_candidate_paths(old.get("Mounts", [])):
            raise RuntimeError("running mount hides reviewed candidate source")
        nginx = self.inspect("ai-voice-nginx")
        if (
            not nginx["State"]["Running"]
            or nginx["Config"].get("StopSignal", "").upper() not in ("SIGQUIT", "QUIT", "3")
        ):
            raise RuntimeError("nginx is not ready for graceful stop")
        redis = self.inspect("ai-voice-redis")
        if not redis["State"]["Running"] or redis["State"].get("Health", {}).get(
            "Status", "healthy"
        ) not in ("healthy", ""):
            raise RuntimeError("redis is not healthy")
        database = self.inspect("ai-voice-db")
        if not database["State"]["Running"] or database["State"].get("Health", {}).get("Status") != "healthy":
            raise RuntimeError("database is not healthy")
        common = set(old["NetworkSettings"]["Networks"]) & set(database["NetworkSettings"]["Networks"])
        if len(common) != 1:
            raise RuntimeError("application and database network differ")
        network, = common
        settings = self.settings()
        self._validate_settings(settings, candidate=False)
        database_url = settings.get("database_url")
        if not isinstance(database_url, str) or "\r" in database_url or "\n" in database_url:
            raise RuntimeError("live database target is invalid")
        parsed = urlsplit(database_url)
        database_env = dict(item.split("=", 1) for item in database["Config"]["Env"])
        expected_database = database_env.get("POSTGRES_DB", database_env.get("POSTGRES_USER", "postgres"))
        network_info = database["NetworkSettings"]["Networks"][network]
        hosts = {
            "db",
            "ai-voice-db",
            network_info.get("IPAddress"),
            *(network_info.get("Aliases") or []),
        }
        if parsed.hostname not in hosts or unquote(parsed.path.lstrip("/")) != expected_database:
            raise RuntimeError("application and database targets differ")
        self.run("docker", "inspect", OLD_IMAGE)
        self.run("docker", "exec", "ai-voice-nginx", "nginx", "-t")
        self.public_health()
        rendered = self.render_compose(override)
        if hidden_candidate_paths(rendered["services"]["app"].get("volumes", [])):
            raise RuntimeError("effective mount hides reviewed candidate source")
        self.platform = self.inspect_platform(OLD_IMAGE)
        self.baseline_packages = self.package_inventory(OLD_IMAGE)
        if self.baseline_packages.get("numpy") != "1.26.4" or any(
            name in self.baseline_packages for name in PACKAGE_VERSIONS
        ):
            raise RuntimeError("baseline package inventory differs")
        self.old = old
        self.previous_settings = settings
        self.previous_rendered = rendered
        self.previous_contract = container_contract(old, candidate=False)
        self.network = network
        self.private_write("previous-override.json", override_bytes)
        self.private_write("previous-app.json", json.dumps(old).encode())
        self.private_write("previous-settings.json", json.dumps(settings).encode())
        self.private_write("previous-rendered.json", json.dumps(rendered).encode())
        self.private_write("previous-contract.json", json.dumps(self.previous_contract).encode())
        self.private_write("target-platform.json", (json.dumps(self.platform, sort_keys=True) + "\n").encode())
        self.private_write("baseline-packages.json", (json.dumps(self.baseline_packages, sort_keys=True) + "\n").encode())
        self.private_write("base-compose.sha256", sha256((self.root / "docker-compose.yml").read_bytes()).encode())
        self.live_schema_env = self.private_write(
            "live-schema.env",
            ("MIGRATION_DATABASE_URL=" + database_url + "\n").encode(),
        )
        if not self.schema_matches(OLD_IMAGE):
            raise RuntimeError("live read-only schema 0004 contract differs")
        print("PREFLIGHT_0004_DIAGNOSTICS_FILTER_SETTINGS_PLATFORM_READY_HTTPS_OK", flush=True)
        return original

    def build_candidate(self) -> str:
        if self.manifest is None or not self.runtime_hashes or not self.model_hashes:
            raise RuntimeError("reviewed assets were not staged")
        self.verify_wheelhouse(self.wheelhouse)
        parent = "ai-phone-t033h-parent:" + self.directory.name
        candidate_tag = "ai-phone-t033h-candidate:" + self.directory.name
        self.run("docker", "tag", OLD_IMAGE, parent)
        if self.inspect(parent)["Id"] != OLD_IMAGE:
            raise RuntimeError("candidate parent image changed")
        context = self.directory / "build"
        context.mkdir(mode=0o700)
        copied = (*RUNTIME_PATHS, *MODEL_ASSET_PATHS)
        for path in copied:
            source = self.assets / path
            target = context / path
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            target.write_bytes(source.read_bytes())
            target.chmod(0o600)
        target_wheels = context / "wheelhouse"
        target_wheels.mkdir(mode=0o700)
        for wheel in WHEELS.values():
            target = target_wheels / wheel["filename"]
            target.write_bytes((self.wheelhouse / wheel["filename"]).read_bytes())
            target.chmod(0o600)
        wheel_files = " ".join("/tmp/t033h-wheels/" + wheel["filename"] for wheel in WHEELS.values())
        dockerfile = (
            "FROM " + parent + "\n"
            "COPY wheelhouse /tmp/t033h-wheels\n"
            "RUN python -m pip install --disable-pip-version-check --no-index --no-deps --only-binary=:all: "
            + wheel_files
            + " && rm -rf /tmp/t033h-wheels\n"
            + "".join("COPY " + path + " /app/" + path + "\n" for path in copied)
        )
        (context / "Dockerfile").write_text(dockerfile, encoding="utf-8", newline="\n")
        (context / "Dockerfile").chmod(0o600)
        try:
            self.run(
                "docker",
                "build",
                "--network",
                "none",
                "--pull=false",
                "--label",
                "org.opencontainers.image.revision=" + self.commit,
                "-t",
                candidate_tag,
                str(context),
                timeout=300,
            )
        finally:
            self.run("docker", "image", "rm", parent, check=False)
        built = self.inspect(candidate_tag)
        image = built["Id"]
        if not image.startswith("sha256:"):
            raise RuntimeError("candidate image has no immutable identity")
        if built["Config"]["Labels"].get("org.opencontainers.image.revision") != self.commit:
            raise RuntimeError("candidate image source label differs")
        self.candidate_packages = self.package_inventory(image)
        self.verify_package_transition(self.baseline_packages, self.candidate_packages)
        text_hashes = {**self.runtime_hashes, **self.protected_runtime_hashes}
        expected_hashes = {**text_hashes, **self.model_hashes}
        probe = self.one_shot(
            "t033h-image-probe-" + self.directory.name,
            image,
            ["-c", source_probe_code(text_hashes, self.model_hashes)],
            synthetic_env=True,
            memory="768m",
        )
        value = strict_json(probe.stdout)
        default_settings = value.get("settings", {})
        if (
            value.get("hashes") != expected_hashes
            or value.get("packages") != self.candidate_packages
            or value.get("model_loaded") is not True
            or default_settings.get("speech_aware_barge_in_enabled") is not False
            or default_settings.get("elevenlabs_stt_filter_background_audio") is not False
        ):
            raise RuntimeError("candidate image source, package, model or defaults differ")
        self.private_write(
            "candidate-packages.json",
            (json.dumps(self.candidate_packages, sort_keys=True) + "\n").encode(),
        )
        print("SPEECH_GATE_CANDIDATE_IMAGE_WHEELS_HASHES_MODEL_IMPORTS_OK", flush=True)
        return image

    def run_candidate_tests(self, image: str) -> None:
        mounts = (
            "type=bind,src=" + str(self.assets / "tests") + ",dst=/audit/tests,readonly",
            "type=bind,src=" + str(self.assets / "scripts") + ",dst=/audit/scripts,readonly",
            "type=bind,src=" + str(self.assets / "docker-compose.yml") + ",dst=/audit/docker-compose.yml,readonly",
            "type=bind,src=" + str(self.assets / "models") + ",dst=/audit/models,readonly",
        )
        arguments = [
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "/audit/tests/stt",
            "/audit/tests/tts",
            "/audit/tests/telephony",
            "/audit/tests/conversation",
            "/audit/tests/evaluation/test_local_vad_audio.py",
        ]
        self.one_shot(
            "t033h-tests-" + self.directory.name,
            image,
            arguments,
            mounts=mounts,
            synthetic_env=True,
            timeout=240,
            memory="768m",
        )
        print("SPEECH_GATE_CANDIDATE_REAL_MODEL_AND_SPEECH_TESTS_PASSED", flush=True)

    def measure_candidate(self, image: str) -> dict:
        result = self.one_shot(
            "t033h-warm-" + self.directory.name,
            image,
            ["-c", measurement_code()],
            synthetic_env=True,
            timeout=120,
            memory="768m",
        )
        value = strict_json(result.stdout)
        numeric = ("rss_delta_bytes", "mean_ms", "p95_ms", "max_ms")
        if (
            value.get("model_loaded") is not True
            or value.get("frames") != 500
            or any(type(value.get(key)) not in (int, float) or value[key] < 0 for key in numeric)
            or value["rss_delta_bytes"] <= 0
            or value["p95_ms"] > GATE_SETTINGS["local_vad_max_inference_ms"]
        ):
            raise RuntimeError("candidate model measurement differs")
        self.capacity(
            available_memory=self.available_memory(),
            measured_rss_delta=int(value["rss_delta_bytes"]),
        )
        self.private_write(
            "model-resource-evidence.json",
            (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
        )
        print("CANDIDATE_MODEL_WARMED_MEASURED_AND_FOUR_WORKER_CAPACITY_OK", flush=True)
        return value

    def candidate_probe(self, override: Path, label: str) -> dict:
        name = "t033h-settings-" + label + "-" + self.directory.name
        text_hashes = {**self.runtime_hashes, **self.protected_runtime_hashes}
        try:
            result = self.compose(
                override,
                "run",
                "--rm",
                "--no-deps",
                "--pull",
                "never",
                "--name",
                name,
                "--entrypoint",
                "python",
                "app",
                "-c",
                source_probe_code(text_hashes, self.model_hashes),
                timeout=120,
            )
            value = strict_json(result.stdout)
        except BaseException:
            try:
                self.remove_container(name)
            except BaseException as cleanup_error:
                self.log_cleanup_failure(cleanup_error)
            raise
        else:
            self.remove_container(name)
            return value

    def prepare_override(self, image: str, original: dict) -> Path:
        candidate = copy.deepcopy(original)
        app = candidate["services"]["app"]
        app["image"] = image
        environment = app.setdefault("environment", {})
        if not isinstance(environment, dict):
            raise RuntimeError("private app environment is not a mapping")
        if any(key in environment for key in GATE_ENV):
            raise RuntimeError("private app already has speech gate environment")
        environment.update(GATE_ENV)
        settings_bytes = (self.assets / "config/settings.py").read_bytes()
        if sha256(settings_bytes) != self.runtime_hashes.get("config/settings.py"):
            raise RuntimeError("candidate settings source differs")
        self.private_write("candidate-settings.py", settings_bytes)
        self.settings_overlay.chmod(0o644)
        volumes = app.setdefault("volumes", [])
        if not isinstance(volumes, list):
            raise RuntimeError("private app mounts are invalid")
        settings_mounts = [
            mount
            for mount in volumes
            if isinstance(mount, dict) and mount.get("target") == SETTINGS_TARGET
        ]
        if len(settings_mounts) != 1:
            raise RuntimeError("private settings mount is not exact")
        volumes[:] = [
            mount
            for mount in volumes
            if not (isinstance(mount, dict) and mount.get("target") == SETTINGS_TARGET)
        ]
        volumes.append(
            {
                "type": "bind",
                "source": str(self.settings_overlay),
                "target": SETTINGS_TARGET,
                "read_only": True,
            }
        )
        path = self.private_write(
            "candidate-override.json",
            (json.dumps(candidate, indent=2) + "\n").encode(),
        )
        rendered = self.render_compose(path)
        if normalized_compose(
            rendered,
            candidate=True,
            settings_source=self.settings_overlay,
        ) != normalized_compose(self.previous_rendered, candidate=False):
            raise RuntimeError("effective Compose configuration changed")
        if hidden_candidate_paths(rendered["services"]["app"].get("volumes", [])):
            raise RuntimeError("candidate mount hides reviewed source or model")
        expected_settings = settings_for(self.previous_settings, candidate=True)
        expected_hashes = {
            **self.runtime_hashes,
            **self.protected_runtime_hashes,
            **self.model_hashes,
        }
        probe = self.candidate_probe(path, "candidate")
        validate_effective_probe(
            probe,
            hashes=expected_hashes,
            settings=expected_settings,
            packages=self.candidate_packages,
        )
        print("SPEECH_GATE_SETTINGS_MOUNTS_PACKAGES_AND_MODEL_LOAD_VERIFIED", flush=True)
        return path

    def recheck_before_stop(self) -> None:
        previous = self.directory / "previous-override.json"
        live = self.root / "docker-compose.override.yml"
        if live.is_symlink() or live.read_bytes() != previous.read_bytes():
            raise RuntimeError("private override changed before cutover")
        if sha256((self.root / "docker-compose.yml").read_bytes()) != (
            self.directory / "base-compose.sha256"
        ).read_text():
            raise RuntimeError("base Compose changed before cutover")
        if sha256((self.root / "nginx/nginx.conf").read_bytes()) != NGINX_HASH:
            raise RuntimeError("nginx hotfix changed before cutover")
        current = self.inspect("ai-voice-app")
        if (
            current["Image"] != OLD_IMAGE
            or not current["State"]["Running"]
            or current["State"].get("Health", {}).get("Status") != "healthy"
            or container_contract(current, candidate=False) != self.previous_contract
            or not typed_mapping_equal(self.settings(), self.previous_settings)
        ):
            raise RuntimeError("running application changed before cutover")
        for name in ("ai-voice-db", "ai-voice-redis", "ai-voice-nginx"):
            if not self.inspect(name)["State"]["Running"]:
                raise RuntimeError("dependency changed before cutover")
        self.run("docker", "exec", "ai-voice-nginx", "nginx", "-t")
        if not self.schema_matches(OLD_IMAGE):
            raise RuntimeError("schema 0004 changed before cutover")
        self.public_health()

    def stop_components(self, *, strict: bool) -> None:
        for name, seconds in (("ai-voice-nginx", "120"), ("ai-voice-app", "60")):
            try:
                self.run(
                    "docker",
                    "stop",
                    "--time",
                    seconds,
                    name,
                    timeout=int(seconds) + 15,
                    check=False,
                )
            except BaseException:
                pass
        try:
            running = [
                name
                for name in ("ai-voice-nginx", "ai-voice-app")
                if self.inspect(name)["State"]["Running"]
            ]
        except BaseException:
            running = ["unverified"]
        if strict and running:
            raise RuntimeError("ingress or application stop could not be verified")

    def install_override(self, source: Path) -> None:
        live = self.root / "docker-compose.override.yml"
        if source.is_symlink() or live.is_symlink():
            raise RuntimeError("override path is a symlink")
        temporary = self.root / (".t033h-override-" + self.directory.name)
        with temporary.open("xb") as output:
            output.write(source.read_bytes())
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, live)

    def running_probe(self) -> dict:
        text_hashes = {**self.runtime_hashes, **self.protected_runtime_hashes}
        return strict_json(
            self.run(
                "docker",
                "exec",
                "ai-voice-app",
                "python",
                "-c",
                source_probe_code(text_hashes, self.model_hashes),
            ).stdout
        )

    def verify_replaced(self, image: str, revision: str, *, candidate: bool) -> None:
        self.wait_ready()
        current = self.inspect("ai-voice-app")
        if (
            current["Image"] != image
            or not current["State"]["Running"]
            or current["State"].get("Health", {}).get("Status") != "healthy"
            or current["Config"]["Labels"].get("org.opencontainers.image.revision") != revision
            or container_contract(
                current,
                candidate=candidate,
                settings_source=self.settings_overlay if candidate else None,
            )
            != self.previous_contract
        ):
            raise RuntimeError("replacement image or runtime contract differs")
        settings = self.settings()
        self._validate_settings(settings, candidate=candidate)
        expected_settings = settings_for(self.previous_settings, candidate=candidate)
        if not typed_mapping_equal(settings, expected_settings):
            raise RuntimeError("replacement effective settings changed")
        if candidate:
            expected_hashes = {
                **self.runtime_hashes,
                **self.protected_runtime_hashes,
                **self.model_hashes,
            }
            validate_effective_probe(
                self.running_probe(),
                hashes=expected_hashes,
                settings=expected_settings,
                packages=self.candidate_packages,
            )
        rendered = self.render_compose(self.root / "docker-compose.override.yml")
        if normalized_compose(
            rendered,
            candidate=candidate,
            settings_source=self.settings_overlay if candidate else None,
        ) != normalized_compose(self.previous_rendered, candidate=False):
            raise RuntimeError("replacement effective Compose changed")
        if not self.schema_matches(image):
            raise RuntimeError("replacement read-only schema 0004 check failed")

    def replace_app(self, image: str, revision: str, *, candidate: bool) -> None:
        self.compose(
            self.root / "docker-compose.override.yml",
            "up",
            "-d",
            "--no-deps",
            "--no-build",
            "--pull",
            "never",
            "--force-recreate",
            "app",
            timeout=120,
        )
        self.verify_replaced(image, revision, candidate=candidate)

    def start_ingress(self) -> None:
        self.run("docker", "start", "ai-voice-nginx")
        self.run("docker", "exec", "ai-voice-nginx", "nginx", "-t")
        self.public_health()

    def force_close_ingress(self) -> bool:
        try:
            self.run(
                "docker",
                "stop",
                "--time",
                "30",
                "ai-voice-nginx",
                timeout=45,
                check=False,
            )
        except BaseException:
            pass
        try:
            return not self.inspect("ai-voice-nginx")["State"]["Running"]
        except BaseException:
            return False

    def recover(self) -> None:
        self.stop_components(strict=True)
        previous = self.directory / "previous-override.json"
        self.install_override(previous)
        self.replace_app(OLD_IMAGE, DEPLOYED_SOURCE, candidate=False)
        self.start_ingress()
        print("PREVIOUS_APP_RESTORED_GATE_ABSENT_DIAGNOSTICS_ON_FILTER_OFF_SCHEMA_0004_UNCHANGED", flush=True)
        print("RECOVERY_IMAGE=" + OLD_IMAGE, flush=True)

    def deploy(self, candidate: str, candidate_override: Path) -> None:
        self.recheck_before_stop()
        print("PRE_CUTOVER_SPEECH_GATE_FINGERPRINTS_OK", flush=True)
        self.private_write("cutover-started", (self.commit + "\n").encode())
        print("CUTOVER_STARTED_STOPPING_BETA_TRAFFIC", flush=True)
        try:
            self.stop_components(strict=True)
            self.install_override(candidate_override)
            self.replace_app(candidate, self.commit, candidate=True)
            self.start_ingress()
            self.private_write(
                "deployed",
                (self.commit + "\n" + candidate + "\n" + str(self.directory) + "\n").encode(),
            )
            print("SPEECH_AWARE_BARGE_IN_DEPLOYED_READY_HTTPS_OK", flush=True)
            print("DEPLOYED_COMMIT=" + self.commit, flush=True)
            print("DEPLOYED_IMAGE=" + candidate, flush=True)
        except BaseException:
            print("SPEECH_GATE_CUTOVER_FAILED_RESTORING_PREVIOUS_APP", flush=True)
            try:
                self.recover()
            except BaseException as recovery_error:
                self.log_cleanup_failure(recovery_error)
                closed = self.force_close_ingress()
                if closed:
                    print("RECOVERY_FAILED_INGRESS_CLOSED", flush=True)
                else:
                    print("RECOVERY_FAILED_INGRESS_CLOSURE_UNVERIFIED", flush=True)
                print("RECOVERY_UNVERIFIED_MANUAL_REVIEW_REQUIRED", flush=True)
            raise

    def execute(self) -> None:
        self.stage = "preflight"
        original = self.preflight()
        self.stage = "assets"
        self.stage_assets()
        self.stage = "wheels"
        self.prepare_wheels(OLD_IMAGE)
        self.stage = "build"
        candidate = self.build_candidate()
        self.private_write("candidate-image", (candidate + "\n").encode())
        self.stage = "tests"
        self.run_candidate_tests(candidate)
        self.stage = "measurement"
        self.measure_candidate(candidate)
        self.stage = "settings"
        candidate_override = self.prepare_override(candidate, original)
        self.stage = "cutover"
        self.deploy(candidate, candidate_override)


def main(commit: str) -> None:
    import fcntl

    if os.geteuid() != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("root and exact 40-hex release commit required")
    os.umask(0o077)
    descriptor = os.open(
        "/run/ai-phone-deployment.lock",
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(descriptor, "r+") as lock:
        metadata = os.fstat(lock.fileno())
        if (
            metadata.st_uid != 0
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_mode & 0o077
        ):
            raise RuntimeError("deployment lock is not root protected")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for previous in Path("/opt").glob("ai-phone-speech-gate-release.*"):
            if (previous / "cutover-started").exists() and not (previous / "deployed").exists():
                raise RuntimeError("prior speech-gate cutover needs review")
        directory = Path(
            tempfile.mkdtemp(prefix="ai-phone-speech-gate-release.", dir="/opt")
        )
        metadata = directory.stat()
        if directory.is_symlink() or metadata.st_uid != 0 or metadata.st_mode & 0o077:
            raise RuntimeError("release directory is not root protected")
        print("RELEASE_DIRECTORY=" + str(directory), flush=True)
        release = Release(directory, commit)
        try:
            release.execute()
        except BaseException as error:
            print("RELEASE_STOPPED_STAGE=" + release.stage, flush=True)
            print("RELEASE_STOPPED_TYPE=" + type(error).__name__, flush=True)
            print("KEEP_PROTECTED_RELEASE_FILES_DO_NOT_PASTE_PRIVATE_LOGS", flush=True)
            raise


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise RuntimeError("one exact release commit argument required")
        main(sys.argv[1])
    except BaseException as error:
        print("RELEASE_EXIT_FAILURE_TYPE=" + type(error).__name__, flush=True)
        sys.exit(1)
