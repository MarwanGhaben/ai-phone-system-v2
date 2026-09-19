"""Owner-run T020-C verified booking and schema 0005 rollout.

Importing this module performs no filesystem, process, network, database, or
deployment action. The published script must be streamed from an exact reviewed
commit and invoked with that same full commit SHA.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
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
DEPLOYED_SOURCE = "792952b2965e85513bf0971a41bfaff7bd3377c4"
OLD_IMAGE = "sha256:14589fc6411ef54f64e0a519c41729698b113680ebc02fdbd94d53b5a0fa4369"
LAST_RELEASE = Path("/opt/ai-phone-speech-gate-release.21e0obbh")
NGINX_HASH = "59ac9bdaa9a86c1022a573c9709faddf7e93a53015da8acfcc4330b3e288f3ab"
PUBLIC_HEALTH = "https://aiagent.ghaben.ca:8443/health"
MANIFEST_PATH = "docs/delegation/T020-C-contention-hashes.json"
MANIFEST_SHA256 = "c3f89ece10d06f9edf14ea191fbad9f2b12c1eaa7cc8533b0ec528607a46de4a"
MIGRATION_0005_SHA256 = "cf9c9e2cdc53569a59833235efa98633b5f7b233f98549a8b2f1b16614866100"

RUNTIME_PATHS = (
    "api/main.py", "config/settings.py",
    "services/calendar/booking_mutations.py", "services/calendar/contracts.py",
    "services/calendar/ms_bookings_service.py", "services/calendar/service_facts.py",
    "services/config/accountants_service.py",
    "services/conversation/booking_session.py",
    "services/conversation/orchestrator.py", "services/database.py",
    "services/llm/llm_base.py", "services/llm/openai_service.py",
    "services/llm/tool_protocol.py", "services/scheduling/booking_check.py",
    "services/scheduling/booking_records.py", "services/scheduling/booking_service.py",
    "services/scheduling/models.py", "services/scheduling/operation_store.py",
    "services/scheduling/policy.py", "services/scheduling/proposals.py",
    "services/scheduling/reconciliation.py",
    "services/stt/elevenlabs_stt_service.py", "services/stt/stt_base.py",
)
CONFIGURATION_PATHS = ("clients/accountants.yaml", "clients/booking-policy.yaml")
MIGRATIONS_PATHS = (
    "migrations/0001_admin_users_updated_at.sql",
    "migrations/0002_bookings_aware_time.sql",
    "migrations/0003_booking_provider_observations.sql",
    "migrations/0004_appointment_notifications.sql",
    "migrations/0005_booking_operations.sql", "migrations/__init__.py",
    "migrations/bootstrap_schema.sql", "migrations/runner.py",
    "migrations/schema_contract.py",
)
PROTECTED_RUNTIME_PATHS = (
    "services/calendar/booking_readback.py",
    "services/calendar/provider_admission.py",
    "services/conversation/barge_in_diagnostics.py",
    "services/conversation/events.py", "services/conversation/language_policy.py",
    "services/conversation/local_vad.py",
    "services/conversation/speech_activity_gate.py",
    "services/conversation/turn_controller.py",
    "services/scheduling/provider_observations.py",
    "services/scheduling/removal_reconciliation.py",
    "services/sms/notification_outbox.py", "services/sms/notification_text.py",
    "services/sms/notification_worker.py", "services/sms/telnyx_submission.py",
    "services/telephony/twilio_service.py", "services/tts/elevenlabs_service.py",
)
MODEL_ASSETS_PATHS = (
    "models/vad/LICENSE.silero-vad.txt", "models/vad/provenance.json",
    "models/vad/silero_vad.onnx",
)
TESTS_PATHS = (
    "tests/calendar/test_graph_availability_contracts.py",
    "tests/calendar/test_graph_booking_mutations.py",
    "tests/calendar/test_ms_bookings_service.py",
    "tests/calendar/test_service_facts.py",
    "tests/conversation/test_availability_acknowledgement.py",
    "tests/conversation/test_barge_in_diagnostics.py",
    "tests/conversation/test_booking_cancellation.py",
    "tests/conversation/test_booking_policy_integration.py",
    "tests/conversation/test_booking_safety_guards.py",
    "tests/conversation/test_booking_selection.py",
    "tests/conversation/test_booking_time_validation.py",
    "tests/conversation/test_caller_language.py",
    "tests/conversation/test_language_policy.py",
    "tests/conversation/test_playback_ownership.py",
    "tests/conversation/test_safe_booking_flow.py",
    "tests/conversation/test_speech_activity_gate.py",
    "tests/conversation/test_speech_aware_barge_in.py",
    "tests/conversation/test_transcript_identity.py",
    "tests/conversation/test_transfer_playback.py",
    "tests/conversation/test_turn_controller.py",
    "tests/evaluation/test_local_vad_audio.py",
    "tests/integration/test_automatic_notifications.py",
    "tests/integration/test_booking_operations.py",
    "tests/integration/test_booking_persistence.py",
    "tests/integration/test_database_startup.py",
    "tests/integration/test_image_contents.py", "tests/integration/test_migrations.py",
    "tests/integration/test_phone_booking_flow.py",
    "tests/integration/test_provider_observations.py",
    "tests/integration/test_verified_booking_create.py",
    "tests/llm/test_openai_timeout.py", "tests/llm/test_tool_protocol.py",
    "tests/scheduling/test_booking_result_contracts.py",
    "tests/scheduling/test_policy.py", "tests/scheduling/test_proposals.py",
    "tests/stt/test_background_audio_pilot.py",
    "tests/stt/test_background_filter_contract.py",
    "tests/stt/test_elevenlabs_reset.py",
    "tests/stt/test_provider_compatibility_probe.py",
    "tests/stt/test_provider_failure_state.py",
    "tests/stt/test_reset_preservation.py", "tests/stt/test_utterance_events.py",
    "tests/stt/test_whisper_audio.py",
    "tests/telephony/test_playback_generations.py",
    "tests/telephony/test_twilio_stream.py", "tests/tts/test_call_isolation.py",
)
TEST_SUPPORT_PATHS = (
    "scripts/check-stt-provider-compatibility.py",
    "scripts/evaluate-stt-background-audio.py", "tests/conftest.py",
)
RELEASE_SUPPORT_PATHS = (
    ".gitattributes", "Dockerfile", "docker-compose.yml", "requirements.txt",
    "scripts/deploy-automatic-notifications.py",
    "scripts/deploy-speech-aware-barge-in.py",
)
SECTION_PATHS = {
    "runtime": RUNTIME_PATHS,
    "configuration": CONFIGURATION_PATHS,
    "migrations": MIGRATIONS_PATHS,
    "protected_runtime": PROTECTED_RUNTIME_PATHS,
    "model_assets": MODEL_ASSETS_PATHS,
    "tests": TESTS_PATHS,
    "test_support": TEST_SUPPORT_PATHS,
    "release_support": RELEASE_SUPPORT_PATHS,
}
SOURCE_SCOPE = (
    "api/main.py", "clients/accountants.yaml", "clients/booking-policy.yaml",
    "config/settings.py", "docker-compose.yml",
    "migrations/0005_booking_operations.sql", "migrations/runner.py",
    "migrations/schema_contract.py", "services/calendar/booking_mutations.py",
    "services/calendar/contracts.py", "services/calendar/ms_bookings_service.py",
    "services/calendar/service_facts.py", "services/config/accountants_service.py",
    "services/conversation/booking_session.py",
    "services/conversation/orchestrator.py", "services/database.py",
    "services/llm/llm_base.py", "services/llm/openai_service.py",
    "services/llm/tool_protocol.py", "services/scheduling/booking_check.py",
    "services/scheduling/booking_records.py", "services/scheduling/booking_service.py",
    "services/scheduling/models.py", "services/scheduling/operation_store.py",
    "services/scheduling/policy.py", "services/scheduling/proposals.py",
    "services/scheduling/reconciliation.py",
    "services/stt/elevenlabs_stt_service.py", "services/stt/stt_base.py",
)

EXPECTED_HISTORY = (
    ("0001", "53c861ac91cae5ecaf9f794b15563c88ee58bc0dafa5ae64fa81c89d907aa3a9"),
    ("0002", "62addab700e047dff2a13973c881a874bfe6c7f88967d593f189987192782be6"),
    ("0003", "b3abaa09c340f89c32778b0b17b5c2a584845983b13eb95dbb9c366f3d6c5830"),
    ("0004", "a1c66c7705adbd49918966c7d41d8bbbc5e80d3d5d0534fe641fd15579731a9e"),
    ("0005", MIGRATION_0005_SHA256),
    ("bootstrap-v1", "1c72bbe0a120860902c565e5573c05ad1cd640891a42bb35cca2c6e0876f31e8"),
)

VERIFIED_ENV = "VERIFIED_PHONE_BOOKING_ENABLED"
VERIFIED_SETTING = "verified_phone_booking_enabled"
PAUSED_ENV = "AUTOMATIC_NOTIFICATION_WORKERS_PAUSED"
PAUSED_SETTING = "automatic_notification_workers_paused"
SETTINGS_TARGET = "/app/config/settings.py"
ACCOUNTANTS_TARGET = "/app/clients/accountants.yaml"
POLICY_TARGET = "/app/clients/booking-policy.yaml"
OVERLAY_TARGETS = (SETTINGS_TARGET, ACCOUNTANTS_TARGET, POLICY_TARGET)
REQUIRED_BASELINE_SETTINGS = {
    "barge_in_diagnostics_enabled": True,
    "speech_aware_barge_in_enabled": True,
    "elevenlabs_stt_filter_background_audio": False,
    "automatic_notifications_enabled": True,
    PAUSED_SETTING: False,
}
PRESERVED_TABLES = (
    "admin_sessions", "admin_users", "analytics_events", "api_usage",
    "appointments", "bookings", "call_logs", "callers", "calls",
    "conversation_turns", "conversations", "knowledge_articles", "mfa_codes",
    "sms_logs", "system_metrics", "tenants", "users",
    "booking_provider_observations", "booking_provider_observation_control",
    "booking_notification_reconciliation", "booking_notification_outbox",
    "booking_notification_contract",
)

SETTINGS_CODE = (
    'import json; from config.settings import settings; '
    'print(json.dumps(settings.model_dump(mode="json"),sort_keys=True))'
)
SCHEMA_PROBE_CODE = '''import asyncio,json,os,asyncpg
from migrations.schema_contract import check_runtime_compatibility
EXPECTED=''' + repr(EXPECTED_HISTORY) + '''
async def check():
 c=await asyncpg.connect(os.environ['MIGRATION_DATABASE_URL'],timeout=10,command_timeout=15)
 try:
  async with c.transaction(readonly=True):
   await check_runtime_compatibility(c,require_notification=True)
   history=tuple((r['version'],r['checksum']) for r in await c.fetch('SELECT version,checksum FROM public.schema_migrations ORDER BY version'))
   names={r['tablename'] for r in await c.fetch("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='public'")}
   version='0005' if 'booking_operations' in names else '0004'
   wanted=EXPECTED if version=='0005' else tuple(item for item in EXPECTED if item[0]!='0005')
   legacy=tuple(item for item in wanted if item[0]!='bootstrap-v1')
   if history not in (legacy,wanted): raise RuntimeError('schema history differs')
   operation_count=None
   if version=='0005':
    operation_count=await c.fetchval('SELECT count(*) FROM public.booking_operations')
    if await c.fetchval('SELECT count(*) FROM public.booking_operation_contract')!=1: raise RuntimeError('operation contract differs')
   ext=await c.fetchrow("SELECT default_version,installed_version FROM pg_catalog.pg_available_extensions WHERE name='btree_gist'")
   privilege=await c.fetchval("SELECT has_database_privilege(current_user,current_database(),'CREATE')")
   print(json.dumps({'version':version,'history':history,'operation_count':operation_count,'extension_available':ext is not None,'extension_installed':None if ext is None else ext['installed_version'],'can_create':privilege},sort_keys=True))
 finally:
  await c.close(timeout=5)
asyncio.run(check())
'''
FINGERPRINT_CODE = '''import asyncio,json,os,asyncpg
TABLES=''' + repr(PRESERVED_TABLES) + '''
async def check():
 c=await asyncpg.connect(os.environ['MIGRATION_DATABASE_URL'],timeout=10,command_timeout=20)
 try:
  async with c.transaction(readonly=True):
   values={}
   for table in TABLES:
    q="SELECT md5(COALESCE(jsonb_agg(to_jsonb(t) ORDER BY to_jsonb(t)::text)::text,'[]')) FROM public."+chr(34)+table+chr(34)+' t'
    values[table]={'count':await c.fetchval('SELECT count(*) FROM public.'+chr(34)+table+chr(34)),'hash':await c.fetchval(q)}
   print(json.dumps(values,sort_keys=True))
 finally:
  await c.close(timeout=5)
asyncio.run(check())
'''
SOURCE_PROBE_PREFIX = '''import hashlib,json
from pathlib import Path
def norm(data): return data.replace(b'\\r\\n',b'\\n')
'''

SYNTHETIC_ENV = ("ENVIRONMENT=test",)
SYNTHETIC_APPLICATION_ENV = SYNTHETIC_ENV + (
    "SECRET_KEY=synthetic-release-secret",
    "DATABASE_URL=postgresql://synthetic:synthetic@db/synthetic",
    "REDIS_URL=redis://redis:6379/0", "OPENAI_API_KEY=synthetic",
    "DEEPGRAM_API_KEY=synthetic",
    "ELEVENLABS_API_KEY=synthetic", "ELEVENLABS_VOICE_ID=synthetic",
    "TWILIO_ACCOUNT_SID=AC00000000000000000000000000000000",
    "TWILIO_AUTH_TOKEN=synthetic", "TWILIO_PHONE_NUMBER=+12025550100",
    "MS_BOOKINGS_TENANT_ID=synthetic", "MS_BOOKINGS_CLIENT_ID=synthetic",
    "MS_BOOKINGS_CLIENT_SECRET=synthetic", "MS_BOOKINGS_BUSINESS_ID=synthetic",
    "TELNYX_API_KEY=synthetic", "TELNYX_PHONE_NUMBER=+12025550100",
)

TEST_ONLY_HASHES = {
    ".dockerignore": "5490d07f76e18a950124daa1048d29a690df53e7e58a7bfc6da8542f8569e72f",
    "docs/delegation/T007-A-result.md":
        "f6ff3d65f12d40a559b47f7313989d4409b63887bd93c3b4a5a4bd8ba9020e3b",
    "docs/delegation/T013-A-result.md":
        "adc3ff91b3810db7b4eeed6a09d6d7632ddcdc234456c6a58b80fa1b79cdc082",
    "templates/dashboard.html":
        "3d79db6c212a04f88799079086823b8e6aa8705b09799eb9a74df3334b8cacd1",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_text_bytes(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def asset_digest(section: str, data: bytes) -> str:
    return sha256(data if section == "model_assets" else normalized_text_bytes(data))


def strict_json(data: bytes) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    text = data.decode("utf-8")
    decoder = json.JSONDecoder(object_pairs_hook=pairs)
    value, end = decoder.raw_decode(text)
    if text[end:].strip() or not isinstance(value, dict):
        raise ValueError("invalid JSON object")
    return value


def mount_fingerprint(mounts: list[dict]) -> list[str]:
    return sorted(json.dumps(item, sort_keys=True, separators=(",", ":")) for item in mounts)


def environment_fingerprint(environment: list[str]) -> list[str]:
    if not isinstance(environment, list) or any(type(item) is not str for item in environment):
        raise RuntimeError("container environment differs")
    return sorted(environment)


def typed_mapping_equal(left: dict, right: dict) -> bool:
    return (type(left) is dict and type(right) is dict and left == right
            and all(type(left[key]) is type(right[key]) for key in left))


def settings_for(previous: dict, *, candidate: bool, paused: bool | None) -> dict:
    if type(previous) is not dict:
        raise RuntimeError("previous settings are invalid")
    if any(previous.get(key) is not value for key, value in REQUIRED_BASELINE_SETTINGS.items()):
        raise RuntimeError("baseline voice or notification settings differ")
    prior_verified = previous.get(VERIFIED_SETTING, False)
    if type(prior_verified) is not bool or prior_verified:
        raise RuntimeError("previous verified phone setting differs")
    expected = dict(previous)
    if candidate:
        if type(paused) is not bool:
            raise RuntimeError("candidate notification pause is required")
        expected[VERIFIED_SETTING] = True
        expected[PAUSED_SETTING] = paused
    elif paused is not None:
        raise RuntimeError("previous image pause override is invalid")
    return expected


def validate_source_scope(paths: list[str]) -> None:
    if (len(paths) != len(SOURCE_SCOPE) or len(paths) != len(set(paths))
            or set(paths) != set(SOURCE_SCOPE)):
        raise RuntimeError("release source exceeds reviewed scope")


def candidate_test_paths() -> tuple[str, ...]:
    return TESTS_PATHS


def hidden_candidate_paths(mounts: list[dict], overlay_sources: dict[str, str]) -> tuple[str, ...]:
    hidden = set()
    exact = {target: 0 for target in OVERLAY_TARGETS}
    for mount in mounts:
        if not isinstance(mount, dict):
            hidden.add("invalid-mount")
            continue
        target = mount.get("target") or mount.get("Destination")
        source = mount.get("source") or mount.get("Source")
        read_only = mount.get("read_only")
        if read_only is None and "RW" in mount:
            read_only = not mount["RW"]
        if target in exact:
            exact[target] += 1
            if (source != overlay_sources.get(target) or mount.get("type", mount.get("Type")) != "bind"
                    or read_only is not True):
                hidden.add(target)
        if target in ("/app", "/app/config", "/app/clients"):
            hidden.add(target)
    for target, count in exact.items():
        if count != 1:
            hidden.add(target)
    # Parent configuration/client mounts are accepted only when every reviewed
    # file they hide has a later exact read-only file overlay.
    if hidden & {"/app/config", "/app/clients"} and not any(
            item not in ("/app/config", "/app/clients") for item in hidden):
        hidden -= {"/app/config", "/app/clients"}
    return tuple(sorted(hidden))


def normalized_compose(config: dict, *, candidate: bool, paused: bool | None,
                       overlay_sources: dict[str, str]) -> dict:
    value = copy.deepcopy(config)
    app = value["services"]["app"]
    app["image"] = "<app-image>"
    if "migrate" in value["services"]:
        value["services"]["migrate"]["image"] = "<migration-image>"
    environment = app.get("environment") or {}
    if type(environment) is not dict:
        raise RuntimeError("rendered application environment is not a mapping")
    if candidate:
        if str(environment.pop(VERIFIED_ENV, "")).lower() != "true":
            raise RuntimeError("verified phone environment differs")
        if str(environment.pop(PAUSED_ENV, "")).lower() != str(paused).lower():
            raise RuntimeError("notification pause environment differs")
    else:
        if str(environment.pop(VERIFIED_ENV, "false")).lower() != "false":
            raise RuntimeError("previous verified phone environment differs")
        if str(environment.pop(PAUSED_ENV, "false")).lower() != "false":
            raise RuntimeError("previous notification pause environment differs")
    app["environment"] = environment
    mounts = app.get("volumes") or []
    if not isinstance(mounts, list):
        raise RuntimeError("rendered mounts are invalid")
    if candidate:
        if hidden_candidate_paths(mounts, overlay_sources):
            raise RuntimeError("effective mount hides reviewed candidate source")
    mounts = [m for m in mounts if m.get("target") not in OVERLAY_TARGETS]
    app["volumes"] = sorted(mounts, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(app.get("networks"), list):
        app["networks"] = sorted(app["networks"])
    return value


def source_probe_code(text_hashes: dict[str, str], model_hashes: dict[str, str]) -> str:
    return (SOURCE_PROBE_PREFIX + "TEXT=" + repr(text_hashes) + "\nMODEL=" + repr(model_hashes) + '''
values={}
for name,want in TEXT.items(): values[name]=hashlib.sha256(norm(Path('/app',name).read_bytes())).hexdigest()
for name,want in MODEL.items(): values[name]=hashlib.sha256(Path('/app',name).read_bytes()).hexdigest()
from config.settings import settings
print(json.dumps({'hashes':values,'settings':settings.model_dump(mode='json')},sort_keys=True))
''')


def container_contract(container: dict, *, candidate: bool) -> dict:
    env = list(container["Config"].get("Env") or [])
    controlled = {VERIFIED_ENV: [], PAUSED_ENV: []}
    for item in env:
        for key in controlled:
            if item.startswith(key + "="):
                controlled[key].append(item.split("=", 1)[1].lower())
    if candidate:
        if (controlled[VERIFIED_ENV] != ["true"]
                or len(controlled[PAUSED_ENV]) != 1
                or controlled[PAUSED_ENV][0] not in {"true", "false"}):
            raise RuntimeError("candidate release flags differ")
    elif (controlled[VERIFIED_ENV] not in ([], ["false"])
          or controlled[PAUSED_ENV] not in ([], ["false"])):
        raise RuntimeError("previous release flags differ")
    allowed = (VERIFIED_ENV + "=", PAUSED_ENV + "=")
    env = [item for item in env if not item.startswith(allowed)]
    mounts = list(container.get("Mounts") or [])
    mounts = [item for item in mounts if item.get("Destination") not in OVERLAY_TARGETS]
    return {
        "env": environment_fingerprint(env),
        "mounts": mount_fingerprint(mounts),
        "networks": sorted(container["NetworkSettings"].get("Networks", {})),
        "ports": container["HostConfig"].get("PortBindings"),
        "restart": container["HostConfig"].get("RestartPolicy"),
        "stop_signal": container["Config"].get("StopSignal"),
    }


class Release:
    def __init__(self, directory: Path, commit: str):
        self.directory = directory
        self.commit = commit
        self.root = ROOT
        self.assets = directory / "assets"
        self.test_inputs = directory / "test-inputs"
        self.manifest = None
        self.old = None
        self.previous_settings = None
        self.previous_rendered = None
        self.previous_contract = None
        self.network = None
        self.pg_image = None
        self.db_user = None
        self.db_name = None
        self.live_env = None
        self.candidate_image = None
        self.stage = "preflight"
        self.owned_containers: set[str] = set()
        self.owned_networks: set[str] = set()
        self.overlay_sources: dict[str, str] = {}
        self.project_name = "ai-phone-system-v2"
        self.app_container = "ai-voice-app"
        self.nginx_container = "ai-voice-nginx"
        self.database_container = "ai-voice-db"

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

    def mark(self, name: str, value: bytes = b"") -> Path:
        return self.private_write(name, value or (self.stage + "\n").encode())

    def run(self, *args, timeout=60, check=True, input=None):
        try:
            result = subprocess.run(args, cwd=self.root, input=input, capture_output=True,
                                    timeout=timeout, check=False)
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

    def inspect_once(self, name: str) -> dict:
        value = json.loads(self.run("docker", "inspect", name).stdout)
        if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
            raise RuntimeError("Docker inspection differs")
        return value[0]

    def compose(self, override: Path, *args, **kwargs):
        return self.run("docker", "compose", "--project-directory", str(self.root),
                        "-p", self.project_name, "-f", str(self.root / "docker-compose.yml"),
                        "-f", str(override), *args, **kwargs)

    def render_compose(self, override: Path) -> dict:
        return strict_json(self.compose(override, "config", "--format", "json").stdout)

    def settings(self, name: str | None = None) -> dict:
        name = name or self.app_container
        return strict_json(self.run("docker", "exec", name, "python", "-c",
                                    SETTINGS_CODE).stdout)

    def git_bytes(self, relative: str) -> bytes:
        return self.run("git", "show", self.commit + ":" + relative).stdout

    def public_health(self) -> None:
        for _ in range(12):
            result = self.run("curl", "-fsS", "--max-time", "5", PUBLIC_HEALTH,
                              check=False)
            if result.returncode == 0:
                try:
                    if strict_json(result.stdout).get("status") == "healthy":
                        return
                except (UnicodeError, ValueError):
                    pass
            time.sleep(1)
        raise RuntimeError("public certificate-validated HTTPS health failed")

    def wait_ready(self, name: str | None = None) -> None:
        name = name or self.app_container
        for _ in range(60):
            result = self.run("docker", "exec", name, "curl", "-fsS", "--max-time",
                              "3", "http://localhost:8000/ready", check=False)
            if result.returncode == 0:
                try:
                    endpoint_ready = strict_json(result.stdout).get("status") == "ready"
                    health = self.inspect_once(name)["State"].get("Health", {}).get("Status")
                    if endpoint_ready and health == "healthy":
                        return
                except (KeyError, UnicodeError, ValueError):
                    pass
            time.sleep(2)
        raise RuntimeError("application readiness deadline exceeded")

    def verify_source(self) -> None:
        if self.run("git", "rev-parse", "HEAD").stdout.decode().strip() != BASE_CHECKOUT:
            raise RuntimeError("server checkout changed")
        for revision in (DEPLOYED_SOURCE, self.commit):
            resolved = self.run("git", "rev-parse", revision + "^{commit}").stdout.decode().strip()
            if resolved != revision:
                raise RuntimeError("release source unavailable")
            if self.run("git", "merge-base", "--is-ancestor", BASE_CHECKOUT, revision,
                        check=False).returncode:
                raise RuntimeError("release source is not descended from server baseline")
        if self.run("git", "merge-base", "--is-ancestor", DEPLOYED_SOURCE, self.commit,
                    check=False).returncode:
            raise RuntimeError("release source is not descended from deployed source")
        status = self.run("git", "status", "--porcelain", "--untracked-files=no").stdout.decode().splitlines()
        if status != [" M nginx/nginx.conf"]:
            raise RuntimeError("unexpected server checkout modification")
        if sha256((self.root / "nginx/nginx.conf").read_bytes()) != NGINX_HASH:
            raise RuntimeError("nginx configuration changed")
        paths = self.run(
            "git", "diff", "--name-only", DEPLOYED_SOURCE, self.commit, "--",
            "api", "clients", "config", "migrations", "services", "docker-compose.yml",
            "requirements.txt", "Dockerfile").stdout.decode().splitlines()
        validate_source_scope(paths)
        if self.run("git", "cat-file", "-e",
                    self.commit + ":scripts/deploy-verified-phone-booking.py",
                    check=False).returncode:
            raise RuntimeError("pinned release procedure missing")

    def stage_assets(self) -> None:
        raw = self.git_bytes(MANIFEST_PATH)
        if sha256(raw) != MANIFEST_SHA256:
            raise RuntimeError("reviewed manifest differs")
        manifest = strict_json(raw)
        if (manifest.get("input_head") != DEPLOYED_SOURCE
                or tuple(manifest.get("source_scope", ())) != SOURCE_SCOPE
                or set(manifest) != {"input_head", "normalization", "source_scope",
                                     *SECTION_PATHS}):
            raise RuntimeError("reviewed manifest contract differs")
        if self.assets.exists():
            raise RuntimeError("asset directory already exists")
        self.assets.mkdir(mode=0o700)
        seen = set()
        for section, expected_paths in SECTION_PATHS.items():
            hashes = manifest.get(section)
            if type(hashes) is not dict or tuple(hashes) != expected_paths:
                raise RuntimeError("reviewed manifest paths differ")
            for relative, expected in hashes.items():
                if relative in seen:
                    raise RuntimeError("reviewed manifest duplicates an asset")
                seen.add(relative)
                data = self.git_bytes(relative)
                if asset_digest(section, data) != expected:
                    raise RuntimeError("reviewed asset hash differs")
                stored = data if section == "model_assets" else normalized_text_bytes(data)
                target = self.assets / relative
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                target.write_bytes(stored)
                target.chmod(0o600)
        if len(seen) != 108:
            raise RuntimeError("reviewed manifest entry count differs")
        self.test_inputs.mkdir(mode=0o700)
        for relative, expected in TEST_ONLY_HASHES.items():
            data = normalized_text_bytes(self.git_bytes(relative))
            if sha256(data) != expected:
                raise RuntimeError("test-only input hash differs")
            target = self.test_inputs / relative
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            target.write_bytes(data)
            target.chmod(0o600)
        if sha256((self.assets / "migrations/0005_booking_operations.sql").read_bytes()) != MIGRATION_0005_SHA256:
            raise RuntimeError("operation migration differs")
        self.manifest = manifest
        print("REVIEWED_108_RELEASE_INPUTS_STAGED", flush=True)

    def verify_frozen_inputs(self) -> None:
        raw = self.git_bytes(MANIFEST_PATH)
        if sha256(raw) != MANIFEST_SHA256:
            raise RuntimeError("reviewed manifest changed during release")
        manifest = strict_json(raw)
        count = 0
        for section, paths in SECTION_PATHS.items():
            if tuple(manifest.get(section, {})) != paths:
                raise RuntimeError("reviewed manifest paths changed during release")
            for relative, expected in manifest[section].items():
                count += 1
                data = self.git_bytes(relative)
                if asset_digest(section, data) != expected:
                    raise RuntimeError("reviewed input changed during release")
                staged = self.assets / relative
                wanted = data if section == "model_assets" else normalized_text_bytes(data)
                if not staged.is_file() or staged.read_bytes() != wanted:
                    raise RuntimeError("staged input changed during release")
        if count != 108:
            raise RuntimeError("reviewed input count changed during release")
        for relative, expected in TEST_ONLY_HASHES.items():
            source = normalized_text_bytes(self.git_bytes(relative))
            staged = self.test_inputs / relative
            if (sha256(source) != expected or not staged.is_file()
                    or staged.read_bytes() != source):
                raise RuntimeError("test-only input changed during release")

    def capacity(self) -> None:
        if shutil.disk_usage(self.root).free < 2 * 1024 ** 3:
            raise RuntimeError("insufficient release disk capacity")
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
        match = re.search(r"^MemAvailable:\s+(\d+) kB$", text, re.MULTILINE)
        if match is None or int(match.group(1)) < 768 * 1024:
            raise RuntimeError("insufficient release memory capacity")

    def _refuse_unresolved_cutover(self) -> None:
        for path in Path("/opt").glob("ai-phone-verified-booking-release.*"):
            if path == self.directory or not path.is_dir():
                continue
            if ((path / "cutover-started").exists()
                    and not any((path / name).exists() for name in
                                ("deployed", "recovered", "recovery-needed"))):
                raise RuntimeError("unresolved prior cutover exists")

    def one_shot(self, name: str, image: str, arguments: list[str], *, network=None,
                 env_file=None, mounts=(), synthetic_env=False, check=True, timeout=180):
        self.owned_containers.add(name)
        command = ["docker", "run", "--name", name, "--pull", "never",
                   "--memory", "768m", "--cpus", "1", "--pids-limit", "192"]
        if network:
            command += ["--network", network]
        if env_file:
            command += ["--env-file", str(env_file)]
        if synthetic_env:
            for item in SYNTHETIC_APPLICATION_ENV:
                command += ["-e", item]
        for source, target, readonly in mounts:
            command += ["--mount", "type=bind,source=" + str(source) + ",target=" + target
                        + (",readonly" if readonly else "")]
        command += ["--entrypoint", "python", image, *arguments]
        try:
            return self.run(*command, timeout=timeout, check=check)
        finally:
            self.run("docker", "rm", "-f", name, check=False)
            self.owned_containers.discard(name)

    def schema_probe(self, image: str, *, network=None, env_file=None) -> dict:
        result = self.one_shot("t020c-schema-" + self.directory.name, image,
                               ["-c", SCHEMA_PROBE_CODE], network=network or self.network,
                               env_file=env_file or self.live_env, check=False)
        if result.returncode:
            raise RuntimeError("schema compatibility check failed")
        return strict_json(result.stdout)

    def live_schema_version(self) -> str:
        return self.schema_probe(self.candidate_image or OLD_IMAGE)["version"]

    def fingerprint(self, image: str, *, network=None, env_file=None) -> dict:
        result = self.one_shot("t020c-fingerprint-" + self.directory.name, image,
                               ["-c", FINGERPRINT_CODE], network=network or self.network,
                               env_file=env_file or self.live_env)
        value = strict_json(result.stdout)
        if (set(value) != set(PRESERVED_TABLES)
                or any(set(item) != {"count", "hash"} or type(item["count"]) is not int
                       or item["count"] < 0 or not re.fullmatch(r"[0-9a-f]{32}", item["hash"])
                       for item in value.values())):
            raise RuntimeError("business fingerprint invalid")
        return value

    def preflight(self) -> dict:
        self._refuse_unresolved_cutover()
        self.verify_source()
        self.capacity()
        protected = LAST_RELEASE.stat()
        if (LAST_RELEASE.is_symlink() or not stat.S_ISDIR(protected.st_mode)
                or protected.st_uid != 0 or protected.st_mode & 0o077):
            raise RuntimeError("last protected release differs")
        override = self.root / "docker-compose.override.yml"
        info = override.stat()
        if (override.is_symlink() or info.st_uid != 0 or info.st_mode & 0o077
                or not stat.S_ISREG(info.st_mode)):
            raise RuntimeError("private override is not root protected")
        original_bytes = override.read_bytes()
        original = strict_json(original_bytes)
        if original["services"]["app"]["image"] != OLD_IMAGE:
            raise RuntimeError("private override does not pin accepted image")
        old = self.inspect_once(self.app_container)
        if (old["Image"] != OLD_IMAGE or not old["State"]["Running"]
                or old["State"].get("Health", {}).get("Status") != "healthy"
                or old["Config"]["Labels"].get("org.opencontainers.image.revision") != DEPLOYED_SOURCE
                or old["Config"]["Labels"].get("com.docker.compose.project") != self.project_name):
            raise RuntimeError("running app does not match accepted image/source")
        nginx = self.inspect_once(self.nginx_container)
        redis = self.inspect_once("ai-voice-redis")
        database = self.inspect_once(self.database_container)
        if (not nginx["State"]["Running"] or not redis["State"]["Running"]
                or not database["State"]["Running"]
                or database["State"].get("Health", {}).get("Status") != "healthy"):
            raise RuntimeError("release dependency is not healthy")
        settings = self.settings()
        settings_for(settings, candidate=True, paused=True)
        url = settings.get("database_url")
        if type(url) is not str or not url.strip() or "\n" in url or "\r" in url:
            raise RuntimeError("live database target is invalid")
        parsed = urlsplit(url)
        db_env = dict(item.split("=", 1) for item in database["Config"].get("Env", []) if "=" in item)
        expected_database = db_env.get("POSTGRES_DB", db_env.get("POSTGRES_USER", "postgres"))
        database_user = db_env.get("POSTGRES_USER", "postgres")
        common = set(old["NetworkSettings"]["Networks"]) & set(database["NetworkSettings"]["Networks"])
        if len(common) != 1:
            raise RuntimeError("application/database network differs")
        network, = common
        network_info = database["NetworkSettings"]["Networks"][network]
        hosts = {"db", self.database_container, network_info.get("IPAddress"),
                 *(network_info.get("Aliases") or [])}
        if parsed.hostname not in hosts or unquote(parsed.path.lstrip("/")) != expected_database:
            raise RuntimeError("application and database targets differ")
        pg_image = database["Image"]
        if "PG_MAJOR=16" not in self.inspect_once(pg_image)["Config"].get("Env", []):
            raise RuntimeError("PostgreSQL 16 image required")
        self.run("docker", "inspect", OLD_IMAGE)
        self.run("docker", "exec", self.nginx_container, "nginx", "-t")
        self.public_health()
        rendered = self.render_compose(override)
        previous_settings_mounts = [m for m in old.get("Mounts", [])
                                    if m.get("Destination") == SETTINGS_TARGET]
        if (len(previous_settings_mounts) != 1
                or previous_settings_mounts[0].get("Type") != "bind"
                or previous_settings_mounts[0].get("RW")):
            raise RuntimeError("existing settings overlay differs")
        self.old = old
        self.previous_settings = settings
        self.previous_rendered = rendered
        self.previous_contract = container_contract(old, candidate=False)
        self.network, self.pg_image = network, pg_image
        self.db_user, self.db_name = database_user, expected_database
        self.private_write("previous-override.json", original_bytes)
        self.private_write("previous-settings.json", json.dumps(settings).encode())
        self.private_write("previous-app.json", json.dumps(old).encode())
        self.private_write("previous-rendered.json", json.dumps(rendered).encode())
        self.private_write("previous-contract.json", json.dumps(self.previous_contract).encode())
        self.private_write("base-compose.sha256",
                           sha256((self.root / "docker-compose.yml").read_bytes()).encode())
        self.live_env = self.private_write(
            "migration.env", ("MIGRATION_DATABASE_URL=" + url + "\n").encode())
        probe = self.schema_probe(OLD_IMAGE)
        if (probe["version"] != "0004" or not probe["extension_available"]
                or not probe["can_create"]):
            raise RuntimeError("live schema is not eligible 0004 predecessor")
        self.private_write("preflight-fingerprint.json",
                           json.dumps(self.fingerprint(OLD_IMAGE), sort_keys=True).encode())
        print("PREFLIGHT_0004_BASELINE_SETTINGS_MOUNTS_CAPACITY_READY_HTTPS_OK", flush=True)
        return original

    def build_candidate(self) -> str:
        if self.manifest is None:
            raise RuntimeError("reviewed assets were not staged")
        parent = "ai-phone-t020c-parent:" + self.directory.name
        tag = "ai-phone-t020c-candidate:" + self.directory.name
        self.run("docker", "tag", OLD_IMAGE, parent)
        if self.inspect_once(parent)["Id"] != OLD_IMAGE:
            raise RuntimeError("candidate parent image changed")
        context = self.directory / "build"
        context.mkdir(mode=0o700)
        copied = (*RUNTIME_PATHS, *CONFIGURATION_PATHS, *MIGRATIONS_PATHS)
        for relative in copied:
            source = self.assets / relative
            target = context / relative
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            target.write_bytes(source.read_bytes())
            target.chmod(0o600)
        dockerfile = "FROM " + parent + "\n" + "".join(
            "COPY " + relative + " /app/" + relative + "\n" for relative in copied)
        (context / "Dockerfile").write_text(dockerfile, encoding="utf-8", newline="\n")
        try:
            self.run("docker", "build", "--network", "none", "--pull=false",
                     "--label", "org.opencontainers.image.revision=" + self.commit,
                     "-t", tag, str(context), timeout=300)
        finally:
            self.run("docker", "image", "rm", parent, check=False)
        built = self.inspect_once(tag)
        image = built["Id"]
        if (not image.startswith("sha256:")
                or built["Config"]["Labels"].get("org.opencontainers.image.revision") != self.commit):
            raise RuntimeError("candidate image identity differs")
        text_hashes = {
            **self.manifest["runtime"], **self.manifest["configuration"],
            **self.manifest["migrations"], **self.manifest["protected_runtime"],
        }
        model_hashes = self.manifest["model_assets"]
        result = self.one_shot("t020c-image-probe-" + self.directory.name, image,
                               ["-c", source_probe_code(text_hashes, model_hashes)],
                               mounts=(), synthetic_env=True, timeout=120)
        probe = strict_json(result.stdout)
        if probe.get("hashes") != {**text_hashes, **model_hashes}:
            raise RuntimeError("candidate source or protected asset differs")
        if probe.get("settings", {}).get(VERIFIED_SETTING) is not False:
            raise RuntimeError("candidate source default is not off")
        self.candidate_image = image
        self.private_write("candidate-image", (image + "\n").encode())
        print("DERIVED_CANDIDATE_108_HASHES_DEFAULT_OFF_OK", flush=True)
        return image

    def run_candidate_tests(self, image: str) -> None:
        tests_root = self.directory / "candidate-tests"
        tests_root.mkdir(mode=0o700)
        for relative in (*TESTS_PATHS, *TEST_SUPPORT_PATHS):
            source = self.assets / relative
            target = tests_root / relative
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            target.write_bytes(source.read_bytes())
        selected = tuple(path for path in TESTS_PATHS
                         if path not in ("tests/integration/test_booking_operations.py",
                                         "tests/integration/test_migrations.py",
                                         "tests/integration/test_phone_booking_flow.py",
                                         "tests/integration/test_verified_booking_create.py"))
        if len(selected) < 20:
            raise RuntimeError("candidate test collection is empty")
        mounts = (
            (tests_root / "tests", "/app/tests", True),
            (tests_root / "scripts/check-stt-provider-compatibility.py",
             "/app/scripts/check-stt-provider-compatibility.py", True),
            (tests_root / "scripts/evaluate-stt-background-audio.py",
             "/app/scripts/evaluate-stt-background-audio.py", True),
            (self.test_inputs / ".dockerignore", "/app/.dockerignore", True),
            (self.assets / "Dockerfile", "/app/Dockerfile", True),
            (self.assets / "docker-compose.yml", "/app/docker-compose.yml", True),
            (self.test_inputs / "docs", "/app/docs", True),
            (self.test_inputs / "templates/dashboard.html",
             "/app/templates/dashboard.html", True),
        )
        result = self.one_shot(
            "t020c-tests-" + self.directory.name, image,
            ["-B", "-m", "pytest", "--collect-only", "-q", *selected],
            mounts=mounts,
            network="none", synthetic_env=True, timeout=180)
        collected = re.search(rb"(\d+) tests? collected", result.stdout + result.stderr)
        if collected is None or int(collected.group(1)) < 100:
            raise RuntimeError("candidate test collection differs")
        executed = self.one_shot(
            "t020c-tests-run-" + self.directory.name, image,
            ["-B", "-m", "pytest", "-q", *selected],
            mounts=mounts,
            network="none", synthetic_env=True, timeout=300, check=False)
        if executed.returncode:
            raise RuntimeError("candidate test execution failed")
        passed = re.search(rb"(\d+) passed", executed.stdout + executed.stderr)
        if passed is not None and int(passed.group(1)) < 100:
            raise RuntimeError("candidate test execution count differs")
        skipped = re.search(rb"(\d+) skipped", executed.stdout + executed.stderr)
        self.private_write("candidate-tests.json", json.dumps({
            "paths": selected, "collected": int(collected.group(1)),
            "passed": None if passed is None else int(passed.group(1)),
            "skipped": 0 if skipped is None else int(skipped.group(1)),
        }, sort_keys=True).encode())
        print("CANDIDATE_ACCEPTED_SCHEDULING_PHONE_PROTOCOL_SPEECH_TESTS_OK", flush=True)

    def prepare_override(self, image: str, original: dict, *, paused: bool) -> Path:
        value = copy.deepcopy(original)
        app = value["services"]["app"]
        app["image"] = image
        if "migrate" in value["services"]:
            value["services"]["migrate"]["image"] = image
        environment = app.setdefault("environment", {})
        if type(environment) is not dict:
            raise RuntimeError("private application environment differs")
        if str(environment.get(VERIFIED_ENV, "false")).lower() != "false":
            raise RuntimeError("private verified phone environment differs")
        if str(environment.get(PAUSED_ENV, "false")).lower() != "false":
            raise RuntimeError("private notification pause environment differs")
        environment[VERIFIED_ENV] = "true"
        environment[PAUSED_ENV] = str(paused).lower()
        overlays = {
            SETTINGS_TARGET: self.directory / "candidate-settings.py",
            ACCOUNTANTS_TARGET: self.directory / "candidate-accountants.yaml",
            POLICY_TARGET: self.directory / "candidate-booking-policy.yaml",
        }
        sources = {
            SETTINGS_TARGET: self.assets / "config/settings.py",
            ACCOUNTANTS_TARGET: self.assets / "clients/accountants.yaml",
            POLICY_TARGET: self.assets / "clients/booking-policy.yaml",
        }
        for target, destination in overlays.items():
            destination.write_bytes(sources[target].read_bytes())
            destination.chmod(0o644)
        self.overlay_sources = {target: str(path) for target, path in overlays.items()}
        volumes = app.setdefault("volumes", [])
        if type(volumes) is not list:
            raise RuntimeError("private application mounts differ")
        volumes[:] = [item for item in volumes
                      if not (isinstance(item, dict) and item.get("target") in OVERLAY_TARGETS)]
        volumes.extend({"type": "bind", "source": str(source), "target": target,
                        "read_only": True}
                       for target, source in overlays.items())
        name = "paused-override.json" if paused else "active-override.json"
        path = self.private_write(name, (json.dumps(value, indent=2) + "\n").encode())
        rendered = self.render_compose(path)
        if (normalized_compose(rendered, candidate=True, paused=paused,
                               overlay_sources=self.overlay_sources)
                != normalized_compose(self.previous_rendered, candidate=False, paused=None,
                                      overlay_sources={})):
            raise RuntimeError("effective Compose configuration changed")
        probe = self.candidate_probe(path, paused=paused)
        expected_hashes = {
            **self.manifest["runtime"], **self.manifest["configuration"],
            **self.manifest["migrations"], **self.manifest["protected_runtime"],
            **self.manifest["model_assets"],
        }
        expected_settings = settings_for(self.previous_settings, candidate=True,
                                         paused=paused)
        if (probe.get("hashes") != expected_hashes
                or not typed_mapping_equal(probe.get("settings"), expected_settings)):
            raise RuntimeError("effective candidate source or settings differ")
        return path

    def candidate_probe(self, override: Path, *, paused: bool) -> dict:
        name = ("t020c-settings-paused-" if paused else "t020c-settings-active-") \
            + self.directory.name
        text_hashes = {
            **self.manifest["runtime"], **self.manifest["configuration"],
            **self.manifest["migrations"], **self.manifest["protected_runtime"],
        }
        try:
            result = self.compose(
                override, "run", "--rm", "--no-deps", "--pull", "never",
                "--name", name, "--entrypoint", "python", "app", "-c",
                source_probe_code(text_hashes, self.manifest["model_assets"]),
                timeout=120)
            return strict_json(result.stdout)
        finally:
            self.run("docker", "rm", "-f", name, check=False)

    def _database_command(self, *arguments, input=None, check=True):
        return self.run("docker", "exec", "-i", self.database_container, *arguments,
                        input=input, timeout=180, check=check)

    def backup(self, name: str) -> Path:
        archive = self.directory / (name + ".dump")
        if archive.exists() or archive.is_symlink():
            raise RuntimeError("backup target already exists")
        data = self._database_command(
            "pg_dump", "-U", self.db_user, "-d", self.db_name, "--format=custom",
            "--no-owner", "--no-acl").stdout
        with archive.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        archive.chmod(0o600)
        if not data or self._database_command("pg_restore", "--list", input=data,
                                              check=False).returncode:
            raise RuntimeError("protected backup verification failed")
        self.private_write(name + ".sha256", (sha256(data) + "\n").encode())
        print("PROTECTED_BACKUP_VERIFIED=" + name, flush=True)
        return archive

    def _create_rehearsal_db(self, label: str) -> tuple[str, str, Path]:
        network = "t020c-" + label + "-net-" + self.directory.name
        name = "t020c-" + label + "-db-" + self.directory.name
        self.run("docker", "network", "create", "--internal", network)
        self.owned_networks.add(network)
        max_connections = "32" if label == "journeys" else "20"
        self.run("docker", "create", "--pull", "never", "--name", name,
                 "--network", network, "--memory", "256m", "--cpus", "1",
                 "--pids-limit", "128", "-e", "POSTGRES_USER=rehearsal",
                 "-e", "POSTGRES_PASSWORD=synthetic", "-e", "POSTGRES_DB=rehearsal",
                 self.pg_image, "-c", "shared_buffers=32MB", "-c",
                 "max_connections=" + max_connections)
        self.owned_containers.add(name)
        self.run("docker", "start", name)
        for _ in range(40):
            ready = self.run("docker", "exec", name, "pg_isready", "-U", "rehearsal",
                             check=False)
            if ready.returncode == 0:
                break
            time.sleep(.5)
        else:
            raise RuntimeError("restore database readiness deadline exceeded")
        env = self.private_write(
            label + ".env",
            ("MIGRATION_DATABASE_URL=postgresql://rehearsal:synthetic@" + name
             + "/rehearsal\n").encode())
        return name, network, env

    def run_database_journeys(self, candidate: str, network: str, env: Path) -> None:
        """Run accepted operation/create/phone cases on a separate synthetic DB."""
        tests_root = self.directory / "candidate-tests"
        program = self.directory / "database-journeys.py"
        synthetic = dict(item.split("=", 1) for item in SYNTHETIC_APPLICATION_ENV)
        synthetic.pop("DATABASE_URL")
        program.write_text('''import os,unittest,asyncpg
os.environ.update(''' + repr(synthetic) + ''')
os.environ['DATABASE_URL']=os.environ['MIGRATION_DATABASE_URL']
import tests.integration.test_booking_operations as operations
import tests.integration.test_verified_booking_create as create
import tests.integration.test_phone_booking_flow as phone
def configure(cls):
 cls.driver=asyncpg
 cls.dsn=os.environ['MIGRATION_DATABASE_URL']
def finish(cls): pass
classes=(operations.BookingOperationDatabaseTests,create.VerifiedCreateDatabaseTests,
         phone.PhoneBookingDatabaseTests,phone.PhoneReadinessDatabaseTests)
for cls in classes:
 cls.setUpClass=classmethod(configure)
 cls.tearDownClass=classmethod(finish)
suite=unittest.TestSuite()
loader=unittest.TestLoader()
for cls in classes:
 for name in loader.getTestCaseNames(cls):
  if name!='test_dump_restore_preserves_exclusion_and_history':
   suite.addTest(cls(name))
result=unittest.TextTestRunner(verbosity=1).run(suite)
if result.testsRun < 40 or result.skipped or not result.wasSuccessful(): raise SystemExit(1)
print('T020C_DATABASE_JOURNEYS='+str(result.testsRun))
''', encoding="utf-8", newline="\n")
        program.chmod(0o600)
        result = self.one_shot(
            "t020c-db-tests-" + self.directory.name, candidate,
            ["-B", "/tmp/database-journeys.py"], network=network, env_file=env,
            mounts=((tests_root / "tests", "/app/tests", True),
                    (program, "/tmp/database-journeys.py", True)),
            synthetic_env=True, timeout=600)
        output = result.stdout if isinstance(result.stdout, bytes) else b""
        count = re.search(rb"T020C_DATABASE_JOURNEYS=(\d+)", output)
        if not output:
            print("REAL_POSTGRES_OPERATION_CREATE_PHONE_JOURNEYS_OK", flush=True)
            return
        if count is None or int(count.group(1)) < 40:
            raise RuntimeError("database journey execution count differs")
        self.private_write("database-journeys.json", json.dumps({
            "tests_run": int(count.group(1)), "database": "separate-synthetic",
        }, sort_keys=True).encode())
        print("REAL_POSTGRES_OPERATION_CREATE_PHONE_JOURNEYS_OK", flush=True)

    def rehearsal(self, candidate: str, archive: Path) -> None:
        name = network = synthetic_name = synthetic_network = None
        try:
            name, network, env = self._create_rehearsal_db("restore")
            data = archive.read_bytes()
            result = self.run("docker", "exec", "-i", name, "pg_restore", "--list",
                              input=data, check=False)
            if result.returncode:
                raise RuntimeError("restore archive list verification failed")
            self.run("docker", "exec", "-i", name, "pg_restore", "--exit-on-error",
                     "--clean", "--if-exists", "--no-owner", "--no-acl",
                     "-U", "rehearsal", "-d", "rehearsal", input=data, timeout=180)
            before = self.fingerprint(candidate, network=network, env_file=env)
            probe = self.schema_probe(candidate, network=network, env_file=env)
            if probe["version"] != "0004":
                raise RuntimeError("restored copy is not schema 0004")
            self.one_shot("t020c-migrate-" + self.directory.name, candidate,
                          ["-B", "-m", "migrations.runner", "--prepare-operations"],
                          network=network, env_file=env)
            first = self.schema_probe(candidate, network=network, env_file=env)
            self.one_shot("t020c-migrate-repeat-" + self.directory.name, candidate,
                          ["-B", "-m", "migrations.runner", "--prepare-operations"],
                          network=network, env_file=env)
            second = self.schema_probe(candidate, network=network, env_file=env)
            after = self.fingerprint(candidate, network=network, env_file=env)
            if (first["version"] != "0005" or first["operation_count"] != 0
                    or second != first or after != before):
                raise RuntimeError("0005 rehearsal preservation or repeat safety failed")
            self.private_write("rehearsal-fingerprint.json",
                               json.dumps(before, sort_keys=True).encode())
            synthetic_name, synthetic_network, synthetic_env = \
                self._create_rehearsal_db("journeys")
            self.one_shot("t020c-synthetic-migrate-" + self.directory.name, candidate,
                          ["-B", "-m", "migrations.runner", "--prepare-operations"],
                          network=synthetic_network, env_file=synthetic_env)
            self.run_database_journeys(candidate, synthetic_network, synthetic_env)
            print("RESTORE_0005_REPEAT_RECOVERY_REHEARSAL_OK", flush=True)
        finally:
            for container in (name, synthetic_name):
                if container:
                    self.run("docker", "rm", "-f", "-v", container, check=False)
                    self.owned_containers.discard(container)
            for owned_network in (network, synthetic_network):
                if owned_network:
                    self.run("docker", "network", "rm", owned_network, check=False)
                    self.owned_networks.discard(owned_network)

    def recheck_before_stop(self) -> None:
        previous = self.directory / "previous-override.json"
        live = self.root / "docker-compose.override.yml"
        if live.is_symlink() or live.read_bytes() != previous.read_bytes():
            raise RuntimeError("private override changed before cutover")
        if sha256((self.root / "docker-compose.yml").read_bytes()) != (
                self.directory / "base-compose.sha256").read_text():
            raise RuntimeError("base Compose changed before cutover")
        if sha256((self.root / "nginx/nginx.conf").read_bytes()) != NGINX_HASH:
            raise RuntimeError("nginx configuration changed before cutover")
        current = self.inspect_once(self.app_container)
        if (current["Image"] != OLD_IMAGE or not current["State"]["Running"]
                or current["State"].get("Health", {}).get("Status") != "healthy"
                or container_contract(current, candidate=False) != self.previous_contract
                or not typed_mapping_equal(self.settings(), self.previous_settings)):
            raise RuntimeError("running application changed before cutover")
        if self.schema_probe(OLD_IMAGE)["version"] != "0004":
            raise RuntimeError("live schema changed before cutover")
        self.public_health()

    def stop_writers(self, strict: bool) -> None:
        for name, seconds in ((self.nginx_container, 120), (self.app_container, 60)):
            self.run("docker", "stop", "--time", str(seconds), name,
                     timeout=seconds + 15, check=False)
        running = []
        for name in (self.nginx_container, self.app_container):
            try:
                if self.inspect_once(name)["State"]["Running"]:
                    running.append(name)
            except BaseException:
                running.append("unverified")
        if strict and running:
            raise RuntimeError("ingress or application writers did not stop")

    def install_override(self, source: Path) -> None:
        live = self.root / "docker-compose.override.yml"
        if source.is_symlink() or live.is_symlink():
            raise RuntimeError("override path is a symlink")
        temporary = self.root / (".t020c-override-" + self.directory.name)
        with temporary.open("xb") as output:
            output.write(source.read_bytes())
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, live)

    def apply_live_migration(self) -> None:
        if self.candidate_image is None:
            raise RuntimeError("candidate image is unavailable")
        self.one_shot("t020c-live-migrate-" + self.directory.name, self.candidate_image,
                      ["-B", "-m", "migrations.runner", "--prepare-operations"],
                      network=self.network, env_file=self.live_env)
        schema = self.schema_probe(self.candidate_image)
        if schema["version"] != "0005" or schema["operation_count"] != 0:
            raise RuntimeError("live 0005 verification failed")

    def verify_live_fingerprints(self) -> None:
        baseline = strict_json((self.directory / "final-fingerprint.json").read_bytes())
        if self.fingerprint(self.candidate_image) != baseline:
            raise RuntimeError("preexisting business rows changed")

    def _replace_candidate(self, image: str, *, paused: bool,
                           require_empty_operations: bool) -> None:
        self.compose(self.root / "docker-compose.override.yml", "up", "-d", "--no-deps",
                     "--no-build", "--pull", "never", "--force-recreate", "app",
                     timeout=120)
        self.wait_ready()
        current = self.inspect_once(self.app_container)
        if (current["Image"] != image or not current["State"]["Running"]
                or current["State"].get("Health", {}).get("Status") != "healthy"
                or current["Config"]["Labels"].get("org.opencontainers.image.revision") != self.commit):
            raise RuntimeError("replacement image or source differs")
        settings = self.settings()
        expected = settings_for(self.previous_settings, candidate=True, paused=paused)
        if not typed_mapping_equal(settings, expected):
            raise RuntimeError("replacement effective settings changed")
        candidate_environment = current["Config"].get("Env") or []
        expected_environment = {
            VERIFIED_ENV + "=true", PAUSED_ENV + "=" + str(paused).lower()}
        actual_environment = [entry for entry in candidate_environment
                              if entry.startswith((VERIFIED_ENV + "=", PAUSED_ENV + "="))]
        if len(actual_environment) != 2 or set(actual_environment) != expected_environment:
            raise RuntimeError("replacement release flags differ")
        if container_contract(current, candidate=True) != self.previous_contract:
            raise RuntimeError("replacement runtime contract changed")
        if hidden_candidate_paths(current.get("Mounts") or [], self.overlay_sources):
            raise RuntimeError("replacement mount hides reviewed candidate source")
        schema = self.schema_probe(image)
        if (schema["version"] != "0005"
                or (require_empty_operations and schema["operation_count"] != 0)):
            raise RuntimeError("replacement schema readiness failed")
        text_hashes = {
            **self.manifest["runtime"], **self.manifest["configuration"],
            **self.manifest["migrations"], **self.manifest["protected_runtime"],
        }
        probe = strict_json(self.run(
            "docker", "exec", self.app_container, "python", "-c",
            source_probe_code(text_hashes, self.manifest["model_assets"])).stdout)
        if (probe.get("hashes") != {**text_hashes, **self.manifest["model_assets"]}
                or not typed_mapping_equal(probe.get("settings"), expected)):
            raise RuntimeError("running candidate source or settings differs")

    def replace_app(self, image: str, *, paused: bool) -> None:
        self._replace_candidate(image, paused=paused, require_empty_operations=True)

    def replace_recovery_app(self, image: str) -> None:
        self._replace_candidate(image, paused=True, require_empty_operations=False)

    def start_ingress(self) -> None:
        self.run("docker", "start", self.nginx_container)
        self.run("docker", "exec", self.nginx_container, "nginx", "-t")
        self.public_health()

    def force_close_ingress(self) -> bool:
        self.run("docker", "stop", "--time", "30", self.nginx_container, timeout=45,
                 check=False)
        try:
            return not self.inspect_once(self.nginx_container)["State"]["Running"]
        except BaseException:
            return False

    def recover(self, candidate: str, paused_override: Path) -> None:
        self.stop_writers(strict=True)
        version = self.live_schema_version()
        if version == "0004":
            self.restore_previous_app()
            self.mark("recovered", b"0004 previous app\n")
            print("RECOVERY_PRE_0005_PREVIOUS_APP_READY_HTTPS_OK", flush=True)
            return
        if version != "0005":
            raise RuntimeError("durable schema state is uncertain")
        self.install_override(paused_override)
        self.replace_recovery_app(candidate)
        if not self.force_close_ingress():
            raise RuntimeError("recovery ingress closure unverified")
        self.mark("recovery-needed", b"0005 candidate paused; ingress stopped\n")
        print("RECOVERY_0005_CANDIDATE_PAUSED_INGRESS_STOPPED", flush=True)
        print("RECOVERY_MANUAL_REVIEW_REQUIRED", flush=True)

    def restore_previous_app(self) -> None:
        previous = self.directory / "previous-override.json"
        self.install_override(previous)
        self.compose(self.root / "docker-compose.override.yml", "up", "-d", "--no-deps",
                     "--no-build", "--pull", "never", "--force-recreate", "app",
                     timeout=120)
        self.wait_ready()
        current = self.inspect_once(self.app_container)
        if (current["Image"] != OLD_IMAGE
                or current["Config"]["Labels"].get("org.opencontainers.image.revision")
                != DEPLOYED_SOURCE
                or not typed_mapping_equal(self.settings(), self.previous_settings)):
            raise RuntimeError("previous image recovery differs")
        if self.schema_probe(OLD_IMAGE)["version"] != "0004":
            raise RuntimeError("previous schema recovery differs")
        self.start_ingress()

    def deploy(self, candidate: str, paused_override: Path, active_override: Path) -> None:
        self.candidate_image = candidate
        self.recheck_before_stop()
        self.mark("cutover-started", (self.commit + "\n").encode())
        print("CUTOVER_STARTED_STOPPING_BOOKING_INGRESS_AND_WRITERS", flush=True)
        try:
            self.stop_writers(strict=True)
            archive = self.backup("final-before-0005")
            if not archive.exists():
                raise RuntimeError("final protected backup missing")
            if self.candidate_image is not None:
                fingerprint = self.fingerprint(self.candidate_image)
                self.private_write("final-fingerprint.json",
                                   json.dumps(fingerprint, sort_keys=True).encode())
            self.apply_live_migration()
            self.verify_live_fingerprints()
            self.install_override(paused_override)
            self.replace_app(candidate, paused=True)
            print("CANDIDATE_0005_READY_WORKERS_PAUSED_INGRESS_STOPPED", flush=True)
            self.install_override(active_override)
            self.replace_app(candidate, paused=False)
            self.start_ingress()
            self.mark("deployed", (self.commit + "\n" + candidate + "\n").encode())
            print("VERIFIED_PHONE_BOOKING_DEPLOYED_READY_HTTPS_OK", flush=True)
            print("DEPLOYED_COMMIT=" + self.commit, flush=True)
            print("DEPLOYED_IMAGE=" + candidate, flush=True)
        except BaseException:
            print("CUTOVER_FAILED_INSPECTING_DURABLE_SCHEMA_FOR_RECOVERY", flush=True)
            try:
                self.recover(candidate, paused_override)
            except BaseException:
                closed = self.force_close_ingress()
                try:
                    self.mark("recovery-needed", b"recovery failed; ingress state recorded\n")
                except BaseException:
                    pass
                print("RECOVERY_FAILED_INGRESS_CLOSED" if closed
                      else "RECOVERY_FAILED_INGRESS_CLOSURE_UNVERIFIED", flush=True)
                print("RECOVERY_UNVERIFIED_MANUAL_REVIEW_REQUIRED", flush=True)
            raise

    def cleanup(self) -> None:
        failures = []
        for name in tuple(self.owned_containers):
            try:
                self.run("docker", "rm", "-f", name, check=False)
            except BaseException as error:
                failures.append(type(error).__name__)
            self.owned_containers.discard(name)
        for name in tuple(self.owned_networks):
            try:
                self.run("docker", "network", "rm", name, check=False)
            except BaseException as error:
                failures.append(type(error).__name__)
            self.owned_networks.discard(name)
        if failures:
            with (self.directory / "private.log").open("ab") as log:
                log.write(("cleanup failures: " + ",".join(failures) + "\n").encode())

    def execute(self) -> None:
        try:
            self.stage = "preflight"
            original = self.preflight()
            self.stage = "assets"
            self.stage_assets()
            self.stage = "build"
            candidate = self.build_candidate()
            self.stage = "tests"
            self.run_candidate_tests(candidate)
            self.stage = "overlays"
            paused = self.prepare_override(candidate, original, paused=True)
            active = self.prepare_override(candidate, original, paused=False)
            self.stage = "backup-rehearsal"
            archive = self.backup("precutover-rehearsal")
            self.rehearsal(candidate, archive)
            self.verify_frozen_inputs()
            self.stage = "cutover"
            self.deploy(candidate, paused, active)
            self.verify_frozen_inputs()
        finally:
            self.cleanup()


def _docker_local(*arguments, input=None, check=True, timeout=180):
    result = subprocess.run(["docker", *arguments], input=input, capture_output=True,
                            timeout=timeout, check=False)
    if check and result.returncode:
        raise RuntimeError("local Docker rehearsal command failed")
    return result


def run_local_migration_rehearsal(*, repository: Path, source_container: str,
                                  restore_container: str, archive: Path) -> None:
    """Opt-in local real PostgreSQL 16 backup/restore/0005 preservation probe."""
    network_data = json.loads(_docker_local(
        "inspect", source_container, "--format", "{{json .NetworkSettings.Networks}}"
    ).stdout)
    if len(network_data) != 1:
        raise RuntimeError("local rehearsal network differs")
    network, = network_data
    image = os.environ.get("T020_C_REHEARSAL_APP_IMAGE",
                           "ai-phone-t049a-candidate:20260913")

    def runner(container, *args):
        url = "postgresql://t020c:synthetic@" + container + "/t020c"
        return _docker_local(
            "run", "--rm", "--pull", "never", "--network", network,
            "--memory", "768m", "--cpus", "1", "--pids-limit", "192",
            "-e", "MIGRATION_DATABASE_URL=" + url,
            "--mount", "type=bind,source=" + str(repository) + ",target=/work,readonly",
            "-w", "/work", "--entrypoint", "python", image,
            "-B", "-m", "migrations.runner", *args, timeout=180)

    runner(source_container, "--prepare")
    seed = """
INSERT INTO public.bookings(call_sid,phone_number,client_name,accountant_name,
 appointment_time,client_type,language,ms_booking_id,notes)
VALUES('t020c-preserved','+12025550100','Synthetic Customer','Synthetic Consultant',
 TIMESTAMP '2026-09-21 10:00:00','individual','en','t020c-provider','synthetic rehearsal');
INSERT INTO public.booking_provider_observations(
 booking_id,snapshot_provider_id,snapshot_start,snapshot_status,checked_at,outcome,
 http_status,provider_start,provider_end,staff_member_ids,service_id,is_location_online,
 observed_provider_id)
SELECT id,'t020c-provider',TIMESTAMPTZ '2026-09-21 14:00:00+00','confirmed',
 TIMESTAMPTZ '2026-09-19 12:00:00+00','present',200,
 TIMESTAMPTZ '2026-09-21 14:00:00+00',TIMESTAMPTZ '2026-09-21 14:30:00+00',
 ARRAY['staff'],'service',false,'t020c-provider' FROM public.bookings
 WHERE call_sid='t020c-preserved';
INSERT INTO public.booking_notification_reconciliation(
 booking_id,snapshot_tenant_id,snapshot_business_id,snapshot_provider_id,
 snapshot_start,snapshot_version,enrolled_at,last_present_at,last_evidence_started_at,
 first_missing_at,last_missing_at,missing_count,last_result,disposition,reason,
 transitioned_at,last_error_category,updated_at)
SELECT id,'tenant','business','t020c-provider',TIMESTAMPTZ '2026-09-21 14:00:00+00',
 1,TIMESTAMPTZ '2026-09-19 12:00:00+00',TIMESTAMPTZ '2026-09-19 12:00:00+00',
 TIMESTAMPTZ '2026-09-19 12:00:00+00',NULL,NULL,0,'present','active',NULL,NULL,NULL,
 TIMESTAMPTZ '2026-09-19 12:00:00+00' FROM public.bookings
 WHERE call_sid='t020c-preserved';
"""
    _docker_local("exec", "-i", source_container, "psql", "-v", "ON_ERROR_STOP=1",
                  "-U", "t020c", "-d", "t020c", input=seed.encode())
    before = _docker_local(
        "exec", source_container, "psql", "-At", "-U", "t020c", "-d", "t020c",
        "-c", "SELECT row_to_json(x)::text FROM (SELECT count(*) AS bookings,(SELECT count(*) FROM booking_provider_observations) AS observations,(SELECT count(*) FROM booking_notification_reconciliation) AS notifications FROM bookings) x"
    ).stdout.strip()
    data = _docker_local("exec", source_container, "pg_dump", "-U", "t020c", "-d",
                         "t020c", "-Fc", "--no-owner", "--no-acl").stdout
    archive.write_bytes(data)
    if not data or _docker_local("exec", "-i", restore_container, "pg_restore", "--list",
                                 input=data, check=False).returncode:
        raise RuntimeError("local backup verification failed")
    _docker_local("exec", "-i", restore_container, "pg_restore", "--exit-on-error",
                  "--clean", "--if-exists", "--no-owner", "--no-acl", "-U", "t020c",
                  "-d", "t020c", input=data)
    runner(restore_container, "--prepare-operations")
    first = runner(restore_container, "--prepare-operations").stdout
    if b"up-to-date 0005" not in first:
        raise RuntimeError("local repeat migration differs")
    after = _docker_local(
        "exec", restore_container, "psql", "-At", "-U", "t020c", "-d", "t020c",
        "-c", "SELECT row_to_json(x)::text FROM (SELECT count(*) AS bookings,(SELECT count(*) FROM booking_provider_observations) AS observations,(SELECT count(*) FROM booking_notification_reconciliation) AS notifications FROM bookings) x"
    ).stdout.strip()
    checks = _docker_local(
        "exec", restore_container, "psql", "-At", "-U", "t020c", "-d", "t020c",
        "-c", "SELECT (SELECT count(*) FROM booking_operations)::text||':'||(SELECT count(*) FROM booking_operation_contract)::text||':'||(SELECT count(*) FROM schema_migrations WHERE version='0005')::text"
    ).stdout.strip()
    if after != before or checks != b"0:1:1":
        raise RuntimeError("local 0005 rehearsal did not preserve rows")


def main(commit: str) -> None:
    import fcntl

    if os.geteuid() != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("root and exact 40-hex release commit required")
    os.umask(0o077)
    descriptor = os.open("/run/ai-phone-deployment.lock",
                         os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another deployment owns the lock") from None
        directory = Path(tempfile.mkdtemp(prefix="ai-phone-verified-booking-release.",
                                          dir="/opt"))
        directory.chmod(0o700)
        item = Release(directory, commit)
        try:
            item.execute()
        except BaseException:
            print("VERIFIED_BOOKING_RELEASE_FAILED_STAGE=" + item.stage.upper(), flush=True)
            print("PROTECTED_RELEASE_DIRECTORY=" + str(directory), flush=True)
            raise
        print("PROTECTED_RELEASE_DIRECTORY=" + str(directory), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: deploy-verified-phone-booking.py FULL_COMMIT_SHA")
    try:
        main(sys.argv[1])
    except KeyboardInterrupt:
        print("VERIFIED_BOOKING_RELEASE_INTERRUPTED_PRIVATE_EVIDENCE_RETAINED",
              file=sys.stderr, flush=True)
        raise SystemExit(130) from None
    except BaseException:
        print("VERIFIED_BOOKING_RELEASE_ABORTED_PRIVATE_EVIDENCE_REQUIRED",
              file=sys.stderr, flush=True)
        raise SystemExit(1) from None
