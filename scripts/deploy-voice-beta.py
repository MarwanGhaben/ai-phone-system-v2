"""Owner-run T034-C app-only availability acknowledgement rollout.

Importing this module is side-effect free. The executable script must be fetched
from the exact reviewed commit passed on the command line. Public output uses only
fixed stage/type markers; command details stay in the protected release directory.
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
DEPLOYED_SOURCE = "2f7e15e1b38f051615fc63c11f7bf6a66d827ad9"
OLD_IMAGE = "sha256:9cecc2353a29f4fefdd95566a7c1365fca8cbd380e9db8777ce08914cf3b14f5"
NGINX_HASH = "59ac9bdaa9a86c1022a573c9709faddf7e93a53015da8acfcc4330b3e288f3ab"
PROJECT = "ai-phone-system-v2"
MANIFEST_PATH = "docs/delegation/T034-C-accepted-hashes.json"
PUBLIC_HEALTH_URL = "https://aiagent.ghaben.ca:8443/health"

RUNTIME_PATHS = (
    "services/conversation/events.py",
    "services/conversation/language_policy.py",
    "services/conversation/orchestrator.py",
    "services/conversation/turn_controller.py",
    "services/stt/elevenlabs_stt_service.py",
    "services/stt/stt_base.py",
    "services/telephony/twilio_service.py",
    "services/tts/elevenlabs_service.py",
)
TEST_PATHS = (
    "tests/conftest.py",
    "tests/conversation/test_availability_acknowledgement.py",
    "tests/conversation/test_booking_cancellation.py",
    "tests/conversation/test_booking_safety_guards.py",
    "tests/conversation/test_booking_time_validation.py",
    "tests/conversation/test_caller_language.py",
    "tests/conversation/test_language_policy.py",
    "tests/conversation/test_playback_ownership.py",
    "tests/conversation/test_transcript_identity.py",
    "tests/conversation/test_transfer_playback.py",
    "tests/conversation/test_turn_controller.py",
    "tests/stt/test_elevenlabs_reset.py",
    "tests/stt/test_reset_preservation.py",
    "tests/stt/test_utterance_events.py",
    "tests/stt/test_whisper_audio.py",
    "tests/telephony/test_playback_generations.py",
    "tests/telephony/test_twilio_stream.py",
    "tests/tts/test_call_isolation.py",
)
TEST_TARGETS = tuple(path for path in TEST_PATHS if path != "tests/conftest.py")
SOURCE_SCOPE = frozenset(RUNTIME_PATHS)
APPLICATION_SCOPE = (
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
}
SETTINGS_CODE = (
    "import json; from config.settings import settings; "
    "print(json.dumps(settings.model_dump(mode='json'),sort_keys=True))"
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


def normalized_bytes(data: bytes) -> bytes:
    """Apply the manifest's only allowed source transformation."""
    return data.replace(b"\r\n", b"\n")


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


def mount_fingerprint(mounts: list[dict]) -> list[str]:
    """Ignore Docker mount ordering while retaining all fields and duplicates."""
    return sorted(json.dumps(mount, sort_keys=True) for mount in mounts)


def environment_fingerprint(environment: list[str]) -> list[str]:
    """Environment ordering is irrelevant; duplicate values remain significant."""
    return sorted(environment)


def semantic_compose(configuration: dict) -> dict:
    """Normalize only the app image and mount list order for comparison."""
    value = copy.deepcopy(configuration)
    app = value["services"]["app"]
    app["image"] = "<app-image>"
    volumes = app.get("volumes", [])
    if not isinstance(volumes, list):
        raise RuntimeError("rendered application mounts are invalid")
    app["volumes"] = sorted(volumes, key=lambda item: json.dumps(item, sort_keys=True))
    return value


def hidden_runtime_paths(mounts: list[dict]) -> tuple[str, ...]:
    """Return candidate files hidden by a running app mount."""
    hidden = []
    for source_path in RUNTIME_PATHS:
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


def container_contract(container: dict) -> dict:
    """Runtime fields which must survive an app-only replacement."""
    config = container["Config"]
    host = container["HostConfig"]
    return {
        "mounts": mount_fingerprint(container.get("Mounts", [])),
        "environment": environment_fingerprint(config.get("Env") or []),
        "command": config.get("Cmd"),
        "entrypoint": config.get("Entrypoint"),
        "ports": host.get("PortBindings"),
        "restart": host.get("RestartPolicy"),
        "networks": sorted(container["NetworkSettings"]["Networks"]),
    }


def source_probe_code(hashes: dict[str, str], *, include_settings: bool) -> str:
    modules = tuple(path[:-3].replace("/", ".") for path in RUNTIME_PATHS)
    return (
        "import hashlib,importlib,json; from pathlib import Path; "
        f"expected={hashes!r}; modules={modules!r}; "
        "actual={p:hashlib.sha256(Path('/app',p).read_bytes().replace(b'\\r\\n',b'\\n')).hexdigest() for p in expected}; "
        "[importlib.import_module(m) for m in modules]; "
        + (
            "from config.settings import settings; "
            "configured=settings.model_dump(mode='json'); "
            if include_settings else "configured=None; "
        )
        + "print(json.dumps({'hashes':actual,'settings':configured},sort_keys=True))"
    )


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
        self.assets = self.directory / "assets"

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
        return json.loads(
            self.run("docker", "exec", name, "python", "-c", SETTINGS_CODE).stdout
        )

    def render_compose(self, override: Path) -> dict:
        result = self.compose(override, "config", "--format", "json")
        return strict_json(result.stdout)

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
                        if (
                            state["Running"]
                            and state.get("Health", {}).get("Status") == "healthy"
                        ):
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
        """Record a safe cleanup category without replacing the primary error."""
        path = self.directory / "private.log"
        with path.open("ab") as log:
            log.write(("cleanup_failure_type=" + type(error).__name__ + "\n").encode())
        path.chmod(0o600)

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
            "512m",
            "--cpus",
            "1",
            "--pids-limit",
            "128",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
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
        status = self.run(
            "git", "status", "--porcelain", "--untracked-files=no"
        ).stdout.decode().strip()
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
            raise RuntimeError("release source exceeds reviewed voice runtime scope")
        self.run("git", "cat-file", "-e", self.commit + ":scripts/deploy-voice-beta.py")
        self.run("git", "cat-file", "-e", self.commit + ":" + MANIFEST_PATH)

    def stage_assets(self) -> None:
        manifest_bytes = normalized_bytes(self.git_bytes(MANIFEST_PATH))
        manifest = strict_json(manifest_bytes)
        if (
            manifest.get("schema") != 1
            or manifest.get("baseline_commit") != DEPLOYED_SOURCE
            or manifest.get("branch") != "codex/phase-4-arabic-voice-quality"
            or set(manifest.get("runtime", {})) != set(RUNTIME_PATHS)
            or set(manifest.get("tests", {})) != set(TEST_PATHS)
        ):
            raise RuntimeError("accepted manifest contract differs")
        if "CRLF converted to LF" not in str(manifest.get("hash_convention", "")):
            raise RuntimeError("accepted manifest normalization differs")
        self.assets.mkdir(mode=0o700)
        self.asset_write("manifest.json", manifest_bytes)
        for section, paths in (("runtime", RUNTIME_PATHS), ("tests", TEST_PATHS)):
            for path in paths:
                data = normalized_bytes(self.git_bytes(path))
                if sha256(data) != manifest[section][path]:
                    raise RuntimeError("reviewed asset hash differs")
                self.asset_write(path, data)
        self.manifest = manifest
        self.runtime_hashes = dict(manifest["runtime"])
        print("REVIEWED_VOICE_ASSETS_STAGED", flush=True)

    def capacity(self) -> None:
        if shutil.disk_usage(self.root).free < 1024**3:
            raise RuntimeError("insufficient release disk capacity")
        match = re.search(
            r"^MemAvailable:\s+(\d+) kB$",
            Path("/proc/meminfo").read_text(),
            re.MULTILINE,
        )
        if match is None or int(match.group(1)) < 512 * 1024:
            raise RuntimeError("insufficient release memory capacity")

    def _validate_settings(self, settings: dict) -> None:
        if any(settings.get(key) != value for key, value in REQUIRED_SETTINGS.items()):
            raise RuntimeError("notification or observation settings differ")

    def schema_matches(self, image: str = OLD_IMAGE) -> bool:
        name = "t034b-schema-" + self.directory.name
        result = self.one_shot(
            name,
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
            or old["Config"]["Labels"].get("org.opencontainers.image.revision")
            != DEPLOYED_SOURCE
            or old["Config"]["Labels"].get("com.docker.compose.project") != PROJECT
        ):
            raise RuntimeError("running app does not match accepted source and image")
        if hidden_runtime_paths(old.get("Mounts", [])):
            raise RuntimeError("running mount hides reviewed voice source")
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
        if (
            not database["State"]["Running"]
            or database["State"].get("Health", {}).get("Status") != "healthy"
        ):
            raise RuntimeError("database is not healthy")
        common = set(old["NetworkSettings"]["Networks"]) & set(
            database["NetworkSettings"]["Networks"]
        )
        if len(common) != 1:
            raise RuntimeError("application and database network differ")
        network, = common
        settings = self.settings()
        self._validate_settings(settings)
        database_url = settings.get("database_url")
        if not isinstance(database_url, str) or "\r" in database_url or "\n" in database_url:
            raise RuntimeError("live database target is invalid")
        parsed = urlsplit(database_url)
        database_env = dict(item.split("=", 1) for item in database["Config"]["Env"])
        expected_database = database_env.get(
            "POSTGRES_DB", database_env.get("POSTGRES_USER", "postgres")
        )
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
        if hidden_runtime_paths(rendered["services"]["app"].get("volumes", [])):
            raise RuntimeError("effective mount hides reviewed voice source")
        self.old = old
        self.previous_settings = settings
        self.previous_rendered = rendered
        self.previous_contract = container_contract(old)
        self.network = network
        self.private_write("previous-override.json", override_bytes)
        self.private_write("previous-app.json", json.dumps(old).encode())
        self.private_write("previous-settings.json", json.dumps(settings).encode())
        self.private_write("previous-rendered.json", json.dumps(rendered).encode())
        self.private_write("previous-contract.json", json.dumps(self.previous_contract).encode())
        self.private_write(
            "base-compose.sha256",
            sha256((self.root / "docker-compose.yml").read_bytes()).encode(),
        )
        self.live_schema_env = self.private_write(
            "live-schema.env",
            ("MIGRATION_DATABASE_URL=" + database_url + "\n").encode(),
        )
        if not self.schema_matches(OLD_IMAGE):
            raise RuntimeError("live read-only schema 0004 contract differs")
        print("PREFLIGHT_0004_SETTINGS_READY_HTTPS_OK", flush=True)
        return original

    def build_candidate(self) -> str:
        if self.manifest is None or not self.runtime_hashes:
            raise RuntimeError("reviewed assets were not staged")
        parent = "ai-phone-t034b-parent:" + self.directory.name
        self.run("docker", "tag", OLD_IMAGE, parent)
        if self.inspect(parent)["Id"] != OLD_IMAGE:
            raise RuntimeError("candidate parent image changed")
        context = self.directory / "build"
        context.mkdir(mode=0o700)
        for path in RUNTIME_PATHS:
            source = self.assets / path
            target = context / path
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            target.write_bytes(source.read_bytes())
            target.chmod(0o600)
        dockerfile = "FROM " + parent + "\n" + "".join(
            "COPY " + path + " /app/" + path + "\n" for path in RUNTIME_PATHS
        )
        (context / "Dockerfile").write_text(dockerfile, encoding="utf-8", newline="\n")
        (context / "Dockerfile").chmod(0o600)
        tag = "ai-phone-t034b-candidate:" + self.directory.name
        self.run(
            "docker",
            "build",
            "--network",
            "none",
            "--pull=false",
            "--label",
            "org.opencontainers.image.revision=" + self.commit,
            "-t",
            tag,
            str(context),
            timeout=300,
        )
        built = self.inspect(tag)
        image = built["Id"]
        if not image.startswith("sha256:"):
            raise RuntimeError("candidate image has no immutable identity")
        if built["Config"]["Labels"].get("org.opencontainers.image.revision") != self.commit:
            raise RuntimeError("candidate image source label differs")
        probe = self.one_shot(
            "t034b-image-probe-" + self.directory.name,
            image,
            ["-c", source_probe_code(self.runtime_hashes, include_settings=False)],
            synthetic_env=True,
        )
        value = json.loads(probe.stdout)
        if value.get("hashes") != self.runtime_hashes:
            raise RuntimeError("candidate image source differs")
        print("VOICE_CANDIDATE_IMAGE_HASHES_IMPORTS_OK", flush=True)
        return image

    def run_candidate_tests(self, image: str) -> None:
        mount = "type=bind,src=" + str(self.assets / "tests") + ",dst=/audit/tests,readonly"
        arguments = [
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            *("/audit/" + path for path in TEST_TARGETS),
        ]
        self.one_shot(
            "t034b-tests-" + self.directory.name,
            image,
            arguments,
            mounts=(mount,),
            synthetic_env=True,
            timeout=180,
        )
        print("VOICE_CANDIDATE_ACCEPTED_TESTS_PASSED", flush=True)

    def candidate_probe(self, override: Path, label: str) -> dict:
        name = "t034b-settings-" + label + "-" + self.directory.name
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
                source_probe_code(self.runtime_hashes, include_settings=True),
                timeout=90,
            )
            value = json.loads(result.stdout)
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
        candidate["services"]["app"]["image"] = image
        path = self.private_write(
            "candidate-override.json",
            (json.dumps(candidate, indent=2) + "\n").encode(),
        )
        rendered = self.render_compose(path)
        if semantic_compose(rendered) != semantic_compose(self.previous_rendered):
            raise RuntimeError("effective Compose configuration changed")
        if hidden_runtime_paths(rendered["services"]["app"].get("volumes", [])):
            raise RuntimeError("candidate mount hides reviewed voice source")
        probe = self.candidate_probe(path, "candidate")
        if (
            probe.get("hashes") != self.runtime_hashes
            or probe.get("settings") != self.previous_settings
        ):
            raise RuntimeError("candidate effective source or settings changed")
        print("EFFECTIVE_SETTINGS_MOUNTS_AND_SOURCE_PRESERVED", flush=True)
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
            or container_contract(current) != self.previous_contract
            or self.settings() != self.previous_settings
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
        failures = []
        for name, seconds in (("ai-voice-nginx", "120"), ("ai-voice-app", "60")):
            try:
                result = self.run(
                    "docker",
                    "stop",
                    "--time",
                    seconds,
                    name,
                    timeout=int(seconds) + 15,
                    check=False,
                )
                if result.returncode:
                    failures.append(name)
            except BaseException:
                failures.append(name)
        try:
            running = [
                name
                for name in ("ai-voice-nginx", "ai-voice-app")
                if self.inspect(name)["State"]["Running"]
            ]
        except BaseException:
            running = ["unverified"]
        if strict and (failures or running):
            raise RuntimeError("ingress or application stop could not be verified")

    def install_override(self, source: Path) -> None:
        live = self.root / "docker-compose.override.yml"
        if source.is_symlink() or live.is_symlink():
            raise RuntimeError("override path is a symlink")
        temporary = self.root / (".t034b-override-" + self.directory.name)
        with temporary.open("xb") as output:
            output.write(source.read_bytes())
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, live)

    def running_probe(self) -> dict:
        return json.loads(
            self.run(
                "docker",
                "exec",
                "ai-voice-app",
                "python",
                "-c",
                source_probe_code(self.runtime_hashes, include_settings=True),
            ).stdout
        )

    def verify_replaced(self, image: str, revision: str) -> None:
        self.wait_ready()
        current = self.inspect("ai-voice-app")
        if (
            current["Image"] != image
            or not current["State"]["Running"]
            or current["State"].get("Health", {}).get("Status") != "healthy"
            or current["Config"]["Labels"].get("org.opencontainers.image.revision")
            != revision
            or container_contract(current) != self.previous_contract
        ):
            raise RuntimeError("replacement image or runtime contract differs")
        settings = self.settings()
        self._validate_settings(settings)
        if settings != self.previous_settings:
            raise RuntimeError("replacement effective settings changed")
        if image != OLD_IMAGE:
            probe = self.running_probe()
            if probe.get("hashes") != self.runtime_hashes:
                raise RuntimeError("replacement effective source changed")
        rendered = self.render_compose(self.root / "docker-compose.override.yml")
        if semantic_compose(rendered) != semantic_compose(self.previous_rendered):
            raise RuntimeError("replacement effective Compose changed")
        if not self.schema_matches(image):
            raise RuntimeError("replacement read-only schema 0004 check failed")

    def replace_app(self, image: str, revision: str) -> None:
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
        self.verify_replaced(image, revision)

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
        self.replace_app(OLD_IMAGE, DEPLOYED_SOURCE)
        self.start_ingress()
        print("PREVIOUS_VOICE_APP_RESTORED_SCHEMA_0004_UNCHANGED", flush=True)
        print("RECOVERY_IMAGE=" + OLD_IMAGE, flush=True)

    def deploy(self, candidate: str, candidate_override: Path) -> None:
        self.recheck_before_stop()
        print("PRE_CUTOVER_VOICE_FINGERPRINTS_OK", flush=True)
        self.private_write("cutover-started", (self.commit + "\n").encode())
        print("CUTOVER_STARTED_STOPPING_BETA_TRAFFIC", flush=True)
        try:
            self.stop_components(strict=True)
            self.install_override(candidate_override)
            self.replace_app(candidate, self.commit)
            self.start_ingress()
            self.private_write(
                "deployed",
                (self.commit + "\n" + candidate + "\n" + str(self.directory) + "\n").encode(),
            )
            print("VOICE_BETA_DEPLOYED_READY_HTTPS_OK", flush=True)
            print("DEPLOYED_COMMIT=" + self.commit, flush=True)
            print("DEPLOYED_IMAGE=" + candidate, flush=True)
        except BaseException:
            print("VOICE_CUTOVER_FAILED_RESTORING_PREVIOUS_APP", flush=True)
            try:
                self.recover()
            except BaseException:
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
        self.stage = "build"
        candidate = self.build_candidate()
        self.private_write("candidate-image", (candidate + "\n").encode())
        self.stage = "tests"
        self.run_candidate_tests(candidate)
        self.stage = "settings"
        candidate_override = self.prepare_override(candidate, original)
        self.stage = "cutover"
        self.deploy(candidate, candidate_override)


def main(commit: str) -> None:
    import fcntl  # Linux-only; local test import remains side-effect free.

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
        for previous in Path("/opt").glob("ai-phone-voice-release.*"):
            if (previous / "cutover-started").exists() and not (
                previous / "deployed"
            ).exists():
                raise RuntimeError("prior voice cutover needs review")
        directory = Path(
            tempfile.mkdtemp(prefix="ai-phone-voice-release.", dir="/opt")
        )
        metadata = directory.stat()
        if (
            directory.is_symlink()
            or metadata.st_uid != 0
            or metadata.st_mode & 0o077
        ):
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
