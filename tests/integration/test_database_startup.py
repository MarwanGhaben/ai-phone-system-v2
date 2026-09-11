"""T005-C application startup, readiness and Compose integration regressions."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
import uuid
import warnings
from unittest import mock

import pytest
import yaml
from fastapi import APIRouter
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
SENTINEL = "T005-C-credential-customer@example.invalid"


class _Logger:
    def remove(self): pass
    def add(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass


class _Settings:
    log_level = "INFO"
    allowed_origins = ["http://example.invalid"]
    environment = "test"
    stt_provider = "test"
    public_domain = ""
    twilio_account_sid = "synthetic"
    twilio_auth_token = "synthetic"
    twilio_phone_number = "+12025550100"
    api_host = "127.0.0.1"
    api_port = 8000
    debug = False

    def model_dump(self):
        return {}


class _PassMiddleware:
    def __init__(self, app, **kwargs):
        self.app = app

    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)


def _load_main(monkeypatch, events, *, compatible=True, database_url=None):
    settings = _Settings()
    if database_url is not None:
        settings.database_url = database_url
    monkeypatch.setitem(
        sys.modules, "config.settings",
        types.SimpleNamespace(get_settings=lambda: settings, settings=settings),
    )
    monkeypatch.setitem(
        sys.modules, "loguru", types.SimpleNamespace(logger=_Logger()))
    monkeypatch.setitem(
        sys.modules, "services.conversation.orchestrator",
        types.SimpleNamespace(get_orchestrator=lambda: object()),
    )
    monkeypatch.setitem(
        sys.modules, "services.security.middleware",
        types.SimpleNamespace(
            SecurityHeadersMiddleware=_PassMiddleware,
            RateLimitMiddleware=_PassMiddleware,
            validate_twilio_signature=lambda: True,
        ),
    )
    router = APIRouter()
    monkeypatch.setitem(
        sys.modules, "services.dashboard.dashboard_routes",
        types.SimpleNamespace(router=router, require_auth=lambda: {}),
    )
    monkeypatch.setitem(
        sys.modules, "services.dashboard.dashboard_service",
        types.SimpleNamespace(get_dashboard_service=lambda: object()),
    )
    monkeypatch.setitem(
        sys.modules, "services.knowledge.faq_service",
        types.SimpleNamespace(get_kb_service=lambda: object()),
    )
    monkeypatch.setitem(
        sys.modules, "services.calendar.ms_bookings_service",
        types.SimpleNamespace(get_calendar_service=lambda: object()),
    )

    state = {"compatible": compatible}
    pool = object()

    async def get_pool():
        events.append("pool")
        return pool

    async def check(candidate=None):
        events.append("check")
        assert candidate is None or candidate is pool
        if not state["compatible"]:
            raise RuntimeError(SENTINEL)

    async def close():
        events.append("close")

    monkeypatch.setitem(
        sys.modules, "services.database",
        types.SimpleNamespace(
            get_db_pool=get_pool,
            check_database_compatibility=check,
            close_db_pool=close,
        ),
    )
    if database_url is not None:
        database_spec = importlib.util.spec_from_file_location(
            "t005_real_database_" + uuid.uuid4().hex, ROOT / "services/database.py")
        database = importlib.util.module_from_spec(database_spec)
        database_spec.loader.exec_module(database)
        monkeypatch.setitem(sys.modules, "services.database", database)
        state["database"] = database

    class Scheduler:
        def start(self):
            events.append("scheduler-start")

        def stop(self):
            events.append("scheduler-stop")

    monkeypatch.setitem(
        sys.modules, "services.sms.reminder_scheduler",
        types.SimpleNamespace(get_reminder_scheduler=lambda: Scheduler()),
    )

    class TTS:
        async def prewarm(self):
            events.append("tts-prewarm")

    monkeypatch.setitem(
        sys.modules, "services.tts.elevenlabs_service",
        types.SimpleNamespace(create_elevenlabs_tts=lambda config: TTS()),
    )

    name = "t005_startup_main_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, ROOT / "api/main.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, state


def test_compatible_database_starts_then_ready_tracks_current_database(monkeypatch):
    events = []
    module, state = _load_main(monkeypatch, events)
    with TestClient(module.app) as client:
        assert module.app.state.ready is True
        assert events[:4] == ["pool", "check", "scheduler-start", "tts-prewarm"]
        assert client.get("/health").status_code == 200
        assert client.get("/ready").json() == {"status": "ready"}
        state["compatible"] = False
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json() == {"status": "unavailable"}
        assert SENTINEL not in response.text
    assert module.app.state.ready is False
    assert events[-2:] == ["scheduler-stop", "close"]


def test_incompatible_database_aborts_before_scheduler_or_tts_and_closes(monkeypatch):
    events = []
    module, _ = _load_main(monkeypatch, events, compatible=False)
    with pytest.raises(RuntimeError, match=SENTINEL):
        with TestClient(module.app):
            raise AssertionError("startup unexpectedly succeeded")
    assert events == ["pool", "check", "close"]
    assert module.app.state.ready is False


def test_database_check_is_read_only_bounded_and_redacts_failures(monkeypatch):
    fake_settings = types.SimpleNamespace(database_url=SENTINEL)
    monkeypatch.setitem(
        sys.modules, "config.settings", types.SimpleNamespace(settings=fake_settings))
    monkeypatch.setitem(
        sys.modules, "loguru", types.SimpleNamespace(logger=_Logger()))
    monkeypatch.setitem(
        sys.modules, "asyncpg", types.SimpleNamespace(Pool=object, create_pool=mock.AsyncMock()))
    name = "t005_database_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, ROOT / "services/database.py")
    database = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(database)

    class Transaction:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False

    class Connection:
        def __init__(self): self.commands = []
        def transaction(self, **kwargs):
            assert kwargs == {"isolation": "repeatable_read", "readonly": True}
            return Transaction()
        async def execute(self, command): self.commands.append(command)

    conn = Connection()

    class Acquire:
        async def __aenter__(self): return conn
        async def __aexit__(self, *args): return False

    pool = types.SimpleNamespace(acquire=lambda: Acquire())
    checker = mock.AsyncMock(side_effect=RuntimeError(SENTINEL))
    monkeypatch.setattr(database, "check_runtime_compatibility", checker)
    with pytest.raises(database.DatabaseReadinessError) as error:
        asyncio.run(database.check_database_compatibility(pool))
    assert str(error.value) == "database unavailable"
    assert SENTINEL not in str(error.value)
    assert conn.commands == [
        "SET LOCAL statement_timeout = '4000ms'; SET LOCAL lock_timeout = '3000ms'; "
        "SET LOCAL search_path = pg_catalog"
    ]
    checker.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(database.check_database_compatibility(pool))


def test_database_pool_acquisition_timeout_is_bounded(monkeypatch):
    fake_settings = types.SimpleNamespace(database_url=SENTINEL)
    monkeypatch.setitem(
        sys.modules, "config.settings", types.SimpleNamespace(settings=fake_settings))
    monkeypatch.setitem(
        sys.modules, "loguru", types.SimpleNamespace(logger=_Logger()))
    monkeypatch.setitem(
        sys.modules, "asyncpg", types.SimpleNamespace(Pool=object, create_pool=mock.AsyncMock()))
    name = "t005_database_timeout_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, ROOT / "services/database.py")
    database = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(database)
    monkeypatch.setattr(database, "READINESS_ACQUIRE_TIMEOUT", 0.01)

    class Acquire:
        async def __aenter__(self):
            await asyncio.Event().wait()
        async def __aexit__(self, *args):
            return False

    pool = types.SimpleNamespace(acquire=lambda: Acquire())
    with pytest.raises(database.DatabaseReadinessError, match="database unavailable"):
        asyncio.run(database.check_database_compatibility(pool))


def test_compose_declares_preparation_gate_and_readiness_ordering():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    postgres, migrate, app, nginx = (
        services["postgres"], services["migrate"], services["app"], services["nginx"])
    assert postgres["volumes"] == ["postgres_data:/var/lib/postgresql/data"]
    assert migrate["build"] == app["build"] == {
        "context": ".", "dockerfile": "Dockerfile"}
    assert migrate["command"] == ["python", "-m", "migrations.runner", "--prepare"]
    assert migrate["restart"] == "no"
    assert migrate["healthcheck"] == {"disable": True}
    assert migrate["depends_on"] == {"postgres": {"condition": "service_healthy"}}
    assert set(migrate["environment"]) == {"MIGRATION_DATABASE_URL"}
    assert not any(key in migrate for key in ("ports", "expose", "volumes", "env_file"))
    assert app["depends_on"]["migrate"]["condition"] == "service_completed_successfully"
    assert app["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert app["depends_on"]["redis"]["condition"] == "service_healthy"
    assert app["healthcheck"]["test"][-1].endswith("/ready")
    assert nginx["depends_on"] == {"app": {"condition": "service_healthy"}}
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "curl -f http://localhost:8000/ready" in dockerfile
    assert "curl -f http://localhost:8000/health" not in dockerfile


@pytest.mark.skipif(
    os.environ.get("T005_RUN_COMPOSE_TESTS") != "1",
    reason="synthetic Compose ordering requires T005_RUN_COMPOSE_TESTS=1",
)
def test_synthetic_compose_failure_prevents_dependent_start(tmp_path):
    docker = shutil.which("docker")
    if docker is None:
        pytest.fail("opt-in Compose test requested but Docker is unavailable")
    context = subprocess.run(
        [docker, "--context", "desktop-linux", "info", "--format", "{{.OSType}}"],
        capture_output=True, text=True, timeout=15, check=False)
    assert context.returncode == 0 and context.stdout.strip() == "linux"
    listing = subprocess.run(
        [docker, "--context", "desktop-linux", "image", "ls", "postgres",
         "--format", "{{json .}}"],
        capture_output=True, text=True, timeout=15, check=False)
    candidates = [json.loads(line) for line in listing.stdout.splitlines() if line]
    assert listing.returncode == 0
    images = [row["Repository"] + ":" + row["Tag"] for row in candidates
              if row.get("Tag", "").startswith("16")]
    if not images:
        pytest.fail("opt-in Compose test needs an existing PostgreSQL 16 image; no pull attempted")
    project = "t005c" + uuid.uuid4().hex
    config = tmp_path / "compose.yml"
    config.write_text(yaml.safe_dump({
        "services": {
            "migrate": {
                "image": images[0], "command": ["sh", "-c", "exit 23"],
                "network_mode": "none", "mem_limit": "48m", "pids_limit": 32,
            },
            "app": {
                "image": images[0], "command": ["sh", "-c", "sleep 30"],
                "network_mode": "none", "mem_limit": "48m", "pids_limit": 32,
                "depends_on": {"migrate": {"condition": "service_completed_successfully"}},
            },
        },
    }), encoding="utf-8")
    base = [
        docker, "--context", "desktop-linux", "compose",
        "--project-name", project, "--file", str(config),
    ]
    try:
        result = subprocess.run(
            [*base, "up", "--no-build", "--pull", "never", "--abort-on-container-exit"],
            capture_output=True, text=True, timeout=30, check=False)
        assert result.returncode != 0
        ps = subprocess.run(
            [*base, "ps", "--all", "--format", "json"],
            capture_output=True, text=True, timeout=15, check=False)
        assert ps.returncode == 0
        output = ps.stdout.strip()
        records = (json.loads(output) if output.startswith("[") else
                   [json.loads(line) for line in output.splitlines() if line])
        migration_records = [row for row in records if row.get("Service") == "migrate"]
        assert len(migration_records) == 1
        assert migration_records[0].get("ExitCode") == 23
        app_records = [row for row in records if row.get("Service") == "app"]
        assert all(row.get("State", "").lower() in ("", "created") for row in app_records)
    finally:
        original_failure = sys.exc_info()[0] is not None
        unresolved = []
        commands = [[*base, "down", "--volumes", "--remove-orphans"]]
        for resource in ("container", "network", "volume"):
            commands.append([
                docker, "--context", "desktop-linux", resource, "ls", "--quiet",
                *( ["--all"] if resource == "container" else [] ),
                "--filter", "label=com.docker.compose.project=" + project,
            ])
        for index, command in enumerate(commands):
            try:
                checked = subprocess.run(command, capture_output=True, text=True,
                                         timeout=30, check=False)
                if checked.returncode or (index > 0 and checked.stdout.strip()):
                    unresolved.append("command " + str(index))
            except (OSError, subprocess.TimeoutExpired):
                unresolved.append("command " + str(index))
        if unresolved:
            message = "T005 Compose cleanup unconfirmed for " + project + ": " + ", ".join(unresolved)
            if original_failure:
                warnings.warn(message)
            else:
                pytest.fail(message)
        else:
            print("T005 Compose cleanup confirmed: " + project)
