"""T004 Docker build-context and copied-image acceptance checks.

The Docker test constructs only harmless synthetic files in a temporary context.
It never sends this repository, its secrets, logs, or runtime data to Docker.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import uuid
import warnings
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCKERIGNORE_PATH = REPOSITORY_ROOT / ".dockerignore"
DOCKERFILE_PATH = REPOSITORY_ROOT / "Dockerfile"
COMPOSE_PATH = REPOSITORY_ROOT / "docker-compose.yml"

PROBE_DOCKERFILE = b"FROM scratch\nWORKDIR /app\nCOPY . .\n"
REQUIRED_SENTINELS = {
    "api/main.py": b"T004_REQUIRED_API_MAIN",
    "migrations/__init__.py": b"T005_REQUIRED_MIGRATIONS_PACKAGE",
    "migrations/runner.py": b"T005_REQUIRED_MIGRATION_RUNNER",
    "migrations/schema_contract.py": b"T005_REQUIRED_SCHEMA_CONTRACT",
    "migrations/0001_admin_users_updated_at.sql": b"T005_REQUIRED_MIGRATION_SQL",
    "migrations/bootstrap_schema.sql": b"T005_REQUIRED_BOOTSTRAP_SQL",
    "services/knowledge/faq_service.py": b"T004_REQUIRED_SERVICE_SOURCE",
    "models/__init__.py": b"T004_REQUIRED_MODEL_SOURCE",
    "config/settings.py": b"T004_REQUIRED_CONFIG_SOURCE",
    "clients/accountants.yaml": b"T004_REQUIRED_ACCOUNTANTS_YAML",
    "clients/iflextax_faq.yaml": b"T004_REQUIRED_FAQ_YAML",
    "templates/dashboard.html": b"T004_REQUIRED_DASHBOARD_TEMPLATE",
    "templates/login.html": b"T004_REQUIRED_LOGIN_TEMPLATE",
    "requirements.txt": b"T004_REQUIRED_REQUIREMENTS",
    "scripts/init_db.sql": b"T004_REQUIRED_INIT_SQL",
    "services/dashboard/db_init.sql": b"T004_REQUIRED_DASHBOARD_SQL",
}
FORBIDDEN_SENTINELS = {
    ".env": b"T004_FORBIDDEN_ROOT_ENV",
    ".env.example": b"T004_FORBIDDEN_ENV_EXAMPLE",
    "nested/app/.env.production": b"T004_FORBIDDEN_NESTED_ENV",
    "tls/server.key": b"T004_FORBIDDEN_PRIVATE_KEY",
    "certificates/client.pem": b"T004_FORBIDDEN_CERTIFICATE",
    "secrets/provider-token.txt": b"T004_FORBIDDEN_SECRET_DIRECTORY",
    "nested/credentials/cloud.json": b"T004_FORBIDDEN_CREDENTIAL_DIRECTORY",
    ".aws/credentials": b"T004_FORBIDDEN_AWS_CREDENTIALS",
    ".ssh/id_rsa": b"T004_FORBIDDEN_SSH_KEY",
    "nginx/ssl/server.crt": b"T004_FORBIDDEN_NGINX_SSL",
    "nginx/www/.well-known/acme-challenge/token": b"T004_FORBIDDEN_ACME_WEBROOT",
    "certbot/conf/account.json": b"T004_FORBIDDEN_CERTBOT_STATE",
    "nested/letsencrypt/account.json": b"T004_FORBIDDEN_LETSENCRYPT_STATE",
    ".git/config": b"T004_FORBIDDEN_GIT_DIRECTORY",
    "nested/worktree/.git": b"T004_FORBIDDEN_GIT_WORKTREE_FILE",
    ".repo-review/checkout/api/main.py": b"T004_FORBIDDEN_REVIEW_CHECKOUT",
    "nested/.worktrees/review/HEAD": b"T004_FORBIDDEN_WORKTREE_DIRECTORY",
    ".agents/notes.md": b"T004_FORBIDDEN_AGENT_METADATA",
    ".codex/config.toml": b"T004_FORBIDDEN_CODEX_METADATA",
    ".claude/settings.json": b"T004_FORBIDDEN_CLAUDE_METADATA",
    ".specify/feature.json": b"T004_FORBIDDEN_SPEC_TOOL_METADATA",
    ".vs/solution.json": b"T004_FORBIDDEN_VS_METADATA",
    ".vscode/settings.json": b"T004_FORBIDDEN_VSCODE_METADATA",
    ".idea/workspace.xml": b"T004_FORBIDDEN_IDEA_METADATA",
    ".venv/pyvenv.cfg": b"T004_FORBIDDEN_DOT_VENV",
    "nested/venv/pyvenv.cfg": b"T004_FORBIDDEN_VENV",
    "nested/env/pyvenv.cfg": b"T004_FORBIDDEN_ENV_VENV",
    "src/__pycache__/module.cpython-312.pyc": b"T004_FORBIDDEN_PYTHON_CACHE",
    ".pytest_cache/v/cache/nodeids": b"T004_FORBIDDEN_PYTEST_CACHE",
    "web/node_modules/package/index.js": b"T004_FORBIDDEN_NODE_MODULES",
    "logs/app.log": b"T004_FORBIDDEN_ROOT_LOG",
    "nginx/logs/access.log": b"T004_FORBIDDEN_NGINX_LOG",
    "nested/logs/worker.log": b"T004_FORBIDDEN_NESTED_LOG",
    "data/reminders.json": b"T004_FORBIDDEN_RUNTIME_DATA",
    "nested/data/customer.json": b"T004_FORBIDDEN_NESTED_DATA",
    "local.db": b"T004_FORBIDDEN_DATABASE",
    "state.sqlite3": b"T004_FORBIDDEN_SQLITE_DATABASE",
    "backups/production.sql": b"T004_FORBIDDEN_BACKUP_SQL",
    "nested/dumps/production.dump": b"T004_FORBIDDEN_DATABASE_DUMP",
    "tests/fixtures/audio/call.wav": b"T004_FORBIDDEN_TEST_FIXTURE",
    "nested/fixtures/provider.json": b"T004_FORBIDDEN_NESTED_FIXTURE",
    "docs/audits/readiness.md": b"T004_FORBIDDEN_AUDIT_DOCUMENT",
    "docs/delegation/task.md": b"T004_FORBIDDEN_DELEGATION_DOCUMENT",
    "specs/feature/spec.md": b"T004_FORBIDDEN_SPEC_DOCUMENT",
    "Deployment Commands to use.txt": b"T004_FORBIDDEN_COMMAND_NOTES",
}


def _run_docker(
    arguments: list[str], *, timeout: int, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["docker", *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"Docker command timed out after {timeout}s: docker {' '.join(arguments)}\n"
            f"stdout: {exc.stdout or ''}\nstderr: {exc.stderr or ''}"
        )


def _require_linux_docker() -> None:
    if shutil.which("docker") is None:
        pytest.fail(
            "T004_RUN_DOCKER_TESTS=1 requested real image verification, but the "
            "docker executable is not on PATH. Install/start nothing automatically; "
            "rerun on an approved local Linux Docker engine."
        )

    context_name = os.environ.get("DOCKER_CONTEXT") or None
    endpoint: str | None = None
    if context_name is None:
        endpoint = os.environ.get("DOCKER_HOST") or None
    if endpoint is None:
        if context_name is None:
            selected_context = _run_docker(["context", "show"], timeout=15)
            if selected_context.returncode != 0:
                pytest.fail(
                    "Could not identify Docker's selected context.\n"
                    f"stdout: {selected_context.stdout}\nstderr: {selected_context.stderr}"
                )
            context_name = selected_context.stdout.strip()

        inspected_context = _run_docker(
            ["context", "inspect", context_name],
            timeout=15,
        )
        if inspected_context.returncode != 0:
            pytest.fail(
                f"Could not inspect Docker context {context_name!r}.\n"
                f"stdout: {inspected_context.stdout}\n"
                f"stderr: {inspected_context.stderr}"
            )
        try:
            context_details = json.loads(inspected_context.stdout)
            endpoint = context_details[0]["Endpoints"]["docker"]["Host"]
            if not isinstance(endpoint, str):
                raise TypeError("Docker endpoint is not a string")
        except (json.JSONDecodeError, IndexError, KeyError, TypeError):
            pytest.fail(
                f"Docker context {context_name!r} returned an unreadable endpoint: "
                f"{inspected_context.stdout!r}"
            )

    assert endpoint is not None
    parsed_endpoint = urlparse(endpoint)
    local_endpoint = parsed_endpoint.scheme in {"unix", "npipe"} or (
        parsed_endpoint.scheme == "tcp"
        and parsed_endpoint.hostname in {"127.0.0.1", "::1", "localhost"}
    )
    if not local_endpoint:
        pytest.fail(
            "T004 image verification may use only an existing local Docker engine; "
            f"the effective endpoint is {endpoint!r}. The test did not change it."
        )

    result = _run_docker(["info", "--format", "{{.OSType}}"], timeout=15)
    if result.returncode != 0:
        pytest.fail(
            "T004_RUN_DOCKER_TESTS=1 requested real image verification, but Docker "
            f"is unreachable (exit {result.returncode}).\nstdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    if result.stdout.strip().lower() != "linux":
        pytest.fail(
            "T004 image verification requires a Linux Docker engine; "
            f"the selected engine reported {result.stdout.strip()!r}."
        )


def _write_synthetic_context(context: Path, dockerignore: bytes) -> None:
    files = {**REQUIRED_SENTINELS, **FORBIDDEN_SENTINELS}
    for relative_path, sentinel in files.items():
        destination = context / Path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(sentinel)

    (context / "Dockerfile").write_bytes(PROBE_DOCKERFILE)
    (context / ".dockerignore").write_bytes(dockerignore)


def _assert_command_succeeded(
    result: subprocess.CompletedProcess[str], description: str
) -> None:
    assert result.returncode == 0, (
        f"{description} failed with exit {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def _export_image_files(
    context: Path,
    image_name: str,
    container_name: str,
    image_candidates: list[str],
    container_candidates: list[str],
) -> dict[str, bytes]:
    image_candidates.append(image_name)
    build = _run_docker(
        ["build", "--no-cache", "--tag", image_name, "--file", "Dockerfile", "."],
        timeout=120,
        cwd=context,
    )
    _assert_command_succeeded(build, f"building probe image {image_name}")

    container_candidates.append(container_name)
    create = _run_docker(
        ["create", "--name", container_name, image_name, "/t004-inspection-only"],
        timeout=30,
    )
    _assert_command_succeeded(create, f"creating inspection container {container_name}")

    archive_path = context / "t004-rootfs.tar"
    export = _run_docker(
        ["export", "--output", str(archive_path), container_name], timeout=60
    )
    _assert_command_succeeded(export, f"exporting inspection container {container_name}")

    files: dict[str, bytes] = {}
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            assert extracted is not None
            files[member.name.lstrip("./")] = extracted.read()
    return files


def _cleanup_docker_resource(kind: str, name: str) -> str | None:
    if kind == "container":
        arguments = ["container", "rm", "--force", name]
    else:
        arguments = ["image", "rm", "--force", name]
    command = ["docker", *arguments]
    command_text = " ".join(command)
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return (
            f"cleanup outcome unknown after 30s timeout: {command_text}. "
            "Verify this exact T004 resource when the local daemon is reachable."
        )
    except OSError as exc:
        return (
            f"cleanup outcome unknown after operating-system error: {command_text}: "
            f"{exc}. Verify this exact T004 resource when the local daemon is reachable."
        )
    if result.returncode == 0:
        return None

    output = f"{result.stdout}\n{result.stderr}".casefold()
    if f"no such {kind}" in output:
        return None
    return (
        f"cleanup failed for exact T004 resource (exit {result.returncode}): "
        f"{command_text}: {result.stderr.strip() or result.stdout.strip()}"
    )


def _finish_docker_cleanup(
    container_candidates: list[str],
    image_candidates: list[str],
    *,
    test_already_failed: bool,
) -> None:
    cleanup_errors = []
    for container_name in reversed(container_candidates):
        error = _cleanup_docker_resource("container", container_name)
        if error:
            cleanup_errors.append(error)
    for image_name in reversed(image_candidates):
        error = _cleanup_docker_resource("image", image_name)
        if error:
            cleanup_errors.append(error)

    if cleanup_errors:
        message = "\n".join(cleanup_errors)
        if test_already_failed:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        else:
            pytest.fail(message)


def _assert_expected_app_build(config_path: Path) -> None:
    try:
        compose = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AssertionError(
            f"services.app.build cannot be validated in {config_path}: {exc}"
        ) from exc

    assert isinstance(compose, dict), "services.app.build requires a Compose mapping"
    services = compose.get("services")
    assert isinstance(services, dict), "services.app.build requires services mapping"
    app = services.get("app")
    assert isinstance(app, dict), "services.app.build requires app service mapping"
    build = app.get("build")
    assert isinstance(build, dict), "services.app.build must be a mapping"
    assert build.get("context") == ".", (
        "services.app.build.context must be exactly '.'; "
        f"found {build.get('context')!r}"
    )
    assert build.get("dockerfile") == "Dockerfile", (
        "services.app.build.dockerfile must be exactly 'Dockerfile'; "
        f"found {build.get('dockerfile')!r}"
    )


def test_project_copy_boundary_and_ignore_precedence_have_not_drifted() -> None:
    """Offline drift guard; this does not inspect a Docker image."""

    dockerfile_lines = [
        line.strip()
        for line in DOCKERFILE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    final_stage = dockerfile_lines[
        max(index for index, line in enumerate(dockerfile_lines) if line.upper().startswith("FROM ")) :
    ]

    assert "WORKDIR /app" in final_stage
    assert "COPY . ." in final_stage
    assert not (REPOSITORY_ROOT / "Dockerfile.dockerignore").exists(), (
        "Dockerfile.dockerignore would override the reviewed root .dockerignore"
    )

    _assert_expected_app_build(COMPOSE_PATH)


@pytest.mark.parametrize(
    "app_build",
    [
        pytest.param(
            {"context": "./different-context", "dockerfile": "Dockerfile"},
            id="changed-context",
        ),
        pytest.param(
            {"context": ".", "dockerfile": "Dockerfile.unreviewed"},
            id="changed-dockerfile",
        ),
        pytest.param(".", id="build-is-not-mapping"),
    ],
)
def test_compose_build_guard_rejects_wrong_app_mapping_despite_decoys(
    tmp_path: Path, app_build: object
) -> None:
    config_path = tmp_path / "compose.yml"
    config = {
        "services": {
            "decoy": {
                "build": {"context": ".", "dockerfile": "Dockerfile"},
            },
            "app": {"build": app_build},
        }
    }
    config_path.write_text(
        "# context: .\n# dockerfile: Dockerfile\n" + yaml.safe_dump(config),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match=r"services\.app\.build"):
        _assert_expected_app_build(config_path)


def test_uncertain_build_result_still_cleans_candidate_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suffix = "a" * 32
    image_name = f"t004-build-exclusions-safe:{suffix}"
    container_name = f"t004-build-exclusions-safe-{suffix}"
    resources = {"images": set()}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[1] == "build":
            resources["images"].add(image_name)
            raise subprocess.TimeoutExpired(command, timeout=120, stderr="lost reply")
        if command[1:4] == ["image", "rm", "--force"]:
            resources["images"].remove(command[4])
            return subprocess.CompletedProcess(command, 0, "removed", "")
        raise AssertionError(f"unexpected Docker command: {command}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    image_candidates: list[str] = []
    container_candidates: list[str] = []

    with pytest.raises(pytest.fail.Exception, match="Docker command timed out"):
        try:
            _export_image_files(
                tmp_path,
                image_name,
                container_name,
                image_candidates,
                container_candidates,
            )
        finally:
            _finish_docker_cleanup(
                container_candidates,
                image_candidates,
                test_already_failed=sys.exc_info()[0] is not None,
            )

    assert resources["images"] == set()
    assert container_candidates == []


def test_uncertain_create_preserves_failure_and_reports_unresolved_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suffix = "b" * 32
    image_name = f"t004-build-exclusions-safe:{suffix}"
    container_name = f"t004-build-exclusions-safe-{suffix}"
    resources = {"images": set(), "containers": set()}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[1] == "build":
            resources["images"].add(image_name)
            return subprocess.CompletedProcess(command, 0, "built", "")
        if command[1] == "create":
            resources["containers"].add(container_name)
            raise subprocess.TimeoutExpired(command, timeout=30, stderr="lost reply")
        if command[1:4] == ["container", "rm", "--force"]:
            raise OSError("daemon disconnected during cleanup")
        if command[1:4] == ["image", "rm", "--force"]:
            resources["images"].remove(command[4])
            return subprocess.CompletedProcess(command, 0, "removed", "")
        raise AssertionError(f"unexpected Docker command: {command}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    image_candidates: list[str] = []
    container_candidates: list[str] = []

    with pytest.warns(RuntimeWarning, match=container_name) as cleanup_warnings:
        with pytest.raises(pytest.fail.Exception, match="Docker command timed out"):
            try:
                _export_image_files(
                    tmp_path,
                    image_name,
                    container_name,
                    image_candidates,
                    container_candidates,
                )
            finally:
                _finish_docker_cleanup(
                    container_candidates,
                    image_candidates,
                    test_already_failed=sys.exc_info()[0] is not None,
                )

    assert resources["images"] == set()
    assert resources["containers"] == {container_name}
    assert "daemon disconnected during cleanup" in str(cleanup_warnings[0].message)


@pytest.mark.parametrize("kind", ["container", "image"])
def test_cleanup_treats_verified_resource_absence_as_benign(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "t004-build-exclusions-safe-" + "c" * 32

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            1,
            "",
            f"Error response from daemon: No such {kind}: {name}",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _cleanup_docker_resource(kind, name) is None


def test_cleanup_timeout_reports_exact_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_name = "t004-build-exclusions-safe:" + "d" * 32

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, timeout=30)

    monkeypatch.setattr(subprocess, "run", fake_run)

    error = _cleanup_docker_resource("image", image_name)
    assert error is not None
    assert "cleanup outcome unknown" in error
    assert f"docker image rm --force {image_name}" in error


@pytest.mark.skipif(
    os.environ.get("T004_RUN_DOCKER_TESTS") != "1",
    reason="real Docker image check requires T004_RUN_DOCKER_TESTS=1",
)
def test_docker_filters_synthetic_context_and_negative_control_detects_leak(
    tmp_path: Path,
) -> None:
    _require_linux_docker()

    suffix = uuid.uuid4().hex
    safe_image = f"t004-build-exclusions-safe:{suffix}"
    unsafe_image = f"t004-build-exclusions-control:{suffix}"
    safe_container = f"t004-build-exclusions-safe-{suffix}"
    unsafe_container = f"t004-build-exclusions-control-{suffix}"
    image_candidates: list[str] = []
    container_candidates: list[str] = []

    try:
        safe_context = tmp_path / "safe-context"
        safe_context.mkdir()
        _write_synthetic_context(safe_context, DOCKERIGNORE_PATH.read_bytes())
        safe_files = _export_image_files(
            safe_context,
            safe_image,
            safe_container,
            image_candidates,
            container_candidates,
        )

        for relative_path, sentinel in REQUIRED_SENTINELS.items():
            image_path = f"app/{relative_path}"
            assert safe_files.get(image_path) == sentinel, (
                f"required runtime asset missing or changed: /{image_path}"
            )

        safe_payloads = tuple(safe_files.values())
        for relative_path, sentinel in FORBIDDEN_SENTINELS.items():
            image_path = f"app/{relative_path}"
            assert image_path not in safe_files, f"forbidden path leaked: /{image_path}"
            assert all(sentinel not in payload for payload in safe_payloads), (
                f"forbidden sentinel bytes leaked from {relative_path}"
            )

        unsafe_context = tmp_path / "unsafe-control-context"
        unsafe_context.mkdir()
        _write_synthetic_context(unsafe_context, b"")
        unsafe_files = _export_image_files(
            unsafe_context,
            unsafe_image,
            unsafe_container,
            image_candidates,
            container_candidates,
        )

        leaked_path = "app/.env"
        assert unsafe_files.get(leaked_path) == FORBIDDEN_SENTINELS[".env"], (
            "negative control did not expose the synthetic .env sentinel; the "
            "inspection would not detect the original COPY . . defect"
        )
    finally:
        _finish_docker_cleanup(
            container_candidates,
            image_candidates,
            test_already_failed=sys.exc_info()[0] is not None,
        )
