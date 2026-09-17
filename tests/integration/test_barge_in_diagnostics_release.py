"""Behavioral tests for the protected T033-F diagnostics rollout."""

from __future__ import annotations

import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import uuid


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "scripts" / "deploy-barge-in-diagnostics.py"
SPEC = importlib.util.spec_from_file_location("t033f_release", SCRIPT)
release = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(release)

COMMIT = "a" * 40
SECRET = "synthetic-$-first\nsecond"


def completed(stdout: bytes = b"", returncode: int = 0):
    return SimpleNamespace(stdout=stdout, stderr=b"", returncode=returncode)


def new_release(path: Path) -> release.Release:
    item = release.Release(path, COMMIT)
    item.root = path / "root"
    item.root.mkdir(parents=True, exist_ok=True)
    return item


def compose_fixture(image: str, settings_source: str = "/old/settings.py") -> dict:
    return {
        "name": release.PROJECT,
        "services": {
            "app": {
                "image": image,
                "command": ["python", "main.py"],
                "environment": {"PRIVATE_VALUE": SECRET, "UNKNOWN": "retained"},
                "networks": {"internal": None},
                "ports": [{"target": 8000, "published": "8000"}],
                "restart": "unless-stopped",
                "volumes": [
                    {
                        "type": "bind", "source": "/private/runtime.env",
                        "target": "/app/.env", "read_only": True,
                        "bind": {"create_host_path": True},
                    },
                    {
                        "type": "bind", "source": "/private/config",
                        "target": "/app/config", "read_only": True,
                    },
                    {
                        "type": "bind", "source": settings_source,
                        "target": release.SETTINGS_TARGET, "read_only": True,
                    },
                ],
                "x-unknown": {"preserve": True},
            },
            "db": {"image": "postgres:16", "x-unknown": "retained"},
        },
        "networks": {"internal": {"name": "existing-internal"}},
        "x-top-level": {"preserve": True},
    }


def container_fixture(image, revision, *, candidate, settings_source):
    environment = ["PRIVATE_VALUE=" + SECRET, "UNKNOWN=retained"]
    if candidate:
        environment.extend(f"{key}={value}" for key, value in release.DIAGNOSTIC_ENV.items())
    return {
        "Image": image,
        "State": {"Running": True, "Health": {"Status": "healthy"}},
        "Config": {
            "Env": environment, "Cmd": ["python", "main.py"], "Entrypoint": None,
            "Labels": {"org.opencontainers.image.revision": revision},
        },
        "HostConfig": {
            "PortBindings": {"8000/tcp": [{"HostPort": "8000"}]},
            "RestartPolicy": {"Name": "unless-stopped"},
        },
        "Mounts": [
            {"Type": "bind", "Source": "/private/runtime.env", "Destination": "/app/.env", "RW": False},
            {"Type": "bind", "Source": "/private/config", "Destination": "/app/config", "RW": False},
            {"Type": "bind", "Source": settings_source, "Destination": release.SETTINGS_TARGET, "RW": False},
        ],
        "NetworkSettings": {"Networks": {"existing-internal": {}}},
    }


class OfflineReleaseTests(unittest.TestCase):
    def test_deployed_settings_without_new_fields_are_valid_but_enabled_is_not(self):
        previous = dict(release.REQUIRED_SETTINGS)
        self.item._validate_settings(previous, candidate=False)
        for key in release.DIAGNOSTIC_SETTINGS:
            with self.subTest(key=key):
                with self.assertRaises(RuntimeError):
                    self.item._validate_settings({**previous, key: True}, candidate=False)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.item = new_release(self.path)

    def test_import_is_side_effect_free(self):
        spec = importlib.util.spec_from_file_location("t033f_import_" + uuid.uuid4().hex, SCRIPT)
        module = importlib.util.module_from_spec(spec)
        with mock.patch("subprocess.run") as run, mock.patch("tempfile.mkdtemp") as mkdir:
            assert spec.loader is not None
            spec.loader.exec_module(module)
        run.assert_not_called()
        mkdir.assert_not_called()

    def test_manifest_and_all_36_entries_match_normalized_worktree(self):
        manifest_path = REPOSITORY / release.MANIFEST_PATH
        self.assertEqual(release.sha256(release.normalized_bytes(manifest_path.read_bytes())), release.MANIFEST_SHA256)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed = {}
        for section in ("runtime", "protected_runtime", "tests", "test_support"):
            for relative, expected in manifest[section].items():
                actual = release.sha256(release.normalized_bytes((REPOSITORY / relative).read_bytes()))
                self.assertEqual(actual, expected, relative)
                observed[relative] = actual
        self.assertEqual(len(observed), 36)

    def _source_runner(self, changed=None, missing=None):
        calls = []

        def run(*arguments, **_kwargs):
            calls.append(arguments)
            if arguments[:3] == ("git", "rev-parse", "HEAD"):
                return completed((release.BASE_CHECKOUT + "\n").encode())
            if arguments[:2] == ("git", "rev-parse"):
                return completed((arguments[2].removesuffix("^{commit}") + "\n").encode())
            if arguments[:2] == ("git", "merge-base"):
                return completed()
            if arguments[:2] == ("git", "status"):
                return completed(b"M nginx/nginx.conf\n")
            if arguments[:3] == ("git", "diff", "--name-only"):
                payload = changed if changed is not None else ("\n".join(sorted(release.SOURCE_SCOPE)) + "\n").encode()
                return completed(payload)
            if arguments[:2] == ("git", "cat-file"):
                path = arguments[-1].split(":", 1)[1]
                if path == missing:
                    raise RuntimeError("command failed; protected log retained")
                return completed()
            self.fail("unexpected command: " + repr(arguments))

        return run, calls

    def test_verify_source_accepts_exact_scope_and_all_six_release_files(self):
        nginx = self.item.root / "nginx" / "nginx.conf"
        nginx.parent.mkdir()
        nginx.write_bytes(b"reviewed nginx")
        self.item.run, calls = self._source_runner()
        with mock.patch.object(release, "NGINX_HASH", release.sha256(b"reviewed nginx")):
            self.item.verify_source()
        checked = {call[-1].split(":", 1)[1] for call in calls if call[:2] == ("git", "cat-file")}
        self.assertEqual(checked, {release.MANIFEST_PATH, *release.RELEASE_PATHS})

    def test_wrong_baseline_extra_scope_and_missing_release_file_stop_verification(self):
        nginx = self.item.root / "nginx" / "nginx.conf"
        nginx.parent.mkdir()
        nginx.write_bytes(b"reviewed nginx")
        self.item.run, _ = self._source_runner()
        original = self.item.run

        def wrong_head(*args, **kwargs):
            if args[:3] == ("git", "rev-parse", "HEAD"):
                return completed(b"0" * 40 + b"\n")
            return original(*args, **kwargs)

        self.item.run = wrong_head
        with self.assertRaisesRegex(RuntimeError, "checkout changed"):
            self.item.verify_source()

        extra = "\n".join(sorted(release.SOURCE_SCOPE | {"services/private.py"})) + "\n"
        self.item.run, _ = self._source_runner(extra.encode())
        with mock.patch.object(release, "NGINX_HASH", release.sha256(b"reviewed nginx")):
            with self.assertRaisesRegex(RuntimeError, "exceeds reviewed diagnostics scope"):
                self.item.verify_source()

        self.item.run, _ = self._source_runner(missing=release.RELEASE_PATHS[-1])
        with mock.patch.object(release, "NGINX_HASH", release.sha256(b"reviewed nginx")):
            with self.assertRaisesRegex(RuntimeError, "command failed"):
                self.item.verify_source()

    def test_staging_reads_and_verifies_every_manifest_asset(self):
        seen = []

        def git_bytes(relative):
            seen.append(relative)
            return (REPOSITORY / relative).read_bytes()

        self.item.git_bytes = git_bytes
        self.item.stage_assets()
        expected = {release.MANIFEST_PATH, *release.RUNTIME_PATHS, *release.PROTECTED_RUNTIME_PATHS, *release.TEST_PATHS, *release.TEST_SUPPORT_PATHS}
        self.assertEqual(set(seen), expected)
        self.assertEqual(len(expected) - 1, 36)

    def test_staging_rejects_hash_mismatch_and_missing_test_support(self):
        def changed(relative):
            if relative == release.PROTECTED_RUNTIME_PATHS[0]:
                return b"changed protected runtime"
            return (REPOSITORY / relative).read_bytes()

        self.item.git_bytes = changed
        with self.assertRaisesRegex(RuntimeError, "reviewed asset hash differs"):
            self.item.stage_assets()

        other = new_release(self.path / "missing")

        def missing(relative):
            if relative == release.TEST_SUPPORT_PATHS[-1]:
                raise RuntimeError("command failed; protected log retained")
            return (REPOSITORY / relative).read_bytes()

        other.git_bytes = missing
        with self.assertRaisesRegex(RuntimeError, "command failed"):
            other.stage_assets()

    def test_staging_accepts_git_normalized_manifest_without_changing_entries(self):
        self.item.git_bytes = lambda path: release.normalized_bytes(
            (REPOSITORY / path).read_bytes()
        )
        self.item.stage_assets()
        self.assertEqual(len(self.item.runtime_hashes), 4)

    def test_hidden_runtime_mounts_allow_only_exact_settings_exception(self):
        harmless = [
            {"Destination": "/app/config", "Source": "/private/config"},
            {"Destination": release.SETTINGS_TARGET, "Source": "/private/settings.py"},
        ]
        self.assertEqual(release.hidden_runtime_paths(harmless), ())
        hidden = release.hidden_runtime_paths([{"target": "/app/services/conversation", "source": "/old"}])
        self.assertIn("services/conversation/orchestrator.py", hidden)
        self.assertIn("services/conversation/turn_controller.py", hidden)
        self.assertNotIn("services/stt/stt_base.py", hidden)

    def test_compose_comparison_ignores_order_and_declared_exceptions_only(self):
        previous = compose_fixture(release.OLD_IMAGE)
        candidate = copy.deepcopy(previous)
        app = candidate["services"]["app"]
        app["image"] = "sha256:candidate"
        app["environment"].update(release.DIAGNOSTIC_ENV)
        app["volumes"][-1]["source"] = str(self.item.settings_overlay)
        app["volumes"].reverse()
        self.assertEqual(
            release.normalized_compose(previous, candidate=False),
            release.normalized_compose(candidate, candidate=True, settings_source=self.item.settings_overlay),
        )
        changed = copy.deepcopy(candidate)
        changed["services"]["app"]["environment"]["PRIVATE_VALUE"] = "changed"
        self.assertNotEqual(
            release.normalized_compose(previous, candidate=False),
            release.normalized_compose(changed, candidate=True, settings_source=self.item.settings_overlay),
        )
        duplicate = copy.deepcopy(candidate)
        non_settings = next(
            value
            for value in duplicate["services"]["app"]["volumes"]
            if value["target"] != release.SETTINGS_TARGET
        )
        duplicate["services"]["app"]["volumes"].append(copy.deepcopy(non_settings))
        self.assertNotEqual(
            release.normalized_compose(previous, candidate=False),
            release.normalized_compose(duplicate, candidate=True, settings_source=self.item.settings_overlay),
        )

    def test_container_contract_requires_exact_environment_and_overlay(self):
        old = container_fixture(release.OLD_IMAGE, release.DEPLOYED_SOURCE, candidate=False, settings_source="/old/settings.py")
        candidate = container_fixture("sha256:candidate", COMMIT, candidate=True, settings_source=str(self.item.settings_overlay))
        self.assertEqual(
            release.container_contract(old, candidate=False),
            release.container_contract(candidate, candidate=True, settings_source=self.item.settings_overlay),
        )
        candidate["Config"]["Env"].append("BARGE_IN_DIAGNOSTICS_ENABLED=true")
        with self.assertRaisesRegex(RuntimeError, "diagnostic environment differs"):
            release.container_contract(candidate, candidate=True, settings_source=self.item.settings_overlay)

    def test_settings_validation_preserves_every_unrelated_value(self):
        previous = {
            "private": SECRET,
            **release.REQUIRED_SETTINGS,
            "barge_in_diagnostics_enabled": False,
            "elevenlabs_stt_filter_background_audio": False,
        }
        self.item.previous_settings = previous
        self.item._validate_settings(previous, candidate=False)
        self.item._validate_settings(release.settings_for(previous, candidate=True), candidate=True)
        drift = release.settings_for(previous, candidate=True)
        drift["private"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "effective settings differ"):
            self.item._validate_settings(drift, candidate=True)
        filtering = dict(previous, elevenlabs_stt_filter_background_audio=True)
        with self.assertRaisesRegex(RuntimeError, "background filter"):
            self.item._validate_settings(filtering, candidate=False)

    def test_prepare_override_preserves_parent_mount_values_and_installs_nested_file(self):
        original = compose_fixture(release.OLD_IMAGE)
        self.item.previous_rendered = copy.deepcopy(original)
        self.item.previous_settings = {
            "private": SECRET,
            **release.REQUIRED_SETTINGS,
            "barge_in_diagnostics_enabled": False,
            "elevenlabs_stt_filter_background_audio": False,
        }
        settings = release.normalized_bytes((REPOSITORY / "config/settings.py").read_bytes())
        target = self.item.assets / "config/settings.py"
        target.parent.mkdir(parents=True)
        target.write_bytes(settings)
        self.item.runtime_hashes = {"config/settings.py": release.sha256(settings)}
        self.item.protected_runtime_hashes = {"protected.py": "abc"}

        def render(path):
            value = json.loads(path.read_text(encoding="utf-8"))
            value["services"]["app"]["volumes"].reverse()
            return value

        self.item.render_compose = render
        self.item.candidate_probe = lambda *_args: {
            "hashes": {**self.item.runtime_hashes, **self.item.protected_runtime_hashes},
            "settings": release.settings_for(self.item.previous_settings, candidate=True),
        }
        path = self.item.prepare_override("sha256:candidate", original)
        app = json.loads(path.read_text(encoding="utf-8"))["services"]["app"]
        self.assertEqual(app["environment"]["PRIVATE_VALUE"], SECRET)
        self.assertEqual({key: app["environment"][key] for key in release.DIAGNOSTIC_ENV}, release.DIAGNOSTIC_ENV)
        self.assertEqual(app["x-unknown"], {"preserve": True})
        self.assertEqual(len([v for v in app["volumes"] if v["target"] == "/app/config"]), 1)
        self.assertEqual([v for v in app["volumes"] if v["target"] == release.SETTINGS_TARGET], [{
            "type": "bind", "source": str(self.item.settings_overlay),
            "target": release.SETTINGS_TARGET, "read_only": True,
        }])
        self.assertEqual(self.item.settings_overlay.read_bytes(), settings)
        if os.name != "nt":
            self.assertEqual(self.item.settings_overlay.stat().st_mode & 0o777, 0o644)

    def test_candidate_test_mounts_only_reviewed_support_and_all_four_suites(self):
        observed = {}

        def one_shot(name, image, arguments, **kwargs):
            observed.update(name=name, image=image, arguments=arguments, kwargs=kwargs)
            return completed()

        self.item.one_shot = one_shot
        self.item.run_candidate_tests("sha256:candidate")
        self.assertEqual(observed["kwargs"].get("network", "none"), "none")
        self.assertEqual(len(observed["kwargs"]["mounts"]), 3)
        for suite in ("stt", "tts", "telephony", "conversation"):
            self.assertIn("/audit/tests/" + suite, observed["arguments"])
        self.assertTrue(any("/audit/scripts" in value for value in observed["kwargs"]["mounts"]))
        self.assertTrue(any("/audit/docker-compose.yml" in value for value in observed["kwargs"]["mounts"]))

    def test_candidate_failure_stops_before_override_or_cutover(self):
        self.item.preflight = mock.Mock(return_value=compose_fixture(release.OLD_IMAGE))
        self.item.stage_assets = mock.Mock()
        self.item.build_candidate = mock.Mock(return_value="sha256:candidate")
        self.item.run_candidate_tests = mock.Mock(side_effect=RuntimeError("test failure"))
        self.item.prepare_override = mock.Mock()
        self.item.deploy = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            self.item.execute()
        self.item.prepare_override.assert_not_called()
        self.item.deploy.assert_not_called()
        self.assertEqual(self.item.stage, "tests")

    def test_schema_probe_is_read_only_and_requires_exact_0004_history(self):
        self.item.network = "internal"
        self.item.live_schema_env = self.path / "schema.env"
        observed = {}

        def one_shot(name, image, arguments, **kwargs):
            observed.update(name=name, image=image, arguments=arguments, kwargs=kwargs)
            return completed(json.dumps({"read_only": True, "history": release.EXPECTED_HISTORY}).encode())

        self.item.one_shot = one_shot
        self.assertTrue(self.item.schema_matches())
        self.assertIn("transaction(readonly=True)", release.SCHEMA_0004_CODE)
        self.assertEqual(observed["kwargs"]["network"], "internal")
        self.assertFalse(observed["kwargs"]["check"])
        self.item.one_shot = lambda *_a, **_k: completed(json.dumps({"read_only": True, "history": release.EXPECTED_HISTORY[:-1]}).encode())
        self.assertFalse(self.item.schema_matches())

    def test_timeout_and_partial_creation_remove_registered_container(self):
        calls = []

        def run(*arguments, **_kwargs):
            calls.append(arguments)
            if arguments[:2] == ("docker", "run"):
                raise RuntimeError("command timed out; protected log retained")
            if arguments[:3] in (("docker", "rm", "-f"), ("docker", "ps", "-a")):
                return completed()
            self.fail("unexpected command")

        self.item.run = run
        name = "t033f-partial-" + self.path.name
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.item.one_shot(name, "sha256:image", ["-c", "pass"])
        self.assertIn(("docker", "rm", "-f", name), calls)
        self.assertIn(("docker", "ps", "-a", "--format", "{{.Names}}"), calls)

    def test_cleanup_failure_does_not_replace_primary_or_log_detail(self):
        self.item.run = mock.Mock(side_effect=RuntimeError("primary failure"))
        self.item.remove_container = mock.Mock(side_effect=RuntimeError("private sentinel"))
        with self.assertRaisesRegex(RuntimeError, "primary failure"):
            self.item.one_shot("registered", "sha256:image", ["-c", "pass"])
        private = (self.item.directory / "private.log").read_text()
        self.assertIn("cleanup_failure_type=RuntimeError", private)
        self.assertNotIn("private sentinel", private)

    def test_stop_uses_verified_final_state_after_already_stopped_errors(self):
        def run(*arguments, **_kwargs):
            if arguments[:2] == ("docker", "stop"):
                return completed(returncode=1)
            self.fail("unexpected command")

        self.item.run = run
        self.item.inspect = lambda _name: {"State": {"Running": False}}
        self.item.stop_components(strict=True)

    def test_effective_replacement_verifier_accepts_only_exact_contract(self):
        old = container_fixture(release.OLD_IMAGE, release.DEPLOYED_SOURCE, candidate=False, settings_source="/old/settings.py")
        current = container_fixture("sha256:candidate", COMMIT, candidate=True, settings_source=str(self.item.settings_overlay))
        self.item.previous_contract = release.container_contract(old, candidate=False)
        self.item.previous_settings = {
            "private": SECRET,
            **release.REQUIRED_SETTINGS,
            "barge_in_diagnostics_enabled": False,
            "elevenlabs_stt_filter_background_audio": False,
        }
        self.item.previous_rendered = compose_fixture(release.OLD_IMAGE)
        self.item.runtime_hashes = {"runtime.py": "aaa"}
        self.item.protected_runtime_hashes = {"protected.py": "bbb"}
        self.item.wait_ready = mock.Mock()
        self.item.inspect = lambda _name: current
        self.item.settings = lambda: release.settings_for(self.item.previous_settings, candidate=True)
        self.item.running_probe = lambda: {"hashes": {**self.item.runtime_hashes, **self.item.protected_runtime_hashes}}
        rendered = compose_fixture("sha256:candidate", str(self.item.settings_overlay))
        rendered["services"]["app"]["environment"].update(release.DIAGNOSTIC_ENV)
        rendered["services"]["app"]["volumes"].reverse()
        self.item.render_compose = lambda _path: rendered
        self.item.schema_matches = lambda image: image == "sha256:candidate"
        self.item.verify_replaced("sha256:candidate", COMMIT, candidate=True)
        current["Config"]["Env"].append("DRIFT=true")
        with self.assertRaisesRegex(RuntimeError, "runtime contract differs"):
            self.item.verify_replaced("sha256:candidate", COMMIT, candidate=True)

    def _cutover(self, *, failure=None, recovery_failure=False):
        previous = b'{"services":{"app":{"image":"old"}}}\n'
        candidate_bytes = b'{"services":{"app":{"image":"candidate"}}}\n'
        previous_path = self.item.directory / "previous-override.json"
        candidate_path = self.item.directory / "candidate-override.json"
        previous_path.write_bytes(previous)
        candidate_path.write_bytes(candidate_bytes)
        state = {"image": release.OLD_IMAGE, "override": previous, "ingress": True, "events": []}
        self.item.recheck_before_stop = lambda: state["events"].append("recheck")

        def stop_components(*, strict):
            state["events"].append(("stop", strict, state["image"]))
            state["ingress"] = False

        def install(path):
            state["override"] = path.read_bytes()
            state["events"].append(("install", path.name))

        def replace(image, revision, *, candidate):
            state["events"].append(("replace", image, revision, candidate))
            if (failure == "readiness" and candidate) or (recovery_failure and not candidate):
                raise RuntimeError("readiness failure")
            state["image"] = image

        def ingress():
            state["events"].append(("ingress", state["image"]))
            state["ingress"] = True
            if failure == "tls" and state["image"] != release.OLD_IMAGE:
                raise RuntimeError("TLS failure")

        self.item.stop_components = stop_components
        self.item.install_override = install
        self.item.replace_app = replace
        self.item.start_ingress = ingress

        def force_close():
            state["events"].append("force-close")
            state["ingress"] = False
            return True

        self.item.force_close_ingress = force_close
        return state, candidate_path, io.StringIO()

    def test_successful_cutover_records_candidate_and_success_marker(self):
        state, candidate, output = self._cutover()
        with redirect_stdout(output):
            self.item.deploy("sha256:candidate", candidate)
        self.assertEqual(state["image"], "sha256:candidate")
        self.assertEqual(state["override"], candidate.read_bytes())
        self.assertTrue(state["ingress"])
        self.assertEqual((self.item.directory / "deployed").read_text().splitlines()[:2], [COMMIT, "sha256:candidate"])
        self.assertIn("BARGE_IN_DIAGNOSTICS_DEPLOYED_READY_HTTPS_OK", output.getvalue())

    def test_readiness_failure_restores_exact_previous_app_and_settings_state(self):
        state, candidate, output = self._cutover(failure="readiness")
        with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "readiness failure"):
            self.item.deploy("sha256:candidate", candidate)
        self.assertEqual(state["image"], release.OLD_IMAGE)
        self.assertEqual(state["override"], (self.item.directory / "previous-override.json").read_bytes())
        self.assertTrue(state["ingress"])
        self.assertIn("PREVIOUS_APP_RESTORED_DIAGNOSTICS_OFF_SCHEMA_0004_UNCHANGED", output.getvalue())
        restores = [event for event in state["events"] if isinstance(event, tuple) and event[0] == "replace"]
        self.assertFalse(restores[-1][-1])

    def test_tls_failure_stops_candidate_before_recovery(self):
        state, candidate, output = self._cutover(failure="tls")
        with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "TLS failure"):
            self.item.deploy("sha256:candidate", candidate)
        self.assertEqual(state["image"], release.OLD_IMAGE)
        self.assertTrue(state["ingress"])
        stops = [event for event in state["events"] if isinstance(event, tuple) and event[0] == "stop"]
        self.assertEqual(stops[-1][2], "sha256:candidate")

    def test_failed_recovery_keeps_ingress_closed_and_never_reports_success(self):
        state, candidate, output = self._cutover(failure="readiness", recovery_failure=True)
        with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "readiness failure"):
            self.item.deploy("sha256:candidate", candidate)
        self.assertFalse(state["ingress"])
        text = output.getvalue()
        self.assertIn("RECOVERY_FAILED_INGRESS_CLOSED", text)
        self.assertIn("RECOVERY_UNVERIFIED_MANUAL_REVIEW_REQUIRED", text)
        self.assertNotIn("BARGE_IN_DIAGNOSTICS_DEPLOYED_READY_HTTPS_OK", text)


@unittest.skipUnless(
    os.environ.get("T033_F_RUN_DOCKER_TESTS") == "1",
    "set T033_F_RUN_DOCKER_TESTS=1 for the isolated Docker/Compose rehearsal",
)
class DockerRehearsalTests(unittest.TestCase):
    """Opt-in local parent/nested mount, replacement, and rollback rehearsal."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temporary.name)
        cls.name = "t033f-local-" + uuid.uuid4().hex[:10]
        cls.project = cls.name
        cls.docker = [shutil.which("docker") or "docker"]
        context = os.environ.get("T033_F_DOCKER_CONTEXT")
        if context:
            cls.docker.extend(("--context", context))
        info = subprocess.run([*cls.docker, "info", "--format", "{{.OSType}}"], capture_output=True, text=True, check=True, timeout=30)
        if info.stdout.strip() != "linux":
            raise AssertionError("T033-F rehearsal requires a Linux Docker engine")
        cls.base = os.environ.get("T033_F_BASE_IMAGE", "ai-phone-t049a-candidate:20260913")
        cls.base_id = subprocess.run([*cls.docker, "image", "inspect", cls.base, "--format", "{{.Id}}"], capture_output=True, text=True, check=True, timeout=30).stdout.strip()
        cls.item = new_release(cls.path)
        cls.item.assets.mkdir()
        for relative in (
            *release.RUNTIME_PATHS,
            *release.PROTECTED_RUNTIME_PATHS,
            *release.TEST_PATHS,
            *release.TEST_SUPPORT_PATHS,
        ):
            target = cls.item.assets / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(release.normalized_bytes((REPOSITORY / relative).read_bytes()))
        manifest = json.loads((REPOSITORY / release.MANIFEST_PATH).read_text())
        cls.item.manifest = manifest
        cls.item.runtime_hashes = dict(manifest["runtime"])
        cls.item.protected_runtime_hashes = dict(manifest["protected_runtime"])
        cls.old_image = release.OLD_IMAGE
        release.OLD_IMAGE = cls.base_id
        cls.item.run = cls._run
        try:
            # The local dependency image predates the deployed voice release.
            # Reconstruct only its reviewed source layer from the real baseline
            # commit; never pretend the older image already contains those files.
            baseline = cls.path / "baseline"
            baseline.mkdir()
            baseline_paths = (
                "config/settings.py",
                "services/conversation/orchestrator.py",
                "services/stt/elevenlabs_stt_service.py",
                *release.PROTECTED_RUNTIME_PATHS,
            )
            for relative in baseline_paths:
                data = subprocess.run(
                    ["git", "show", release.DEPLOYED_SOURCE + ":" + relative],
                    cwd=REPOSITORY, capture_output=True, check=True, timeout=30,
                ).stdout
                target = baseline / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(release.normalized_bytes(data))
            cls._run("docker", "tag", cls.base_id, cls.name + "-seed")
            seed_id = cls._run(
                "docker", "image", "inspect", cls.name + "-seed",
                "--format", "{{.Id}}",
            ).stdout.decode().strip()
            if seed_id != cls.base_id:
                raise AssertionError("local dependency tag differs")
            (baseline / "Dockerfile").write_text(
                "FROM " + cls.name + "-seed\n" + "".join(
                    "COPY " + relative + " /app/" + relative + "\n"
                    for relative in baseline_paths
                ), encoding="utf-8",
            )
            cls._run(
                "docker", "build", "--pull=false", "--network", "none",
                "--label", "org.opencontainers.image.revision=" + release.DEPLOYED_SOURCE,
                "-t", cls.name + "-baseline", str(baseline), timeout=120,
            )
            cls.base_id = cls._run(
                "docker", "image", "inspect", cls.name + "-baseline",
                "--format", "{{.Id}}",
            ).stdout.decode().strip()
            release.OLD_IMAGE = cls.base_id
            cls.candidate = cls.item.build_candidate()
        except BaseException:
            cls._cleanup(False)
            release.OLD_IMAGE = cls.old_image
            cls.temporary.cleanup()
            raise

    @classmethod
    def _run(cls, *arguments, timeout=60, check=True):
        if arguments[0] != "docker":
            raise RuntimeError("unexpected local release command")
        result = subprocess.run([*cls.docker, *arguments[1:]], cwd=REPOSITORY, capture_output=True, timeout=timeout, check=False)
        if check and result.returncode:
            raise RuntimeError((result.stdout + result.stderr).decode(errors="replace"))
        return result

    @classmethod
    def _cleanup(cls, verify=True):
        compose = cls.path / "compose.json"
        if compose.exists():
            subprocess.run([*cls.docker, "compose", "-p", cls.project, "-f", str(compose), "down", "--remove-orphans"], capture_output=True, timeout=60)
        for name in (
            cls.name + "-app",
            "t033f-image-probe-" + cls.path.name,
            "t033f-tests-" + cls.path.name,
        ):
            subprocess.run([*cls.docker, "rm", "-f", name], capture_output=True, timeout=30)
        for image in ("ai-phone-t033f-candidate:" + cls.path.name, "ai-phone-t033f-parent:" + cls.path.name, cls.name + "-baseline", cls.name + "-seed"):
            subprocess.run([*cls.docker, "image", "rm", "-f", image], capture_output=True, timeout=60)
        if verify:
            inventory = "\n".join(subprocess.run(command, capture_output=True, text=True, check=True, timeout=30).stdout for command in (
                [*cls.docker, "container", "ls", "-a", "--format", "{{.Names}}"],
                [*cls.docker, "network", "ls", "--format", "{{.Name}}"],
                [*cls.docker, "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
            ))
            if cls.name in inventory or cls.path.name in inventory:
                raise AssertionError("T033-F Docker resources were not removed")

    @classmethod
    def tearDownClass(cls):
        try:
            cls._cleanup()
        finally:
            release.OLD_IMAGE = cls.old_image
            cls.temporary.cleanup()

    def test_parent_nested_settings_candidate_import_and_exact_rollback(self):
        parent_config = self.path / "parent-config"
        shutil.copytree(REPOSITORY / "config", parent_config)
        (parent_config / "settings.py").write_text("raise RuntimeError('parent hidden')\n")
        old_settings = self.path / "old-settings.py"
        old_settings.write_bytes((self.path / "baseline/config/settings.py").read_bytes())
        nested = self.path / "candidate-settings.py"
        nested.write_bytes((REPOSITORY / "config/settings.py").read_bytes())
        previous_service = {"services": {"app": {
            "image": self.base_id, "container_name": self.name + "-app",
            "entrypoint": ["python", "-c"],
            "command": ["import time; time.sleep(300)"],
            "environment": dict(value.split("=", 1) for value in release.SYNTHETIC_ENV),
            "volumes": [
                {"type": "bind", "source": str(parent_config), "target": "/app/config", "read_only": True},
                {"type": "bind", "source": str(old_settings), "target": release.SETTINGS_TARGET, "read_only": True},
            ],
            "networks": ["isolated"],
        }}, "networks": {"isolated": {"internal": True}}}
        compose = self.path / "compose.json"

        def up(image, *, candidate):
            service = copy.deepcopy(previous_service)
            service["services"]["app"]["image"] = image
            if candidate:
                service["services"]["app"]["environment"].update(release.DIAGNOSTIC_ENV)
                service["services"]["app"]["volumes"][-1]["source"] = str(nested)
            compose.write_text(json.dumps(service), encoding="utf-8")
            self._run("docker", "compose", "-p", self.project, "-f", str(compose), "up", "-d", "--no-deps", "--no-build", "--pull", "never", "--force-recreate", "app", timeout=90)
            return json.loads(self._run("docker", "inspect", self.name + "-app").stdout)[0]

        self.item.run_candidate_tests(self.candidate)
        old = up(self.base_id, candidate=False)
        candidate = up(self.candidate, candidate=True)
        probe = self._run(
            "docker", "exec", self.name + "-app", "python", "-c",
            release.source_probe_code(
                {**self.item.runtime_hashes, **self.item.protected_runtime_hashes},
                include_settings=True,
            ),
        )
        value = json.loads(probe.stdout)
        self.assertEqual(
            value["hashes"],
            {**self.item.runtime_hashes, **self.item.protected_runtime_hashes},
        )
        self.assertTrue(value["settings"]["barge_in_diagnostics_enabled"])
        self.assertFalse(value["settings"]["elevenlabs_stt_filter_background_audio"])
        self.assertTrue(candidate["State"]["Running"])
        restored = up(self.base_id, candidate=False)
        self.assertTrue(old["State"]["Running"])
        self.assertTrue(restored["State"]["Running"])
        self.assertEqual(old["Image"], self.base_id)
        self.assertEqual(candidate["Image"], self.candidate)
        self.assertEqual(restored["Image"], self.base_id)
        self.assertEqual(
            release.container_contract(old, candidate=False),
            release.container_contract(restored, candidate=False),
        )
        restored_settings = json.loads(
            self._run(
                "docker", "exec", self.name + "-app", "python", "-c",
                release.SETTINGS_CODE,
            ).stdout
        )
        self.assertNotIn("barge_in_diagnostics_enabled", restored_settings)
        self.assertNotIn("elevenlabs_stt_filter_background_audio", restored_settings)


if __name__ == "__main__":
    unittest.main()
