"""Behavioral tests for the bounded T034-B app-only voice release."""
from __future__ import annotations

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
SOURCE = REPOSITORY / "scripts" / "deploy-voice-beta.py"
SPEC = importlib.util.spec_from_file_location("t034b_voice_release", SOURCE)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)

COMMIT = "f" * 40
SECRET = "synthetic-$-first\nsecond"


def completed(stdout=b"", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=b"", returncode=returncode)


def new_release(path: Path) -> release.Release:
    item = release.Release(path, COMMIT)
    item.root = path / "root"
    item.root.mkdir(parents=True, exist_ok=True)
    return item


def compose_fixture(image: str) -> dict:
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
                        "type": "bind",
                        "source": "/private/runtime.env",
                        "target": "/app/.env",
                        "read_only": True,
                        "bind": {"create_host_path": True},
                    },
                    {
                        "type": "bind",
                        "source": "/private/settings.py",
                        "target": "/app/config/settings.py",
                        "read_only": True,
                    },
                ],
                "x-unknown": {"preserve": True},
            },
            "db": {"image": "postgres:16", "x-unknown": "retained"},
        },
        "networks": {"internal": {"name": "existing-internal"}},
        "x-top-level": {"preserve": True},
    }


class OfflineReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.item = new_release(self.path)

    def test_import_has_no_subprocess_filesystem_or_network_side_effect(self):
        spec = importlib.util.spec_from_file_location(
            "t034b_import_check_" + uuid.uuid4().hex, SOURCE
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch("subprocess.run") as run, mock.patch("tempfile.mkdtemp") as mkdir:
            spec.loader.exec_module(module)
        run.assert_not_called()
        mkdir.assert_not_called()

    def test_ready_http_waits_for_docker_health_before_replacement_validation(self):
        self.item.run = mock.Mock(return_value=completed(b'{"status":"ready"}'))
        self.item.inspect = mock.Mock(side_effect=[
            {"State": {"Running": True, "Health": {"Status": "starting"}}},
            {"State": {"Running": True, "Health": {"Status": "healthy"}}},
        ])
        with mock.patch.object(release.time, "sleep"):
            self.item.wait_ready()
        self.assertEqual(self.item.inspect.call_count, 2)
        self.assertEqual(self.item.run.call_count, 2)

    def test_current_manifest_entries_match_normalized_working_tree(self):
        manifest = json.loads(
            (REPOSITORY / release.MANIFEST_PATH).read_text(encoding="utf-8")
        )
        observed = {}
        for section in ("runtime", "tests"):
            for relative, expected in manifest[section].items():
                actual = release.sha256(
                    release.normalized_bytes((REPOSITORY / relative).read_bytes())
                )
                self.assertEqual(actual, expected, relative)
                observed[relative] = actual
        self.assertEqual(len(observed), 26)

    def test_verify_source_accepts_only_exact_runtime_scope(self):
        nginx = self.item.root / "nginx" / "nginx.conf"
        nginx.parent.mkdir()
        nginx.write_bytes(b"reviewed nginx")
        expected_nginx = release.NGINX_HASH

        def run(*arguments, **_kwargs):
            if arguments[:3] == ("git", "rev-parse", "HEAD"):
                return completed((release.BASE_CHECKOUT + "\n").encode())
            if arguments[:2] == ("git", "rev-parse"):
                return completed((arguments[2].removesuffix("^{commit}") + "\n").encode())
            if arguments[:2] == ("git", "merge-base"):
                return completed()
            if arguments[:2] == ("git", "status"):
                return completed(b"M nginx/nginx.conf\n")
            if arguments[:3] == ("git", "diff", "--name-only"):
                return completed(b"services/conversation/orchestrator.py\n")
            if arguments[:2] == ("git", "cat-file"):
                return completed()
            self.fail("unexpected command: " + repr(arguments))

        self.item.run = run
        with mock.patch.object(release, "NGINX_HASH", release.sha256(b"reviewed nginx")):
            self.item.verify_source()

            original = self.item.run

            def added(*arguments, **kwargs):
                value = original(*arguments, **kwargs)
                if arguments[:3] == ("git", "diff", "--name-only"):
                    value.stdout += b"services/other.py\n"
                return value

            self.item.run = added
            with self.assertRaisesRegex(RuntimeError, "exceeds reviewed voice runtime scope"):
                self.item.verify_source()
        self.assertEqual(expected_nginx, release.NGINX_HASH)

    @unittest.skipUnless(shutil.which("git"), "real source-history regression requires Git")
    def test_incremental_release_source_guard_with_real_git_history(self):
        root = self.item.root

        def git(*args):
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, check=True, timeout=15,
            )

        git("init")
        git("config", "user.name", "Synthetic Release Test")
        git("config", "user.email", "synthetic@example.invalid")
        git("config", "core.autocrlf", "false")
        paths = (*release.RUNTIME_PATHS, release.MANIFEST_PATH,
                 "scripts/deploy-voice-beta.py", "nginx/nginx.conf")
        for relative in paths:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("baseline\n", encoding="utf-8")
        git("add", ".")
        git("-c", "commit.gpgsign=false", "commit", "-m", "baseline")
        base = git("rev-parse", "HEAD").stdout.decode().strip()
        for relative in release.RUNTIME_PATHS:
            (root / relative).write_text("previous voice release\n", encoding="utf-8")
        git("add", ".")
        git("-c", "commit.gpgsign=false", "commit", "-m", "previous voice release")
        deployed = git("rev-parse", "HEAD").stdout.decode().strip()
        (root / "services/conversation/orchestrator.py").write_text(
            "acknowledgement release\n", encoding="utf-8"
        )
        git("add", ".")
        git("-c", "commit.gpgsign=false", "commit", "-m", "one runtime change")
        candidate = git("rev-parse", "HEAD").stdout.decode().strip()
        # An otherwise accepted voice file must not become an allowed change.
        (root / "services/stt/stt_base.py").write_text("unreviewed change\n", encoding="utf-8")
        git("add", ".")
        git("-c", "commit.gpgsign=false", "commit", "-m", "extra runtime change")
        extra = git("rev-parse", "HEAD").stdout.decode().strip()
        git("checkout", "--detach", base)
        (root / "nginx/nginx.conf").write_bytes(b"reviewed nginx\n")

        def run(*arguments, **kwargs):
            return subprocess.run(
                arguments, cwd=root, capture_output=True,
                check=kwargs.get("check", True), timeout=15,
            )

        self.item.run = run
        self.item.commit = candidate
        with mock.patch.multiple(
            release, BASE_CHECKOUT=base, DEPLOYED_SOURCE=deployed,
            NGINX_HASH=release.sha256(b"reviewed nginx\n"),
        ):
            self.item.verify_source()
            self.item.commit = extra
            with self.assertRaisesRegex(RuntimeError, "exceeds reviewed voice runtime scope"):
                self.item.verify_source()

    def test_staging_rejects_any_hash_mismatch(self):
        self.item.git_bytes = lambda relative: (REPOSITORY / relative).read_bytes()
        self.item.stage_assets()
        self.assertEqual(set(self.item.runtime_hashes), set(release.RUNTIME_PATHS))

        other = new_release(self.path / "other")

        def changed(relative):
            data = (REPOSITORY / relative).read_bytes()
            if relative == release.RUNTIME_PATHS[0]:
                return data + b"\n# drift\n"
            return data

        other.git_bytes = changed
        with self.assertRaisesRegex(RuntimeError, "reviewed asset hash differs"):
            other.stage_assets()

    def test_hidden_runtime_mounts_are_refused(self):
        harmless = [{"Destination": "/app/config/settings.py", "Source": "/private/s"}]
        self.assertEqual(release.hidden_runtime_paths(harmless), ())
        hidden = release.hidden_runtime_paths(
            [{"target": "/app/services/conversation", "source": "/old/conversation"}]
        )
        self.assertIn("services/conversation/orchestrator.py", hidden)
        self.assertIn("services/conversation/turn_controller.py", hidden)
        self.assertNotIn("services/stt/stt_base.py", hidden)

    def test_semantic_compose_ignores_mount_order_only(self):
        baseline = compose_fixture(release.OLD_IMAGE)
        reordered = json.loads(json.dumps(baseline))
        reordered["services"]["app"]["image"] = "sha256:candidate"
        reordered["services"]["app"]["volumes"].reverse()
        self.assertEqual(
            release.semantic_compose(baseline), release.semantic_compose(reordered)
        )
        changed = json.loads(json.dumps(reordered))
        changed["services"]["app"]["volumes"][0]["source"] = "/different"
        self.assertNotEqual(
            release.semantic_compose(baseline), release.semantic_compose(changed)
        )
        duplicate = json.loads(json.dumps(reordered))
        duplicate["services"]["app"]["volumes"].append(
            dict(duplicate["services"]["app"]["volumes"][0])
        )
        self.assertNotEqual(
            release.semantic_compose(baseline), release.semantic_compose(duplicate)
        )

    def test_prepare_override_preserves_unknown_fields_dollars_and_newlines(self):
        original = compose_fixture(release.OLD_IMAGE)
        baseline = json.loads(json.dumps(original))
        self.item.previous_rendered = baseline
        self.item.previous_settings = {
            "private": SECRET,
            **release.REQUIRED_SETTINGS,
        }
        self.item.runtime_hashes = {"voice.py": "abc"}

        def render(path):
            value = json.loads(path.read_text(encoding="utf-8"))
            value["services"]["app"]["volumes"].reverse()
            return value

        self.item.render_compose = render
        self.item.candidate_probe = lambda *_args: {
            "hashes": self.item.runtime_hashes,
            "settings": self.item.previous_settings,
        }
        output = self.item.prepare_override("sha256:candidate", original)
        written = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(written["services"]["app"]["environment"]["PRIVATE_VALUE"], SECRET)
        self.assertEqual(written["services"]["app"]["x-unknown"], {"preserve": True})
        self.assertEqual(written["x-top-level"], {"preserve": True})

    def test_prepare_override_refuses_settings_source_and_mount_drift(self):
        original = compose_fixture(release.OLD_IMAGE)
        self.item.previous_rendered = json.loads(json.dumps(original))
        self.item.previous_settings = {"private": SECRET, **release.REQUIRED_SETTINGS}
        self.item.runtime_hashes = {"voice.py": "abc"}
        self.item.render_compose = lambda path: json.loads(path.read_text())
        self.item.candidate_probe = lambda *_args: {
            "hashes": {"voice.py": "wrong"},
            "settings": self.item.previous_settings,
        }
        with self.assertRaisesRegex(RuntimeError, "source or settings changed"):
            self.item.prepare_override("sha256:candidate", original)

        settings = new_release(self.path / "settings-drift")
        settings.previous_rendered = json.loads(json.dumps(original))
        settings.previous_settings = self.item.previous_settings
        settings.runtime_hashes = self.item.runtime_hashes
        settings.render_compose = lambda path: json.loads(path.read_text())
        settings.candidate_probe = lambda *_args: {
            "hashes": settings.runtime_hashes,
            "settings": {"private": "different", **release.REQUIRED_SETTINGS},
        }
        with self.assertRaisesRegex(RuntimeError, "source or settings changed"):
            settings.prepare_override("sha256:candidate", original)

        other = new_release(self.path / "mount-drift")
        other.previous_rendered = json.loads(json.dumps(original))
        other.previous_settings = self.item.previous_settings
        other.runtime_hashes = self.item.runtime_hashes

        def changed(path):
            value = json.loads(path.read_text())
            value["services"]["app"]["volumes"][0]["source"] = "/different"
            return value

        other.render_compose = changed
        other.candidate_probe = self.item.candidate_probe
        with self.assertRaisesRegex(RuntimeError, "Compose configuration changed"):
            other.prepare_override("sha256:candidate", original)

    def test_stale_override_blocks_before_any_stop(self):
        expected = b'{"services":{"app":{"image":"old"}}}'
        (self.item.directory / "previous-override.json").write_bytes(expected)
        (self.item.root / "docker-compose.override.yml").write_bytes(b"{}")
        stop = mock.Mock()
        self.item.stop_components = stop
        with self.assertRaisesRegex(RuntimeError, "override changed before cutover"):
            self.item.deploy("sha256:candidate", self.item.directory / "candidate.json")
        stop.assert_not_called()

    def test_candidate_test_failure_stops_before_prepare_or_cutover(self):
        original = compose_fixture(release.OLD_IMAGE)
        self.item.preflight = mock.Mock(return_value=original)
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
            payload = {"read_only": True, "history": release.EXPECTED_HISTORY}
            return completed(json.dumps(payload).encode())

        self.item.one_shot = one_shot
        self.assertTrue(self.item.schema_matches())
        self.assertIn("transaction(readonly=True)", release.SCHEMA_0004_CODE)
        self.assertIn(
            "check_runtime_compatibility(c,require_notification=True)",
            release.SCHEMA_0004_CODE,
        )
        self.assertEqual(observed["kwargs"]["network"], "internal")
        self.assertFalse(observed["kwargs"]["check"])

        self.item.one_shot = lambda *_a, **_k: completed(
            json.dumps({"read_only": True, "history": release.EXPECTED_HISTORY[:-1]}).encode()
        )
        self.assertFalse(self.item.schema_matches())
        self.item.one_shot = lambda *_a, **_k: completed(
            json.dumps({"read_only": False, "history": release.EXPECTED_HISTORY}).encode()
        )
        self.assertFalse(self.item.schema_matches())

    def test_timeout_and_partial_creation_always_remove_registered_container(self):
        calls = []

        def run(*arguments, **_kwargs):
            calls.append(arguments)
            if arguments[:2] == ("docker", "run"):
                raise RuntimeError("command timed out; protected log retained")
            if arguments[:3] == ("docker", "rm", "-f"):
                return completed()
            if arguments[:3] == ("docker", "ps", "-a"):
                return completed()
            self.fail("unexpected command")

        self.item.run = run
        name = "t034b-partial-" + self.path.name
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.item.one_shot(name, "sha256:image", ["-c", "pass"])
        self.assertIn(("docker", "rm", "-f", name), calls)
        self.assertIn(("docker", "ps", "-a", "--format", "{{.Names}}"), calls)

    def test_cleanup_failure_does_not_replace_primary_failure(self):
        self.item.run = mock.Mock(side_effect=RuntimeError("primary test failure"))
        self.item.remove_container = mock.Mock(
            side_effect=RuntimeError("cleanup diagnostic sentinel")
        )
        with self.assertRaisesRegex(RuntimeError, "primary test failure"):
            self.item.one_shot("registered-name", "sha256:image", ["-c", "pass"])
        private = (self.item.directory / "private.log").read_text()
        self.assertIn("cleanup_failure_type=RuntimeError", private)
        self.assertNotIn("cleanup diagnostic sentinel", private)

    def test_settings_probe_registers_name_and_cleans_partial_creation(self):
        observed = []

        def compose(_override, *arguments, **_kwargs):
            observed.append(arguments)
            raise RuntimeError("probe creation failed")

        self.item.compose = compose
        self.item.remove_container = lambda name: observed.append(("removed", name))
        with self.assertRaisesRegex(RuntimeError, "probe creation failed"):
            self.item.candidate_probe(self.path / "candidate.json", "candidate")
        command = observed[0]
        name = "t034b-settings-candidate-" + self.path.name
        self.assertEqual(command[command.index("--name") + 1], name)
        self.assertEqual(observed[-1], ("removed", name))

    def test_actual_replacement_verifier_accepts_exact_runtime_contract(self):
        mounts = [
            {
                "Type": "bind", "Source": "/private/runtime.env",
                "Destination": "/app/.env", "RW": False,
            },
            {
                "Type": "bind", "Source": "/private/settings.py",
                "Destination": "/app/config/settings.py", "RW": False,
            },
        ]
        current = {
            "Image": "sha256:candidate",
            "State": {"Running": True, "Health": {"Status": "healthy"}},
            "Config": {
                "Env": ["PRIVATE_VALUE=" + SECRET, "UNKNOWN=retained"],
                "Cmd": ["python", "main.py"],
                "Entrypoint": None,
                "Labels": {"org.opencontainers.image.revision": COMMIT},
            },
            "HostConfig": {
                "PortBindings": {"8000/tcp": [{"HostPort": "8000"}]},
                "RestartPolicy": {"Name": "unless-stopped"},
            },
            "Mounts": mounts,
            "NetworkSettings": {"Networks": {"existing-internal": {}}},
        }
        self.item.previous_contract = release.container_contract(current)
        self.item.previous_settings = {"private": SECRET, **release.REQUIRED_SETTINGS}
        self.item.previous_rendered = compose_fixture(release.OLD_IMAGE)
        self.item.runtime_hashes = {"voice.py": "abc"}
        self.item.wait_ready = mock.Mock()
        self.item.inspect = lambda _name: current
        self.item.settings = lambda: self.item.previous_settings
        self.item.running_probe = lambda: {"hashes": self.item.runtime_hashes}
        rendered = compose_fixture("sha256:candidate")
        rendered["services"]["app"]["volumes"].reverse()
        self.item.render_compose = lambda _path: rendered
        self.item.schema_matches = lambda image: image == "sha256:candidate"
        self.item.verify_replaced("sha256:candidate", COMMIT)

        current["Config"]["Env"].append("DRIFT=true")
        with self.assertRaisesRegex(RuntimeError, "runtime contract differs"):
            self.item.verify_replaced("sha256:candidate", COMMIT)

    def _cutover(self, *, failure=None, recovery_failure=False):
        previous = b'{"services":{"app":{"image":"old"}}}\n'
        candidate_bytes = b'{"services":{"app":{"image":"candidate"}}}\n'
        previous_path = self.item.directory / "previous-override.json"
        candidate_path = self.item.directory / "candidate-override.json"
        previous_path.write_bytes(previous)
        candidate_path.write_bytes(candidate_bytes)
        state = {
            "image": release.OLD_IMAGE,
            "override": previous,
            "ingress": True,
            "events": [],
        }
        self.item.recheck_before_stop = lambda: state["events"].append("recheck")

        def stop_components(*, strict):
            state["events"].append(("stop", strict, state["image"]))
            state["ingress"] = False

        def install(path):
            state["override"] = path.read_bytes()
            state["events"].append(("install", path.name))

        def replace(image, revision):
            state["events"].append(("replace", image, revision))
            is_candidate = image != release.OLD_IMAGE
            if (failure == "readiness" and is_candidate) or (
                recovery_failure and not is_candidate
            ):
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
        output = io.StringIO()
        return state, candidate_path, output

    def test_successful_replacement_records_candidate_and_success_markers(self):
        state, candidate, output = self._cutover()
        with redirect_stdout(output):
            self.item.deploy("sha256:candidate", candidate)
        self.assertEqual(state["image"], "sha256:candidate")
        self.assertEqual(state["override"], candidate.read_bytes())
        self.assertTrue(state["ingress"])
        self.assertEqual(
            (self.item.directory / "deployed").read_text().splitlines()[:2],
            [COMMIT, "sha256:candidate"],
        )
        self.assertIn("VOICE_BETA_DEPLOYED_READY_HTTPS_OK", output.getvalue())

    def test_candidate_readiness_failure_restores_exact_previous_app(self):
        state, candidate, output = self._cutover(failure="readiness")
        with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "readiness failure"):
            self.item.deploy("sha256:candidate", candidate)
        self.assertEqual(state["image"], release.OLD_IMAGE)
        self.assertEqual(
            state["override"], (self.item.directory / "previous-override.json").read_bytes()
        )
        self.assertTrue(state["ingress"])
        self.assertIn("PREVIOUS_VOICE_APP_RESTORED_SCHEMA_0004_UNCHANGED", output.getvalue())
        replacements = [event for event in state["events"] if isinstance(event, tuple) and event[0] == "replace"]
        old_index = state["events"].index(replacements[-1])
        self.assertEqual(state["events"][old_index - 2][0], "stop")

    def test_post_restart_tls_failure_closes_candidate_then_recovers(self):
        state, candidate, output = self._cutover(failure="tls")
        with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "TLS failure"):
            self.item.deploy("sha256:candidate", candidate)
        self.assertEqual(state["image"], release.OLD_IMAGE)
        self.assertTrue(state["ingress"])
        stop_events = [event for event in state["events"] if isinstance(event, tuple) and event[0] == "stop"]
        self.assertEqual(stop_events[-1][2], "sha256:candidate")

    def test_failed_recovery_leaves_ingress_closed_and_never_reports_success(self):
        state, candidate, output = self._cutover(
            failure="readiness", recovery_failure=True
        )
        with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "readiness failure"):
            self.item.deploy("sha256:candidate", candidate)
        self.assertFalse(state["ingress"])
        text = output.getvalue()
        self.assertIn("RECOVERY_FAILED_INGRESS_CLOSED", text)
        self.assertIn("RECOVERY_UNVERIFIED_MANUAL_REVIEW_REQUIRED", text)
        self.assertNotIn("VOICE_BETA_DEPLOYED_READY_HTTPS_OK", text)


@unittest.skipUnless(
    os.environ.get("T034_RUN_DOCKER_TESTS") == "1",
    "set T034_RUN_DOCKER_TESTS=1 for the isolated Docker rehearsal",
)
class DockerRehearsalTests(unittest.TestCase):
    """Opt-in local derived-image, accepted-test, replacement and rollback rehearsal."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temporary.name)
        cls.name = "t034b-local-" + uuid.uuid4().hex[:10]
        cls.project = cls.name
        cls.candidate = ""
        cls.old_image = release.OLD_IMAGE
        cls.old_project = release.PROJECT
        cls.item = new_release(cls.path)
        cls.item.assets.mkdir()
        for relative in (*release.RUNTIME_PATHS, *release.TEST_PATHS):
            target = cls.item.assets / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(release.normalized_bytes((REPOSITORY / relative).read_bytes()))
        manifest = json.loads((REPOSITORY / release.MANIFEST_PATH).read_text())
        cls.item.manifest = manifest
        cls.item.runtime_hashes = dict(manifest["runtime"])
        docker = shutil.which("docker")
        if docker is None:
            raise AssertionError("T034 Docker rehearsal requested but Docker is unavailable")
        context = os.environ.get("T034_DOCKER_CONTEXT", "desktop-linux")
        cls.docker = [docker, "--context", context]
        info = subprocess.run(
            [*cls.docker, "info", "--format", "{{.OSType}}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        if info.stdout.strip() != "linux":
            raise AssertionError("T034 Docker rehearsal requires a Linux Docker engine")
        cls.base = os.environ.get(
            "T034_BASE_IMAGE", "ai-phone-t049a-candidate:20260913"
        )
        inspect = subprocess.run(
            [*cls.docker, "image", "inspect", cls.base, "--format", "{{.Id}}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        cls.base_id = inspect.stdout.strip()
        release.OLD_IMAGE = cls.base_id
        release.PROJECT = cls.project
        cls.item.run = cls._run
        try:
            cls.candidate = cls.item.build_candidate()
        except BaseException:
            cls._cleanup(verify=False)
            release.OLD_IMAGE = cls.old_image
            release.PROJECT = cls.old_project
            cls.temporary.cleanup()
            raise

    @classmethod
    def _run(cls, *arguments, timeout=60, check=True):
        if not arguments or arguments[0] != "docker":
            raise RuntimeError("unexpected local release command")
        result = subprocess.run(
            [*cls.docker, *arguments[1:]],
            cwd=REPOSITORY,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if check and result.returncode:
            # This harness uses synthetic credentials only. Keep pytest's failure
            # output as well as stderr so image-only regressions are diagnosable.
            raise RuntimeError((result.stdout + result.stderr).decode(errors="replace"))
        return result

    @classmethod
    def _cleanup(cls, *, verify):
        compose = cls.path / "compose.json"
        if compose.exists():
            subprocess.run(
                [
                    *cls.docker, "compose", "-p", cls.project,
                    "-f", str(compose), "down", "--remove-orphans",
                ],
                capture_output=True,
                timeout=60,
            )
        for container in (
            cls.name + "-app",
            cls.name + "-probe",
            "t034b-image-probe-" + cls.path.name,
            "t034b-tests-" + cls.path.name,
        ):
            subprocess.run(
                [*cls.docker, "rm", "-f", container],
                capture_output=True,
                timeout=30,
            )
        for image in (
            "ai-phone-t034b-candidate:" + cls.path.name,
            "ai-phone-t034b-parent:" + cls.path.name,
        ):
            subprocess.run(
                [*cls.docker, "image", "rm", "-f", image],
                capture_output=True,
                timeout=60,
            )
        if verify:
            containers = subprocess.run(
                [*cls.docker, "container", "ls", "--all", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout
            networks = subprocess.run(
                [*cls.docker, "network", "ls", "--format", "{{.Name}}"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout
            images = subprocess.run(
                [*cls.docker, "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout
            if cls.name in containers or cls.project in networks or cls.path.name in images:
                raise AssertionError("T034 Docker rehearsal resource cleanup unverified")

    @classmethod
    def tearDownClass(cls):
        try:
            cls._cleanup(verify=True)
        finally:
            release.OLD_IMAGE = cls.old_image
            release.PROJECT = cls.old_project
            cls.temporary.cleanup()

    def test_derived_assets_tests_mount_contract_and_rollback(self):
        self.item.run_candidate_tests(self.candidate)
        marker = self.path / "synthetic.env"
        marker.write_text("VOICE_TEST_VALUE=synthetic-$-first\\nsecond\n", encoding="utf-8")
        service = {
            "services": {
                "app": {
                    "image": self.base_id,
                    "container_name": self.name + "-app",
                    "entrypoint": ["python", "-c"],
                    "command": ["import time; time.sleep(300)"],
                    "environment": {"VOICE_TEST_VALUE": SECRET},
                    "volumes": [
                        {
                            "type": "bind", "source": str(marker),
                            "target": "/audit/synthetic.env", "read_only": True,
                        }
                    ],
                    "networks": ["isolated"],
                }
            },
            "networks": {"isolated": {"internal": True}},
        }
        compose = self.path / "compose.json"
        compose.write_text(json.dumps(service), encoding="utf-8")

        def up(image):
            service["services"]["app"]["image"] = image
            compose.write_text(json.dumps(service), encoding="utf-8")
            self._run(
                "docker", "compose", "-p", self.project, "-f", str(compose),
                "up", "-d", "--no-deps", "--no-build", "--pull", "never",
                "--force-recreate", "app", timeout=90,
            )
            return json.loads(self._run("docker", "inspect", self.name + "-app").stdout)[0]

        old = up(self.base_id)
        candidate = up(self.candidate)
        restored = up(self.base_id)
        self.assertEqual(candidate["Image"], self.candidate)
        self.assertEqual(restored["Image"], self.base_id)
        self.assertEqual(release.container_contract(old), release.container_contract(candidate))
        self.assertEqual(release.container_contract(old), release.container_contract(restored))
        probe = self.item.one_shot(
            self.name + "-probe",
            self.candidate,
            ["-c", release.source_probe_code(self.item.runtime_hashes, include_settings=False)],
            synthetic_env=True,
        )
        self.assertEqual(json.loads(probe.stdout)["hashes"], self.item.runtime_hashes)


if __name__ == "__main__":
    unittest.main()
