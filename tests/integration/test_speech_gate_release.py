"""Behavioral contracts for the protected T033-H speech-gate rollout."""

from __future__ import annotations

import ast
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import uuid


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "scripts" / "deploy-speech-aware-barge-in.py"
SPEC = importlib.util.spec_from_file_location("t033h_release", SCRIPT)
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


def old_settings() -> dict:
    return {
        "private": SECRET,
        **release.REQUIRED_SETTINGS,
        "barge_in_diagnostics_enabled": True,
        "elevenlabs_stt_filter_background_audio": False,
    }


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
                        "type": "bind",
                        "source": "/private/runtime.env",
                        "target": "/app/.env",
                        "read_only": True,
                        "bind": {"create_host_path": True},
                    },
                    {
                        "type": "bind",
                        "source": "/private/config",
                        "target": "/app/config",
                        "read_only": True,
                    },
                    {
                        "type": "bind",
                        "source": settings_source,
                        "target": release.SETTINGS_TARGET,
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


def container_fixture(image: str, revision: str, *, candidate: bool, settings_source: str) -> dict:
    environment = ["PRIVATE_VALUE=" + SECRET, "UNKNOWN=retained"]
    if candidate:
        environment.extend(f"{key}={value}" for key, value in release.GATE_ENV.items())
    return {
        "Image": image,
        "State": {"Running": True, "Health": {"Status": "healthy"}},
        "Config": {
            "Env": environment,
            "Cmd": ["python", "main.py"],
            "Entrypoint": None,
            "Labels": {"org.opencontainers.image.revision": revision},
        },
        "HostConfig": {
            "PortBindings": {"8000/tcp": [{"HostPort": "8000"}]},
            "RestartPolicy": {"Name": "unless-stopped"},
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": "/private/runtime.env",
                "Destination": "/app/.env",
                "RW": False,
            },
            {
                "Type": "bind",
                "Source": "/private/config",
                "Destination": "/app/config",
                "RW": False,
            },
            {
                "Type": "bind",
                "Source": settings_source,
                "Destination": release.SETTINGS_TARGET,
                "RW": False,
            },
        ],
        "NetworkSettings": {"Networks": {"existing-internal": {}}},
    }


class OfflineReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.item = new_release(self.path)

    def test_import_is_side_effect_free(self):
        spec = importlib.util.spec_from_file_location("t033h_import_" + uuid.uuid4().hex, SCRIPT)
        module = importlib.util.module_from_spec(spec)
        with mock.patch("subprocess.run") as run, mock.patch("tempfile.mkdtemp") as mkdir:
            assert spec.loader is not None
            spec.loader.exec_module(module)
        run.assert_not_called()
        mkdir.assert_not_called()

    def test_manifest_and_all_51_entries_match_with_raw_model_assets(self):
        manifest_path = REPOSITORY / release.MANIFEST_PATH
        manifest_bytes = release.normalized_text_bytes(manifest_path.read_bytes())
        self.assertEqual(release.sha256(manifest_bytes), release.MANIFEST_SHA256)
        manifest = json.loads(manifest_bytes)
        observed = {}
        for section in release.ASSET_SECTIONS:
            for relative, expected in manifest[section].items():
                data = (REPOSITORY / relative).read_bytes()
                actual = release.sha256(
                    data if section == "model_assets" else release.normalized_text_bytes(data)
                )
                self.assertEqual(actual, expected, relative)
                observed[relative] = actual
        self.assertEqual(len(observed), 51)
        self.assertEqual(len(observed), len(set(observed)))

    def test_all_four_booking_method_spans_match_frozen_hashes(self):
        manifest = json.loads(
            (REPOSITORY / release.MANIFEST_PATH).read_text(encoding="utf-8")
        )
        source = release.normalized_text_bytes(
            (REPOSITORY / "services/conversation/orchestrator.py").read_bytes()
        ).decode("utf-8")
        lines = source.splitlines(keepends=True)
        expected = manifest["booking_method_hashes"]
        nodes = {
            node.name: node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in expected
        }
        self.assertEqual(set(nodes), set(expected))
        for name, node in nodes.items():
            span = "".join(lines[node.lineno - 1 : node.end_lineno]).encode()
            self.assertEqual(release.sha256(span), expected[name], name)

    def test_binary_corruption_is_not_normalized_away(self):
        original = b"onnx\r\npayload"
        self.assertNotEqual(
            release.asset_digest("model_assets", original),
            release.asset_digest("model_assets", original.replace(b"\r\n", b"\n")),
        )
        self.assertEqual(
            release.asset_digest("runtime", original),
            release.asset_digest("runtime", original.replace(b"\r\n", b"\n")),
        )

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
                payload = changed if changed is not None else (
                    "\n".join(sorted(release.SOURCE_SCOPE)) + "\n"
                ).encode()
                return completed(payload)
            if arguments[:2] == ("git", "cat-file"):
                path = arguments[-1].split(":", 1)[1]
                if path == missing:
                    raise RuntimeError("command failed; protected log retained")
                return completed()
            self.fail("unexpected command: " + repr(arguments))

        return run, calls

    def test_source_requires_exact_scope_manifest_and_four_release_files(self):
        nginx = self.item.root / "nginx" / "nginx.conf"
        nginx.parent.mkdir()
        nginx.write_bytes(b"reviewed nginx")
        self.item.run, calls = self._source_runner()
        with mock.patch.object(release, "NGINX_HASH", release.sha256(b"reviewed nginx")):
            self.item.verify_source()
        checked = {
            call[-1].split(":", 1)[1]
            for call in calls
            if call[:2] == ("git", "cat-file")
        }
        self.assertEqual(checked, {release.MANIFEST_PATH, *release.RELEASE_PATHS})
        self.assertEqual(len(release.RELEASE_PATHS), 4)

    def test_extra_scope_or_missing_release_file_stops_verification(self):
        nginx = self.item.root / "nginx" / "nginx.conf"
        nginx.parent.mkdir()
        nginx.write_bytes(b"reviewed nginx")
        extra = "\n".join(sorted(release.SOURCE_SCOPE | {"services/private.py"})) + "\n"
        self.item.run, _ = self._source_runner(extra.encode())
        with mock.patch.object(release, "NGINX_HASH", release.sha256(b"reviewed nginx")):
            with self.assertRaisesRegex(RuntimeError, "scope"):
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
        expected = {release.MANIFEST_PATH}
        for values in release.SECTION_PATHS.values():
            expected.update(values)
        self.assertEqual(set(seen), expected)
        self.assertEqual(len(expected) - 1, 51)

    def test_staging_rejects_changed_raw_model_and_missing_support(self):
        def changed(relative):
            if relative == release.MODEL_ASSET_PATHS[0]:
                return (REPOSITORY / relative).read_bytes() + b"\n"
            return (REPOSITORY / relative).read_bytes()

        self.item.git_bytes = changed
        with self.assertRaisesRegex(RuntimeError, "reviewed asset hash differs"):
            self.item.stage_assets()

        other = new_release(self.path / "missing")

        def missing(relative):
            if relative == release.BUILD_SUPPORT_PATHS[-1]:
                raise RuntimeError("command failed; protected log retained")
            return (REPOSITORY / relative).read_bytes()

        other.git_bytes = missing
        with self.assertRaisesRegex(RuntimeError, "command failed"):
            other.stage_assets()

        missing_model = new_release(self.path / "missing-model")

        def without_model(relative):
            if relative == release.MODEL_ASSET_PATHS[0]:
                raise RuntimeError("command failed; protected log retained")
            return (REPOSITORY / relative).read_bytes()

        missing_model.git_bytes = without_model
        with self.assertRaisesRegex(RuntimeError, "command failed"):
            missing_model.stage_assets()

    def test_platform_selection_is_exact_and_rejects_python_or_architecture_drift(self):
        expected = {
            "implementation": "cpython",
            "python": "3.12.14",
            "machine": "x86_64",
            "system": "Linux",
            "libc_name": "glibc",
            "libc_version": "2.36",
            "compatible": {
                wheel["filename"]: True for wheel in release.WHEELS.values()
            },
        }
        self.assertEqual(release.select_wheels(expected), release.WHEELS)
        for key, value in (
            ("python", "3.11.9"),
            ("machine", "aarch64"),
            ("system", "Darwin"),
            ("libc_version", "2.27"),
            (
                "compatible",
                {**expected["compatible"], next(iter(expected["compatible"])): False},
            ),
        ):
            with self.subTest(key=key):
                changed = dict(expected, **{key: value})
                with self.assertRaisesRegex(RuntimeError, "platform"):
                    release.select_wheels(changed)

    def test_wheelhouse_requires_exact_filenames_and_sha256(self):
        wheelhouse = self.path / "wheelhouse"
        wheelhouse.mkdir()
        payloads = {
            key: ("synthetic-wheel-" + key).encode()
            for key in release.WHEELS
        }
        for key, wheel in release.WHEELS.items():
            (wheelhouse / wheel["filename"]).write_bytes(payloads[key])
        with mock.patch.object(
            release,
            "WHEELS",
            {
                key: {**value, "sha256": release.sha256(payloads[key])}
                for key, value in release.WHEELS.items()
            },
        ):
            evidence = self.item.verify_wheelhouse(wheelhouse)
            self.assertEqual(set(evidence), set(release.WHEELS))
            first = next(iter(release.WHEELS.values()))["filename"]
            (wheelhouse / first).write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "wheel integrity"):
                self.item.verify_wheelhouse(wheelhouse)
        (wheelhouse / "unexpected.whl").write_bytes(b"unexpected")
        with self.assertRaisesRegex(RuntimeError, "wheel set"):
            self.item.verify_wheelhouse(wheelhouse)

    def test_package_transition_allows_only_three_additions_and_preserves_numpy(self):
        baseline = {"numpy": "1.26.4", "packaging": "25.0", "sympy": "1.14.0"}
        candidate = {**baseline, **release.PACKAGE_VERSIONS}
        self.item.verify_package_transition(baseline, candidate)
        for changed in (
            {**candidate, "numpy": "2.0.0"},
            {**candidate, "extra": "1"},
            {**candidate, "protobuf": "7.36.0"},
        ):
            with self.assertRaisesRegex(RuntimeError, "package"):
                self.item.verify_package_transition(baseline, changed)
        with self.assertRaisesRegex(RuntimeError, "baseline package"):
            self.item.verify_package_transition({**baseline, "protobuf": "6"}, candidate)

    def test_wheel_preparation_does_not_require_optional_symbolic_extra(self):
        self.item.platform = {"synthetic": True}
        self.item.baseline_packages = {"numpy": "1.26.4", "packaging": "26.3"}
        self.item.one_shot = mock.Mock(return_value=completed())
        self.item.verify_wheelhouse = mock.Mock(return_value={})
        with mock.patch.object(release, "select_wheels", return_value=release.WHEELS):
            self.item.prepare_wheels("sha256:baseline")
        self.item.one_shot.assert_called_once()
        self.assertTrue((self.path / "wheel-evidence.json").is_file())

    def test_old_settings_must_lack_all_six_gate_fields(self):
        previous = old_settings()
        self.item._validate_settings(previous, candidate=False)
        for key in release.GATE_SETTINGS:
            with self.subTest(key=key):
                with self.assertRaisesRegex(RuntimeError, "old settings"):
                    self.item._validate_settings({**previous, key: False}, candidate=False)

    def test_candidate_adds_exact_typed_values_and_preserves_intersection(self):
        previous = old_settings()
        self.item.previous_settings = previous
        candidate = release.settings_for(previous, candidate=True)
        self.item._validate_settings(candidate, candidate=True)
        drift = dict(candidate, private="changed")
        with self.assertRaisesRegex(RuntimeError, "candidate effective settings"):
            self.item._validate_settings(drift, candidate=True)
        bool_as_int = dict(candidate, local_vad_speech_duration_ms=True)
        with self.assertRaisesRegex(RuntimeError, "candidate effective settings"):
            self.item._validate_settings(bool_as_int, candidate=True)

    def test_compose_and_container_comparisons_preserve_mount_counts_and_private_env(self):
        previous = compose_fixture(release.OLD_IMAGE)
        candidate = copy.deepcopy(previous)
        app = candidate["services"]["app"]
        app["image"] = "sha256:candidate"
        app["environment"].update(release.GATE_ENV)
        app["volumes"][-1]["source"] = str(self.item.settings_overlay)
        app["volumes"].reverse()
        self.assertEqual(
            release.normalized_compose(previous, candidate=False),
            release.normalized_compose(
                candidate,
                candidate=True,
                settings_source=self.item.settings_overlay,
            ),
        )
        duplicate = copy.deepcopy(candidate)
        non_settings = next(
            mount
            for mount in duplicate["services"]["app"]["volumes"]
            if mount["target"] != release.SETTINGS_TARGET
        )
        duplicate["services"]["app"]["volumes"].append(copy.deepcopy(non_settings))
        self.assertNotEqual(
            release.normalized_compose(previous, candidate=False),
            release.normalized_compose(
                duplicate,
                candidate=True,
                settings_source=self.item.settings_overlay,
            ),
        )
        old = container_fixture(
            release.OLD_IMAGE,
            release.DEPLOYED_SOURCE,
            candidate=False,
            settings_source="/old/settings.py",
        )
        new = container_fixture(
            "sha256:candidate",
            COMMIT,
            candidate=True,
            settings_source=str(self.item.settings_overlay),
        )
        self.assertEqual(
            release.container_contract(old, candidate=False),
            release.container_contract(
                new, candidate=True, settings_source=self.item.settings_overlay
            ),
        )
        new["Config"]["Env"][0] = "PRIVATE_VALUE=changed"
        self.assertNotEqual(
            release.container_contract(old, candidate=False),
            release.container_contract(
                new, candidate=True, settings_source=self.item.settings_overlay
            ),
        )

    def test_hidden_candidate_paths_allow_only_nested_settings_exception(self):
        harmless = [
            {"Destination": "/app/config", "Source": "/private/config"},
            {"Destination": release.SETTINGS_TARGET, "Source": "/private/settings.py"},
        ]
        self.assertEqual(release.hidden_candidate_paths(harmless), ())
        hidden = release.hidden_candidate_paths(
            [{"target": "/app/models", "source": "/old/models"}]
        )
        self.assertIn("models/vad/silero_vad.onnx", hidden)

    def test_capacity_rejects_low_disk_memory_and_measured_four_worker_budget(self):
        with mock.patch.object(
            release.shutil,
            "disk_usage",
            return_value=SimpleNamespace(free=release.MIN_FREE_DISK_BYTES - 1),
        ):
            with self.assertRaisesRegex(RuntimeError, "disk"):
                self.item.capacity(available_memory=release.MIN_PREFLIGHT_MEMORY_BYTES)
        with mock.patch.object(
            release.shutil,
            "disk_usage",
            return_value=SimpleNamespace(free=release.MIN_FREE_DISK_BYTES),
        ):
            with self.assertRaisesRegex(RuntimeError, "memory"):
                self.item.capacity(available_memory=1)
            self.item.capacity(available_memory=release.MIN_PREFLIGHT_MEMORY_BYTES)
            with self.assertRaisesRegex(RuntimeError, "four-worker"):
                self.item.capacity(
                    available_memory=release.POST_BUILD_RESERVE_BYTES + 4 * 64 * 1024**2 - 1,
                    measured_rss_delta=64 * 1024**2,
                )

    def test_candidate_tests_include_real_model_and_all_four_speech_directories(self):
        observed = {}

        def one_shot(name, image, arguments, **kwargs):
            observed.update(name=name, image=image, arguments=arguments, kwargs=kwargs)
            return completed()

        self.item.one_shot = one_shot
        self.item.run_candidate_tests("sha256:candidate")
        self.assertEqual(observed["kwargs"].get("network", "none"), "none")
        for suite in ("stt", "tts", "telephony", "conversation"):
            self.assertIn("/audit/tests/" + suite, observed["arguments"])
        self.assertIn("/audit/tests/evaluation/test_local_vad_audio.py", observed["arguments"])
        self.assertTrue(any("/audit/models" in value for value in observed["kwargs"]["mounts"]))

    def test_effective_probe_requires_true_flag_exact_packages_and_model_load(self):
        expected_hashes = {"runtime.py": "aaa", "models/vad/silero_vad.onnx": "bbb"}
        baseline = {"numpy": "1.26.4", "packaging": "25.0", "sympy": "1.14.0"}
        packages = {**baseline, **release.PACKAGE_VERSIONS}
        settings = release.settings_for(old_settings(), candidate=True)
        value = {
            "hashes": expected_hashes,
            "settings": settings,
            "packages": packages,
            "model_loaded": True,
        }
        release.validate_effective_probe(
            value,
            hashes=expected_hashes,
            settings=settings,
            packages=packages,
        )
        for key, bad in (
            ("model_loaded", False),
            ("settings", {**settings, "speech_aware_barge_in_enabled": False}),
        ):
            changed = dict(value, **{key: bad})
            with self.assertRaisesRegex(RuntimeError, "effective candidate"):
                release.validate_effective_probe(
                    changed,
                    hashes=expected_hashes,
                    settings=settings,
                    packages=packages,
                )

    def test_execute_does_no_asset_download_build_or_cutover_before_preflight(self):
        self.item.preflight = mock.Mock(side_effect=RuntimeError("preflight refusal"))
        self.item.stage_assets = mock.Mock()
        self.item.prepare_wheels = mock.Mock()
        self.item.build_candidate = mock.Mock()
        self.item.deploy = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "preflight refusal"):
            self.item.execute()
        self.item.stage_assets.assert_not_called()
        self.item.prepare_wheels.assert_not_called()
        self.item.build_candidate.assert_not_called()
        self.item.deploy.assert_not_called()
        self.assertEqual(self.item.stage, "preflight")

    def test_candidate_failure_stops_before_override_or_cutover(self):
        self.item.preflight = mock.Mock(return_value=compose_fixture(release.OLD_IMAGE))
        self.item.stage_assets = mock.Mock()
        self.item.prepare_wheels = mock.Mock()
        self.item.build_candidate = mock.Mock(return_value="sha256:candidate")
        self.item.run_candidate_tests = mock.Mock(side_effect=RuntimeError("test failure"))
        self.item.prepare_override = mock.Mock()
        self.item.deploy = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            self.item.execute()
        self.item.prepare_override.assert_not_called()
        self.item.deploy.assert_not_called()
        self.assertEqual(self.item.stage, "tests")

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
        name = "t033h-partial-" + self.path.name
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.item.one_shot(name, "sha256:image", ["-c", "pass"])
        self.assertIn(("docker", "rm", "-f", name), calls)

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
            "stop_failed": False,
        }
        self.item.recheck_before_stop = lambda: state["events"].append("recheck")

        def stop_components(*, strict):
            state["events"].append(("stop", strict, state["image"]))
            if failure == "stop" and not state["stop_failed"]:
                state["stop_failed"] = True
                raise RuntimeError("stop failure")
            state["ingress"] = False

        def install(path):
            state["events"].append(("install", path.name))
            if failure == "install" and path == candidate_path:
                raise RuntimeError("install failure")
            state["override"] = path.read_bytes()

        def replace(image, revision, *, candidate):
            state["events"].append(("replace", image, revision, candidate))
            if (failure == "replace" and candidate) or (recovery_failure and not candidate):
                raise RuntimeError("replace failure")
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
        self.item.force_close_ingress = lambda: state.update(ingress=False) or True
        return state, candidate_path, io.StringIO()

    def test_successful_cutover_records_candidate_and_marker(self):
        state, candidate, output = self._cutover()
        with redirect_stdout(output):
            self.item.deploy("sha256:candidate", candidate)
        self.assertEqual(state["image"], "sha256:candidate")
        self.assertTrue(state["ingress"])
        self.assertIn("SPEECH_AWARE_BARGE_IN_DEPLOYED_READY_HTTPS_OK", output.getvalue())

    def test_each_cutover_phase_failure_restores_exact_previous_state(self):
        for phase in ("stop", "install", "replace", "tls"):
            with self.subTest(phase=phase):
                other = new_release(self.path / phase)
                original = self.item
                self.item = other
                try:
                    state, candidate, output = self._cutover(failure=phase)
                    with redirect_stdout(output), self.assertRaises(RuntimeError):
                        self.item.deploy("sha256:candidate", candidate)
                    self.assertEqual(state["image"], release.OLD_IMAGE)
                    self.assertEqual(
                        state["override"],
                        (self.item.directory / "previous-override.json").read_bytes(),
                    )
                    self.assertTrue(state["ingress"])
                    self.assertNotIn(
                        "SPEECH_AWARE_BARGE_IN_DEPLOYED_READY_HTTPS_OK",
                        output.getvalue(),
                    )
                finally:
                    self.item = original

    def test_failed_recovery_keeps_ingress_closed_and_public_output_redacted(self):
        state, candidate, output = self._cutover(failure="replace", recovery_failure=True)
        with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "replace failure"):
            self.item.deploy("sha256:candidate", candidate)
        self.assertFalse(state["ingress"])
        text = output.getvalue()
        self.assertIn("RECOVERY_FAILED_INGRESS_CLOSED", text)
        self.assertIn("RECOVERY_UNVERIFIED_MANUAL_REVIEW_REQUIRED", text)
        self.assertNotIn(SECRET, text)


@unittest.skipUnless(
    os.environ.get("T033_H_RUN_DOCKER_TESTS") == "1",
    "set T033_H_RUN_DOCKER_TESTS=1 for the isolated Docker/Compose rehearsal",
)
class DockerRehearsalTests(unittest.TestCase):
    """Opt-in real-wheel, model, nested-mount, replacement and rollback rehearsal."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temporary.name)
        cls.name = "t033h-local-" + uuid.uuid4().hex[:10]
        cls.project = cls.name
        docker = os.environ.get("T033_H_DOCKER", shutil.which("docker") or "docker")
        cls.docker = [docker]
        context = os.environ.get("T033_H_DOCKER_CONTEXT")
        if context:
            cls.docker.extend(("--context", context))
        info = cls._host_run("info", "--format", "{{.OSType}}", timeout=30)
        if info.stdout.strip() != b"linux":
            raise AssertionError("T033-H rehearsal requires a Linux Docker engine")
        cls.base = os.environ.get("T033_H_BASE_IMAGE", "ai-phone-t049a-candidate:20260913")
        cls.base_id = cls._host_run(
            "image", "inspect", cls.base, "--format", "{{.Id}}", timeout=30
        ).stdout.decode().strip()
        cls.item = new_release(cls.path)
        cls.item.run = cls._run
        cls.item.assets.mkdir()
        manifest = json.loads((REPOSITORY / release.MANIFEST_PATH).read_text(encoding="utf-8"))
        for section, paths in release.SECTION_PATHS.items():
            for relative in paths:
                target = cls.item.assets / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                data = (REPOSITORY / relative).read_bytes()
                target.write_bytes(
                    data if section == "model_assets" else release.normalized_text_bytes(data)
                )
        cls.item.manifest = manifest
        cls.item.runtime_hashes = dict(manifest["runtime"])
        cls.item.protected_runtime_hashes = dict(manifest["protected_runtime"])
        cls.item.model_hashes = dict(manifest["model_assets"])
        cls.item.platform = {
            "implementation": "cpython",
            "python": "3.12.14",
            "machine": "x86_64",
            "system": "Linux",
            "libc_name": "glibc",
            "libc_version": "2.36",
            "compatible": {
                wheel["filename"]: True for wheel in release.WHEELS.values()
            },
        }
        cls.old_image = release.OLD_IMAGE
        try:
            baseline = cls.path / "baseline"
            baseline.mkdir()
            cls.baseline = baseline
            baseline_paths = (
                "config/settings.py",
                "services/conversation/orchestrator.py",
                *release.PROTECTED_RUNTIME_PATHS,
            )
            git = os.environ.get("T033_H_GIT", shutil.which("git") or "git")
            for relative in baseline_paths:
                data = __import__("subprocess").run(
                    [git, "show", release.DEPLOYED_SOURCE + ":" + relative],
                    cwd=REPOSITORY,
                    capture_output=True,
                    check=True,
                    timeout=30,
                ).stdout
                target = baseline / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(release.normalized_text_bytes(data))
            cls._run("docker", "tag", cls.base_id, cls.name + "-seed")
            (baseline / "Dockerfile").write_text(
                "FROM " + cls.name + "-seed\n"
                + "".join(
                    "COPY " + relative + " /app/" + relative + "\n"
                    for relative in baseline_paths
                ),
                encoding="utf-8",
            )
            cls._run(
                "docker",
                "build",
                "--pull=false",
                "--network",
                "none",
                "--label",
                "org.opencontainers.image.revision=" + release.DEPLOYED_SOURCE,
                "-t",
                cls.name + "-baseline",
                str(baseline),
                timeout=180,
            )
            cls.base_id = cls._run(
                "docker", "image", "inspect", cls.name + "-baseline", "--format", "{{.Id}}"
            ).stdout.decode().strip()
            release.OLD_IMAGE = cls.base_id
            cls.item.baseline_packages = cls.item.package_inventory(cls.base_id)
            cls.item.prepare_wheels(cls.base_id)
            cls.candidate = cls.item.build_candidate()
        except BaseException:
            cls._cleanup(False)
            release.OLD_IMAGE = cls.old_image
            cls.temporary.cleanup()
            raise

    @classmethod
    def _host_run(cls, *arguments, timeout=60, check=True):
        import subprocess

        result = subprocess.run(
            [*cls.docker, *arguments],
            cwd=REPOSITORY,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if check and result.returncode:
            raise RuntimeError((result.stdout + result.stderr).decode(errors="replace"))
        return result

    @classmethod
    def _run(cls, *arguments, timeout=60, check=True):
        if arguments[0] != "docker":
            raise RuntimeError("unexpected local release command")
        return cls._host_run(*arguments[1:], timeout=timeout, check=check)

    @classmethod
    def _cleanup(cls, verify=True):
        compose = cls.path / "compose.json"
        if compose.exists():
            cls._host_run(
                "compose",
                "-p",
                cls.project,
                "-f",
                str(compose),
                "down",
                "--remove-orphans",
                timeout=60,
                check=False,
            )
        for name in (
            cls.name + "-app",
            "t033h-platform-" + cls.path.name,
            "t033h-packages-" + cls.path.name,
            "t033h-wheels-" + cls.path.name,
            "t033h-image-probe-" + cls.path.name,
            "t033h-tests-" + cls.path.name,
            "t033h-warm-" + cls.path.name,
        ):
            cls._host_run("rm", "-f", name, timeout=30, check=False)
        for image in (
            "ai-phone-t033h-candidate:" + cls.path.name,
            "ai-phone-t033h-parent:" + cls.path.name,
            cls.name + "-baseline",
            cls.name + "-seed",
        ):
            cls._host_run("image", "rm", "-f", image, timeout=60, check=False)
        if verify:
            inventory = "\n".join(
                cls._host_run(*command, timeout=30).stdout.decode()
                for command in (
                    ("container", "ls", "-a", "--format", "{{.Names}}"),
                    ("image", "ls", "--format", "{{.Repository}}:{{.Tag}}"),
                )
            )
            if cls.name in inventory or cls.path.name in inventory:
                raise AssertionError("T033-H Docker resources were not removed")

    @classmethod
    def tearDownClass(cls):
        try:
            cls._cleanup()
        finally:
            release.OLD_IMAGE = cls.old_image
            cls.temporary.cleanup()

    def test_real_candidate_model_tests_effective_load_and_exact_image_rollback(self):
        self.item.run_candidate_tests(self.candidate)
        # The release runs on Linux; this driver may run on Windows. Read the
        # actual Docker Linux host's memory rather than Windows /proc.
        def linux_available_memory():
            result = self.item.one_shot(
                "t033h-memory-" + self.path.name,
                self.candidate,
                ["-c", "from pathlib import Path; print(next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))"],
            )
            return int(result.stdout.strip())

        self.item.available_memory = linux_available_memory
        measurement = self.item.measure_candidate(self.candidate)
        print("LOCAL_LINUX_MODEL_MEASUREMENT=" + json.dumps(measurement, sort_keys=True))
        self.assertTrue(measurement["model_loaded"])
        self.assertGreater(measurement["rss_delta_bytes"], 0)
        self.assertLess(measurement["p95_ms"], 20.0)
        packages = self.item.package_inventory(self.candidate)
        self.item.verify_package_transition(self.item.baseline_packages, packages)
        restored = self._run(
            "docker", "image", "inspect", self.base_id, "--format", "{{.Id}}"
        ).stdout.decode().strip()
        self.assertEqual(restored, self.base_id)

        parent_config = self.path / "parent-config"
        shutil.copytree(REPOSITORY / "config", parent_config)
        (parent_config / "settings.py").write_text(
            "raise RuntimeError('parent settings must stay hidden')\n",
            encoding="utf-8",
        )
        old_settings_path = self.path / "old-settings.py"
        old_settings_path.write_bytes(
            (self.baseline / "config/settings.py").read_bytes()
        )
        candidate_settings_path = self.path / "candidate-settings.py"
        candidate_settings_path.write_bytes(
            (REPOSITORY / "config/settings.py").read_bytes()
        )
        service = {
            "services": {
                "app": {
                    "image": self.base_id,
                    "container_name": self.name + "-app",
                    "entrypoint": ["python", "-c"],
                    "command": ["import time; time.sleep(300)"],
                    "environment": dict(
                        value.split("=", 1) for value in release.SYNTHETIC_ENV
                    ),
                    "volumes": [
                        {
                            "type": "bind",
                            "source": str(parent_config),
                            "target": "/app/config",
                            "read_only": True,
                        },
                        {
                            "type": "bind",
                            "source": str(old_settings_path),
                            "target": release.SETTINGS_TARGET,
                            "read_only": True,
                        },
                    ],
                    "networks": ["isolated"],
                }
            },
            "networks": {"isolated": {"internal": True}},
        }
        compose = self.path / "compose.json"

        def up(image: str, *, candidate: bool):
            value = copy.deepcopy(service)
            app = value["services"]["app"]
            app["image"] = image
            if candidate:
                app["environment"].update(release.GATE_ENV)
                app["volumes"][-1]["source"] = str(candidate_settings_path)
            compose.write_text(json.dumps(value), encoding="utf-8")
            self._run(
                "docker",
                "compose",
                "-p",
                self.project,
                "-f",
                str(compose),
                "up",
                "-d",
                "--no-deps",
                "--no-build",
                "--pull",
                "never",
                "--force-recreate",
                "app",
                timeout=90,
            )
            return json.loads(
                self._run("docker", "inspect", self.name + "-app").stdout
            )[0]

        old = up(self.base_id, candidate=False)
        current = up(self.candidate, candidate=True)
        text_hashes = {
            **self.item.runtime_hashes,
            **self.item.protected_runtime_hashes,
        }
        probe = release.strict_json(
            self._run(
                "docker",
                "exec",
                self.name + "-app",
                "python",
                "-c",
                release.source_probe_code(text_hashes, self.item.model_hashes),
            ).stdout
        )
        self.assertTrue(probe["settings"]["speech_aware_barge_in_enabled"])
        self.assertFalse(
            probe["settings"]["elevenlabs_stt_filter_background_audio"]
        )
        self.assertTrue(probe["model_loaded"])
        self.assertEqual(probe["packages"], packages)
        self.assertTrue(current["State"]["Running"])

        rolled_back = up(self.base_id, candidate=False)
        self.assertEqual(old["Image"], self.base_id)
        self.assertEqual(current["Image"], self.candidate)
        self.assertEqual(rolled_back["Image"], self.base_id)
        self.assertEqual(
            release.container_contract(old, candidate=False),
            release.container_contract(rolled_back, candidate=False),
        )
        restored_settings = release.strict_json(
            self._run(
                "docker",
                "exec",
                self.name + "-app",
                "python",
                "-c",
                release.SETTINGS_CODE,
            ).stdout
        )
        for key in release.GATE_SETTINGS:
            self.assertNotIn(key, restored_settings)


if __name__ == "__main__":
    unittest.main()
