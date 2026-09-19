"""Behavioral and opt-in Docker rehearsal for the T020-C release procedure."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock
import uuid


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "scripts" / "deploy-verified-phone-booking.py"
SPEC = importlib.util.spec_from_file_location("deploy_verified_phone_booking", SCRIPT)
release = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(release)


def completed(stdout=b"", returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout, b"")


class ManifestAndPureContractTests(unittest.TestCase):
    def test_manifest_is_exact_and_all_108_inputs_match(self):
        raw = (REPOSITORY / release.MANIFEST_PATH).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), release.MANIFEST_SHA256)
        manifest = json.loads(raw)
        entries = []
        for section in release.SECTION_PATHS:
            self.assertEqual(tuple(manifest[section]), release.SECTION_PATHS[section])
            for relative, expected in manifest[section].items():
                data = (REPOSITORY / relative).read_bytes()
                self.assertEqual(release.asset_digest(section, data), expected)
                entries.append(relative)
        self.assertEqual(len(entries), 108)
        self.assertEqual(len(entries), len(set(entries)))
        self.assertEqual(tuple(manifest["source_scope"]), release.SOURCE_SCOPE)

    def test_text_normalization_and_model_raw_bytes_are_distinct(self):
        self.assertEqual(release.normalized_text_bytes(b"a\r\nb\n"), b"a\nb\n")
        self.assertEqual(
            release.asset_digest("runtime", b"a\r\nb\n"),
            release.asset_digest("runtime", b"a\nb\n"),
        )
        self.assertNotEqual(
            release.asset_digest("model_assets", b"a\r\nb\n"),
            release.asset_digest("model_assets", b"a\nb\n"),
        )

    def test_strict_json_rejects_duplicate_keys_and_trailing_data(self):
        with self.assertRaises(ValueError):
            release.strict_json(b'{"a":1,"a":2}')
        with self.assertRaises(ValueError):
            release.strict_json(b'{"a":1} trailing')
        self.assertEqual(release.strict_json(b'{"a":1}'), {"a": 1})

    def test_semantic_fingerprints_ignore_order_and_preserve_duplicates(self):
        first = [{"target": "/a", "source": "x"}, {"target": "/b", "source": "y"}]
        second = list(reversed(first))
        self.assertEqual(release.mount_fingerprint(first), release.mount_fingerprint(second))
        self.assertNotEqual(
            release.mount_fingerprint(first),
            release.mount_fingerprint(first + [first[0]]),
        )
        env = ["A=1", "B=2", "A=1"]
        self.assertEqual(release.environment_fingerprint(env), sorted(env))
        self.assertNotEqual(
            release.environment_fingerprint(env),
            release.environment_fingerprint(["A=1", "B=2"]),
        )

    def test_settings_variants_change_only_reviewed_flags(self):
        previous = {
            "barge_in_diagnostics_enabled": True,
            "speech_aware_barge_in_enabled": True,
            "elevenlabs_stt_filter_background_audio": False,
            "automatic_notifications_enabled": True,
            "automatic_notification_workers_paused": False,
            "database_url": "postgresql://private.invalid/db",
        }
        paused = release.settings_for(previous, candidate=True, paused=True)
        active = release.settings_for(previous, candidate=True, paused=False)
        old = release.settings_for(previous, candidate=False, paused=None)
        self.assertTrue(paused[release.VERIFIED_SETTING])
        self.assertTrue(paused[release.PAUSED_SETTING])
        self.assertFalse(active[release.PAUSED_SETTING])
        self.assertEqual(old, previous)
        for key in previous:
            if key != release.PAUSED_SETTING:
                self.assertEqual(paused[key], previous[key])

    def test_settings_reject_preexisting_verified_enablement_or_voice_drift(self):
        previous = dict(release.REQUIRED_BASELINE_SETTINGS)
        previous[release.VERIFIED_SETTING] = True
        with self.assertRaisesRegex(RuntimeError, "verified phone setting"):
            release.settings_for(previous, candidate=True, paused=True)
        previous = dict(release.REQUIRED_BASELINE_SETTINGS)
        previous["speech_aware_barge_in_enabled"] = False
        with self.assertRaisesRegex(RuntimeError, "baseline voice"):
            release.settings_for(previous, candidate=True, paused=True)

    def test_hidden_candidate_paths_detect_parent_and_missing_exact_overlays(self):
        expected = {
            release.SETTINGS_TARGET: "/protected/settings.py",
            release.ACCOUNTANTS_TARGET: "/protected/accountants.yaml",
            release.POLICY_TARGET: "/protected/booking-policy.yaml",
        }
        mounts = [
            {"type": "bind", "source": source, "target": target, "read_only": True}
            for target, source in expected.items()
        ]
        self.assertEqual(release.hidden_candidate_paths(mounts, expected), ())
        self.assertIn("/app", release.hidden_candidate_paths(
            mounts + [{"type": "bind", "source": "/tmp", "target": "/app"}], expected))
        self.assertIn(release.POLICY_TARGET, release.hidden_candidate_paths(mounts[:-1], expected))

    def test_source_scope_requires_exact_paths_without_duplicates(self):
        release.validate_source_scope(list(release.SOURCE_SCOPE))
        with self.assertRaisesRegex(RuntimeError, "scope"):
            release.validate_source_scope(list(release.SOURCE_SCOPE) + ["requirements.txt"])
        with self.assertRaisesRegex(RuntimeError, "scope"):
            release.validate_source_scope(list(release.SOURCE_SCOPE) + [release.SOURCE_SCOPE[0]])

    def test_candidate_test_selection_is_explicit_and_nonempty(self):
        paths = release.candidate_test_paths()
        self.assertGreater(len(paths), 20)
        self.assertEqual(len(paths), len(set(paths)))
        self.assertIn("tests/integration/test_phone_booking_flow.py", paths)
        self.assertIn("tests/llm/test_tool_protocol.py", paths)

    def test_embedded_database_probes_compile_before_cutover(self):
        compile(release.SCHEMA_PROBE_CODE, "schema-probe", "exec")
        compile(release.FINGERPRINT_CODE, "fingerprint-probe", "exec")
        self.assertEqual(release.EXPECTED_HISTORY[-1], (
            "bootstrap-v1", "1c72bbe0a120860902c565e5573c05ad1cd640891a42bb35cca2c6e0876f31e8"))

    def test_embedded_schema_probe_accepts_legacy_and_bootstrapped_histories_only(self):
        for operations in (False, True):
            for bootstrap in (False, True):
                for corrupt in (False, True):
                    with self.subTest(operations=operations, bootstrap=bootstrap, corrupt=corrupt):
                        history = [dict(version=v, checksum=c) for v, c in release.EXPECTED_HISTORY
                                   if (operations or v != "0005") and (bootstrap or v != "bootstrap-v1")]
                        if corrupt:
                            history[0]["checksum"] = "0" * 64
                        conn = mock.Mock()
                        conn.transaction.return_value = mock.MagicMock()
                        conn.transaction.return_value.__aenter__ = mock.AsyncMock()
                        conn.transaction.return_value.__aexit__ = mock.AsyncMock(return_value=False)
                        conn.fetch = mock.AsyncMock(side_effect=[history, (
                            [{"tablename": "booking_operations"}] if operations else [])])
                        conn.fetchval = mock.AsyncMock(side_effect=[0, 1, True] if operations else [True])
                        conn.fetchrow = mock.AsyncMock(return_value={"installed_version": None})
                        conn.close = mock.AsyncMock()
                        with mock.patch("asyncpg.connect", mock.AsyncMock(return_value=conn)), \
                                mock.patch("migrations.schema_contract.check_runtime_compatibility", mock.AsyncMock()), \
                                mock.patch.dict(os.environ, {"MIGRATION_DATABASE_URL": "synthetic"}), \
                                contextlib.redirect_stdout(io.StringIO()):
                            if corrupt:
                                with self.assertRaisesRegex(RuntimeError, "history"):
                                    exec(release.SCHEMA_PROBE_CODE, {})
                            else:
                                exec(release.SCHEMA_PROBE_CODE, {})


class ReleaseObjectTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)
        self.item = release.Release(self.path, "a" * 40)
        self.item.root = REPOSITORY

    def tearDown(self):
        self.temporary.cleanup()

    def test_private_write_is_exclusive_root_only_and_rejects_symlink(self):
        target = self.item.private_write("private.json", b"secret")
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
        with self.assertRaises(FileExistsError):
            self.item.private_write("private.json", b"replacement")
        link = self.path / "link"
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest("symlink creation unavailable")
        with self.assertRaises(RuntimeError):
            self.item.private_write("link", b"data")

    def test_stage_assets_reads_every_entry_from_pinned_commit(self):
        calls = []

        def git_bytes(relative):
            calls.append(relative)
            return (REPOSITORY / relative).read_bytes()

        self.item.git_bytes = git_bytes
        self.item.stage_assets()
        self.assertEqual(len(calls), 109 + len(release.TEST_ONLY_HASHES))
        self.assertEqual(calls[0], release.MANIFEST_PATH)
        self.assertEqual(len(set(calls[1:])), 108 + len(release.TEST_ONLY_HASHES))
        self.assertEqual(
            (self.item.assets / "services/conversation/orchestrator.py").read_bytes(),
            release.normalized_text_bytes(
                (REPOSITORY / "services/conversation/orchestrator.py").read_bytes()),
        )
        for relative in release.TEST_ONLY_HASHES:
            self.assertEqual(
                (self.item.test_inputs / relative).read_bytes(),
                release.normalized_text_bytes((REPOSITORY / relative).read_bytes()),
            )

    def test_test_only_inputs_are_independently_pinned(self):
        self.item.git_bytes = lambda relative: (REPOSITORY / relative).read_bytes()
        original = release.TEST_ONLY_HASHES[".dockerignore"]
        with mock.patch.dict(release.TEST_ONLY_HASHES, {".dockerignore": "0" * 64}):
            with self.assertRaisesRegex(RuntimeError, "test-only input hash"):
                self.item.stage_assets()
        self.assertEqual(release.TEST_ONLY_HASHES[".dockerignore"], original)

    def test_stage_assets_rejects_one_hash_drift_without_refreshing_manifest(self):
        def git_bytes(relative):
            data = (REPOSITORY / relative).read_bytes()
            if relative == "services/llm/tool_protocol.py":
                return data + b"\n# drift"
            return data

        self.item.git_bytes = git_bytes
        with self.assertRaisesRegex(RuntimeError, "reviewed asset hash"):
            self.item.stage_assets()

    def test_post_stage_verification_rechecks_pinned_and_staged_inputs(self):
        self.item.git_bytes = lambda relative: (REPOSITORY / relative).read_bytes()
        self.item.stage_assets()
        self.item.verify_frozen_inputs()
        staged = self.item.assets / "services/llm/tool_protocol.py"
        staged.write_bytes(staged.read_bytes() + b"\n# changed")
        with self.assertRaisesRegex(RuntimeError, "staged input"):
            self.item.verify_frozen_inputs()

    def test_normalized_compose_allows_only_images_flags_and_controlled_overlays(self):
        baseline = {
            "services": {
                "app": {"image": "old", "environment": {"A": "1"}, "volumes": [
                    {"type": "bind", "source": "/private/config", "target": "/app/config",
                     "read_only": True},
                ], "networks": ["z", "a"]},
                "migrate": {"image": "old"},
            },
            "networks": {"a": {"name": "a"}, "z": {"name": "z"}},
        }
        candidate = json.loads(json.dumps(baseline))
        candidate["services"]["app"]["image"] = "candidate"
        candidate["services"]["migrate"]["image"] = "candidate"
        candidate["services"]["app"]["environment"].update({
            release.VERIFIED_ENV: "true", release.PAUSED_ENV: "true"})
        overlays = {
            release.SETTINGS_TARGET: str(self.path / "settings.py"),
            release.ACCOUNTANTS_TARGET: str(self.path / "accountants.yaml"),
            release.POLICY_TARGET: str(self.path / "booking-policy.yaml"),
        }
        candidate["services"]["app"]["volumes"].extend(
            {"type": "bind", "source": source, "target": target, "read_only": True}
            for target, source in overlays.items())
        self.assertEqual(
            release.normalized_compose(candidate, candidate=True, paused=True,
                                       overlay_sources=overlays),
            release.normalized_compose(baseline, candidate=False, paused=None,
                                       overlay_sources={}),
        )
        candidate["services"]["app"]["ports"] = ["9999:9999"]
        self.assertNotEqual(
            release.normalized_compose(candidate, candidate=True, paused=True,
                                       overlay_sources=overlays),
            release.normalized_compose(baseline, candidate=False, paused=None,
                                       overlay_sources={}),
        )

    def test_predecessor_false_flags_normalize_but_true_verified_is_rejected(self):
        baseline = {"services": {"app": {
            "image": "old",
            "environment": {release.VERIFIED_ENV: "false", release.PAUSED_ENV: "false"},
            "volumes": [{"type": "bind", "source": "/old/settings.py",
                         "target": release.SETTINGS_TARGET, "read_only": True}],
        }}}
        normalized = release.normalized_compose(
            baseline, candidate=False, paused=None, overlay_sources={})
        self.assertEqual(normalized["services"]["app"]["environment"], {})
        self.assertEqual(normalized["services"]["app"]["volumes"], [])
        baseline["services"]["app"]["environment"][release.VERIFIED_ENV] = "true"
        with self.assertRaisesRegex(RuntimeError, "verified phone"):
            release.normalized_compose(
                baseline, candidate=False, paused=None, overlay_sources={})

    def test_candidate_overlay_validation_rejects_wrong_source_write_and_duplicate(self):
        sources = {target: str(self.path / (str(index) + ".asset"))
                   for index, target in enumerate(release.OVERLAY_TARGETS)}
        mounts = [{"type": "bind", "source": source, "target": target,
                   "read_only": True} for target, source in sources.items()]
        self.assertEqual(release.hidden_candidate_paths(mounts, sources), ())
        wrong = json.loads(json.dumps(mounts))
        wrong[0]["source"] = "/wrong"
        self.assertIn(wrong[0]["target"], release.hidden_candidate_paths(wrong, sources))
        writable = json.loads(json.dumps(mounts))
        writable[1]["read_only"] = False
        self.assertIn(writable[1]["target"], release.hidden_candidate_paths(writable, sources))
        duplicate = mounts + [dict(mounts[2])]
        self.assertIn(mounts[2]["target"], release.hidden_candidate_paths(duplicate, sources))

    def test_candidate_test_invocation_mounts_every_pinned_repository_input(self):
        self.item.assets = REPOSITORY
        self.item.test_inputs = REPOSITORY
        self.item.one_shot = mock.Mock(return_value=completed(b"909 tests collected"))
        self.item.run_candidate_tests("sha256:synthetic")
        targets = {target for _source, target, _readonly
                   in self.item.one_shot.call_args_list[0].kwargs["mounts"]}
        self.assertEqual(
            {"/app/.dockerignore", "/app/Dockerfile", "/app/docker-compose.yml",
             "/app/docs", "/app/templates/dashboard.html"} - targets,
            set(),
        )

    def test_live_pre_stop_polling_is_allowed_and_post_stop_change_is_rejected(self):
        baseline = {"booking_provider_observations": {"count": 1, "hash": "a" * 32}}
        advanced = {"booking_provider_observations": {"count": 1, "hash": "b" * 32}}
        (self.path / "final-fingerprint.json").write_text(json.dumps(baseline))
        self.item.candidate_image = "sha256:candidate"
        self.item.fingerprint = lambda _image: advanced
        with self.assertRaisesRegex(RuntimeError, "preexisting business rows"):
            self.item.verify_live_fingerprints()

    def test_cleanup_removes_only_registered_resources(self):
        calls = []
        self.item.run = lambda *args, **kwargs: calls.append(args) or completed()
        self.item.owned_containers.update(("t020c-a", "t020c-b"))
        self.item.owned_networks.add("t020c-net")
        self.item.cleanup()
        self.assertEqual(
            {call[-1] for call in calls if call[:3] == ("docker", "rm", "-f")},
            {"t020c-a", "t020c-b"},
        )
        self.assertIn(("docker", "network", "rm", "t020c-net"), calls)
        self.assertFalse(any("jordan-repairs" in str(call) for call in calls))

    def test_journey_database_reserves_control_connections_for_twenty_way_cases(self):
        calls = []
        self.item.pg_image = "postgres:16-alpine"
        self.item.run = lambda *args, **kwargs: calls.append(args) or completed()
        self.item._create_rehearsal_db("journeys")
        create = next(call for call in calls if call[:3] == ("docker", "create", "--pull"))
        self.assertIn("max_connections=32", create)

    def test_readiness_waits_for_endpoint_and_container_health(self):
        states = iter(("starting", "healthy"))
        inspections = []
        self.item.run = lambda *args, **kwargs: completed(b'{"status":"ready"}')
        def inspect(_name):
            inspections.append(_name)
            return {"State": {"Health": {"Status": next(states)}}}
        self.item.inspect_once = inspect
        with mock.patch.object(release.time, "sleep"):
            self.item.wait_ready("synthetic-app")
        self.assertEqual(inspections, ["synthetic-app", "synthetic-app"])

    def test_database_journey_program_rejects_skipped_required_cases(self):
        tests = self.path / "candidate-tests/tests"
        tests.mkdir(parents=True)
        self.item.one_shot = mock.Mock()
        self.item.run_database_journeys(
            "sha256:synthetic", "synthetic-network", self.path / "synthetic.env")
        program = (self.path / "database-journeys.py").read_text(encoding="utf-8")
        self.assertIn("result.skipped", program)


class RecoveryStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)
        self.item = release.Release(self.path, "a" * 40)
        self.previous = self.path / "previous-override.json"
        self.paused = self.path / "paused-override.json"
        self.active = self.path / "active-override.json"
        for path in (self.previous, self.paused, self.active):
            path.write_text(path.name, encoding="utf-8")
        self.events = []
        self.schema = "0004"
        self.ingress = True
        self.item.recheck_before_stop = lambda: self.events.append("recheck")
        self.item.stop_writers = lambda strict: self.events.append(("stop", strict)) or setattr(self, "ingress", False)
        def backup(name):
            self.events.append(("backup", name))
            target = self.path / (name + ".dump")
            target.write_bytes(b"synthetic archive")
            return target
        self.item.backup = backup
        self.item.apply_live_migration = lambda: self.events.append("migrate") or setattr(self, "schema", "0005")
        self.item.live_schema_version = lambda: self.schema
        self.item.install_override = lambda path: self.events.append(("install", path.name))
        self.item.replace_app = lambda image, *, paused: self.events.append(("replace", image, paused))
        self.item.replace_recovery_app = lambda image: self.item.replace_app(image, paused=True)
        self.item.verify_live_fingerprints = lambda: self.events.append("fingerprints")
        self.item.fingerprint = lambda _image: {"bookings": {"count": 1, "hash": "a" * 32}}
        self.item.start_ingress = lambda: self.events.append("ingress") or setattr(self, "ingress", True)
        self.item.force_close_ingress = lambda: setattr(self, "ingress", False) or True
        self.item.mark = lambda name, value=b"": self.events.append(("mark", name))
        self.item.restore_previous_app = lambda: (
            self.events.append(("install", "previous-override.json")),
            setattr(self, "ingress", True),
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _fail(self, stage):
        original = {
            "backup": self.item.backup,
            "migrate": self.item.apply_live_migration,
            "paused": self.item.replace_app,
            "active": self.item.replace_app,
            "ingress": self.item.start_ingress,
        }
        if stage == "backup":
            self.item.backup = lambda _name: (_ for _ in ()).throw(RuntimeError("PRIVATE backup"))
        elif stage == "migrate":
            self.item.apply_live_migration = lambda: (_ for _ in ()).throw(RuntimeError("PRIVATE migrate"))
        elif stage in ("paused", "active"):
            def replace(image, *, paused):
                if paused == (stage == "paused"):
                    raise RuntimeError("PRIVATE replacement")
                return original[stage](image, paused=paused)
            self.item.replace_app = replace
        else:
            self.item.start_ingress = lambda: (_ for _ in ()).throw(RuntimeError("PRIVATE TLS"))

    def test_success_stops_ingress_migrates_paused_then_active_and_restores_ingress(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.item.deploy("sha256:candidate", self.paused, self.active)
        self.assertEqual(self.schema, "0005")
        self.assertTrue(self.ingress)
        self.assertLess(self.events.index("migrate"),
                        self.events.index(("replace", "sha256:candidate", True)))
        self.assertLess(self.events.index(("replace", "sha256:candidate", True)),
                        self.events.index(("replace", "sha256:candidate", False)))
        self.assertIn("VERIFIED_PHONE_BOOKING_DEPLOYED_READY_HTTPS_OK", output.getvalue())

    def test_backup_failure_restores_previous_without_migrating(self):
        self._fail("backup")
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "PRIVATE backup"):
            self.item.deploy("sha256:candidate", self.paused, self.active)
        self.assertEqual(self.schema, "0004")
        self.assertTrue(self.ingress)
        self.assertIn(("install", "previous-override.json"), self.events)
        self.assertNotIn("migrate", self.events)
        self.assertNotIn("PRIVATE", output.getvalue())

    def test_transaction_failure_uses_durable_schema_inspection(self):
        self._fail("migrate")
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "PRIVATE migrate"):
            self.item.deploy("sha256:candidate", self.paused, self.active)
        self.assertEqual(self.schema, "0004")
        self.assertTrue(self.ingress)
        self.assertIn("RECOVERY_PRE_0005_PREVIOUS_APP_READY_HTTPS_OK", output.getvalue())

    def test_uncertain_migration_outcome_recovers_from_observed_0005_state(self):
        def uncertain_migration():
            self.schema = "0005"
            raise RuntimeError("PRIVATE uncertain migration")

        self.item.apply_live_migration = uncertain_migration
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaisesRegex(
                RuntimeError, "PRIVATE uncertain migration"):
            self.item.deploy("sha256:candidate", self.paused, self.active)
        self.assertFalse(self.ingress)
        self.assertIn(("replace", "sha256:candidate", True), self.events)
        self.assertIn("RECOVERY_0005_CANDIDATE_PAUSED_INGRESS_STOPPED",
                      output.getvalue())

    def test_failure_after_0005_keeps_candidate_paused_and_ingress_stopped(self):
        self._fail("active")
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "PRIVATE replacement"):
            self.item.deploy("sha256:candidate", self.paused, self.active)
        self.assertEqual(self.schema, "0005")
        self.assertFalse(self.ingress)
        self.assertEqual(self.events[-2], ("replace", "sha256:candidate", True))
        self.assertIn("RECOVERY_0005_CANDIDATE_PAUSED_INGRESS_STOPPED", output.getvalue())
        self.assertNotIn("PRIVATE", output.getvalue())

    def test_late_health_failure_retains_0005_candidate_and_closes_ingress(self):
        self._fail("ingress")
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "PRIVATE TLS"):
            self.item.deploy("sha256:candidate", self.paused, self.active)
        self.assertEqual(self.schema, "0005")
        self.assertFalse(self.ingress)
        self.assertIn(("replace", "sha256:candidate", True), self.events)
        self.assertIn("RECOVERY_0005_CANDIDATE_PAUSED_INGRESS_STOPPED", output.getvalue())

    def test_failed_post_0005_recovery_reports_closed_ingress_without_masking_error(self):
        attempts = 0

        def failed_replacement(_image, *, paused):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("PRIVATE initial replacement" if attempts == 1
                               else "PRIVATE recovery")

        self.item.replace_app = failed_replacement
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaisesRegex(
                RuntimeError, "PRIVATE initial replacement"):
            self.item.deploy("sha256:candidate", self.paused, self.active)
        self.assertFalse(self.ingress)
        self.assertIn("RECOVERY_FAILED_INGRESS_CLOSED", output.getvalue())
        self.assertIn("RECOVERY_UNVERIFIED_MANUAL_REVIEW_REQUIRED", output.getvalue())
        self.assertNotIn("PRIVATE", output.getvalue())


@unittest.skipUnless(
    os.environ.get("T020_C_RUN_DOCKER_TESTS") == "1",
    "set T020_C_RUN_DOCKER_TESTS=1 for the isolated PostgreSQL/Compose rehearsal",
)
class DockerRehearsalTests(unittest.TestCase):
    """Real PostgreSQL 16 backup/restore/0005 and derived-image mount rehearsal."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name)
        cls.name = "t020c-local-" + uuid.uuid4().hex[:10]
        cls.source = cls.name + "-source"
        cls.restore = cls.name + "-restore"
        cls.network = cls.name + "-net"
        cls.image = "postgres:16-alpine"
        cls.images = set()
        cls.project = cls.name + "-compose"
        cls.docker("network", "create", "--internal", cls.network)
        for name in (cls.source, cls.restore):
            cls.docker("create", "--name", name, "--network", cls.network,
                       "--memory", "256m", "--cpus", "1", "--pids-limit", "128",
                       "-e", "POSTGRES_USER=t020c", "-e", "POSTGRES_PASSWORD=synthetic",
                       "-e", "POSTGRES_DB=t020c", cls.image,
                       "-c", "shared_buffers=32MB", "-c", "max_connections=20")
            cls.docker("start", name)
            for _ in range(40):
                result = cls.docker("exec", name, "pg_isready", "-U", "t020c",
                                    check=False)
                if result.returncode == 0:
                    break
                __import__("time").sleep(.25)
            else:
                raise RuntimeError("PostgreSQL rehearsal readiness failed")

    @classmethod
    def docker(cls, *args, check=True, input=None, timeout=120):
        result = subprocess.run(["docker", *args], cwd=REPOSITORY, input=input,
                                capture_output=True, timeout=timeout, check=False)
        if check and result.returncode:
            raise RuntimeError((result.stdout + result.stderr).decode(errors="replace"))
        return result

    @classmethod
    def tearDownClass(cls):
        try:
            compose = cls.path / "compose.json"
            if compose.exists():
                cls.docker("compose", "-p", cls.project, "-f", str(compose),
                           "down", "--remove-orphans", check=False)
            for name in (cls.source, cls.restore):
                cls.docker("rm", "-f", "-v", name, check=False)
            cls.docker("network", "rm", cls.network, check=False)
            for image in cls.images:
                cls.docker("image", "rm", "-f", image, check=False)
            names = cls.docker("ps", "-a", "--format", "{{.Names}}").stdout.decode().splitlines()
            if cls.source in names or cls.restore in names or any(cls.name in name for name in names):
                raise AssertionError("T020-C rehearsal containers remain")
        finally:
            cls.temp.cleanup()

    def test_real_0004_backup_restore_0005_repeat_and_preservation(self):
        # The deployment script's opt-in rehearsal is exercised end to end with
        # synthetic data and no provider network.
        release.run_local_migration_rehearsal(
            repository=REPOSITORY,
            source_container=self.source,
            restore_container=self.restore,
            archive=self.path / "schema0004.dump",
        )

    def test_real_derived_candidate_nested_mount_replacement_and_rollback(self):
        app_base = os.environ.get(
            "T020_C_REHEARSAL_APP_IMAGE", "ai-phone-t049a-candidate:20260913")
        speech_spec = importlib.util.spec_from_file_location(
            "t020c_speech_release", REPOSITORY / "scripts/deploy-speech-aware-barge-in.py")
        speech_release = importlib.util.module_from_spec(speech_spec)
        assert speech_spec.loader is not None
        speech_spec.loader.exec_module(speech_release)
        base_tag = self.name + "-accepted-base"
        base_context = self.path / "accepted-base"
        base_context.mkdir()
        wheelhouse = base_context / "wheelhouse"
        wheelhouse.mkdir()
        downloads = [(item["filename"], item["url"])
                     for item in speech_release.WHEELS.values()]
        download_code = (
            "import pathlib,urllib.request; items=" + repr(downloads) + "; "
            "root=pathlib.Path('/wheelhouse'); "
            "[(root/name).write_bytes(urllib.request.urlopen(url,timeout=60).read()) "
            "for name,url in items]"
        )
        self.docker(
            "run", "--rm", "--network", "bridge", "--pull", "never",
            "--mount", "type=bind,source=" + str(wheelhouse) + ",target=/wheelhouse",
            "--entrypoint", "python", app_base, "-c", download_code,
        )
        for item in speech_release.WHEELS.values():
            self.assertEqual(hashlib.sha256(
                (wheelhouse / item["filename"]).read_bytes()).hexdigest(), item["sha256"])
        copied = (*release.PROTECTED_RUNTIME_PATHS, *release.MODEL_ASSETS_PATHS)
        for relative in copied:
            target = base_context / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPOSITORY / relative).read_bytes())
        (base_context / "Dockerfile").write_text(
            "FROM " + app_base + "\n"
            "COPY wheelhouse /tmp/t020c-wheels\n"
            "RUN python -m pip install --disable-pip-version-check --no-index --no-deps "
            "--only-binary=:all: /tmp/t020c-wheels/*.whl && rm -rf /tmp/t020c-wheels\n"
            "LABEL org.opencontainers.image.revision=" + release.DEPLOYED_SOURCE + "\n"
            + "".join(
                "COPY " + relative + " /app/" + relative + "\n" for relative in copied),
            encoding="utf-8",
        )
        self.docker("build", "--network", "none", "--pull=false", "-t", base_tag,
                    str(base_context))
        self.images.add(base_tag)
        base_id = self.docker(
            "image", "inspect", base_tag, "--format", "{{.Id}}"
        ).stdout.decode().strip()

        release_dir = self.path / "release"
        release_dir.mkdir()
        class LocalRelease(release.Release):
            def start_ingress(local_self):
                local_self.run("docker", "start", local_self.nginx_container)
                if not local_self.inspect_once(local_self.nginx_container)["State"]["Running"]:
                    raise RuntimeError("synthetic ingress did not start")

        item = LocalRelease(release_dir, "a" * 40)
        item.root = REPOSITORY
        item.git_bytes = lambda relative: (REPOSITORY / relative).read_bytes()
        item.run = lambda *args, **kwargs: self.docker(
            *args[1:], check=kwargs.get("check", True), input=kwargs.get("input"),
            timeout=kwargs.get("timeout", 120))
        old_image = release.OLD_IMAGE
        release.OLD_IMAGE = base_id
        try:
            item.stage_assets()
            candidate = item.build_candidate()
            self.images.update(("ai-phone-t020c-candidate:" + release_dir.name,
                                candidate))
        finally:
            release.OLD_IMAGE = old_image

        parent_config = self.path / "parent-config"
        parent_clients = self.path / "parent-clients"
        shutil = __import__("shutil")
        shutil.copytree(REPOSITORY / "config", parent_config)
        shutil.copytree(REPOSITORY / "clients", parent_clients)
        (parent_config / "settings.py").write_text(
            "raise RuntimeError('parent settings must be hidden')\n", encoding="utf-8")
        (parent_clients / "accountants.yaml").write_text("invalid: parent\n", encoding="utf-8")
        (parent_clients / "booking-policy.yaml").write_text("invalid: parent\n", encoding="utf-8")
        old_files = {}
        candidate_files = {}
        for target, relative in {
            release.SETTINGS_TARGET: "config/settings.py",
            release.ACCOUNTANTS_TARGET: "clients/accountants.yaml",
            release.POLICY_TARGET: "clients/booking-policy.yaml",
        }.items():
            old = self.path / ("old-" + Path(relative).name)
            new = self.path / ("candidate-" + Path(relative).name)
            old.write_bytes((REPOSITORY / relative).read_bytes())
            new.write_bytes((item.assets / relative).read_bytes())
            old_files[target] = old
            candidate_files[target] = new

        environment = dict(value.split("=", 1)
                           for value in release.SYNTHETIC_APPLICATION_ENV)
        environment.update({
            "BOOKING_OBSERVATION_ENABLED": "true",
            "BOOKING_OBSERVATION_INTERVAL_SECONDS": "60",
            "BOOKING_OBSERVATION_FRESHNESS_SECONDS": "180",
            "BARGE_IN_DIAGNOSTICS_ENABLED": "true",
            "SPEECH_AWARE_BARGE_IN_ENABLED": "true",
            "ELEVENLABS_STT_FILTER_BACKGROUND_AUDIO": "false",
            "AUTOMATIC_NOTIFICATIONS_ENABLED": "true",
            release.PAUSED_ENV: "false",
        })
        compose = self.path / "compose.json"

        def up(image, *, candidate_mode):
            env = dict(environment)
            files = candidate_files if candidate_mode else old_files
            if candidate_mode:
                env[release.VERIFIED_ENV] = "true"
                env[release.PAUSED_ENV] = "true"
            mounts = [
                {"type": "bind", "source": str(parent_config),
                 "target": "/app/config", "read_only": True},
                {"type": "bind", "source": str(parent_clients),
                 "target": "/app/clients", "read_only": True},
            ]
            mounts.extend({"type": "bind", "source": str(source), "target": target,
                           "read_only": True} for target, source in files.items())
            value = {"services": {"app": {
                "image": image, "container_name": self.name + "-app",
                "entrypoint": ["python", "-c"],
                "command": ["import time; time.sleep(300)"],
                "environment": env, "volumes": mounts,
            }}}
            compose.write_text(json.dumps(value), encoding="utf-8")
            self.docker("compose", "-p", self.project, "-f", str(compose), "up", "-d",
                        "--no-deps", "--no-build", "--pull", "never", "--force-recreate",
                        "app")
            return json.loads(self.docker("inspect", self.name + "-app").stdout)[0]

        old = up(base_id, candidate_mode=False)
        current = up(candidate, candidate_mode=True)
        text_hashes = {
            **item.manifest["runtime"], **item.manifest["configuration"],
            **item.manifest["migrations"], **item.manifest["protected_runtime"],
        }
        probe = release.strict_json(self.docker(
            "exec", self.name + "-app", "python", "-c",
            release.source_probe_code(text_hashes, item.manifest["model_assets"])
        ).stdout)
        self.assertEqual(probe["hashes"], {**text_hashes, **item.manifest["model_assets"]})
        self.assertTrue(probe["settings"][release.VERIFIED_SETTING])
        self.assertTrue(probe["settings"][release.PAUSED_SETTING])
        rolled_back = up(base_id, candidate_mode=False)
        self.assertEqual(old["Image"], base_id)
        self.assertEqual(current["Image"], candidate)
        self.assertEqual(rolled_back["Image"], base_id)
        self.assertEqual(
            release.container_contract(old, candidate=False),
            release.container_contract(rolled_back, candidate=False),
        )
        self.__class__.release_item = item
        self.__class__.candidate_image = candidate
        self.__class__.base_image = base_id
        self.__class__.baseline_environment = environment
        self.__class__.parent_config = parent_config
        self.__class__.parent_clients = parent_clients

    def test_z_actual_release_methods_start_recover_and_preserve_operations(self):
        item = self.release_item
        candidate = self.candidate_image
        base_id = self.base_image
        old_image = release.OLD_IMAGE
        release.OLD_IMAGE = base_id
        item.pg_image = self.image
        item.network = self.network
        item.database_container = self.source
        try:
            item.run_candidate_tests(candidate)
            candidate_evidence = json.loads(
                (item.directory / "candidate-tests.json").read_text(encoding="utf-8"))
            self.assertEqual(candidate_evidence["passed"], 843)
            self.assertEqual(candidate_evidence["skipped"], 5)
            archive = self.path / "actual-method-schema0004.dump"
            archive.write_bytes(self.docker(
                "exec", self.source, "pg_dump", "-U", "t020c", "-d", "t020c",
                "-Fc", "--no-owner", "--no-acl").stdout)
            item.rehearsal(candidate, archive)

            root = self.path / "actual-root"
            root.mkdir()
            app_name = self.name + "-actual-app"
            ingress_name = self.name + "-actual-ingress"
            project = self.project + "-actual"
            base_compose = root / "docker-compose.yml"
            base_compose.write_text(json.dumps({
                "services": {"app": {}},
                "networks": {"rehearsal": {"external": True, "name": self.network}},
            }), encoding="utf-8")
            old_settings = self.path / "actual-old-settings.py"
            old_settings.write_bytes((REPOSITORY / "config/settings.py").read_bytes())
            environment = dict(self.baseline_environment)
            database_url = "postgresql://t020c:synthetic@" + self.source + "/t020c"
            environment["DATABASE_URL"] = database_url
            environment[release.PAUSED_ENV] = "false"
            original = {"services": {"app": {
                "image": base_id,
                "container_name": app_name,
                "command": ["uvicorn", "api.main:app", "--host", "0.0.0.0",
                            "--port", "8000", "--workers", "1"],
                "environment": environment,
                "networks": ["rehearsal"],
                "volumes": [
                    {"type": "bind", "source": str(self.parent_config),
                     "target": "/app/config", "read_only": True},
                    {"type": "bind", "source": str(self.parent_clients),
                     "target": "/app/clients", "read_only": True},
                    {"type": "bind", "source": str(old_settings),
                     "target": release.SETTINGS_TARGET, "read_only": True},
                ],
            }}}
            live_override = root / "docker-compose.override.yml"
            live_override.write_text(json.dumps(original), encoding="utf-8")
            item.root = root
            item.project_name = project
            item.app_container = app_name
            item.nginx_container = ingress_name
            item.live_env = item.private_write(
                "actual-live.env", ("MIGRATION_DATABASE_URL=" + database_url + "\n").encode())
            item.candidate_image = candidate
            item.run("docker", "create", "--name", ingress_name, "--network", "none",
                     self.image, "sleep", "600")
            item.owned_containers.add(ingress_name)
            item.run("docker", "start", ingress_name)
            item.compose(live_override, "up", "-d", "--no-deps", "--no-build",
                         "--pull", "never", "--force-recreate", "app", timeout=120)
            item.wait_ready()
            old = item.inspect_once(app_name)
            item.previous_settings = item.settings()
            item.previous_rendered = item.render_compose(live_override)
            item.previous_contract = release.container_contract(old, candidate=False)
            item.private_write("previous-override.json", live_override.read_bytes())
            paused = item.prepare_override(candidate, original, paused=True)
            active = item.prepare_override(candidate, original, paused=False)

            item.recover(candidate, paused)
            self.assertEqual(item.live_schema_version(), "0004")
            item.stop_writers(strict=True)
            item.private_write("final-fingerprint.json", json.dumps(
                item.fingerprint(candidate), sort_keys=True).encode())
            item.apply_live_migration()
            item.verify_live_fingerprints()
            item.install_override(paused)
            item.replace_app(candidate, paused=True)
            item.install_override(active)
            item.replace_app(candidate, paused=False)
            item.start_ingress()

            insert = """
INSERT INTO public.booking_operations(
 operation_id,tenant_id,business_id,staff_id,service_id,action,proposal_id,
 proposal_revision,payload_hash,payload_snapshot,confirmation_identity,
 confirmed_at,issued_at,expires_at,admitted_at,starts_at,ends_at,pre_buffer,
 post_buffer,claim_span,state)
VALUES('00000000-0000-0000-0000-000000000020','tenant','business','staff',
 'service','create','unknown-outcome',1,repeat('a',64),'{}','{}',
 now(),now(),now()+interval '1 minute',now(),now()+interval '1 day',
 now()+interval '1 day 30 minutes',interval '0',interval '0',
 tstzrange(now()+interval '1 day',now()+interval '1 day 30 minutes','[)'),
 'pending');
"""
            self.docker("exec", "-i", self.source, "psql", "-v", "ON_ERROR_STOP=1",
                        "-U", "t020c", "-d", "t020c", input=insert.encode())
            item.recover(candidate, paused)
            schema = item.schema_probe(candidate)
            self.assertEqual(schema["version"], "0005")
            self.assertEqual(schema["operation_count"], 1)
            self.assertTrue(item.settings()[release.PAUSED_SETTING])
            self.assertFalse(item.inspect_once(ingress_name)["State"]["Running"])
        finally:
            if hasattr(item, "root") and (item.root / "docker-compose.yml").exists():
                item.compose(item.root / "docker-compose.override.yml", "down",
                             "--remove-orphans", check=False)
            item.cleanup()
            release.OLD_IMAGE = old_image


if __name__ == "__main__":
    unittest.main()
