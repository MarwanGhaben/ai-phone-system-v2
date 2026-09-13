"""Offline release-boundary tests; never contact Docker, Git, or a provider."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[2] / 'scripts' / 'deploy-provider-observations.py'
SPEC = importlib.util.spec_from_file_location('t018c_release', SOURCE)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
COMMIT = 'a' * 40
SECRET = 'sentinel-$-line1\nline2'


def fixture(tmp: Path) -> release.Release:
    item = release.Release(tmp, COMMIT)
    item.old = {
        'Image': release.OLD_IMAGE,
        'State': {'Running': True, 'Health': {'Status': 'healthy'}},
        'Config': {'Env': ['TOKEN=' + SECRET, 'OTHER=x'], 'Labels': {}},
        'Mounts': [
            {'Source': '/opt/runtime.env', 'Destination': '/app/.env', 'RW': False},
            {'Source': '/opt/cert', 'Destination': '/cert', 'RW': False},
        ],
        'NetworkSettings': {'Networks': {'internal': {}}},
        'HostConfig': {'PortBindings': {'8000/tcp': None}},
    }
    item.previous_settings = {'token': SECRET, 'database_url': 'postgresql://db/live'}
    item.network = 'internal'
    item.live_env = tmp / 'migration.env'
    item.pg_image = 'sha256:' + 'b' * 64
    return item


def override(image: str) -> dict:
    return {'services': {
        'app': {'image': image, 'environment': {'TOKEN': SECRET, 'OTHER': 'x'}},
        'migrate': {'image': image},
    }}


def rendered(config: dict) -> dict:
    value = json.loads(json.dumps(config))
    value['services']['app']['volumes'] = [
        {'source': '/opt/runtime.env', 'target': '/app/.env', 'read_only': True},
        {'source': '/opt/cert', 'target': '/cert', 'read_only': True},
    ]
    value['services']['app']['networks'] = ['internal']
    value['networks'] = {'internal': {'name': 'internal'}}
    return value


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.item = fixture(self.path)

    def test_exact_candidate_and_fallback_copy_scope(self):
        item = self.item
        expected = (
            'api/main.py', 'config/settings.py', 'migrations/runner.py',
            'migrations/schema_contract.py',
            'migrations/0003_booking_provider_observations.sql',
            'services/calendar/booking_readback.py',
            'services/scheduling/provider_observations.py',
            'services/dashboard/dashboard_routes.py', 'templates/dashboard.html',
        )
        self.assertEqual(release.RUNTIME_PATHS, expected)
        self.assertEqual(release.FALLBACK_PATHS, ('migrations/schema_contract.py',))
        self.assertEqual(release.SOURCE_SCOPE, set(expected) | {'docker-compose.yml'})
        calls = []
        def run(*args, **kwargs):
            calls.append(args)
            if args[:2] == ('git', 'show'):
                return SimpleNamespace(stdout=b'asset\n', returncode=0)
            return SimpleNamespace(stdout=b'', returncode=0)
        item.run = run
        item.inspect = lambda name: {'Id': release.OLD_IMAGE if 'parent' in name else 'sha256:' + 'c' * 64,
                                     'Config': {'Labels': {'org.opencontainers.image.revision':
                                                           COMMIT if 'candidate' in name else release.DEPLOYED_SOURCE}}}
        self.assertEqual(item.build('candidate', release.RUNTIME_PATHS), 'sha256:' + 'c' * 64)
        self.assertEqual(item.build('fallback', release.FALLBACK_PATHS), 'sha256:' + 'c' * 64)
        self.assertEqual([c[2] for c in calls if c[:2] == ('git', 'show')],
                         [COMMIT + ':' + p for p in (*release.RUNTIME_PATHS, *release.FALLBACK_PATHS)])
        self.assertEqual([c[0:6] for c in calls if c[:2] == ('docker', 'build')],
                         [('docker', 'build', '--network', 'none', '--pull=false', '--label')] * 2)
        candidate_dockerfile = (self.path / 'build-candidate' / 'Dockerfile').read_text()
        fallback_dockerfile = (self.path / 'build-fallback' / 'Dockerfile').read_text()
        self.assertEqual(candidate_dockerfile.count('COPY '), 9)
        self.assertEqual(fallback_dockerfile.count('COPY '), 1)
        self.assertIn('COPY migrations/schema_contract.py /app/migrations/schema_contract.py', fallback_dockerfile)
        self.assertNotIn('booking_readback', fallback_dockerfile)
        with self.assertRaises(RuntimeError):
            item.build('candidate', release.RUNTIME_PATHS + ('requirements.txt',))

    def test_readonly_schema_probes_keep_checks_under_python_optimization(self):
        for probe in (release.CHECK_0002, release.CHECK_0003,
                      release.CHECK_EMPTY_OBSERVATIONS):
            tree = ast.parse(probe)
            self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))
            compile(probe, '<probe>', 'exec', optimize=2)

    def test_effective_compose_settings_and_secret_preservation(self):
        item = self.item
        previous = override(release.OLD_IMAGE)
        (self.path / 'previous-override.json').write_text(json.dumps(previous))
        seen = []
        def compose(path, *args, **kwargs):
            config = rendered(json.loads(path.read_text()))
            seen.append((path.name, args))
            return SimpleNamespace(stdout=json.dumps(config).encode())
        item.compose = compose
        item.settings_probe = lambda path, label: release.settings_for(
            item.previous_settings, candidate=label == 'candidate')
        candidate = item.prepare_override('candidate', 'sha256:candidate', 'sha256:candidate', previous)
        fallback = item.prepare_override('fallback', 'sha256:fallback', 'sha256:candidate', previous)
        self.assertEqual(json.loads(candidate.read_text())['services']['app']['environment']['TOKEN'], SECRET)
        self.assertEqual(json.loads(fallback.read_text())['services']['app']['environment'], previous['services']['app']['environment'])
        self.assertEqual(json.loads(candidate.read_text())['services']['app']['environment'] | {},
                         previous['services']['app']['environment'] | release.OBSERVATION_ENV)
        self.assertEqual(len(seen), 4)
        (self.path / 'candidate-override.json').unlink()
        item.settings_probe = lambda path, label: {'token': 'changed'}
        with self.assertRaisesRegex(RuntimeError, 'effective application settings changed'):
            item.prepare_override('candidate', 'sha256:candidate', 'sha256:candidate', previous)

    def test_effective_compose_rejects_unrelated_environment_or_mount_changes(self):
        base = rendered(override(release.OLD_IMAGE))
        proposed = rendered(override('sha256:candidate'))
        proposed['services']['app']['environment'].update(release.OBSERVATION_ENV)
        self.assertEqual(release.normalized_compose(base, candidate=False),
                         release.normalized_compose(proposed, candidate=True))
        proposed['services']['app']['environment']['OTHER'] = 'tampered'
        self.assertNotEqual(release.normalized_compose(base, candidate=False),
                            release.normalized_compose(proposed, candidate=True))
        proposed['services']['app']['environment']['OTHER'] = 'x'
        proposed['services']['app']['volumes'].pop()
        self.assertNotEqual(release.normalized_compose(base, candidate=False),
                            release.normalized_compose(proposed, candidate=True))
        self.assertEqual(proposed['services']['app']['environment']['TOKEN'], SECRET)

    def test_settings_probe_runs_python_without_app_or_dependency_startup(self):
        item = self.item
        observed = []
        item.compose = lambda path, *args, **kwargs: (
            observed.append(args) or SimpleNamespace(stdout=b'{"test":true}'))
        item.remove_container = lambda name: observed.append(('removed', name))
        self.assertEqual(item.settings_probe(self.path / 'override', 'candidate'), {'test': True})
        args = observed[0]
        self.assertIn('--no-deps', args)
        self.assertNotIn('--no-build', args)  # Unsupported by compose run.
        self.assertNotIn('--build', args)
        self.assertEqual(args[args.index('--pull') + 1], 'never')
        self.assertEqual(args[args.index('--entrypoint') + 1], 'python')
        self.assertEqual(observed[-1][0], 'removed')

    def test_replacement_accepts_same_mounts_in_different_inspect_order(self):
        item = self.item
        current = json.loads(json.dumps(item.old))
        current['Image'] = 'sha256:candidate'
        current['Mounts'].reverse()
        current['Config']['Env'].extend(k + '=' + v for k, v in release.OBSERVATION_ENV.items())
        item.wait_ready = lambda: None
        item.inspect = lambda name: current
        item.settings = lambda: release.settings_for(item.previous_settings, candidate=True)
        item.verify_replaced('sha256:candidate', candidate=True)
        current['Mounts'][0]['RW'] = True
        with self.assertRaisesRegex(RuntimeError, 'mounts differ'):
            item.verify_replaced('sha256:candidate', candidate=True)

    def test_preflight_rejects_wrong_private_image_before_live_action(self):
        item = self.item
        item.root = self.path
        item.verify_source = lambda: None
        item.capacity = lambda: None
        private = self.path / 'docker-compose.override.yml'
        private.write_text(json.dumps(override('sha256:wrong')))
        private.chmod(0o600)
        item.inspect = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, 'root protected|pin running image'):
            item.preflight()
        item.inspect.assert_not_called()

    def test_preflight_refusal_has_no_mutating_docker_calls(self):
        item = self.item
        item.verify_source = mock.Mock(side_effect=RuntimeError('drift'))
        item.run = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, 'drift'):
            item.preflight()
        item.run.assert_not_called()
        self.assertEqual(list(self.path.iterdir()), [])

    def test_rehearsal_preserves_aware_naive_and_repeats_migration(self):
        item = self.item
        item.pg_image = 'sha256:postgres16'
        archive = self.path / 'rehearsal.dump'
        archive.write_bytes(b'fake archive')
        operations = []
        before = {'hashes': {table: '0' * 32 for table in release.BUSINESS_TABLES}, 'aware': 4, 'naive': 2}
        item.run = lambda *args, **kwargs: (operations.append(args) or SimpleNamespace(stdout=b'', returncode=0))
        item.wait_for_rehearsal_database = lambda name: operations.append(('wait-authenticated', name))
        item.stream = lambda command, **kwargs: operations.append(tuple(command))
        item.fingerprint = lambda *args: before
        item.schema_matches = lambda image, **kwargs: True
        item.one_shot = lambda image, network, env, args: (
            operations.append((image, *args)) or SimpleNamespace(stdout=b'up-to-date 0003\n'))
        item.remove_container = lambda name: operations.append(('remove-container', name))
        item.remove_network = lambda name: operations.append(('remove-network', name))
        item.rehearsal('sha256:candidate', 'sha256:fallback', archive)
        self.assertEqual(sum(op[0] == 'sha256:candidate' and '-m' in op for op in operations), 2)
        self.assertTrue(any(op[0] == 'sha256:candidate' and release.CHECK_EMPTY_OBSERVATIONS in op for op in operations))
        self.assertEqual(operations[-2][0], 'remove-container')
        self.assertEqual(operations[-1][0], 'remove-network')

    def test_rehearsal_partial_create_and_timeout_always_clean_up(self):
        item = self.item
        archive = self.path / 'rehearsal.dump'
        archive.write_bytes(b'fake archive')
        operations = []
        def fail(*args, **kwargs):
            operations.append(args)
            if args[:3] == ('docker', 'run', '-d'):
                raise RuntimeError('timeout')
            return SimpleNamespace(stdout=b'', returncode=0)
        item.run = fail
        item.remove_container = lambda name: operations.append(('remove-container', name))
        item.remove_network = lambda name: operations.append(('remove-network', name))
        with self.assertRaisesRegex(RuntimeError, 'timeout'):
            item.rehearsal('candidate', 'fallback', archive)
        self.assertEqual(operations[-2][0], 'remove-container')
        self.assertEqual(operations[-1][0], 'remove-network')

    def test_timed_out_one_shot_removes_its_named_container(self):
        item = self.item
        removed = []
        item.run = mock.Mock(side_effect=RuntimeError('command timed out'))
        item.remove_container = removed.append
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            item.one_shot('candidate', 'isolated', self.path / 'synthetic.env', ['-c', 'pass'])
        self.assertEqual(removed, ['t018c-job-' + self.path.name])

    def test_pre_stop_failure_never_marks_cutover(self):
        item = self.item
        item.recheck_before_stop = mock.Mock(side_effect=RuntimeError('fingerprint drift'))
        item.stop_writers = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, 'fingerprint drift'):
            item.deploy('candidate', self.path / 'candidate', 'fallback', self.path / 'fallback')
        item.stop_writers.assert_not_called()
        self.assertFalse((self.path / 'cutover-started').exists())

    def _cutover_harness(self, *, migrate_fails=False, post_migration_fails=False,
                         predecessor=False, current=True):
        item = self.item
        calls = []
        item.recheck_before_stop = lambda: calls.append('recheck')
        item.stop_writers = lambda **kwargs: calls.append(('stop', kwargs['strict']))
        item.fingerprint = lambda *args: {'hashes': {}, 'aware': 4, 'naive': 2}
        item.backup = lambda name: calls.append(('backup', name))
        def one_shot(image, network, env, args):
            calls.append(('migrate', image))
            if migrate_fails:
                raise RuntimeError('migration failed')
        item.one_shot = one_shot
        item.schema_matches = lambda image, **kwargs: predecessor if not kwargs['current'] else current
        item.install_override = lambda path: calls.append(('override', path.name))
        def replace(image, **kwargs):
            calls.append(('replace', image))
            if post_migration_fails and kwargs['candidate']:
                raise RuntimeError('candidate readiness failed')
        item.replace_app = replace
        item.run = lambda *args, **kwargs: (calls.append(args) or SimpleNamespace(stdout=b'', returncode=0))
        item.public_health = lambda: calls.append('https')
        return calls

    def test_migration_failure_recovers_old_image_and_keeps_original_failure(self):
        calls = self._cutover_harness(migrate_fails=True, predecessor=True)
        with self.assertRaisesRegex(RuntimeError, 'migration failed'):
            self.item.deploy('candidate', self.path / 'candidate', 'fallback', self.path / 'fallback')
        self.assertIn(('backup', 'final-after-writer-stop.dump'), calls)
        self.assertIn(('replace', release.OLD_IMAGE), calls)
        self.assertEqual(calls.count('https'), 1)
        self.assertTrue((self.path / 'cutover-started').exists())

    def test_post_migration_failure_recovers_compatible_fallback(self):
        calls = self._cutover_harness(post_migration_fails=True, current=True)
        with self.assertRaisesRegex(RuntimeError, 'candidate readiness failed'):
            self.item.deploy('candidate', self.path / 'candidate', 'fallback', self.path / 'fallback')
        self.assertIn(('replace', 'fallback'), calls)
        self.assertIn(('override', 'fallback'), calls)
        self.assertNotIn(('replace', release.OLD_IMAGE), calls)

    def test_ambiguous_schema_leaves_ingress_stopped_and_keeps_original_failure(self):
        calls = self._cutover_harness(migrate_fails=True, predecessor=False, current=False)
        with self.assertRaisesRegex(RuntimeError, 'migration failed'):
            self.item.deploy('candidate', self.path / 'candidate', 'fallback', self.path / 'fallback')
        self.assertEqual([c for c in calls if isinstance(c, tuple) and c[0] == 'stop'],
                         [('stop', True), ('stop', False)])
        self.assertFalse(any(isinstance(c, tuple) and c[0] == 'replace' for c in calls))
        self.assertNotIn('https', calls)

    def test_safe_markers_never_print_private_environment(self):
        calls = self._cutover_harness(migrate_fails=True, predecessor=True)
        with mock.patch('builtins.print') as printer:
            with self.assertRaisesRegex(RuntimeError, 'migration failed'):
                self.item.deploy('candidate', self.path / 'candidate', 'fallback', self.path / 'fallback')
        published = '\n'.join(' '.join(str(arg) for arg in call.args) for call in printer.call_args_list)
        self.assertNotIn(SECRET, published)
        self.assertIn('RECOVERY_READY_HTTPS_OK', published)
        self.assertIn(('replace', release.OLD_IMAGE), calls)

    def test_failed_recovery_recloses_ingress_and_keeps_cutover_failure(self):
        calls = self._cutover_harness(post_migration_fails=True, current=True)
        self.item.public_health = mock.Mock(side_effect=RuntimeError(SECRET))
        original_run = self.item.run

        def run(*args, **kwargs):
            result = original_run(*args, **kwargs)
            if args[:3] == ('docker', 'stop', '--time') and args[-1] == 'ai-voice-nginx':
                raise RuntimeError('stop request interrupted')
            return result

        self.item.run = run
        with self.assertRaisesRegex(RuntimeError, 'candidate readiness failed'):
            self.item.deploy('candidate', self.path / 'candidate', 'fallback', self.path / 'fallback')
        self.assertIn(('docker', 'start', 'ai-voice-nginx'), calls)
        self.assertEqual(calls[-2:], [('docker', 'stop', '--time', '30', 'ai-voice-nginx'),
                                     ('docker', 'stop', '--time', '30', 'ai-voice-app')])


if __name__ == '__main__':
    unittest.main()
