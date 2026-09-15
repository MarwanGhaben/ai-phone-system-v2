"""Offline T049-B release-boundary tests; no Docker, Git or providers are contacted."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import tempfile
import time
import unittest
import uuid
from unittest import mock


SOURCE = Path(__file__).resolve().parents[2] / 'scripts' / 'deploy-automatic-notifications.py'
SPEC = importlib.util.spec_from_file_location('t049b_release', SOURCE)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
COMMIT = 'a' * 40
SECRET = 'sentinel-$-line1\nline2'
OLD_SETTINGS_SOURCE = '/opt/ai-phone-observation-release.accepted/candidate-settings.py'


def fixture(tmp: Path) -> release.Release:
    item = release.Release(tmp, COMMIT)
    item.old = {
        'Image': release.OLD_IMAGE,
        'State': {'Running': True, 'Health': {'Status': 'healthy'}},
        'Config': {
            'Env': ['TOKEN=' + SECRET, 'OTHER=x', *(
                key + '=' + value for key, value in release.OBSERVATION_ENV.items())],
            'Labels': {'org.opencontainers.image.revision': release.DEPLOYED_SOURCE},
        },
        'Mounts': [
            {'Type': 'bind', 'Source': '/opt/runtime.env',
             'Destination': '/app/.env', 'RW': False},
            {'Type': 'bind', 'Source': '/opt/config',
             'Destination': '/app/config', 'RW': False},
            {'Type': 'bind', 'Source': OLD_SETTINGS_SOURCE,
             'Destination': release.SETTINGS_TARGET, 'RW': False},
        ],
        'NetworkSettings': {'Networks': {'internal': {}}},
        'HostConfig': {'PortBindings': {'8000/tcp': None}},
    }
    item.previous_settings = {
        'token': SECRET, 'database_url': 'postgresql://db/live',
        **release.OBSERVATION_SETTINGS,
    }
    item.previous_settings_source = OLD_SETTINGS_SOURCE
    item.network = 'internal'
    item.live_env = tmp / 'migration.env'
    item.pg_image = 'sha256:' + 'b' * 64
    return item


def override(image: str) -> dict:
    return {'services': {
        'app': {
            'image': image,
            'environment': {'TOKEN': SECRET, 'OTHER': 'x', **release.OBSERVATION_ENV},
            'volumes': [
                {'type': 'bind', 'source': OLD_SETTINGS_SOURCE,
                 'target': release.SETTINGS_TARGET, 'read_only': True},
            ],
        },
        'migrate': {'image': image},
    }}


def rendered(config: dict, *, reverse=False) -> dict:
    value = json.loads(json.dumps(config))
    volumes = value['services']['app'].get('volumes', []) + [
        {'type': 'bind', 'source': '/opt/runtime.env',
         'target': '/app/.env', 'read_only': True},
        {'type': 'bind', 'source': '/opt/config',
         'target': '/app/config', 'read_only': True},
    ]
    if reverse:
        volumes.reverse()
    value['services']['app']['volumes'] = volumes
    value['services']['app']['networks'] = ['internal']
    value['networks'] = {'internal': {'name': 'internal'}}
    return value


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.item = fixture(self.path)
        self.item.settings_overlay.write_text('# reviewed settings\n')
        self.item.settings_overlay.chmod(0o644)

    def test_exact_candidate_copy_scope_and_networkless_build(self):
        self.assertEqual(len(release.RUNTIME_PATHS), 19)
        self.assertEqual(release.SOURCE_SCOPE,
                         set(release.RUNTIME_PATHS) | {'docker-compose.yml'})
        self.assertIn('migrations/0004_appointment_notifications.sql',
                      release.RUNTIME_PATHS)
        calls = []

        def run(*args, **kwargs):
            calls.append(args)
            if args[:2] == ('git', 'show'):
                return SimpleNamespace(stdout=b'asset\n', returncode=0)
            return SimpleNamespace(stdout=b'', returncode=0)

        self.item.run = run
        self.item.inspect = lambda name: {
            'Id': release.OLD_IMAGE if 'parent' in name else 'sha256:' + 'c' * 64,
            'Config': {'Labels': {'org.opencontainers.image.revision': COMMIT}},
        }
        self.assertEqual(self.item.build('candidate', release.RUNTIME_PATHS),
                         'sha256:' + 'c' * 64)
        copied = [call[2] for call in calls if call[:2] == ('git', 'show')]
        self.assertEqual(copied, [COMMIT + ':' + path for path in release.RUNTIME_PATHS])
        build = next(call for call in calls if call[:2] == ('docker', 'build'))
        self.assertIn(('docker', 'build', '--network', 'none', '--pull=false'),
                      (build[:5],))
        dockerfile = (self.path / 'build-candidate' / 'Dockerfile').read_text()
        self.assertEqual(dockerfile.count('COPY '), 19)
        self.assertNotIn('requirements', dockerfile)
        with self.assertRaises(RuntimeError):
            self.item.build('candidate', release.RUNTIME_PATHS + ('requirements.txt',))

    def test_readonly_schema_probes_survive_python_optimization(self):
        for probe in (release.CHECK_0003, release.CHECK_0004,
                      release.CHECK_EMPTY_NOTIFICATIONS,
                      release.CHECK_PAUSED_BOOKING_PATH, release.FINGERPRINT_CODE):
            tree = ast.parse(probe)
            self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))
            compile(probe, '<probe>', 'exec', optimize=2)

    def test_active_and_paused_settings_only_add_three_values(self):
        active = release.settings_for(self.item.previous_settings, paused=False)
        paused = release.settings_for(self.item.previous_settings, paused=True)
        self.assertFalse(active[release.PAUSED_SETTING])
        self.assertTrue(paused[release.PAUSED_SETTING])
        self.assertEqual(active['token'], SECRET)
        self.assertEqual(paused['automatic_notifications_interval_seconds'], 60)
        wrong = dict(self.item.previous_settings, booking_observation_enabled=False)
        with self.assertRaisesRegex(RuntimeError, 'observation settings differ'):
            release.settings_for(wrong, paused=True)

    def test_normalization_accepts_reordered_mounts_and_preserves_duplicates(self):
        baseline = rendered(override(release.OLD_IMAGE))
        proposed_config = override('candidate')
        proposed = proposed_config['services']['app']
        proposed['environment'].update(release.NOTIFICATION_ENV)
        proposed['environment'][release.PAUSED_ENV] = 'false'
        proposed['volumes'][0]['source'] = str(self.item.settings_overlay)
        candidate = rendered(proposed_config, reverse=True)
        self.assertEqual(
            release.normalized_compose(
                baseline, paused=None, settings_source=OLD_SETTINGS_SOURCE),
            release.normalized_compose(
                candidate, paused=False, settings_source=self.item.settings_overlay))
        candidate['services']['app']['volumes'].append(
            dict(candidate['services']['app']['volumes'][0]))
        self.assertNotEqual(
            release.normalized_compose(
                baseline, paused=None, settings_source=OLD_SETTINGS_SOURCE),
            release.normalized_compose(
                candidate, paused=False, settings_source=self.item.settings_overlay))

    def test_normalization_rejects_missing_duplicate_or_writable_settings_mount(self):
        config = override('candidate')
        config['services']['app']['environment'].update(release.NOTIFICATION_ENV)
        config['services']['app']['environment'][release.PAUSED_ENV] = 'true'
        config['services']['app']['volumes'][0]['source'] = str(self.item.settings_overlay)
        value = rendered(config)
        for mutate in (
            lambda mounts: mounts.pop(0),
            lambda mounts: mounts.append(dict(mounts[0])),
            lambda mounts: mounts[0].update(read_only=False),
        ):
            with self.subTest(mutate=mutate):
                changed = json.loads(json.dumps(value))
                mutate(changed['services']['app']['volumes'])
                with self.assertRaisesRegex(RuntimeError, 'pinned settings mount differs'):
                    release.normalized_compose(
                        changed, paused=True, settings_source=self.item.settings_overlay)

    def test_prepare_overrides_replace_one_nested_mount_and_preserve_secrets(self):
        previous = override(release.OLD_IMAGE)
        (self.path / 'previous-override.json').write_text(json.dumps(previous))

        def compose(path, *args, **kwargs):
            return SimpleNamespace(stdout=json.dumps(rendered(
                json.loads(path.read_text()), reverse=path.name.startswith('active'))).encode())

        self.item.compose = compose
        self.item.settings_probe = lambda path, label: release.settings_for(
            self.item.previous_settings, paused=label == 'paused')
        active = self.item.prepare_override('active', 'candidate', previous)
        paused = self.item.prepare_override('paused', 'candidate', previous)
        for path, expected in ((active, 'false'), (paused, 'true')):
            config = json.loads(path.read_text())
            app = config['services']['app']
            self.assertEqual(app['environment']['TOKEN'], SECRET)
            self.assertEqual(app['environment'][release.PAUSED_ENV], expected)
            overlays = [m for m in app['volumes']
                        if m.get('target') == release.SETTINGS_TARGET]
            self.assertEqual(len(overlays), 1)
            self.assertEqual(overlays[0]['source'], str(self.item.settings_overlay))
            self.assertTrue(overlays[0]['read_only'])

    def test_settings_probe_is_no_dependency_python_command(self):
        observed = []
        self.item.compose = lambda path, *args, **kwargs: (
            observed.append(args) or SimpleNamespace(stdout=b'{"test":true}'))
        self.item.remove_container = lambda name: observed.append(('removed', name))
        self.assertEqual(self.item.settings_probe(self.path / 'active', 'active'),
                         {'test': True})
        command = observed[0]
        self.assertIn('--no-deps', command)
        self.assertNotIn('--build', command)
        self.assertEqual(command[command.index('--pull') + 1], 'never')
        self.assertEqual(command[command.index('--entrypoint') + 1], 'python')

    def test_replacement_accepts_mount_reorder_and_requires_pause_setting(self):
        current = json.loads(json.dumps(self.item.old))
        current['Image'] = 'candidate'
        current['Mounts'].reverse()
        overlay = next(m for m in current['Mounts']
                       if m['Destination'] == release.SETTINGS_TARGET)
        overlay['Source'] = str(self.item.settings_overlay)
        current['Config']['Env'].extend(
            key + '=' + value for key, value in release.NOTIFICATION_ENV.items())
        current['Config']['Env'].append(release.PAUSED_ENV + '=true')
        current['Config']['Labels']['org.opencontainers.image.revision'] = COMMIT
        self.item.wait_ready = lambda: None
        self.item.inspect = lambda name: current
        self.item.settings = lambda: release.settings_for(
            self.item.previous_settings, paused=True)
        self.item.verify_replaced('candidate', paused=True)
        current['Config']['Env'][-1] = release.PAUSED_ENV + '=false'
        with self.assertRaisesRegex(RuntimeError, 'pause differs'):
            self.item.verify_replaced('candidate', paused=True)

    def test_preflight_refusal_has_no_mutating_calls(self):
        self.item.verify_source = mock.Mock(side_effect=RuntimeError('drift'))
        self.item.run = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, 'drift'):
            self.item.preflight()
        self.item.run.assert_not_called()

    def test_timed_out_one_shot_removes_named_container(self):
        removed = []
        self.item.run = mock.Mock(side_effect=RuntimeError('command timed out'))
        self.item.remove_container = removed.append
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            self.item.one_shot('candidate', 'isolated', self.path / 'env', ['-c', 'pass'])
        self.assertEqual(removed, ['t049b-job-' + self.path.name])

    def test_rehearsal_migrates_repeats_starts_paused_and_cleans_up(self):
        archive = self.path / 'rehearsal.dump'
        archive.write_bytes(b'archive')
        operations = []
        preserved = {
            'hashes': {table: '0' * 32 for table in release.PRESERVED_TABLES},
            'row_counts': {table: 1 for table in release.PRESERVED_TABLES},
            'aware': 2, 'naive': 1,
        }
        self.item.run = lambda *args, **kwargs: (
            operations.append(args) or SimpleNamespace(stdout=b'', returncode=0))
        self.item.wait_for_rehearsal_database = lambda name: operations.append(('wait', name))
        self.item.stream = lambda command, **kwargs: operations.append(tuple(command))
        self.item.fingerprint = lambda *args: preserved
        self.item.schema_matches = lambda *args, **kwargs: True
        self.item.one_shot = lambda image, network, env, args, **kwargs: (
            operations.append((image, *args)) or
            SimpleNamespace(stdout=b'up-to-date 0004\n', returncode=0))
        self.item.wait_ready = lambda name='ai-voice-app': operations.append(('ready', name))
        self.item.settings = lambda name='ai-voice-app': {
            release.PAUSED_SETTING: True,
            'automatic_notifications_enabled': True,
            'automatic_notifications_interval_seconds': 60,
            'booking_observation_enabled': True,
            'booking_observation_interval_seconds': 60,
            'booking_observation_freshness_seconds': 180,
        }
        self.item.remove_container = lambda name: operations.append(('remove-container', name))
        self.item.remove_network = lambda name: operations.append(('remove-network', name))
        self.item.rehearsal('candidate', archive)
        self.assertTrue(any(op[:3] == ('docker', 'run', '-d') and '--env-file' in op
                            for op in operations))
        self.assertTrue(any(op[0] == 'ready' for op in operations))
        self.assertTrue(any(release.CHECK_PAUSED_BOOKING_PATH in op for op in operations))
        self.assertEqual([op[0] for op in operations[-3:]],
                         ['remove-container', 'remove-container', 'remove-network'])

    def test_rehearsal_partial_start_always_cleans_all_resources(self):
        archive = self.path / 'rehearsal.dump'
        archive.write_bytes(b'archive')
        operations = []

        def fail(*args, **kwargs):
            operations.append(args)
            if args[:3] == ('docker', 'run', '-d'):
                raise RuntimeError('partial start')
            return SimpleNamespace(stdout=b'', returncode=0)

        self.item.run = fail
        self.item.remove_container = lambda name: operations.append(('remove-container', name))
        self.item.remove_network = lambda name: operations.append(('remove-network', name))
        with self.assertRaisesRegex(RuntimeError, 'partial start'):
            self.item.rehearsal('candidate', archive)
        self.assertEqual([op[0] for op in operations[-3:]],
                         ['remove-container', 'remove-container', 'remove-network'])

    def _cutover(self, *, migration_failure=False, paused_failure=False,
                 active_failure=False, predecessor=False, current=True):
        calls = []
        self.item.recheck_before_stop = lambda: calls.append('recheck')
        self.item.stop_writers = lambda **kwargs: calls.append(('stop', kwargs['strict']))
        preserved = {'hashes': {}, 'row_counts': {}, 'aware': 2, 'naive': 1}
        self.item.fingerprint = lambda *args: preserved
        self.item.backup = lambda name: calls.append(('backup', name))

        def one_shot(image, network, env, args, **kwargs):
            calls.append(('one-shot', tuple(args)))
            if migration_failure and '--prepare' in args:
                raise RuntimeError('migration failed')
            return SimpleNamespace(stdout=b'', returncode=0)

        self.item.one_shot = one_shot
        self.item.schema_matches = lambda image, **kwargs: (
            current if kwargs['current'] else predecessor)
        self.item.install_override = lambda path: calls.append(('override', path.name))
        replacements = {'paused': 0, 'active': 0}

        def replace(image, *, paused):
            label = 'old' if paused is None else 'paused' if paused else 'active'
            calls.append(('replace', label))
            if paused is True:
                replacements['paused'] += 1
                if paused_failure and replacements['paused'] == 1:
                    raise RuntimeError('paused readiness failed')
            if paused is False:
                replacements['active'] += 1
                if active_failure:
                    raise RuntimeError('active readiness failed')

        self.item.replace_app = replace
        self.item.run = lambda *args, **kwargs: (
            calls.append(args) or SimpleNamespace(stdout=b'', returncode=0))
        self.item.public_health = lambda: calls.append('https')
        return calls

    def test_pre_stop_failure_never_marks_cutover(self):
        self.item.recheck_before_stop = mock.Mock(side_effect=RuntimeError('fingerprint drift'))
        self.item.stop_writers = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, 'fingerprint drift'):
            self.item.deploy('candidate', self.path / 'active', self.path / 'paused')
        self.item.stop_writers.assert_not_called()
        self.assertFalse((self.path / 'cutover-started').exists())

    def test_success_activates_paused_before_explicit_active_restart(self):
        calls = self._cutover(current=True)
        self.item.deploy('candidate', self.path / 'active', self.path / 'paused')
        self.assertLess(calls.index(('replace', 'paused')), calls.index(('replace', 'active')))
        self.assertIn(('stop', True), calls[calls.index(('replace', 'paused')) + 1:])
        self.assertEqual(calls.count('https'), 2)
        self.assertTrue((self.path / 'deployed').exists())

    def test_migration_failure_recovers_predecessor_and_keeps_original_error(self):
        calls = self._cutover(migration_failure=True, predecessor=True, current=False)
        with self.assertRaisesRegex(RuntimeError, 'migration failed'):
            self.item.deploy('candidate', self.path / 'active', self.path / 'paused')
        self.assertIn(('replace', 'old'), calls)
        self.assertNotIn(('replace', 'active'), calls)

    def test_paused_startup_failure_retries_only_paused_candidate(self):
        calls = self._cutover(paused_failure=True, current=True)
        with self.assertRaisesRegex(RuntimeError, 'paused readiness failed'):
            self.item.deploy('candidate', self.path / 'active', self.path / 'paused')
        self.assertEqual(calls.count(('replace', 'paused')), 2)
        self.assertNotIn(('replace', 'active'), calls)

    def test_active_failure_recovers_paused_without_reactivating_active(self):
        calls = self._cutover(active_failure=True, current=True)
        with self.assertRaisesRegex(RuntimeError, 'active readiness failed'):
            self.item.deploy('candidate', self.path / 'active', self.path / 'paused')
        self.assertEqual(calls.count(('replace', 'active')), 1)
        self.assertEqual(calls.count(('replace', 'paused')), 2)

    def test_ambiguous_schema_keeps_ingress_stopped(self):
        calls = self._cutover(migration_failure=True, predecessor=False, current=False)
        with self.assertRaisesRegex(RuntimeError, 'migration failed'):
            self.item.deploy('candidate', self.path / 'active', self.path / 'paused')
        self.assertFalse(any(call == ('replace', 'old') for call in calls))
        self.assertFalse(any(call == ('replace', 'paused') for call in calls))
        self.assertNotIn('https', calls)

    def test_release_has_no_live_restore_requeue_or_provider_probe(self):
        source = SOURCE.read_text()
        self.assertNotIn('pg_restore', source[source.index('def deploy'):])
        self.assertNotRegex(source, r"UPDATE\s+public\.booking_notification_outbox")
        self.assertNotIn('graph.microsoft.com', source.lower())
        self.assertNotIn('api.telnyx.com', source.lower())

    def test_public_failure_markers_do_not_expose_private_values(self):
        self._cutover(migration_failure=True, predecessor=True, current=False)
        with mock.patch('builtins.print') as printer:
            with self.assertRaisesRegex(RuntimeError, 'migration failed'):
                self.item.deploy('candidate', self.path / 'active', self.path / 'paused')
        public = '\n'.join(' '.join(str(arg) for arg in call.args)
                           for call in printer.call_args_list)
        self.assertNotIn(SECRET, public)
        self.assertIn('RECOVERY_PRE_0004_READY_HTTPS_OK', public)


@unittest.skipUnless(os.environ.get('T049_RUN_DOCKER_TESTS') == '1',
                     'real release rehearsal requires T049_RUN_DOCKER_TESTS=1')
class DockerReleaseTests(unittest.TestCase):
    """Run actual build/rehearsal methods against disposable local PostgreSQL."""

    def test_actual_restored_0003_to_0004_paused_rehearsal_and_cleanup(self):
        docker = shutil.which('docker')
        if docker is None:
            self.fail('opt-in T049 release test requested but Docker is unavailable')
        base = [docker, '--context', 'desktop-linux']

        def call(args, *, timeout=60, input_data=None, check=True):
            result = subprocess.run(
                [*base, *args], input=input_data, capture_output=True,
                timeout=timeout, check=False)
            if check and result.returncode:
                self.fail('local Docker release harness command failed')
            return result

        info = call(['info', '--format', '{{.OSType}}'], timeout=15)
        self.assertEqual(info.stdout.strip(), b'linux')
        accepted = json.loads(call([
            'image', 'inspect', 'ai-phone-t049a-candidate:20260913']).stdout)[0]
        self.assertEqual(
            accepted['Id'],
            'sha256:dbaab7a24155d5f5c3a6af2a8daaaba83ec6f9562069a039aa6163e651fce615')
        images = call(['image', 'ls', 'postgres', '--format', '{{json .}}'])
        rows = [json.loads(line) for line in images.stdout.splitlines() if line]
        candidates = [row['Repository'] + ':' + row['Tag'] for row in rows
                      if row.get('Tag', '').startswith('16')]
        if not candidates:
            self.fail('opt-in T049 release test needs an existing PostgreSQL 16 image')
        postgres = json.loads(call(['image', 'inspect', candidates[0]]).stdout)[0]
        self.assertIn('PG_MAJOR=16', postgres['Config']['Env'])

        suffix = uuid.uuid4().hex
        seed_network = 't049b-seed-net-' + suffix
        seed_db = 't049b-seed-db-' + suffix
        built_tag = ''
        parent_tag = ''
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            built_tag = 'ai-phone-t049b-candidate:' + directory.name
            parent_tag = 'ai-phone-t049b-parent:' + directory.name
            archive = directory / 'source-0003.dump'
            item = release.Release(directory, COMMIT)
            item.root = Path(__file__).resolve().parents[2]
            item.pg_image = postgres['Id']

            def local_run(*args, timeout=60, check=True):
                if args[:2] == ('git', 'show'):
                    prefix, path = args[2].split(':', 1)
                    self.assertEqual(prefix, COMMIT)
                    return SimpleNamespace(
                        stdout=(item.root / path).read_bytes(), returncode=0)
                docker_args = list(args[1:]) if args and args[0] == 'docker' else None
                if docker_args is None:
                    raise RuntimeError('unexpected local release command')
                result = call(docker_args, timeout=timeout, check=False)
                with (directory / 'private.log').open('ab') as log:
                    log.write(result.stdout + result.stderr)
                if result.returncode and release.CHECK_0003 in docker_args:
                    # This harness has synthetic data/credentials and no live mounts.
                    diagnostic = call([
                        'exec', 't049b-db-' + directory.name, 'psql', '-X',
                        '-U', 'rehearsal', '-d', 'rehearsal', '-c',
                        "SELECT k.conname, c.check_definitions->>k.conname AS recorded, "
                        "pg_catalog.pg_get_constraintdef(k.oid,false) AS actual "
                        "FROM pg_catalog.pg_constraint k CROSS JOIN "
                        "public.booking_provider_observation_control c "
                        "WHERE k.contype='c' AND c.check_definitions ? k.conname "
                        "AND c.check_definitions->>k.conname IS DISTINCT FROM "
                        "pg_catalog.pg_get_constraintdef(k.oid,false)"], check=False)
                    self.fail('synthetic predecessor probe failed: ' +
                              result.stderr.decode(errors='replace')[-2500:] +
                              diagnostic.stdout.decode(errors='replace'))
                if check and result.returncode:
                    raise RuntimeError('local release command failed')
                if result.returncode == 0 and release.CHECK_PAUSED_BOOKING_PATH in docker_args:
                    # Recovery uses this same paused image. Preserve durable
                    # accepted/unknown outcomes across a real application restart.
                    db_name = 't049b-db-' + directory.name
                    query = ['exec', db_name, 'psql', '-X', '-A', '-t', '-U',
                             'rehearsal', '-d', 'rehearsal', '-v', 'ON_ERROR_STOP=1', '-c']
                    call([*query, "UPDATE public.booking_notification_outbox SET "
                        "state=CASE WHEN kind='confirmation' THEN 'accepted' ELSE 'unknown' END, "
                        "attempts=1,provider_message_id=CASE WHEN kind='confirmation' THEN 'synthetic-sms' END, "
                        "accepted_at=CASE WHEN kind='confirmation' THEN CURRENT_TIMESTAMP END"])
                    snapshot = "SELECT jsonb_agg(to_jsonb(j) ORDER BY id) FROM public.booking_notification_outbox j"
                    before = call([*query, snapshot]).stdout
                    states = call([*query, "SELECT state FROM public.booking_notification_outbox ORDER BY state"]).stdout
                    self.assertEqual(states.strip().splitlines(), [b'accepted', b'unknown'])
                    call(['restart', 't049b-app-' + directory.name], timeout=90)
                    item.wait_ready('t049b-app-' + directory.name)
                    self.assertEqual(before, call([*query, snapshot]).stdout)
                return result

            item.run = local_run
            try:
                call(['network', 'create', '--internal', seed_network])
                call([
                    'run', '-d', '--name', seed_db, '--pull', 'never',
                    '--network', seed_network, '--memory', '192m', '--cpus', '0.5',
                    '--pids-limit', '96', '--tmpfs',
                    '/var/lib/postgresql/data:size=128m',
                    '-e', 'PGDATA=/var/lib/postgresql/data/pgdata',
                    '-e', 'POSTGRES_USER=rehearsal',
                    '-e', 'POSTGRES_PASSWORD=synthetic',
                    '-e', 'POSTGRES_DB=rehearsal', postgres['Id'],
                    '-c', 'shared_buffers=16MB', '-c', 'max_connections=12'])
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    ready = call([
                        'exec', '-e', 'PGPASSWORD=synthetic', seed_db,
                        'psql', '-X', '-w', '-A', '-t', '-h', '127.0.0.1',
                        '-U', 'rehearsal', '-d', 'rehearsal',
                        '-c', 'SELECT current_database()'], timeout=5, check=False)
                    if ready.returncode == 0 and ready.stdout.strip() == b'rehearsal':
                        break
                    time.sleep(0.25)
                else:
                    self.fail('seed PostgreSQL did not become ready')
                dsn = 'postgresql://rehearsal:synthetic@' + seed_db + ':5432/rehearsal'
                call([
                    'run', '--rm', '--pull', 'never', '--network', seed_network,
                    '-e', 'MIGRATION_DATABASE_URL=' + dsn,
                    '--entrypoint', 'python', accepted['Id'],
                    '-m', 'migrations.runner', '--prepare'], timeout=120)
                seed_sql = """
DROP TABLE public.booking_notification_outbox,
 public.booking_notification_reconciliation,
 public.booking_notification_contract;
-- Model the deployed legacy-adoption history, not a fresh-install ledger.
-- Only this isolated synthetic seed is changed; no release path rewrites history.
DELETE FROM public.schema_migrations WHERE version IN ('0004','bootstrap-v1');
INSERT INTO public.bookings
 (call_sid,appointment_time,status,ms_booking_id,appointment_time_utc)
VALUES
 ('t049b-aware',TIMESTAMP '2099-01-08 16:00:00','confirmed','t049b-aware',
  TIMESTAMPTZ '2099-01-08 16:00:00+00'),
 ('t049b-naive',TIMESTAMP '2099-01-09 16:00:00','confirmed','t049b-naive',NULL);
INSERT INTO public.booking_provider_observations
 (booking_id,snapshot_provider_id,snapshot_start,snapshot_status,checked_at,
  outcome,provider_start,provider_end,observed_provider_id)
SELECT id,ms_booking_id,appointment_time_utc,'confirmed',CURRENT_TIMESTAMP,
 'present',appointment_time_utc,appointment_time_utc+INTERVAL '30 minutes',ms_booking_id
FROM public.bookings WHERE call_sid='t049b-aware';
"""
                call(['exec', '-e', 'PGPASSWORD=synthetic', seed_db,
                      'psql', '-X', '-w', '-h', '127.0.0.1', '-U', 'rehearsal',
                      '-d', 'rehearsal', '-v', 'ON_ERROR_STOP=1', '-c', seed_sql])
                dump = call(['exec', seed_db, 'pg_dump', '-U', 'rehearsal',
                             '-d', 'rehearsal', '-Fc'], timeout=120)
                archive.write_bytes(dump.stdout)
                with (mock.patch.object(release, 'OLD_IMAGE', accepted['Id']),
                      mock.patch.dict(os.environ, {'DOCKER_CONTEXT': 'desktop-linux'})):
                    candidate = item.build('candidate', release.RUNTIME_PATHS)
                    item.rehearsal(candidate, archive)
            finally:
                call(['rm', '-f', 't049b-app-' + directory.name], check=False)
                call(['rm', '-f', 't049b-db-' + directory.name], check=False)
                call(['network', 'rm', 't049b-net-' + directory.name], check=False)
                call(['rm', '-f', seed_db], check=False)
                call(['network', 'rm', seed_network], check=False)
                call(['image', 'rm', '-f', built_tag], check=False)
                call(['image', 'rm', '-f', parent_tag], check=False)
                for kind, args in (
                    ('container', ['container', 'ls', '--all',
                                   '--format', '{{.Names}}']),
                    ('network', ['network', 'ls', '--format', '{{.Name}}']),
                ):
                    listing = call(args, check=False).stdout.decode()
                    self.assertNotIn(directory.name, listing,
                                     kind + ' cleanup unverified')


if __name__ == '__main__':
    unittest.main()
