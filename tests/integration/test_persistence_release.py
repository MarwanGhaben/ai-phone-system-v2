"""Cutover failure injection at Docker/process boundaries; no server access."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def release(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / 'scripts'
    base = load('release_base_test', scripts / 'deploy-booking-guards.py')
    module = load('persistence_release_test', scripts / 'deploy-booking-persistence.py')
    root = tmp_path / 'repo'
    root.mkdir()
    base.ROOT = module.ROOT = root
    base.OLD_IMAGE = module.OLD_IMAGE
    monkeypatch.setattr(base.time, 'sleep', lambda _: None)
    directory = tmp_path / 'release'
    directory.mkdir()
    (root / 'nginx').mkdir()
    (root / 'nginx/nginx.conf').write_text('synthetic nginx')
    base.NGINX_HASH = hashlib.sha256((root / 'nginx/nginx.conf').read_bytes()).hexdigest()
    (root / 'docker-compose.yml').write_text('synthetic compose')
    (directory / 'compose.sha256').write_text(hashlib.sha256((root / 'docker-compose.yml').read_bytes()).hexdigest())
    settings = {'database_url': 'postgresql://synthetic', 'secret_key': 'synthetic-$secret'}
    previous = {'services': {'app': {'image': module.OLD_IMAGE,
                'environment': {'SECRET_KEY': 'synthetic-$$secret'},
                'volumes': ['/private/runtime.env:/app/.env:ro']}}}
    previous_bytes = json.dumps(previous).encode()
    (directory / 'previous-settings.json').write_text(json.dumps(settings))
    (directory / 'previous-override.json').write_bytes(previous_bytes)
    (root / 'docker-compose.override.yml').write_bytes(previous_bytes)
    overrides = []
    for label in ('candidate', 'fallback'):
        config = json.loads(previous_bytes)
        config['services']['app']['image'] = 'sha256:' + label
        path = directory / (label + '-override.json')
        path.write_text(json.dumps(config))
        overrides.append(path)
    state = {'image': module.OLD_IMAGE, 'schema': 'old', 'fault': None,
             'running': {'ai-voice-app': True, 'ai-voice-nginx': True}, 'commands': []}

    def execute(args, **kwargs):
        args = list(args)
        state['commands'].append(args)
        code, output = 0, b''
        if args[:2] == ['docker', 'inspect']:
            output = json.dumps([{'Image': state['image'],
                       'State': {'Running': state['running'].get(args[-1], False)}}]).encode()
        elif args[:2] == ['docker', 'stop']:
            state['running'][args[-1]] = False
        elif args[:2] == ['docker', 'start']:
            state['running'][args[-1]] = True
        elif args[:3] == ['docker', 'exec', 'ai-voice-app']:
            output = json.dumps(settings if args[-1] == base.SETTINGS else {'status': 'ready'}).encode()
        elif 'pg_dump' in ' '.join(args):
            assert not state['running']['ai-voice-app']
            assert not state['running']['ai-voice-nginx']
            kwargs['stdout'].write(b'synthetic archive')
            code = int(state['fault'] == 'backup')
        elif args[:2] == ['docker', 'run']:
            image = args[args.index('--entrypoint') + 2]
            if args[-1] == module.CHECK_SCHEMA:
                compatible = ('old' if image == module.OLD_IMAGE else 'new')
                code = int(state['schema'] != compatible)
            elif args[-1] == '--prepare':
                if state['fault'] != 'migration-before-commit':
                    state['schema'] = 'new'
                if state['fault'] == 'unknown-schema':
                    state['schema'] = 'unknown'
                if state['fault'] == 'ambiguous-commit':
                    raise subprocess.TimeoutExpired(args, 90)
                code = int(state['fault'] in ('migration-before-commit', 'unknown-schema'))
        elif args[:2] == ['docker', 'compose'] and 'up' in args:
            override = Path(args[args.index('-f', args.index('-f') + 1) + 1])
            image = json.loads(override.read_text())['services']['app']['image']
            if image == 'sha256:candidate':
                assert state['schema'] == 'new'
            code = int(state['fault'] == 'replacement' and image == 'sha256:candidate')
            if code == 0:
                state['image'] = image
                state['running']['ai-voice-app'] = True
        elif args[0] == 'curl':
            output = b'{"status":"healthy"}'
            code = int(state['fault'] == 'https' and state['image'] == 'sha256:candidate')
        return subprocess.CompletedProcess(args, code, output, b'')

    monkeypatch.setattr(module.subprocess, 'run', execute)
    deployment = module.release_class(base)(directory)
    deployment.network = 'synthetic-net'
    deployment.live_env = directory / 'migration.env'
    return module, deployment, state, overrides, previous_bytes


@pytest.mark.parametrize('fault,expected', [
    ('backup', 'old'), ('migration-before-commit', 'old'),
    ('ambiguous-commit', 'fallback'), ('replacement', 'fallback'), ('https', 'fallback'),
])
def test_failure_recovers_to_image_that_matches_actual_schema(release, fault, expected, capsys):
    module, deployment, state, overrides, previous = release
    state['fault'] = fault
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
        deployment.deploy('sha256:candidate', overrides[0], 'sha256:fallback', overrides[1])
    image = module.OLD_IMAGE if expected == 'old' else 'sha256:fallback'
    assert state['image'] == image
    assert state['running'] == {'ai-voice-app': True, 'ai-voice-nginx': True}
    installed = json.loads((module.ROOT / 'docker-compose.override.yml').read_text())
    original = json.loads(previous)
    original['services']['app']['image'] = image
    assert installed == original
    assert not (deployment.directory / 'deployed').exists()
    assert 'RECOVERY_READY_HTTPS_OK' in capsys.readouterr().out
    assert not any('pg_restore' in command and '--list' not in command for command in state['commands'])
    if fault == 'ambiguous-commit':
        run_index = next(i for i, c in enumerate(state['commands']) if c[-1] == '--prepare')
        assert state['commands'][run_index + 1][:3] == ['docker', 'rm', '-f']


def test_unknown_schema_keeps_ingress_stopped_and_never_reports_recovery(release, capsys):
    module, deployment, state, overrides, previous = release
    state['fault'] = 'unknown-schema'
    with pytest.raises(RuntimeError):
        deployment.deploy('sha256:candidate', overrides[0], 'sha256:fallback', overrides[1])
    assert not any(state['running'].values())
    assert (module.ROOT / 'docker-compose.override.yml').read_bytes() == previous
    assert 'RECOVERY_READY_HTTPS_OK' not in capsys.readouterr().out


def test_success_migrates_before_replacement_preserves_config_and_records_release(release, capsys):
    module, deployment, state, overrides, previous = release
    deployment.deploy('sha256:candidate', overrides[0], 'sha256:fallback', overrides[1])
    assert state['schema'] == 'new'
    assert state['image'] == 'sha256:candidate'
    assert all(state['running'].values())
    assert (deployment.directory / 'deployed').read_text().splitlines() == [module.APP_COMMIT, 'sha256:candidate']
    assert 'BOOKING_PERSISTENCE_DEPLOYED_READY_HTTPS_OK' in capsys.readouterr().out


def test_changed_override_aborts_before_stopping_or_migrating(release):
    module, deployment, state, overrides, _ = release
    (module.ROOT / 'docker-compose.override.yml').write_text('changed')
    with pytest.raises(RuntimeError):
        deployment.deploy('sha256:candidate', overrides[0], 'sha256:fallback', overrides[1])
    assert state['schema'] == 'old'
    assert all(state['running'].values())
    assert not any(c[:2] == ['docker', 'stop'] for c in state['commands'])
