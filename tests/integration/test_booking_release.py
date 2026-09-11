"""App-only deployment scenarios with Docker/process boundaries simulated."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def release(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / 'scripts/deploy-booking-guards.py'
    spec = importlib.util.spec_from_file_location('booking_release', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path / 'repo'
    module.ROOT.mkdir()
    directory = tmp_path / 'release'
    directory.mkdir()
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    deployment = module.Release(directory)
    settings = {'secret_key': 'synthetic-$secret', 'default_language': 'ar'}
    previous = {'services': {'app': {'image': module.OLD_IMAGE,
                'environment': {'SECRET_KEY': 'synthetic-$$secret'},
                'volumes': ['/private/runtime.env:/app/.env:ro']},
                'migrate': {'image': module.OLD_IMAGE}}}
    previous_bytes = json.dumps(previous).encode()
    (directory / 'previous-override.json').write_bytes(previous_bytes)
    (module.ROOT / 'docker-compose.override.yml').write_bytes(previous_bytes)
    (directory / 'previous-settings.json').write_text(json.dumps(settings))
    previous['services']['app']['image'] = 'sha256:candidate'
    candidate = directory / 'candidate-override.json'
    candidate.write_text(json.dumps(previous))
    state = {'image': module.OLD_IMAGE, 'commands': [], 'fault': None, 'rollback': False}

    def execute(args, **kwargs):
        args = list(args)
        state['commands'].append(args)
        code, output = 0, b''
        if args[:2] == ['docker', 'inspect']:
            output = json.dumps([{'Image': state['image']}]).encode()
        elif args[:3] == ['docker', 'exec', 'ai-voice-app']:
            output = json.dumps(settings if args[-1] == module.SETTINGS else {'status': 'ready'}).encode()
            if args[-1] == module.SETTINGS and state['fault'] == 'running-settings' and not state['rollback']:
                output = b'{"default_language":"changed"}'
        elif args[:2] == ['docker', 'compose']:
            if 'up' in args:
                state['rollback'] = str(directory / 'previous-override.json') in args
                if not state['rollback'] and state['fault'] == 'replace':
                    code = 1
                else:
                    state['image'] = module.OLD_IMAGE if state['rollback'] else 'sha256:candidate'
            elif 'config' in args:
                output = json.dumps({'services': {'app': {
                    'volumes': [{'source': '/private/runtime.env', 'target': '/app/.env', 'read_only': True}],
                    'networks': {'net': {}}}}, 'networks': {'net': {'name': 'actual_net'}}}).encode()
            elif 'run' in args:
                output = json.dumps(settings if state['fault'] != 'probe-settings' else {}).encode()
        elif args[0] == 'curl':
            code = int(state['fault'] == 'https' and not state['rollback'])
            output = b'{"status":"healthy"}'
        return subprocess.CompletedProcess(args, code, output, b'')

    monkeypatch.setattr(module.subprocess, 'run', execute)
    return module, deployment, state, candidate, previous_bytes


@pytest.mark.parametrize('fault', ['replace', 'running-settings', 'https'])
def test_failed_app_release_restores_image_and_exact_override(release, fault, capsys):
    module, deployment, state, candidate, previous_bytes = release
    state['fault'] = fault
    with pytest.raises(RuntimeError):
        deployment.cutover('a' * 40, 'sha256:candidate', candidate)
    assert state['image'] == module.OLD_IMAGE
    assert (module.ROOT / 'docker-compose.override.yml').read_bytes() == previous_bytes
    assert not (deployment.directory / 'deployed').exists()
    assert 'PREVIOUS_APP_RESTORED_DATABASE_UNCHANGED' in capsys.readouterr().out
    stopped = [args[-1] for args in state['commands'] if args[:2] == ['docker', 'stop']]
    assert set(stopped) <= {'ai-voice-app', 'ai-voice-nginx'}
    assert not any(args[0] == 'git' or 'migrate' in args or 'pg_restore' in args for args in state['commands'])


def test_success_pins_only_app_image_and_records_source(release, capsys):
    module, deployment, state, candidate, previous_bytes = release
    deployment.cutover('a' * 40, 'sha256:candidate', candidate)
    assert state['image'] == 'sha256:candidate'
    installed = json.loads((module.ROOT / 'docker-compose.override.yml').read_text())
    expected = json.loads(previous_bytes)
    expected['services']['app']['image'] = 'sha256:candidate'
    assert installed == expected
    assert (deployment.directory / 'deployed').read_text().splitlines() == ['a' * 40, 'sha256:candidate']
    output = capsys.readouterr().out
    assert 'BOOKING_GUARDS_DEPLOYED_READY_HTTPS_OK' in output
    assert 'synthetic' not in output


def test_changed_override_blocks_before_downtime(release):
    module, deployment, state, candidate, _ = release
    (module.ROOT / 'docker-compose.override.yml').write_text('{}')
    with pytest.raises(RuntimeError, match='state changed'):
        deployment.cutover('a' * 40, 'sha256:candidate', candidate)
    assert not any(args[:2] == ['docker', 'stop'] for args in state['commands'])


@pytest.mark.parametrize('fault', ['probe-settings', 'mounts'])
def test_probe_drift_blocks_before_downtime_and_cleans_probe(release, fault):
    module, deployment, state, candidate, previous_bytes = release
    state['fault'] = fault
    old = {'Mounts': [{'Source': '/private/runtime.env', 'Destination': '/app/.env', 'RW': False}],
           'NetworkSettings': {'Networks': {'actual_net': {}}}}
    if fault == 'mounts':
        old['Mounts'] = []
    with pytest.raises(RuntimeError):
        deployment.prepare_override('sha256:candidate', old, json.loads(previous_bytes))
    assert (module.ROOT / 'docker-compose.override.yml').read_bytes() == previous_bytes
    assert not any(args[:2] == ['docker', 'stop'] for args in state['commands'])
    if fault == 'probe-settings':
        assert any(args[:3] == ['docker', 'rm', '-f'] for args in state['commands'])
