"""Cutover failure-path tests; Docker/Git are simulated external boundaries."""
import base64
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def cutover(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / 'scripts/cutover-database-release.py'
    spec = importlib.util.spec_from_file_location('cutover_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path / 'repo'
    module.RELEASE = tmp_path / 'release'
    module.ROOT.mkdir()
    module.RELEASE.mkdir()
    (module.ROOT / '.git/info').mkdir(parents=True)
    monkeypatch.setattr(module.os, 'geteuid', lambda: 0, raising=False)
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: None)
    for name, value in {'commit': module.COMMIT, 'rehearsal-passed': module.COMMIT,
                        'candidate-image-id': module.IMAGE,
                        'previous-image-id': 'sha256:old'}.items():
        (module.RELEASE / name).write_text(value)
    state = {'image': 'sha256:old', 'head': module.BASE, 'commands': [], 'fault': None}
    settings = {'database_url': 'postgresql://synthetic:password@postgres/test',
                'default_language': 'en'}
    env = {'SECRET_KEY': 'synthetic-$secret', 'DATABASE_URL': settings['database_url']}

    def execute(args, **kwargs):
        args = list(args)
        state['commands'].append(args)
        code, output = 0, b''
        if args[:2] == ['docker', 'inspect']:
            name = args[2]
            if name == 'ai-voice-app':
                value = {'Image': state['image'], 'State': {'Health': {'Status': 'healthy'}},
                         'Mounts': [], 'Config': {'Cmd': ['uvicorn', 'api.main:app'],
                         'Env': [k+'='+v for k, v in env.items()], 'Healthcheck': {},
                         'Labels': {'com.docker.compose.project': 'ai-phone-system-v2'}},
                         'NetworkSettings': {'Networks': {'test_network': {}}}}
            elif name == 'ai-voice-nginx':
                value = {'Config': {'StopSignal': 'SIGQUIT'}}
            else:
                value = {'Id': module.IMAGE}
            output = json.dumps([value]).encode()
        elif args[:3] == ['git', 'rev-parse', 'HEAD']:
            output = state['head'].encode()
        elif args[:2] == ['git', 'switch']:
            state['head'] = module.COMMIT if '-c' in args else module.BASE
        elif args[:3] == ['docker', 'exec', 'ai-voice-app']:
            if args[-1] == module.SETTINGS:
                output = json.dumps(settings).encode()
            elif args[-1].endswith('/ready'):
                output = b'{"status":"ready"}'
            elif args[-1].endswith('/health'):
                output = b'{"status":"healthy"}'
            else:
                output = base64.b64encode(b'DEFAULT_LANGUAGE=en\n')
        elif args[:2] == ['docker', 'compose']:
            if 'config' in args:
                output = json.dumps({'services': {'app': {'environment': {
                    k: v.replace('$', '$$') for k, v in env.items()},
                    'volumes': [{'source': str(module.RELEASE / 'runtime.env'),
                                 'target': '/app/.env', 'read_only': True}],
                    'networks': {'ai-voice-network': {}}}}, 'networks': {
                    'ai-voice-network': {'name': 'test_network'}}}).encode()
            elif args[-1] == module.SETTINGS:
                value = dict(settings)
                if state['fault'] == 'settings':
                    value['default_language'] = 'auto'
                output = json.dumps(value).encode()
            elif 'run' in args and args[-1] == 'migrate':
                code = int(state['fault'] == 'migration')
            elif 'up' in args:
                restoring = any('rollback-override.json' in a for a in args)
                if not restoring and state['fault'] == 'replacement':
                    code = 1
                else:
                    state['image'] = 'sha256:old' if restoring else module.IMAGE
        elif args[:3] == ['docker', 'exec', 'ai-voice-db'] and 'sh' in args:
            kwargs['stdout'].write(b'synthetic dump')
        elif args[0] == 'curl':
            output = b'{"status":"healthy"}'
            if state['fault'] == 'proxy' and state['image'] == module.IMAGE:
                code = 1
        return subprocess.CompletedProcess(args, code, output, b'')

    monkeypatch.setattr(module.subprocess, 'run', execute)
    return module, state


def test_setting_drift_aborts_before_stopping_any_container(cutover):
    module, state = cutover
    state['fault'] = 'settings'
    with pytest.raises(RuntimeError, match='settings differ'):
        module.main()
    assert not any(c[:2] == ['docker', 'stop'] for c in state['commands'])
    assert not (module.RELEASE / 'cutover-started').exists()
    assert not (module.ROOT / 'docker-compose.override.yml').exists()


@pytest.mark.parametrize('fault', ['migration', 'replacement', 'proxy'])
def test_failed_cutover_restores_old_app_without_database_rewind(cutover, fault, capsys):
    module, state = cutover
    state['fault'] = fault
    with pytest.raises(RuntimeError):
        module.main()
    assert state['image'] == 'sha256:old'
    assert state['head'] == module.BASE
    assert not (module.RELEASE / 'deployed').exists()
    assert (module.RELEASE / 'final-database.dump').read_bytes() == b'synthetic dump'
    assert not any('pg_restore' in c and '--version' not in c for c in state['commands'])
    assert ['docker', 'start', 'ai-voice-nginx'] in state['commands']
    assert 'PREVIOUS_APP_RESTORED_DATABASE_NOT_REWOUND' in capsys.readouterr().out


def test_success_records_exact_image_and_keeps_database_and_redis_running(cutover, capsys):
    module, state = cutover
    module.main()
    assert state['image'] == module.IMAGE
    assert state['head'] == module.COMMIT
    assert (module.RELEASE / 'deployed').read_text().splitlines() == [module.COMMIT, module.IMAGE]
    stopped = [c[-1] for c in state['commands'] if c[:2] == ['docker', 'stop']]
    assert stopped == ['ai-voice-nginx', 'ai-voice-app']
    override = json.loads((module.ROOT / 'docker-compose.override.yml').read_text())
    assert override['services']['app']['environment']['SECRET_KEY'] == 'synthetic-$$secret'
    assert 'synthetic-$secret' not in capsys.readouterr().out
