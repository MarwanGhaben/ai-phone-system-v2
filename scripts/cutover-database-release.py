"""One-off, owner-run cutover for the rehearsed phase-2 beta image."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/opt/ai-phone-system-v2')
RELEASE = Path('/opt/ai-phone-release.02W6q2')
COMMIT = '1480eeb0e36596e26d0dedde77f08beee91aec5b'
BASE = '1affd83320e157343691a44934fac84f7aaa2490'
IMAGE = 'sha256:01cee33a5a13aea37919568f6d00e0f2c12a8e98bd6eb93ab063b73f51711605'
BRANCH = 'codex/phase-2-database-foundation'
SETTINGS = 'import json; from config.settings import settings; print(json.dumps(settings.model_dump(mode="json"),sort_keys=True))'


def run(*args, timeout=60, check=True):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, timeout=timeout)
    with (RELEASE / 'cutover-private.log').open('ab') as log:
        log.write(result.stdout + result.stderr)
    if check and result.returncode:
        raise RuntimeError('command failed; private log retained')
    return result


def inspect(name):
    return json.loads(run('docker', 'inspect', name).stdout)[0]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    path.chmod(0o600)


def compose_escape(value):
    # Compose interpolates dollar signs even inside JSON strings.
    if isinstance(value, str):
        return value.replace('$', '$$')
    if isinstance(value, dict):
        return {k: compose_escape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [compose_escape(v) for v in value]
    return value


def wait_health(endpoint, seconds=180):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = run('docker', 'exec', 'ai-voice-app', 'curl', '-fsS',
                     '--max-time', '8', 'http://127.0.0.1:8000/' + endpoint,
                     check=False)
        if result.returncode == 0:
            body = json.loads(result.stdout)
            if body.get('status') == ('ready' if endpoint == 'ready' else 'healthy'):
                return
        time.sleep(2)
    raise RuntimeError('application health deadline exceeded')


def main():
    os.umask(0o077)
    if os.geteuid() != 0 or not RELEASE.is_dir():
        raise RuntimeError('expected root account and release directory')
    for filename, expected in [('commit', COMMIT), ('rehearsal-passed', COMMIT),
                               ('candidate-image-id', IMAGE)]:
        if (RELEASE / filename).read_text().strip() != expected:
            raise RuntimeError('release evidence mismatch')
    if (RELEASE / 'cutover-started').exists():
        raise RuntimeError('cutover already attempted; review its result before retrying')
    if run('git', 'rev-parse', 'HEAD').stdout.decode().strip() != BASE:
        raise RuntimeError('unexpected server checkout')
    if run('git', 'status', '--porcelain', '--untracked-files=no').stdout.strip():
        raise RuntimeError('tracked server changes')
    if run('git', 'branch', '--list', BRANCH).stdout.strip():
        raise RuntimeError('phase branch already exists locally')
    # The only live source mount is config; it must not change during checkout.
    if run('git', 'diff', BASE, COMMIT, '--', 'config').stdout.strip():
        raise RuntimeError('unexpected mounted configuration change')
    old = inspect('ai-voice-app')
    nginx = inspect('ai-voice-nginx')
    if old['Image'] != (RELEASE / 'previous-image-id').read_text().strip():
        raise RuntimeError('running image changed since rehearsal')
    if old['State'].get('Health', {}).get('Status') != 'healthy':
        raise RuntimeError('old app is not healthy')
    if nginx['Config'].get('StopSignal', '').upper() not in ('SIGQUIT', 'QUIT', '3'):
        raise RuntimeError('nginx graceful stop signal not verified')
    image = inspect(IMAGE)
    if image['Id'] != IMAGE:
        raise RuntimeError('candidate image unavailable')
    if old['Config'].get('Entrypoint') != image.get('Config', {}).get('Entrypoint'):
        raise RuntimeError('unexpected entrypoint change')
    project = old['Config']['Labels']['com.docker.compose.project']
    if project != 'ai-phone-system-v2':
        raise RuntimeError('unexpected Compose project')
    if (ROOT / 'docker-compose.override.yml').exists():
        raise RuntimeError('existing automatic override must be reviewed')
    old_settings = json.loads(run('docker', 'exec', 'ai-voice-app', 'python',
                                  '-c', SETTINGS).stdout)
    env = dict(item.split('=', 1) for item in old['Config']['Env'])
    # Preserve the old settings file as a runtime-only secret, never in an image.
    raw = run('docker', 'exec', 'ai-voice-app', 'python', '-c',
              'import base64,pathlib; p=pathlib.Path("/app/.env"); '
              'print(base64.b64encode(p.read_bytes() if p.is_file() else b"").decode())').stdout
    runtime_env = RELEASE / 'runtime.env'
    runtime_env.write_bytes(base64.b64decode(raw.strip(), validate=True))
    runtime_env.chmod(0o600)
    app_override = {'image': IMAGE, 'environment': env,
                    'command': old['Config']['Cmd'],
                    'volumes': [str(runtime_env) + ':/app/.env:ro'],
                    'healthcheck': {'test': ['CMD', 'curl', '-f', 'http://localhost:8000/ready']}}
    candidate = RELEASE / 'candidate-override.json'
    write_json(candidate, compose_escape({'services': {
        'app': app_override,
        'migrate': {'image': IMAGE, 'environment': {
            'MIGRATION_DATABASE_URL': old_settings['database_url']}}}}))
    rollback = RELEASE / 'rollback-override.json'
    old_override = dict(app_override, image=old['Image'], healthcheck={
        'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health']})
    write_json(rollback, compose_escape({'services': {'app': old_override}}))
    common = ['docker', 'compose', '--project-directory', str(ROOT), '-p', project]
    new_cmd = common + ['-f', str(RELEASE / 'source/docker-compose.yml'), '-f', str(candidate)]
    old_cmd = common + ['-f', str(RELEASE / 'previous-compose.yml'), '-f', str(rollback)]
    rendered = json.loads(run(*new_cmd, 'config', '--format', 'json').stdout)
    app = rendered['services']['app']
    original_mounts = {(m['Source'], m['Destination'], not m['RW']) for m in old['Mounts']}
    new_mounts = {(m['source'], m['target'], m.get('read_only', False)) for m in app['volumes']}
    if new_mounts != original_mounts | {(str(runtime_env), '/app/.env', True)}:
        raise RuntimeError('application mounts differ from running container')
    if set(app['networks']) != {'ai-voice-network'}:
        raise RuntimeError('unexpected application network')
    if rendered['networks']['ai-voice-network']['name'] not in old['NetworkSettings']['Networks']:
        raise RuntimeError('rendered network differs from running network')
    for k, v in env.items():
        # Some Compose versions re-escape dollars when serializing config JSON.
        if app['environment'].get(k) not in (v, compose_escape(v)):
            raise RuntimeError('environment interpolation mismatch')
    probe = 't005c-settings-check-02w6q2'
    reserved = {probe, 't005c-live-migrate-02w6q2'}
    names = run('docker', 'container', 'ls', '-a', '--format', '{{.Names}}').stdout.decode().splitlines()
    if reserved.intersection(names):
        raise RuntimeError('a reserved cutover container already exists')
    try:
        result = run(*new_cmd, 'run', '--rm', '--no-deps', '--pull', 'never',
                     '--name', probe, '--entrypoint', 'python', 'app', '-c', SETTINGS,
                     timeout=90)
        if json.loads(result.stdout) != old_settings:
            raise RuntimeError('effective application settings differ')
    finally:
        # This exact name is reserved for the configuration-only probe.
        run('docker', 'rm', '-f', probe, check=False)
        names = run('docker', 'container', 'ls', '-a', '--format', '{{.Names}}').stdout.decode().splitlines()
        if probe in names:
            raise RuntimeError('settings probe cleanup failed')
    print('EFFECTIVE_SETTINGS_AND_MOUNTS_PRESERVED', flush=True)
    # Prepare protection before creating an automatic Compose override with secrets.
    exclude = ROOT / '.git/info/exclude'
    with exclude.open('a', encoding='utf-8') as f:
        f.write('\n/docker-compose.override.yml\n')
    (RELEASE / 'cutover-started').write_text(COMMIT + '\n')
    switched = False
    try:
        print('STOPPING_TEST_TRAFFIC_AND_APPLICATION', flush=True)
        run('docker', 'stop', '--time', '120', 'ai-voice-nginx', timeout=135)
        run('docker', 'stop', '--time', '60', 'ai-voice-app', timeout=75)
        with (RELEASE / 'final-database.dump').open('xb') as backup:
            result = subprocess.run(['docker', 'exec', 'ai-voice-db', 'sh', '-c',
                'exec pg_dump -Fc -U "$POSTGRES_USER" -d "$POSTGRES_DB"'],
                stdout=backup, stderr=subprocess.PIPE, timeout=120)
        if result.returncode or (RELEASE / 'final-database.dump').stat().st_size == 0:
            raise RuntimeError('final database backup failed')
        digest = hashlib.sha256((RELEASE / 'final-database.dump').read_bytes()).hexdigest()
        (RELEASE / 'final-database.sha256').write_text(digest + '  final-database.dump\n')
        print('FINAL_BACKUP_AFTER_WRITER_STOP_OK', flush=True)
        run(*new_cmd, 'run', '--rm', '--no-deps', '--pull', 'never',
            '--name', 't005c-live-migrate-02w6q2', 'migrate', timeout=90)
        print('LIVE_DATABASE_MIGRATION_OK', flush=True)
        run(*new_cmd, 'up', '-d', '--no-deps', '--no-build', '--pull', 'never',
            '--force-recreate', 'app', timeout=120)
        wait_health('ready')
        if inspect('ai-voice-app')['Image'] != IMAGE:
            raise RuntimeError('unexpected running candidate image')
        if json.loads(run('docker', 'exec', 'ai-voice-app', 'python', '-c', SETTINGS).stdout) != old_settings:
            raise RuntimeError('running settings changed')
        run('git', 'switch', '-c', BRANCH, COMMIT)
        switched = True
        (ROOT / 'docker-compose.override.yml').write_bytes(candidate.read_bytes())
        (ROOT / 'docker-compose.override.yml').chmod(0o600)
        run('docker', 'start', 'ai-voice-nginx')
        run('docker', 'exec', 'ai-voice-nginx', 'nginx', '-t')
        for attempt in range(20):
            response = run('curl', '-fsS', '--max-time', '5',
                           'http://127.0.0.1/health', check=False)
            if response.returncode == 0 and json.loads(response.stdout).get('status') == 'healthy':
                break
            time.sleep(1)
        else:
            raise RuntimeError('nginx proxy health failed')
        (RELEASE / 'deployed').write_text(COMMIT + '\n' + IMAGE + '\n')
        print('DEPLOYMENT_COMPLETE_DATABASE_READY_PROXY_OK', flush=True)
        print('DEPLOYED_COMMIT=' + COMMIT, flush=True)
        print('DEPLOYED_IMAGE=' + IMAGE, flush=True)
    except BaseException:
        print('CUTOVER_FAILED_RESTORING_PREVIOUS_APP', flush=True)
        run('docker', 'rm', '-f', 't005c-live-migrate-02w6q2', check=False)
        run('docker', 'stop', '--time', '30', 'ai-voice-nginx', check=False)
        run(*old_cmd, 'up', '-d', '--no-deps', '--no-build', '--pull', 'never',
            '--force-recreate', 'app', timeout=120)
        wait_health('health')
        if switched:
            run('git', 'switch', 'main')
        (ROOT / 'docker-compose.override.yml').write_bytes(rollback.read_bytes())
        (ROOT / 'docker-compose.override.yml').chmod(0o600)
        run('docker', 'start', 'ai-voice-nginx')
        run('curl', '-fsS', '--max-time', '10', 'http://127.0.0.1/health')
        print('PREVIOUS_APP_RESTORED_DATABASE_NOT_REWOUND', flush=True)
        raise


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if isinstance(exc, RuntimeError):
            # RuntimeError messages in this script are fixed, credential-free text.
            print('CUTOVER_REASON=' + str(exc), flush=True)
        else:
            print('CUTOVER_FAILURE_TYPE=' + type(exc).__name__, flush=True)
        print('CUTOVER_STOPPED_REVIEW_PRIVATE_LOG_LOCALLY_DO_NOT_PASTE_SECRETS', flush=True)
        sys.exit(1)
