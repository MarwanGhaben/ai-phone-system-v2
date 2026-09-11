"""Owner-run, app-only T020-A release; preserves the existing private override."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path('/opt/ai-phone-system-v2')
BASE = '1480eeb0e36596e26d0dedde77f08beee91aec5b'
OLD_IMAGE = 'sha256:01cee33a5a13aea37919568f6d00e0f2c12a8e98bd6eb93ab063b73f51711605'
NGINX_HASH = '59ac9bdaa9a86c1022a573c9709faddf7e93a53015da8acfcc4330b3e288f3ab'
APP_PATH = 'services/conversation/orchestrator.py'
RUNTIME_PATHS = [APP_PATH]
SETTINGS = 'import json; from config.settings import settings; print(json.dumps(settings.model_dump(mode="json"),sort_keys=True))'


class Release:
    def __init__(self, directory):
        self.directory = directory

    def run(self, *args, timeout=60, check=True):
        result = subprocess.run(args, cwd=ROOT, capture_output=True, timeout=timeout)
        with (self.directory / 'private.log').open('ab') as log:
            log.write(result.stdout + result.stderr)
        if check and result.returncode:
            raise RuntimeError('command failed; protected log retained')
        return result

    def inspect(self, name):
        return json.loads(self.run('docker', 'inspect', name).stdout)[0]

    def compose(self, override, *args, **kwargs):
        return self.run('docker', 'compose', '--project-directory', str(ROOT),
                        '-p', 'ai-phone-system-v2', '-f', str(ROOT / 'docker-compose.yml'),
                        '-f', str(override), *args, **kwargs)

    def settings(self):
        return json.loads(self.run('docker', 'exec', 'ai-voice-app',
                                   'python', '-c', SETTINGS).stdout)

    def wait_ready(self):
        for _ in range(60):
            response = self.run('docker', 'exec', 'ai-voice-app', 'curl', '-fsS',
                                '--max-time', '3', 'http://localhost:8000/ready', check=False)
            if response.returncode == 0 and json.loads(response.stdout).get('status') == 'ready':
                return
            time.sleep(2)
        raise RuntimeError('readiness deadline exceeded')

    def public_health(self):
        for _ in range(12):
            response = self.run('curl', '-fsS', '--max-time', '5',
                                'https://aiagent.ghaben.ca:8443/health', check=False)
            if response.returncode == 0 and json.loads(response.stdout).get('status') == 'healthy':
                return
            time.sleep(1)
        raise RuntimeError('public HTTPS health failed')

    def remove_probe(self, name):
        self.run('docker', 'rm', '-f', name, check=False)
        names = self.run('docker', 'ps', '-a', '--format', '{{.Names}}').stdout.decode().splitlines()
        if name in names:
            raise RuntimeError('probe cleanup failed')

    def preflight(self, commit):
        if self.run('git', 'rev-parse', 'HEAD').stdout.decode().strip() != BASE:
            raise RuntimeError('server checkout changed; do not rerun blindly')
        if self.run('git', 'rev-parse', commit + '^{commit}').stdout.decode().strip() != commit:
            raise RuntimeError('candidate commit unavailable')
        if self.run('git', 'status', '--porcelain', '--untracked-files=no').stdout.decode().strip() != 'M nginx/nginx.conf':
            raise RuntimeError('unexpected tracked server changes')
        if hashlib.sha256((ROOT / 'nginx/nginx.conf').read_bytes()).hexdigest() != NGINX_HASH:
            raise RuntimeError('HTTPS configuration changed')
        # Only the one reviewed runtime file may differ from the deployed source.
        changed = self.run('git', 'diff', '--name-only', BASE, commit, '--',
                           'api', 'services', 'config', 'models', 'clients', 'migrations',
                           'requirements.txt', 'Dockerfile', 'docker-compose.yml').stdout.decode().splitlines()
        if changed != sorted(RUNTIME_PATHS):
            raise RuntimeError('candidate exceeds app-only scope')
        old = self.inspect('ai-voice-app')
        if old['Image'] != OLD_IMAGE or old['State']['Health']['Status'] != 'healthy':
            raise RuntimeError('expected healthy previous image is not running')
        if old['Config']['Labels'].get('com.docker.compose.project') != 'ai-phone-system-v2':
            raise RuntimeError('unexpected Compose project')
        nginx = self.inspect('ai-voice-nginx')
        if not nginx['State']['Running'] or nginx['Config'].get('StopSignal', '').upper() not in ('SIGQUIT', 'QUIT', '3'):
            raise RuntimeError('nginx graceful stop not verified')
        if shutil.disk_usage(ROOT).free < 1024 ** 3:
            raise RuntimeError('at least 1 GiB free disk required')
        override = ROOT / 'docker-compose.override.yml'
        if override.is_symlink() or override.stat().st_uid != 0 or override.stat().st_mode & 0o077:
            raise RuntimeError('private override must be a protected root-owned file')
        configuration = json.loads(override.read_text())
        if configuration['services']['app']['image'] != OLD_IMAGE:
            raise RuntimeError('private override image differs')
        (self.directory / 'previous-override.json').write_bytes(override.read_bytes())
        (self.directory / 'previous-settings.json').write_text(json.dumps(self.settings()))
        (self.directory / 'previous-app.json').write_text(json.dumps(old))
        self.run('docker', 'tag', OLD_IMAGE, 'ai-phone-rollback:t020a-' + self.directory.name)
        self.public_health()
        return old, configuration

    def build_and_test(self, commit):
        build = self.directory / 'build'
        build.mkdir()
        source = self.run('git', 'show', commit + ':' + APP_PATH).stdout
        (build / 'orchestrator.py').write_bytes(source)
        base_tag = 'ai-phone-rollback:t020a-' + self.directory.name
        if self.inspect(base_tag)['Id'] != OLD_IMAGE:
            raise RuntimeError('build base image mismatch')
        (build / 'Dockerfile').write_text(
            'FROM ' + base_tag + '\nCOPY orchestrator.py /app/' + APP_PATH + '\n')
        tag = 'ai-phone-candidate:t020a-' + self.directory.name
        print('BUILDING_APP_ONLY_CANDIDATE_CURRENT_APP_UNCHANGED', flush=True)
        self.run('docker', 'build', '--network', 'none', '--pull=false',
                 '--label', 'org.opencontainers.image.revision=' + commit,
                 '-t', tag, str(build), timeout=300)
        image = self.inspect(tag)['Id']
        tests = self.directory / 'tests'
        (tests / 'conversation').mkdir(parents=True)
        for path in ['conftest.py', 'conversation/test_booking_safety_guards.py',
                     'conversation/test_booking_time_validation.py']:
            (tests / path).write_bytes(self.run('git', 'show', commit + ':tests/' + path).stdout)
        probe = 't020a-tests-' + self.directory.name
        try:
            self.run('docker', 'run', '--rm', '--name', probe, '--network', 'none',
                     '--memory', '384m', '--cpus', '1', '--pull', 'never',
                     '--mount', 'type=bind,src=' + str(tests) + ',dst=/audit/tests,readonly',
                     '--entrypoint', 'python', image, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                     '/audit/tests', timeout=120)
        finally:
            self.remove_probe(probe)
        print('CANDIDATE_IMAGE_BOOKING_TESTS_PASSED', flush=True)
        return image

    def prepare_override(self, image, old, configuration):
        configuration['services']['app']['image'] = image
        candidate = self.directory / 'candidate-override.json'
        candidate.write_text(json.dumps(configuration, indent=2) + '\n')
        rendered = json.loads(self.compose(candidate, 'config', '--format', 'json').stdout)
        app = rendered['services']['app']
        expected_mounts = {(m['Source'], m['Destination'], not m['RW']) for m in old['Mounts']}
        mounts = {(m['source'], m['target'], m.get('read_only', False)) for m in app['volumes']}
        if mounts != expected_mounts:
            raise RuntimeError('app mounts changed')
        networks = {rendered['networks'][n]['name'] for n in app['networks']}
        if networks != set(old['NetworkSettings']['Networks']):
            raise RuntimeError('app networks changed')
        probe = 't020a-settings-' + self.directory.name
        try:
            result = self.compose(candidate, 'run', '--rm', '--no-deps', '--pull', 'never',
                                  '--name', probe, '--entrypoint', 'python', 'app', '-c', SETTINGS)
            if json.loads(result.stdout) != json.loads((self.directory / 'previous-settings.json').read_text()):
                raise RuntimeError('candidate effective settings changed')
        finally:
            self.remove_probe(probe)
        print('EFFECTIVE_SETTINGS_AND_MOUNTS_PRESERVED', flush=True)
        return candidate

    def replace_app(self, override, image):
        self.compose(override, 'up', '-d', '--no-deps', '--no-build', '--pull', 'never',
                     '--force-recreate', 'app', timeout=120)
        self.wait_ready()
        if self.inspect('ai-voice-app')['Image'] != image:
            raise RuntimeError('replacement image mismatch')
        if self.settings() != json.loads((self.directory / 'previous-settings.json').read_text()):
            raise RuntimeError('replacement settings changed')

    def cutover(self, commit, image, candidate):
        previous = self.directory / 'previous-override.json'
        live_override = ROOT / 'docker-compose.override.yml'
        if live_override.read_bytes() != previous.read_bytes() or self.inspect('ai-voice-app')['Image'] != OLD_IMAGE:
            raise RuntimeError('deployment state changed during preparation')
        print('STOPPING_TEST_INGRESS_AND_REPLACING_APP', flush=True)
        (self.directory / 'cutover-started').write_text(commit + '\n')
        try:
            self.run('docker', 'stop', '--time', '120', 'ai-voice-nginx', timeout=135)
            self.run('docker', 'stop', '--time', '60', 'ai-voice-app', timeout=75)
            self.replace_app(candidate, image)
            live_override.write_bytes(candidate.read_bytes())
            self.run('docker', 'start', 'ai-voice-nginx')
            self.run('docker', 'exec', 'ai-voice-nginx', 'nginx', '-t')
            self.public_health()
            (self.directory / 'deployed').write_text(commit + '\n' + image + '\n')
            print('BOOKING_GUARDS_DEPLOYED_READY_HTTPS_OK', flush=True)
            print('DEPLOYED_COMMIT=' + commit, flush=True)
            print('DEPLOYED_IMAGE=' + image, flush=True)
        except BaseException:
            print('CUTOVER_FAILED_RESTORING_PREVIOUS_APP', flush=True)
            self.run('docker', 'stop', '--time', '30', 'ai-voice-nginx', check=False)
            live_override.write_bytes(previous.read_bytes())
            self.replace_app(previous, OLD_IMAGE)
            self.run('docker', 'start', 'ai-voice-nginx')
            self.public_health()
            print('PREVIOUS_APP_RESTORED_DATABASE_UNCHANGED', flush=True)
            raise


def main(commit):
    import fcntl
    if os.geteuid() != 0 or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise RuntimeError('root and full candidate commit required')
    os.umask(0o077)
    with open('/run/ai-phone-deployment.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = Path(tempfile.mkdtemp(prefix='ai-phone-booking-release.', dir='/opt'))
        print('RELEASE_DIRECTORY=' + str(directory), flush=True)
        release = Release(directory)
        old, configuration = release.preflight(commit)
        image = release.build_and_test(commit)
        candidate = release.prepare_override(image, old, configuration)
        release.cutover(commit, image, candidate)


if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except BaseException as exc:
        print('RELEASE_STOPPED_TYPE=' + type(exc).__name__, flush=True)
        print('KEEP_PROTECTED_RELEASE_FILES_DO_NOT_PASTE_PRIVATE_LOGS', flush=True)
        sys.exit(1)
