"""Owner-run T018-C 0003 rollout; no action occurs on import.

The script itself is fetched from an exact reviewed commit. Every staged source
file is read from that same commit. Public output contains only fixed markers.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from urllib.parse import unquote, urlsplit


ROOT = Path('/opt/ai-phone-system-v2')
BASE_CHECKOUT = '1480eeb0e36596e26d0dedde77f08beee91aec5b'
DEPLOYED_SOURCE = '075d8cf6fc29c4cbdf69b5c82baa130d5c0c540d'
OLD_IMAGE = 'sha256:f65fd1b449e2e107d2137158fc05bdbcc376c301b74c97e4a56c57a73b576a21'
NGINX_HASH = '59ac9bdaa9a86c1022a573c9709faddf7e93a53015da8acfcc4330b3e288f3ab'
SQL_HASHES = {
    'migrations/bootstrap_schema.sql': '1c72bbe0a120860902c565e5573c05ad1cd640891a42bb35cca2c6e0876f31e8',
    'migrations/0001_admin_users_updated_at.sql': '53c861ac91cae5ecaf9f794b15563c88ee58bc0dafa5ae64fa81c89d907aa3a9',
    'migrations/0002_bookings_aware_time.sql': '62addab700e047dff2a13973c881a874bfe6c7f88967d593f189987192782be6',
    'migrations/0003_booking_provider_observations.sql': 'b3abaa09c340f89c32778b0b17b5c2a584845983b13eb95dbb9c366f3d6c5830',
}
RUNTIME_PATHS = (
    'api/main.py',
    'config/settings.py',
    'migrations/runner.py',
    'migrations/schema_contract.py',
    'migrations/0003_booking_provider_observations.sql',
    'services/calendar/booking_readback.py',
    'services/scheduling/provider_observations.py',
    'services/dashboard/dashboard_routes.py',
    'templates/dashboard.html',
)
SOURCE_SCOPE = frozenset((*RUNTIME_PATHS, 'docker-compose.yml'))
FALLBACK_PATHS = ('migrations/schema_contract.py',)
OBSERVATION_ENV = {
    'BOOKING_OBSERVATION_ENABLED': 'true',
    'BOOKING_OBSERVATION_INTERVAL_SECONDS': '60',
    'BOOKING_OBSERVATION_FRESHNESS_SECONDS': '180',
}
OBSERVATION_SETTINGS = {
    'booking_observation_enabled': True,
    'booking_observation_interval_seconds': 60,
    'booking_observation_freshness_seconds': 180,
}
SETTINGS_TARGET = '/app/config/settings.py'
BUSINESS_TABLES = (
    'admin_sessions', 'admin_users', 'analytics_events', 'api_usage',
    'appointments', 'bookings', 'call_logs', 'callers', 'calls',
    'conversation_turns', 'conversations', 'knowledge_articles', 'mfa_codes',
    'sms_logs', 'system_metrics', 'tenants', 'users',
)
SETTINGS = ('import json; from config.settings import settings; '
            'print(json.dumps(settings.model_dump(mode="json"),sort_keys=True))')

# Read-only probes are deliberately separate: the old image knows only 0002.
CHECK_0002 = '''import asyncio,os,asyncpg
from migrations.schema_contract import check_runtime_compatibility
async def check():
 c=await asyncpg.connect(os.environ['MIGRATION_DATABASE_URL'],timeout=10,command_timeout=10)
 try:
  async with c.transaction(readonly=True):
   await check_runtime_compatibility(c)
   rows=await c.fetch('SELECT version,checksum FROM public.schema_migrations ORDER BY version')
   if [(r['version'],r['checksum']) for r in rows]!=[
    ('0001','53c861ac91cae5ecaf9f794b15563c88ee58bc0dafa5ae64fa81c89d907aa3a9'),
    ('0002','62addab700e047dff2a13973c881a874bfe6c7f88967d593f189987192782be6')]:
    raise RuntimeError('0002 history differs')
   if await c.fetchval("SELECT to_regclass('public.booking_provider_observations')") is not None:
    raise RuntimeError('0003 table already exists')
 finally:
  await c.close(timeout=5)
asyncio.run(check())
'''
CHECK_0003 = '''import asyncio,os,asyncpg
from migrations.schema_contract import check_runtime_compatibility
async def check():
 c=await asyncpg.connect(os.environ['MIGRATION_DATABASE_URL'],timeout=10,command_timeout=10)
 try:
  async with c.transaction(readonly=True):
   await check_runtime_compatibility(c)
   rows=await c.fetch('SELECT version,checksum FROM public.schema_migrations ORDER BY version')
   if [(r['version'],r['checksum']) for r in rows]!=[
    ('0001','53c861ac91cae5ecaf9f794b15563c88ee58bc0dafa5ae64fa81c89d907aa3a9'),
    ('0002','62addab700e047dff2a13973c881a874bfe6c7f88967d593f189987192782be6'),
    ('0003','b3abaa09c340f89c32778b0b17b5c2a584845983b13eb95dbb9c366f3d6c5830')]:
    raise RuntimeError('0003 history differs')
 finally:
  await c.close(timeout=5)
asyncio.run(check())
'''
FINGERPRINT_CODE = '''import asyncio,json,os,asyncpg
TABLES = ''' + repr(BUSINESS_TABLES) + '''
async def check():
 c=await asyncpg.connect(os.environ['MIGRATION_DATABASE_URL'],timeout=10,command_timeout=15)
 try:
  async with c.transaction(readonly=True):
   hashes={}
   for table in TABLES:
    sql="SELECT md5(COALESCE(jsonb_agg(to_jsonb(t) ORDER BY to_jsonb(t)::text)::text,'[]')) FROM public."+chr(34)+table+chr(34)+" t"
    hashes[table]=await c.fetchval(sql)
   counts=await c.fetchrow('SELECT count(*) FILTER (WHERE appointment_time_utc IS NOT NULL) AS aware, count(*) FILTER (WHERE appointment_time_utc IS NULL AND appointment_time IS NOT NULL) AS naive FROM public.bookings')
   print(json.dumps({'hashes':hashes,'aware':counts['aware'],'naive':counts['naive']},sort_keys=True))
 finally:
  await c.close(timeout=5)
asyncio.run(check())
'''
CHECK_EMPTY_OBSERVATIONS = '''import asyncio,os,asyncpg
async def check():
 c=await asyncpg.connect(os.environ['MIGRATION_DATABASE_URL'],timeout=10,command_timeout=10)
 try:
  async with c.transaction(readonly=True):
   if await c.fetchval('SELECT count(*) FROM public.booking_provider_observations') != 0:
    raise RuntimeError('observation rehearsal is not empty')
 finally:
  await c.close(timeout=5)
asyncio.run(check())
'''


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def settings_for(previous: dict, *, candidate: bool) -> dict:
    expected = dict(previous)
    if candidate:
        # Old image has no observation fields; do not weaken any other key.
        if OBSERVATION_SETTINGS.keys() & previous.keys():
            raise RuntimeError('unexpected previous observation settings')
        expected.update(OBSERVATION_SETTINGS)
    return expected


def normalized_compose(config: dict, *, candidate: bool, settings_source=None) -> dict:
    value = copy.deepcopy(config)
    app = value['services']['app']
    app['image'] = '<app-image>'
    environment = app.get('environment') or {}
    if not isinstance(environment, dict):
        raise RuntimeError('rendered app environment is not a mapping')
    if candidate:
        for key, expected in OBSERVATION_ENV.items():
            if str(environment.pop(key, None)) != expected:
                raise RuntimeError('observation environment differs')
    app['environment'] = environment
    if candidate and settings_source is not None:
        mounts = app.get('volumes', [])
        overlays = [m for m in mounts if m.get('target') == SETTINGS_TARGET]
        if (len(overlays) != 1 or overlays[0].get('source') != str(settings_source)
                or overlays[0].get('type') != 'bind'
                or overlays[0].get('read_only') is not True):
            raise RuntimeError('pinned settings mount differs')
        app['volumes'] = [m for m in mounts if m.get('target') != SETTINGS_TARGET]
    value['services']['migrate']['image'] = '<migration-image>'
    return value


class Release:
    def __init__(self, directory: Path, commit: str):
        self.directory = directory
        self.commit = commit
        self.root = ROOT
        self.old = None
        self.previous_settings = None
        self.network = None
        self.pg_image = None
        self.live_env = None
        self.db = None
        self.stage = 'preflight'
        self.settings_overlay = self.directory / 'candidate-settings.py'

    def private_write(self, name: str, data: bytes) -> Path:
        path = self.directory / name
        if path.is_symlink():
            raise RuntimeError('private path is a symlink')
        with path.open('xb') as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        path.chmod(0o600)
        return path

    def run(self, *args, timeout=60, check=True):
        try:
            result = subprocess.run(args, cwd=self.root, capture_output=True,
                                    timeout=timeout, check=False)
        except subprocess.TimeoutExpired as error:
            with (self.directory / 'private.log').open('ab') as log:
                log.write((error.stdout or b'') + (error.stderr or b''))
            raise RuntimeError('command timed out; protected log retained') from None
        except OSError:
            raise RuntimeError('command unavailable; protected log retained') from None
        with (self.directory / 'private.log').open('ab') as log:
            log.write(result.stdout + result.stderr)
        if check and result.returncode:
            raise RuntimeError('command failed; protected log retained')
        return result

    def inspect(self, name: str) -> dict:
        return json.loads(self.run('docker', 'inspect', name).stdout)[0]

    def compose(self, override: Path, *args, **kwargs):
        return self.run('docker', 'compose', '--project-directory', str(self.root),
                        '-p', 'ai-phone-system-v2', '-f', str(self.root / 'docker-compose.yml'),
                        '-f', str(override), *args, **kwargs)

    def settings(self) -> dict:
        return json.loads(self.run('docker', 'exec', 'ai-voice-app',
                                   'python', '-c', SETTINGS).stdout)

    def public_health(self) -> None:
        for _ in range(12):
            result = self.run('curl', '-fsS', '--max-time', '5',
                              'https://aiagent.ghaben.ca:8443/health', check=False)
            if result.returncode == 0:
                try:
                    if json.loads(result.stdout).get('status') == 'healthy':
                        return
                except ValueError:
                    pass
            time.sleep(1)
        raise RuntimeError('public certificate-validated HTTPS health failed')

    def wait_ready(self) -> None:
        for _ in range(60):
            result = self.run('docker', 'exec', 'ai-voice-app', 'curl', '-fsS',
                              '--max-time', '3', 'http://localhost:8000/ready', check=False)
            if result.returncode == 0:
                try:
                    if json.loads(result.stdout).get('status') == 'ready':
                        return
                except ValueError:
                    pass
            time.sleep(2)
        raise RuntimeError('application readiness deadline exceeded')

    def verify_source(self) -> None:
        if self.run('git', 'rev-parse', 'HEAD').stdout.decode().strip() != BASE_CHECKOUT:
            raise RuntimeError('server checkout changed')
        for revision in (DEPLOYED_SOURCE, self.commit):
            if self.run('git', 'rev-parse', revision + '^{commit}').stdout.decode().strip() != revision:
                raise RuntimeError('release source unavailable')
            if self.run('git', 'merge-base', '--is-ancestor', BASE_CHECKOUT, revision,
                        check=False).returncode != 0:
                raise RuntimeError('release source is not descended from server baseline')
        if self.run('git', 'merge-base', '--is-ancestor', DEPLOYED_SOURCE, self.commit,
                    check=False).returncode != 0:
            raise RuntimeError('release source is not descended from deployed app')
        if self.run('git', 'status', '--porcelain', '--untracked-files=no').stdout.decode().strip() != 'M nginx/nginx.conf':
            raise RuntimeError('unexpected server checkout modification')
        if sha256((self.root / 'nginx/nginx.conf').read_bytes()) != NGINX_HASH:
            raise RuntimeError('nginx hotfix changed')
        paths = self.run('git', 'diff', '--name-only', DEPLOYED_SOURCE, self.commit, '--',
                         'api', 'config', 'migrations', 'services', 'templates',
                         'docker-compose.yml', 'requirements.txt', 'Dockerfile').stdout.decode().splitlines()
        if len(paths) != len(SOURCE_SCOPE) or set(paths) != SOURCE_SCOPE:
            raise RuntimeError('release source exceeds reviewed runtime scope')
        for path, expected in SQL_HASHES.items():
            data = self.run('git', 'show', self.commit + ':' + path).stdout
            if b'\r' in data or sha256(data) != expected:
                raise RuntimeError('release SQL differs from reviewed LF bytes')
        compose_source = self.run('git', 'show', self.commit + ':docker-compose.yml').stdout
        if any(key.encode() not in compose_source for key in OBSERVATION_ENV):
            raise RuntimeError('reviewed Compose observation mapping missing')
        self.run('git', 'cat-file', '-e', self.commit + ':scripts/deploy-provider-observations.py')

    def capacity(self) -> None:
        if shutil.disk_usage(self.root).free < 1024 ** 3:
            raise RuntimeError('insufficient release disk capacity')
        meminfo = Path('/proc/meminfo').read_text()
        match = re.search(r'^MemAvailable:\s+(\d+) kB$', meminfo, re.MULTILINE)
        if match is None or int(match.group(1)) < 512 * 1024:
            raise RuntimeError('insufficient release memory capacity')

    def preflight(self) -> dict:
        self.verify_source()
        self.capacity()
        override = self.root / 'docker-compose.override.yml'
        if (override.is_symlink() or override.stat().st_uid != 0
                or override.stat().st_mode & 0o077 or not stat.S_ISREG(override.stat().st_mode)):
            raise RuntimeError('private override is not root protected')
        original = override.read_bytes()
        configuration = json.loads(original)
        if configuration['services']['app']['image'] != OLD_IMAGE:
            raise RuntimeError('private override does not pin running image')
        old = self.inspect('ai-voice-app')
        if (old['Image'] != OLD_IMAGE or not old['State']['Running']
                or old['State'].get('Health', {}).get('Status') != 'healthy'
                or old['Config']['Labels'].get('org.opencontainers.image.revision') != DEPLOYED_SOURCE
                or old['Config']['Labels'].get('com.docker.compose.project') != 'ai-phone-system-v2'):
            raise RuntimeError('running app does not match accepted image/source')
        nginx = self.inspect('ai-voice-nginx')
        if (not nginx['State']['Running']
                or nginx['Config'].get('StopSignal', '').upper() not in ('SIGQUIT', 'QUIT', '3')):
            raise RuntimeError('nginx is not ready for graceful stop')
        if not self.inspect('ai-voice-redis')['State']['Running']:
            raise RuntimeError('redis is not running')
        self.run('docker', 'exec', 'ai-voice-nginx', 'nginx', '-t')
        previous_settings = self.settings()
        url = previous_settings['database_url']
        if not isinstance(url, str) or '\r' in url or '\n' in url:
            raise RuntimeError('invalid live database target')
        parsed = urlsplit(url)
        db = self.inspect('ai-voice-db')
        db_env = dict(item.split('=', 1) for item in db['Config']['Env'])
        expected_db = db_env.get('POSTGRES_DB', db_env.get('POSTGRES_USER', 'postgres'))
        common = set(old['NetworkSettings']['Networks']) & set(db['NetworkSettings']['Networks'])
        if (len(common) != 1 or not db['State']['Running']
                or db['State'].get('Health', {}).get('Status') != 'healthy'):
            raise RuntimeError('database network or health changed')
        network, = common
        network_info = db['NetworkSettings']['Networks'][network]
        hosts = {'db', 'ai-voice-db', network_info['IPAddress'], *(network_info.get('Aliases') or [])}
        if parsed.hostname not in hosts or unquote(parsed.path.lstrip('/')) != expected_db:
            raise RuntimeError('application and database targets differ')
        pg_image = db['Image']
        if 'PG_MAJOR=16' not in self.inspect(pg_image)['Config']['Env']:
            raise RuntimeError('PostgreSQL 16 image required')
        self.run('docker', 'inspect', OLD_IMAGE)
        self.public_health()
        self.old, self.previous_settings = old, previous_settings
        self.network, self.pg_image, self.db = network, pg_image, db
        self.private_write('previous-override.json', original)
        self.private_write('previous-settings.json', json.dumps(previous_settings).encode())
        self.private_write('previous-app.json', json.dumps(old).encode())
        self.private_write('compose.sha256', sha256((self.root / 'docker-compose.yml').read_bytes()).encode())
        self.live_env = self.private_write('migration.env',
                                           ('MIGRATION_DATABASE_URL=' + url + '\n').encode())
        if not self.schema_matches(OLD_IMAGE, current=False):
            raise RuntimeError('live database is not the exact 0002 predecessor')
        print('PREFLIGHT_0002_AND_HTTPS_OK', flush=True)
        return configuration

    def remove_container(self, name: str) -> None:
        self.run('docker', 'rm', '-f', name, check=False)
        remaining = self.run('docker', 'ps', '-a', '--format', '{{.Names}}')
        if name in remaining.stdout.decode().splitlines():
            raise RuntimeError('tracked container cleanup unverified')

    def remove_network(self, name: str) -> None:
        self.run('docker', 'network', 'rm', name, check=False)
        remaining = self.run('docker', 'network', 'ls', '--format', '{{.Name}}')
        if name in remaining.stdout.decode().splitlines():
            raise RuntimeError('tracked network cleanup unverified')

    def one_shot(self, image: str, network: str, env_file: Path,
                 arguments: list[str], *, check=True):
        name = 't018c-job-' + self.directory.name
        try:
            return self.run('docker', 'run', '--name', name, '--pull', 'never',
                            '--network', network, '--memory', '192m', '--cpus', '1',
                            '--pids-limit', '96', '--env-file', str(env_file),
                            '--entrypoint', 'python', image, *arguments,
                            timeout=120, check=check)
        finally:
            self.remove_container(name)

    def schema_matches(self, image: str, *, current: bool,
                       network: str | None = None, env_file: Path | None = None) -> bool:
        result = self.one_shot(image, network or self.network,
                               env_file or self.live_env,
                               ['-c', CHECK_0003 if current else CHECK_0002],
                               check=False)
        return result.returncode == 0

    def fingerprint(self, image: str, network: str, env_file: Path) -> dict:
        data = self.one_shot(image, network, env_file, ['-c', FINGERPRINT_CODE]).stdout
        value = json.loads(data)
        if (set(value['hashes']) != set(BUSINESS_TABLES)
                or any(not re.fullmatch('[0-9a-f]{32}', digest)
                       for digest in value['hashes'].values())
                or not isinstance(value['aware'], int)
                or not isinstance(value['naive'], int)):
            raise RuntimeError('business fingerprint invalid')
        return value

    def build(self, label: str, paths: tuple[str, ...]) -> str:
        if (label == 'candidate' and paths != RUNTIME_PATHS
                or label == 'fallback' and paths != FALLBACK_PATHS):
            raise RuntimeError('image copy scope differs')
        parent = 'ai-phone-t018c-parent:' + self.directory.name
        if label == 'candidate':
            self.run('docker', 'tag', OLD_IMAGE, parent)
        if self.inspect(parent)['Id'] != OLD_IMAGE:
            raise RuntimeError('build parent image changed')
        context = self.directory / ('build-' + label)
        context.mkdir(mode=0o700)
        for path in paths:
            target = context / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.run('git', 'show', self.commit + ':' + path).stdout)
        dockerfile = 'FROM ' + parent + '\n' + ''.join(
            'COPY ' + path + ' /app/' + path + '\n' for path in paths)
        (context / 'Dockerfile').write_text(dockerfile, encoding='utf-8', newline='\n')
        tag = 'ai-phone-t018c-' + label + ':' + self.directory.name
        labels = (['--label', 'org.opencontainers.image.revision=' + self.commit]
                  if label == 'candidate' else
                  ['--label', 'ai-phone.observation-fallback-source=' + self.commit])
        self.run('docker', 'build', '--network', 'none', '--pull=false',
                 *labels, '-t', tag, str(context), timeout=300)
        built = self.inspect(tag)
        image = built['Id']
        if not image.startswith('sha256:'):
            raise RuntimeError('built image has no immutable ID')
        revision = built['Config']['Labels'].get('org.opencontainers.image.revision')
        if revision != (self.commit if label == 'candidate' else DEPLOYED_SOURCE):
            raise RuntimeError('built image source label differs')
        return image

    def settings_probe(self, override: Path, label: str) -> dict:
        name = 't018c-settings-' + label + '-' + self.directory.name
        try:
            result = self.compose(override, 'run', '--rm', '--no-deps',
                                  '--pull', 'never',
                                  '--name', name, '--entrypoint', 'python',
                                  'app', '-c', SETTINGS, timeout=90)
            return json.loads(result.stdout)
        finally:
            self.remove_container(name)

    def prepare_override(self, label: str, image: str, migration_image: str,
                         original: dict) -> Path:
        if label not in ('candidate', 'fallback'):
            raise RuntimeError('unknown release variant')
        config = copy.deepcopy(original)
        app = config['services']['app']
        app['image'] = image
        config['services']['migrate']['image'] = migration_image
        if label == 'candidate':
            environment = app.setdefault('environment', {})
            if not isinstance(environment, dict) or OBSERVATION_ENV.keys() & environment.keys():
                raise RuntimeError('private observation environment differs')
            environment.update(OBSERVATION_ENV)
            if any(m['Destination'] == SETTINGS_TARGET for m in self.old['Mounts']):
                raise RuntimeError('previous app already has a settings file overlay')
            source = self.run('git', 'show', self.commit + ':config/settings.py').stdout
            self.private_write('candidate-settings.py', source)
            # This is reviewed application source, not configuration values or
            # credentials. Keep its directory private; permit image users to read
            # the bind-mounted file while leaving the original config mount intact.
            self.settings_overlay.chmod(0o644)
            mounts = app.setdefault('volumes', [])
            if not isinstance(mounts, list) or any(
                    isinstance(m, dict) and m.get('target') == SETTINGS_TARGET for m in mounts):
                raise RuntimeError('private settings mount already exists')
            mounts.append({'type': 'bind', 'source': str(self.settings_overlay),
                           'target': SETTINGS_TARGET, 'read_only': True})
        path = self.private_write(label + '-override.json',
                                  (json.dumps(config, indent=2) + '\n').encode())
        previous = self.directory / 'previous-override.json'
        baseline = json.loads(self.compose(previous, 'config', '--format', 'json').stdout)
        rendered = json.loads(self.compose(path, 'config', '--format', 'json').stdout)
        if (normalized_compose(rendered, candidate=(label == 'candidate'),
                               settings_source=self.settings_overlay)
                != normalized_compose(baseline, candidate=False)):
            raise RuntimeError('effective Compose configuration changed')
        rendered_app = rendered['services']['app']
        old_mounts = {(m['Source'], m['Destination'], not m['RW']) for m in self.old['Mounts']}
        if label == 'candidate':
            old_mounts.add((str(self.settings_overlay), SETTINGS_TARGET, True))
        new_mounts = {(m['source'], m['target'], m.get('read_only', False))
                      for m in rendered_app['volumes']}
        new_networks = {rendered['networks'][n]['name'] for n in rendered_app['networks']}
        if (new_mounts != old_mounts
                or new_networks != set(self.old['NetworkSettings']['Networks'])):
            raise RuntimeError('effective mounts or networks changed')
        if self.settings_probe(path, label) != settings_for(
                self.previous_settings, candidate=(label == 'candidate')):
            raise RuntimeError('effective application settings changed')
        return path

    def stream(self, command: list[str], *, source=None, destination=None,
               timeout=180) -> None:
        try:
            with (self.directory / 'private.log').open('ab') as log:
                result = subprocess.run(command, cwd=self.root, stdin=source,
                                        stdout=destination or log, stderr=log,
                                        timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired):
            raise RuntimeError('backup or restore command unavailable or timed out') from None
        if result.returncode:
            raise RuntimeError('backup or restore command failed')

    def backup(self, name: str) -> Path:
        path = self.directory / name
        with path.open('xb') as output:
            self.stream(['docker', 'exec', 'ai-voice-db', 'sh', '-c',
                         'exec pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc'],
                        destination=output)
            output.flush()
            os.fsync(output.fileno())
        path.chmod(0o600)
        if path.stat().st_size == 0:
            raise RuntimeError('empty protected backup')
        with path.open('rb') as source:
            self.stream(['docker', 'exec', '-i', 'ai-voice-db', 'pg_restore', '--list'],
                        source=source)
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
        self.private_write(name + '.sha256', (digest.hexdigest() + '\n').encode())
        print('PROTECTED_BACKUP_VERIFIED=' + name, flush=True)
        return path

    def wait_for_rehearsal_database(self, name: str) -> None:
        for _ in range(60):
            result = self.run('docker', 'exec', '-e', 'PGPASSWORD=synthetic',
                              '-e', 'PGCONNECT_TIMEOUT=2', name,
                              'psql', '-X', '-w', '-A', '-t', '-h', '127.0.0.1',
                              '-U', 'rehearsal', '-d', 'rehearsal',
                              '-v', 'ON_ERROR_STOP=1', '-c', 'SELECT current_database()',
                              timeout=5, check=False)
            if result.returncode == 0 and result.stdout.strip() == b'rehearsal':
                return
            time.sleep(1)
        raise RuntimeError('authenticated rehearsal database not ready')

    def rehearsal(self, candidate: str, fallback: str, archive: Path) -> None:
        network = 't018c-net-' + self.directory.name
        db = 't018c-db-' + self.directory.name
        env_file = self.private_write(
            'synthetic.env',
            ('MIGRATION_DATABASE_URL=postgresql://rehearsal:synthetic@'
             + db + ':5432/rehearsal\n').encode())
        try:
            self.run('docker', 'network', 'create', '--internal', network)
            self.run('docker', 'run', '-d', '--name', db, '--pull', 'never',
                     '--network', network, '--memory', '192m', '--cpus', '0.5',
                     '--pids-limit', '96', '--tmpfs', '/var/lib/postgresql/data:size=128m',
                     '-e', 'PGDATA=/var/lib/postgresql/data/pgdata',
                     '-e', 'POSTGRES_USER=rehearsal', '-e', 'POSTGRES_PASSWORD=synthetic',
                     '-e', 'POSTGRES_DB=rehearsal', self.pg_image,
                     '-c', 'shared_buffers=16MB', '-c', 'max_connections=12')
            self.wait_for_rehearsal_database(db)
            with archive.open('rb') as source:
                self.stream(['docker', 'exec', '-i', '-e', 'PGPASSWORD=synthetic',
                             db, 'pg_restore', '-h', '127.0.0.1', '--no-password',
                             '--exit-on-error', '--single-transaction', '--no-owner',
                             '--no-acl', '-U', 'rehearsal', '-d', 'rehearsal'], source=source)
            before = self.fingerprint(OLD_IMAGE, network, env_file)
            if before['aware'] < 1 or before['naive'] < 1:
                raise RuntimeError('rehearsal lacks expected aware/naive booking history')
            if not self.schema_matches(OLD_IMAGE, current=False,
                                       network=network, env_file=env_file):
                raise RuntimeError('restored predecessor schema differs')
            self.one_shot(candidate, network, env_file,
                          ['-m', 'migrations.runner', '--prepare'])
            repeat = self.one_shot(candidate, network, env_file,
                                   ['-m', 'migrations.runner', '--prepare'])
            if repeat.stdout.strip() != b'up-to-date 0003':
                raise RuntimeError('0003 migration repeat safety failed')
            if not (self.schema_matches(candidate, current=True,
                                        network=network, env_file=env_file)
                    and self.schema_matches(fallback, current=True,
                                            network=network, env_file=env_file)):
                raise RuntimeError('candidate or fallback schema incompatible')
            after = self.fingerprint(candidate, network, env_file)
            if after != before:
                raise RuntimeError('restored business rows changed')
            self.one_shot(candidate, network, env_file,
                          ['-c', CHECK_EMPTY_OBSERVATIONS])
            print('RESTORED_0002_TO_0003_AND_FALLBACK_OK', flush=True)
        finally:
            try:
                self.remove_container(db)
            finally:
                self.remove_network(network)
            print('REHEARSAL_RESOURCES_REMOVED', flush=True)

    def recheck_before_stop(self) -> None:
        previous = self.directory / 'previous-override.json'
        live = self.root / 'docker-compose.override.yml'
        if live.is_symlink() or live.read_bytes() != previous.read_bytes():
            raise RuntimeError('private override changed before cutover')
        if sha256((self.root / 'docker-compose.yml').read_bytes()) != (
                self.directory / 'compose.sha256').read_text():
            raise RuntimeError('base Compose changed before cutover')
        if sha256((self.root / 'nginx/nginx.conf').read_bytes()) != NGINX_HASH:
            raise RuntimeError('nginx hotfix changed before cutover')
        old = self.inspect('ai-voice-app')
        if (old['Image'] != OLD_IMAGE or not old['State']['Running']
                or old['State'].get('Health', {}).get('Status') != 'healthy'
                or old['Mounts'] != self.old['Mounts']
                or old['NetworkSettings']['Networks'] != self.old['NetworkSettings']['Networks']
                or old['HostConfig'].get('PortBindings') != self.old['HostConfig'].get('PortBindings')
                or old['Config']['Env'] != self.old['Config']['Env']
                or self.settings() != self.previous_settings):
            raise RuntimeError('running app changed before cutover')
        db = self.inspect('ai-voice-db')
        if (db['Image'] != self.pg_image or not db['State']['Running']
                or db['State'].get('Health', {}).get('Status') != 'healthy'
                or db['NetworkSettings']['Networks'] != self.db['NetworkSettings']['Networks']
                or not self.inspect('ai-voice-redis')['State']['Running']
                or not self.inspect('ai-voice-nginx')['State']['Running']):
            raise RuntimeError('database, redis or ingress changed before cutover')
        self.run('docker', 'exec', 'ai-voice-nginx', 'nginx', '-t')
        if not self.schema_matches(OLD_IMAGE, current=False):
            raise RuntimeError('live predecessor schema changed before cutover')
        self.public_health()

    def stop_writers(self, *, strict: bool) -> None:
        for name, seconds in (('ai-voice-nginx', '120'), ('ai-voice-app', '60')):
            self.run('docker', 'stop', '--time', seconds, name,
                     timeout=int(seconds) + 15, check=strict)
        if any(self.inspect(name)['State']['Running']
               for name in ('ai-voice-nginx', 'ai-voice-app')):
            raise RuntimeError('ingress or app stop could not be verified')

    def install_override(self, source: Path) -> None:
        live = self.root / 'docker-compose.override.yml'
        if live.is_symlink() or source.is_symlink():
            raise RuntimeError('override path is a symlink')
        temporary = self.root / ('.t018c-override-' + self.directory.name)
        with temporary.open('xb') as output:
            output.write(source.read_bytes())
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, live)

    def verify_replaced(self, image: str, *, candidate: bool) -> None:
        self.wait_ready()
        current = self.inspect('ai-voice-app')
        # Docker inspect may reorder mounts on recreation. Compare every field.
        expected_mounts = sorted(json.dumps(mount, sort_keys=True)
                                 for mount in self.old['Mounts'])
        mounted = current['Mounts']
        if candidate:
            overlays = [m for m in mounted if m['Destination'] == SETTINGS_TARGET]
            if (len(overlays) != 1 or overlays[0]['Source'] != str(self.settings_overlay)
                    or overlays[0].get('Type') != 'bind' or overlays[0]['RW']):
                raise RuntimeError('replacement pinned settings mount differs')
            mounted = [m for m in mounted if m['Destination'] != SETTINGS_TARGET]
        actual_mounts = sorted(json.dumps(mount, sort_keys=True) for mount in mounted)
        if (current['Image'] != image or not current['State']['Running']
                or actual_mounts != expected_mounts
                or set(current['NetworkSettings']['Networks'])
                != set(self.old['NetworkSettings']['Networks'])
                or current['HostConfig'].get('PortBindings')
                != self.old['HostConfig'].get('PortBindings')):
            raise RuntimeError('replacement image, ports, networks or mounts differ')
        old_env = dict(item.split('=', 1) for item in self.old['Config']['Env'])
        new_env = dict(item.split('=', 1) for item in current['Config']['Env'])
        if candidate:
            for key, expected in OBSERVATION_ENV.items():
                if new_env.pop(key, None) != expected:
                    raise RuntimeError('running observation environment differs')
        if new_env != old_env or self.settings() != settings_for(
                self.previous_settings, candidate=candidate):
            raise RuntimeError('running effective settings changed')

    def replace_app(self, image: str, *, candidate: bool) -> None:
        self.compose(self.root / 'docker-compose.override.yml',
                     'up', '-d', '--no-deps', '--no-build', '--pull', 'never',
                     '--force-recreate', 'app', timeout=120)
        self.verify_replaced(image, candidate=candidate)

    def recover(self, fallback: str, fallback_override: Path) -> None:
        self.stop_writers(strict=False)
        if self.schema_matches(OLD_IMAGE, current=False):
            image = OLD_IMAGE
            override = self.directory / 'previous-override.json'
        elif self.schema_matches(fallback, current=True):
            image = fallback
            override = fallback_override
        else:
            print('RECOVERY_REQUIRES_REVIEW_INGRESS_LEFT_STOPPED', flush=True)
            raise RuntimeError('schema does not match a verified recovery image')
        self.install_override(override)
        self.replace_app(image, candidate=False)
        self.run('docker', 'start', 'ai-voice-nginx')
        self.run('docker', 'exec', 'ai-voice-nginx', 'nginx', '-t')
        self.public_health()
        print('RECOVERY_READY_HTTPS_OK', flush=True)
        print('RECOVERY_IMAGE=' + image, flush=True)

    def deploy(self, candidate: str, candidate_override: Path,
               fallback: str, fallback_override: Path) -> None:
        self.recheck_before_stop()
        print('PRE_CUTOVER_FINGERPRINTS_OK', flush=True)
        self.private_write('cutover-started', (self.commit + '\n').encode())
        print('CUTOVER_STARTED_STOPPING_BETA_TRAFFIC', flush=True)
        try:
            self.stop_writers(strict=True)
            before = self.fingerprint(OLD_IMAGE, self.network, self.live_env)
            self.backup('final-after-writer-stop.dump')
            self.one_shot(candidate, self.network, self.live_env,
                          ['-m', 'migrations.runner', '--prepare'])
            if not self.schema_matches(candidate, current=True):
                raise RuntimeError('live 0003 contract did not pass')
            if self.fingerprint(candidate, self.network, self.live_env) != before:
                raise RuntimeError('live business rows changed by migration')
            print('LIVE_0003_AND_BUSINESS_ROWS_OK', flush=True)
            self.install_override(candidate_override)
            self.replace_app(candidate, candidate=True)
            self.run('docker', 'start', 'ai-voice-nginx')
            self.run('docker', 'exec', 'ai-voice-nginx', 'nginx', '-t')
            self.public_health()
            self.private_write('deployed',
                               (self.commit + '\n' + candidate + '\n'
                                + str(self.directory) + '\n').encode())
            print('PROVIDER_OBSERVATIONS_DEPLOYED_READY_HTTPS_OK', flush=True)
            print('DEPLOYED_COMMIT=' + self.commit, flush=True)
            print('DEPLOYED_IMAGE=' + candidate, flush=True)
        except BaseException:
            print('CUTOVER_FAILED_CHECKING_COMPATIBLE_RECOVERY', flush=True)
            try:
                self.recover(fallback, fallback_override)
            except BaseException:
                # Recovery can fail after nginx was restarted (for example on
                # the HTTPS check). Close ingress again; attempt both stops even
                # if one Docker command fails, without hiding the cutover error.
                for name in ('ai-voice-nginx', 'ai-voice-app'):
                    try:
                        self.run('docker', 'stop', '--time', '30', name,
                                 timeout=45, check=False)
                    except BaseException:
                        pass
                print('RECOVERY_UNVERIFIED_INGRESS_MUST_REMAIN_STOPPED', flush=True)
            raise

    def execute(self) -> None:
        self.stage = 'preflight'
        original = self.preflight()
        self.stage = 'build'
        candidate = self.build('candidate', RUNTIME_PATHS)
        fallback = self.build('fallback', FALLBACK_PATHS)
        self.stage = 'settings'
        candidate_override = self.prepare_override('candidate', candidate, candidate, original)
        fallback_override = self.prepare_override('fallback', fallback, candidate, original)
        self.private_write('images.json',
                           json.dumps({'candidate': candidate, 'fallback': fallback}).encode())
        self.stage = 'rehearsal'
        archive = self.backup('rehearsal.dump')
        self.rehearsal(candidate, fallback, archive)
        self.stage = 'cutover'
        self.deploy(candidate, candidate_override, fallback, fallback_override)


def main(commit: str) -> None:
    import fcntl  # Linux-only; importing this module for local tests is read-only.
    if os.geteuid() != 0 or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise RuntimeError('root and exact 40-hex release commit required')
    os.umask(0o077)
    lock_path = '/run/ai-phone-deployment.lock'
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'r+') as lock:
        metadata = os.fstat(lock.fileno())
        if (metadata.st_uid != 0 or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_mode & 0o077):
            raise RuntimeError('deployment lock is not root protected')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for previous in Path('/opt').glob('ai-phone-observation-release.*'):
            if (previous / 'cutover-started').exists() and not (previous / 'deployed').exists():
                raise RuntimeError('prior cutover needs review before another attempt')
        directory = Path(tempfile.mkdtemp(prefix='ai-phone-observation-release.', dir='/opt'))
        if (directory.is_symlink() or directory.stat().st_uid != 0
                or directory.stat().st_mode & 0o077):
            raise RuntimeError('release directory is not root protected')
        print('RELEASE_DIRECTORY=' + str(directory), flush=True)
        release = Release(directory, commit)
        try:
            release.execute()
        except BaseException as error:
            print('RELEASE_STOPPED_STAGE=' + release.stage, flush=True)
            print('RELEASE_STOPPED_TYPE=' + type(error).__name__, flush=True)
            print('KEEP_PROTECTED_RELEASE_FILES_DO_NOT_PASTE_PRIVATE_LOGS', flush=True)
            raise


if __name__ == '__main__':
    try:
        if len(sys.argv) != 2:
            raise RuntimeError('one exact release commit argument required')
        main(sys.argv[1])
    except BaseException as error:
        # Also report failures before the release object exists (lock, argument,
        # prior-cutover guard); never print exception text/private values.
        print('RELEASE_EXIT_FAILURE_TYPE=' + type(error).__name__, flush=True)
        sys.exit(1)
