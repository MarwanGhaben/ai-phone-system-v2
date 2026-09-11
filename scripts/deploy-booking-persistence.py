"""Owner-run T005-D rollout, pinned to the verified phase-3 beta deployment."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import types
from urllib.parse import unquote, urlsplit

ROOT = Path('/opt/ai-phone-system-v2')
APP_COMMIT = '075d8cf6fc29c4cbdf69b5c82baa130d5c0c540d'
OLD_IMAGE = 'sha256:8c0b299785bed1703cb897205591e63121256772f138021e687f6b71e80d7264'
PATHS = [
    'migrations/runner.py', 'migrations/schema_contract.py',
    'migrations/0002_bookings_aware_time.sql',
    'services/scheduling/__init__.py', 'services/scheduling/booking_records.py',
    'services/conversation/orchestrator.py', 'services/dashboard/dashboard_routes.py',
]
CHECK_SCHEMA = '''import asyncio,os,asyncpg
from migrations.schema_contract import check_runtime_compatibility
async def check():
    c=await asyncpg.connect(os.environ['MIGRATION_DATABASE_URL'],timeout=10,command_timeout=10)
    try:
        async with c.transaction(readonly=True):
            await check_runtime_compatibility(c)
    finally:
        await c.close(timeout=5)
asyncio.run(check())
'''


def load_base(commit):
    result = subprocess.run(['git', 'show', commit + ':scripts/deploy-booking-guards.py'],
                            cwd=ROOT, capture_output=True, timeout=15, check=True)
    module = types.ModuleType('persistence_release_base')
    exec(compile(result.stdout, 'deploy-booking-guards.py', 'exec'), module.__dict__)
    module.OLD_IMAGE = OLD_IMAGE
    module.RUNTIME_PATHS = PATHS
    return module


def release_class(base):
    class PersistenceRelease(base.Release):
        def build(self, label, paths):
            build = self.directory / label
            build.mkdir()
            for path in paths:
                target = build / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(self.run('git', 'show', APP_COMMIT + ':' + path).stdout)
            tag = 'ai-phone-t005d-' + label + ':' + self.directory.name
            parent = 'ai-phone-rollback:t020a-' + self.directory.name
            if self.inspect(parent)['Id'] != OLD_IMAGE:
                raise RuntimeError('build parent changed')
            (build / 'Dockerfile').write_text('FROM ' + parent + '\n' + ''.join(
                'COPY ' + path + ' /app/' + path + '\n' for path in paths))
            self.run('docker', 'build', '--network', 'none', '--pull=false', '-t', tag,
                     '--label', 'ai-phone.persistence-source=' + APP_COMMIT,
                     *(['--label', 'org.opencontainers.image.revision=' + APP_COMMIT]
                       if label == 'candidate' else []),
                     str(build), timeout=300)
            return self.inspect(tag)['Id']

        def one_shot(self, image, network, env_file, arguments, *, check=True):
            name = 't005d-job-' + self.directory.name
            try:
                return self.run('docker', 'run', '--name', name, '--pull', 'never',
                                '--network', network, '--memory', '192m', '--cpus', '1',
                                '--pids-limit', '96', '--env-file', str(env_file),
                                '--entrypoint', 'python', image, *arguments,
                                timeout=90, check=check)
            finally:
                # Terminate the actual container even if the Docker client timed out.
                self.remove_probe(name)

        def schema_matches(self, image):
            return self.one_shot(image, self.network, self.live_env,
                                 ['-c', CHECK_SCHEMA], check=False).returncode == 0

        def initialize_database(self, old):
            settings = json.loads((self.directory / 'previous-settings.json').read_text())
            url = settings['database_url']
            parsed = urlsplit(url)
            db = self.inspect('ai-voice-db')
            env = dict(item.split('=', 1) for item in db['Config']['Env'])
            expected_db = env.get('POSTGRES_DB', env.get('POSTGRES_USER', 'postgres'))
            common = set(old['NetworkSettings']['Networks']) & set(db['NetworkSettings']['Networks'])
            if len(common) != 1 or db['State'].get('Health', {}).get('Status') != 'healthy':
                raise RuntimeError('database network/health changed')
            self.network, = common
            network = db['NetworkSettings']['Networks'][self.network]
            hosts = {'db', 'ai-voice-db', network['IPAddress'], *(network.get('Aliases') or [])}
            if parsed.hostname not in hosts or unquote(parsed.path.lstrip('/')) != expected_db:
                raise RuntimeError('backup database does not match application')
            if '\n' in url or '\r' in url:
                raise RuntimeError('invalid database URL')
            self.live_env = self.directory / 'migration.env'
            self.live_env.write_text('MIGRATION_DATABASE_URL=' + url + '\n')
            self.pg_image = db['Image']
            if 'PG_MAJOR=16' not in self.inspect(self.pg_image)['Config']['Env']:
                raise RuntimeError('PostgreSQL 16 required')
            if not self.schema_matches(OLD_IMAGE):
                raise RuntimeError('live schema is not the expected 0001 contract')

        def stream(self, command, *, source=None, destination=None):
            with (self.directory / 'private.log').open('ab') as log:
                result = subprocess.run(command, cwd=ROOT, stdin=source,
                                        stdout=destination or log, stderr=log, timeout=120)
            if result.returncode:
                raise RuntimeError('backup or restore command failed')

        def backup(self, name):
            archive = self.directory / name
            with archive.open('xb') as output:
                self.stream(['docker', 'exec', 'ai-voice-db', 'sh', '-c',
                             'exec pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc'],
                            destination=output)
                output.flush()
                os.fsync(output.fileno())
            if archive.stat().st_size == 0:
                raise RuntimeError('empty backup')
            with archive.open('rb') as source:
                self.stream(['docker', 'exec', '-i', 'ai-voice-db', 'pg_restore', '--list'], source=source)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            (self.directory / (name + '.sha256')).write_text(digest + '\n')
            print('PROTECTED_BACKUP_OK=' + name, flush=True)
            return archive

        def wait_for_rehearsal_database(self, db):
            # The image's temporary initialization server only listens on a Unix
            # socket. Require the final TCP server and the actual database instead.
            for _ in range(60):
                probe = self.run('docker', 'exec', '-e', 'PGPASSWORD=synthetic',
                                 '-e', 'PGCONNECT_TIMEOUT=2', db,
                                 'psql', '-X', '-w', '-A', '-t', '-h', '127.0.0.1',
                                 '-U', 'rehearsal', '-d', 'rehearsal',
                                 '-v', 'ON_ERROR_STOP=1', '-c', 'SELECT current_database()',
                                 timeout=5, check=False)
                if probe.returncode == 0 and probe.stdout.strip() == b'rehearsal':
                    print('REHEARSAL_DATABASE_READY', flush=True)
                    return
                time.sleep(1)
            raise RuntimeError('rehearsal database startup failed')

        def rehearsal(self, candidate, fallback, archive):
            network = 't005d-net-' + self.directory.name
            db = 't005d-db-' + self.directory.name
            env_file = self.directory / 'synthetic.env'
            env_file.write_text('MIGRATION_DATABASE_URL=postgresql://rehearsal:synthetic@'
                                + db + ':5432/rehearsal\n')
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
                                 db, 'pg_restore', '-h', '127.0.0.1', '--no-password', '--exit-on-error',
                                 '--single-transaction', '--no-owner', '--no-acl',
                                 '-U', 'rehearsal', '-d', 'rehearsal'], source=source)
                fingerprint_query = ("SELECT md5(COALESCE(jsonb_agg(to_jsonb(b) "
                                     "- 'appointment_time_utc' ORDER BY id)::text,'[]')) "
                                     "FROM public.bookings b")
                query = ['docker', 'exec', db, 'psql', '-X', '-A', '-t',
                         '-v', 'ON_ERROR_STOP=1', '-U', 'rehearsal', '-d', 'rehearsal', '-c']
                before = self.run(*query, fingerprint_query).stdout.strip()
                self.one_shot(OLD_IMAGE, network, env_file, ['-c', CHECK_SCHEMA])
                self.one_shot(candidate, network, env_file, ['-m', 'migrations.runner', '--prepare'])
                after = self.run(*query, fingerprint_query).stdout.strip()
                populated = self.run(*query, 'SELECT count(*) FROM public.bookings '
                                     'WHERE appointment_time_utc IS NOT NULL').stdout.strip()
                if not before or after != before or populated != b'0':
                    raise RuntimeError('rehearsal changed historical booking data')
                for image in (candidate, fallback):
                    self.one_shot(image, network, env_file, ['-c', CHECK_SCHEMA])
                print('RESTORED_BACKUP_MIGRATION_AND_FALLBACK_OK', flush=True)
            finally:
                try:
                    self.remove_probe(db)
                finally:
                    self.run('docker', 'network', 'rm', network, check=False)
                    result = self.run('docker', 'network', 'inspect', network, check=False)
                    if result.returncode == 0 or b'not found' not in result.stderr.lower():
                        raise RuntimeError('rehearsal network cleanup unverified')

        def install_override(self, path):
            # Atomic replacement; retain restrictive permissions and escaped values.
            live = ROOT / 'docker-compose.override.yml'
            temporary = ROOT / ('.t005d-override-' + self.directory.name)
            with temporary.open('xb') as output:
                output.write(path.read_bytes())
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, live)

        def recover(self, fallback, fallback_override):
            self.run('docker', 'stop', '--time', '30', 'ai-voice-nginx', check=False)
            self.run('docker', 'stop', '--time', '30', 'ai-voice-app', check=False)
            if any(self.inspect(name)['State']['Running']
                   for name in ('ai-voice-app', 'ai-voice-nginx')):
                raise RuntimeError('recovery could not verify writer/ingress stop')
            if self.schema_matches(fallback):
                image, override = fallback, fallback_override
            elif self.schema_matches(OLD_IMAGE):
                image, override = OLD_IMAGE, self.directory / 'previous-override.json'
            else:
                print('RECOVERY_REQUIRES_REVIEW_INGRESS_LEFT_STOPPED', flush=True)
                raise RuntimeError('no verified compatible schema')
            self.replace_app(override, image)
            self.install_override(override)
            self.run('docker', 'start', 'ai-voice-nginx')
            self.public_health()
            print('RECOVERY_READY_HTTPS_OK', flush=True)
            print('RECOVERY_IMAGE=' + image, flush=True)

        def deploy(self, image, candidate_override, fallback, fallback_override):
            previous = self.directory / 'previous-override.json'
            if ((ROOT / 'docker-compose.override.yml').read_bytes() != previous.read_bytes()
                    or self.inspect('ai-voice-app')['Image'] != OLD_IMAGE
                    or self.settings() != json.loads((self.directory / 'previous-settings.json').read_text())):
                raise RuntimeError('server state changed during rehearsal')
            if (hashlib.sha256((ROOT / 'nginx/nginx.conf').read_bytes()).hexdigest() != base.NGINX_HASH
                    or hashlib.sha256((ROOT / 'docker-compose.yml').read_bytes()).hexdigest()
                    != (self.directory / 'compose.sha256').read_text()):
                raise RuntimeError('proxy or Compose configuration changed during rehearsal')
            if not self.schema_matches(OLD_IMAGE):
                raise RuntimeError('live schema changed during rehearsal')
            print('STOPPING_TEST_TRAFFIC_FOR_DATABASE_CUTOVER', flush=True)
            (self.directory / 'cutover-started').write_text(APP_COMMIT + '\n')
            try:
                self.run('docker', 'stop', '--time', '120', 'ai-voice-nginx', timeout=135)
                self.run('docker', 'stop', '--time', '60', 'ai-voice-app', timeout=75)
                self.backup('final-after-writer-stop.dump')
                self.one_shot(image, self.network, self.live_env, ['-m', 'migrations.runner', '--prepare'])
                print('LIVE_DATABASE_0002_OK', flush=True)
                self.replace_app(candidate_override, image)
                self.install_override(candidate_override)
                self.run('docker', 'start', 'ai-voice-nginx')
                self.run('docker', 'exec', 'ai-voice-nginx', 'nginx', '-t')
                self.public_health()
                (self.directory / 'deployed').write_text(APP_COMMIT + '\n' + image + '\n')
                print('BOOKING_PERSISTENCE_DEPLOYED_READY_HTTPS_OK', flush=True)
                print('DEPLOYED_COMMIT=' + APP_COMMIT, flush=True)
                print('DEPLOYED_IMAGE=' + image, flush=True)
            except BaseException:
                print('CUTOVER_FAILED_CHECKING_SAFE_RECOVERY', flush=True)
                self.recover(fallback, fallback_override)
                raise
    return PersistenceRelease


def main(commit):
    import fcntl
    if os.geteuid() != 0 or not re.fullmatch('[0-9a-f]{40}', commit):
        raise RuntimeError('root and exact release commit required')
    os.umask(0o077)
    with open('/run/ai-phone-deployment.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base = load_base(commit)
        directory = Path(tempfile.mkdtemp(prefix='ai-phone-persistence-release.', dir='/opt'))
        print('RELEASE_DIRECTORY=' + str(directory), flush=True)
        release = release_class(base)(directory)
        print('VERIFYING_RUNNING_IMAGE_SETTINGS_AND_SCHEMA', flush=True)
        old, configuration = release.preflight(APP_COMMIT)
        (directory / 'compose.sha256').write_text(
            hashlib.sha256((ROOT / 'docker-compose.yml').read_bytes()).hexdigest())
        release.initialize_database(old)
        print('BUILDING_CANDIDATE_AND_COMPATIBLE_FALLBACK', flush=True)
        image = release.build('candidate', PATHS)
        fallback = release.build('fallback', ['migrations/schema_contract.py'])
        # Future ordinary Compose starts must not invoke the old migration runner.
        configuration['services']['migrate']['image'] = image
        fallback_override = release.prepare_override(fallback, old, copy.deepcopy(configuration))
        fallback_override = fallback_override.rename(directory / 'fallback-override.json')
        candidate_override = release.prepare_override(image, old, copy.deepcopy(configuration))
        release.rehearsal(image, fallback, release.backup('rehearsal.dump'))
        (directory / 'images.json').write_text(json.dumps({'candidate': image, 'fallback': fallback}))
        release.deploy(image, candidate_override, fallback, fallback_override)


if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except BaseException as error:
        print('RELEASE_STOPPED_TYPE=' + type(error).__name__, flush=True)
        print('KEEP_PROTECTED_RELEASE_FILES_DO_NOT_PASTE_PRIVATE_LOGS', flush=True)
        sys.exit(1)
