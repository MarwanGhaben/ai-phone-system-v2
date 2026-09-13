"""Read-only comparison against the saved preflight of the reported release."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path('/opt/ai-phone-system-v2')
RELEASE = Path('/opt/ai-phone-observation-release.oomp1k6t')
OLD_IMAGE = 'sha256:f65fd1b449e2e107d2137158fc05bdbcc376c301b74c97e4a56c57a73b576a21'
NGINX_HASH = '59ac9bdaa9a86c1022a573c9709faddf7e93a53015da8acfcc4330b3e288f3ab'
SCHEMA_CHECK = '''import asyncio,asyncpg
from config.settings import settings
from migrations.schema_contract import check_runtime_compatibility
async def check():
 c=await asyncpg.connect(settings.database_url,timeout=10,command_timeout=10)
 try:
  async with c.transaction(readonly=True):
   await check_runtime_compatibility(c)
 finally:
  await c.close(timeout=5)
asyncio.run(check())
'''


def read_json(path):
    return json.loads(path.read_text())


def run_json(args):
    result = subprocess.run(args, capture_output=True, timeout=20)
    if result.returncode:
        raise RuntimeError('read command failed')
    return json.loads(result.stdout)


def same_mounts(before, after):
    return sorted(json.dumps(v, sort_keys=True) for v in before) == sorted(
        json.dumps(v, sort_keys=True) for v in after)


def app_checks(old, current, previous_settings, settings):
    return {
        'expected_image': current['Image'] == OLD_IMAGE,
        'running': current['State']['Running'] is True,
        'healthy': current['State'].get('Health', {}).get('Status') == 'healthy',
        'mount_list_order_matches': old['Mounts'] == current['Mounts'],
        'mount_contents_match': same_mounts(old['Mounts'], current['Mounts']),
        'networks_match': old['NetworkSettings']['Networks'] == current['NetworkSettings']['Networks'],
        'ports_match': old['HostConfig'].get('PortBindings') == current['HostConfig'].get('PortBindings'),
        'environment_order_matches': old['Config']['Env'] == current['Config']['Env'],
        'environment_contents_match': sorted(old['Config']['Env']) == sorted(current['Config']['Env']),
        'effective_settings_match': settings == previous_settings,
    }


def first_db_snapshot(log):
    decoder = json.JSONDecoder()
    position = 0
    while position < len(log):
        match = re.search(r'[\[{]', log[position:])
        if match is None:
            return None
        position += match.start()
        try:
            item, consumed = decoder.raw_decode(log[position:])
        except ValueError:
            position += 1
            continue
        position += consumed
        if (isinstance(item, list) and len(item) == 1 and isinstance(item[0], dict)
                and item[0].get('Name') == '/ai-voice-db'):
            return item[0]
    return None


def main():
    old = read_json(RELEASE / 'previous-app.json')
    previous_settings = read_json(RELEASE / 'previous-settings.json')
    current = run_json(['docker', 'inspect', 'ai-voice-app'])[0]
    settings = run_json(['docker', 'exec', 'ai-voice-app', 'python', '-c',
                         'import json; from config.settings import settings; '
                         'print(json.dumps(settings.model_dump(mode="json")))'])
    report = {'status': 'READ_ONLY_PRE_CUTOVER_DIAGNOSTIC',
              'cutover_started': (RELEASE / 'cutover-started').exists(),
              'deployed_marker': (RELEASE / 'deployed').exists(),
              'app_checks': app_checks(old, current, previous_settings, settings)}
    override = ROOT / 'docker-compose.override.yml'
    report['file_checks'] = {
        'override_not_symlink': not override.is_symlink(),
        'override_bytes_match': override.read_bytes() == (RELEASE / 'previous-override.json').read_bytes(),
        'compose_hash_matches': hashlib.sha256((ROOT / 'docker-compose.yml').read_bytes()).hexdigest()
            == (RELEASE / 'compose.sha256').read_text(),
        'nginx_hash_matches': hashlib.sha256((ROOT / 'nginx/nginx.conf').read_bytes()).hexdigest() == NGINX_HASH}
    db = run_json(['docker', 'inspect', 'ai-voice-db'])[0]
    with (RELEASE / 'private.log').open('rb') as stream:
        log = stream.read(8 * 1024 * 1024).decode('utf-8', errors='replace')
    original_db = first_db_snapshot(log)
    report['database_checks'] = {'baseline_found': original_db is not None,
        'running': db['State']['Running'] is True,
        'healthy': db['State'].get('Health', {}).get('Status') == 'healthy'}
    if original_db is not None:
        report['database_checks'].update({
            'image_matches': db['Image'] == original_db['Image'],
            'networks_match': db['NetworkSettings']['Networks'] == original_db['NetworkSettings']['Networks']})
    report['redis_running'] = run_json(['docker', 'inspect', 'ai-voice-redis'])[0]['State']['Running'] is True
    report['nginx_running'] = run_json(['docker', 'inspect', 'ai-voice-nginx'])[0]['State']['Running'] is True
    for key, args in (
        ('nginx_config_valid', ['docker', 'exec', 'ai-voice-nginx', 'nginx', '-t']),
        ('running_image_schema_compatible', ['docker', 'exec', 'ai-voice-app', 'python', '-c', SCHEMA_CHECK]),
        ('public_https_healthy', ['curl', '-fsS', '--max-time', '5', 'https://aiagent.ghaben.ca:8443/health']),
    ):
        result = subprocess.run(args, capture_output=True, timeout=25)
        success = result.returncode == 0
        if key == 'public_https_healthy' and success:
            try:
                success = json.loads(result.stdout).get('status') == 'healthy'
            except ValueError:
                success = False
        report[key] = success
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'DIAGNOSTIC_UNAVAILABLE', 'error_type': type(error).__name__}))
