"""Read-only, value-free diagnosis of the reported T018-C settings refusal."""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path('/opt/ai-phone-system-v2')
RELEASE = Path('/opt/ai-phone-observation-release._jjs4r2g')
OBS_KEYS = frozenset(('BOOKING_OBSERVATION_ENABLED',
                      'BOOKING_OBSERVATION_INTERVAL_SECONDS',
                      'BOOKING_OBSERVATION_FRESHNESS_SECONDS'))
LOG_CATEGORIES = {
    'unsupported_cli_option': r'unknown (?:flag|option)|unrecognized arguments',
    'compose_configuration_error': r'validating .*:|services\..* must be|invalid interpolation',
    'container_name_invalid': r'Invalid container name|invalid reference format',
    'container_name_conflict': r'container name .* already in use',
    'image_unavailable': r'No such image|pull access denied',
    'mount_error': r'invalid mount config|error mounting|mount .* denied',
    'python_import_error': r'ModuleNotFoundError:|ImportError:',
    'settings_validation_error': r'ValidationError|validation error[s]? for Settings',
    'docker_permission_error': r'permission denied while trying to connect',
    'memory_error': r'out of memory|cannot allocate memory|"OOMKilled"\s*:\s*true',
    'disk_full': r'no space left on device',
}


def command(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, timeout=30)
    if result.returncode:
        return None
    try:
        return json.loads(result.stdout)
    except (ValueError, TypeError):
        return None


def render(path):
    return command(['docker', 'compose', '--project-directory', str(ROOT),
                    '-p', 'ai-phone-system-v2', '-f', str(ROOT / 'docker-compose.yml'),
                    '-f', str(path), 'config', '--format', 'json'])


def comparison(before, after):
    """Only fixed categories/counts; never output arbitrary keys or values."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {'rendered': False}
    old = json.loads(json.dumps(before))
    new = json.loads(json.dumps(after))
    old_app, new_app = old['services']['app'], new['services']['app']
    old_env, new_env = old_app.get('environment') or {}, new_app.get('environment') or {}
    if not isinstance(old_env, dict) or not isinstance(new_env, dict):
        return {'rendered': True, 'environment_mapping': False}
    expected = {'BOOKING_OBSERVATION_ENABLED': 'true',
                'BOOKING_OBSERVATION_FRESHNESS_SECONDS': '180',
                'BOOKING_OBSERVATION_INTERVAL_SECONDS': '60'}
    correct = all(str(new_env.get(key)) == value for key, value in expected.items())
    for key in OBS_KEYS:
        new_env.pop(key, None)
    env_changes = sum(old_env.get(key) != new_env.get(key)
                      or (key in old_env) != (key in new_env)
                      for key in set(old_env) | set(new_env))
    sections = ('volumes', 'networks', 'ports', 'depends_on', 'build',
                'command', 'entrypoint', 'healthcheck', 'labels', 'env_file')
    app_changes = [key for key in sections if old_app.get(key) != new_app.get(key)]
    old_app['image'] = new_app['image'] = '<image>'
    old['services']['migrate']['image'] = new['services']['migrate']['image'] = '<image>'
    return {'rendered': True, 'observation_values_correct': correct,
            'preexisting_observation_keys': bool(OBS_KEYS & old_env.keys()),
            'unrelated_environment_changes': env_changes,
            'changed_app_sections': app_changes,
            'normalized_config_matches': old == new}


def settings_objects(log):
    """Recover complete JSON stdout objects from concatenated private command logs."""
    decoder = json.JSONDecoder()
    found = []
    position = 0
    while position < len(log):
        match = re.search(r'[\[{]', log[position:])
        if match is None:
            break
        position += match.start()
        try:
            value, consumed = decoder.raw_decode(log[position:])
        except ValueError:
            position += 1
            continue
        position += consumed
        if isinstance(value, dict) and 'database_url' in value and 'openai_model' in value:
            found.append(value)
    return found


def main():
    report = {'status': 'READ_ONLY_SETTINGS_DIAGNOSTIC',
              'cutover_started': (RELEASE / 'cutover-started').exists(),
              'deployed_marker': (RELEASE / 'deployed').exists()}
    candidates = [(name, RELEASE / (name + '-override.json'))
                  for name in ('candidate', 'fallback')]
    report['override_files'] = {name: path.is_file() for name, path in candidates}
    previous = render(RELEASE / 'previous-override.json')
    for name, path in candidates:
        if path.is_file():
            current = render(path)
            if name == 'candidate':
                report['candidate_compose'] = comparison(previous, current)
                if isinstance(current, dict):
                    saved_app = json.loads((RELEASE / 'previous-app.json').read_text())
                    app_config = current['services']['app']
                    old_mounts = {(m['Source'], m['Destination'], not m['RW'])
                                  for m in saved_app['Mounts']}
                    new_mounts = {(m['source'], m['target'], m.get('read_only', False))
                                  for m in app_config['volumes']}
                    report['candidate_mounts_match'] = old_mounts == new_mounts
                    report['candidate_networks_match'] = (
                        {current['networks'][n]['name'] for n in app_config['networks']}
                        == set(saved_app['NetworkSettings']['Networks']))
            else:
                # Fallback does not add observation settings.
                old, new = json.loads(json.dumps(previous)), current
                if isinstance(old, dict) and isinstance(new, dict):
                    for service in ('app', 'migrate'):
                        old['services'][service]['image'] = new['services'][service]['image'] = '<image>'
                    report['fallback_compose_matches'] = old == new
    with (RELEASE / 'private.log').open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 2 * 1024 * 1024))
        log = stream.read().decode('utf-8', errors='replace')
    report['log_categories'] = [name for name, pattern in LOG_CATEGORIES.items()
                                if re.search(pattern, log, re.IGNORECASE)]
    observed = settings_objects(log)
    report['settings_outputs_found'] = len(observed)
    if observed:
        previous_settings = json.loads((RELEASE / 'previous-settings.json').read_text())
        last = dict(observed[-1])
        report['last_output_has_observation_settings'] = 'booking_observation_enabled' in last
        for key in ('booking_observation_enabled', 'booking_observation_interval_seconds',
                    'booking_observation_freshness_seconds'):
            last.pop(key, None)
        report['last_output_matches_previous_settings'] = last == previous_settings
    app = command(['docker', 'inspect', 'ai-voice-app'])
    if isinstance(app, list) and len(app) == 1:
        report['previous_app_still_running'] = (
            app[0].get('Image') == 'sha256:f65fd1b449e2e107d2137158fc05bdbcc376c301b74c97e4a56c57a73b576a21'
            and app[0].get('State', {}).get('Running') is True)
        report['previous_app_healthy'] = app[0].get('State', {}).get('Health', {}).get('Status') == 'healthy'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'DIAGNOSTIC_UNAVAILABLE', 'error_type': type(error).__name__}))
