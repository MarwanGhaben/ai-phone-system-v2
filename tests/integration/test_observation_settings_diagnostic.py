"""Diagnostic outputs only fixed categories, counts and booleans."""
import contextlib
import importlib.util
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SOURCE = Path(__file__).resolve().parents[2] / 'scripts/inspect-observation-settings.py'
spec = importlib.util.spec_from_file_location('settings_diagnostic', SOURCE)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
SECRET = 'PRIVATE_PAYLOAD-$-\nnot-for-output'


def configs():
    old = {'services': {'app': {'image': 'old', 'environment': {'KEY': SECRET}},
                        'migrate': {'image': 'old'}}}
    new = json.loads(json.dumps(old))
    new['services']['app']['image'] = 'new'
    new['services']['migrate']['image'] = 'new'
    new['services']['app']['environment'].update({
        'BOOKING_OBSERVATION_ENABLED': 'true',
        'BOOKING_OBSERVATION_INTERVAL_SECONDS': '60',
        'BOOKING_OBSERVATION_FRESHNESS_SECONDS': '180'})
    return old, new


def test_comparison_reports_mismatch_without_payload_or_unknown_keys():
    old, new = configs()
    assert diagnostic.comparison(old, new)['normalized_config_matches'] is True
    new['services']['app']['environment'][SECRET] = SECRET
    new['services']['app']['volumes'] = [SECRET]
    result = diagnostic.comparison(old, new)
    assert result['normalized_config_matches'] is False
    assert result['unrelated_environment_changes'] == 1
    assert result['changed_app_sections'] == ['volumes']
    assert SECRET not in json.dumps(result)
    assert old['services']['app']['environment'] == {'KEY': SECRET}


def test_log_decoder_handles_concatenated_json_and_unrelated_errors():
    expected = {'database_url': SECRET, 'openai_model': 'synthetic'}
    log = 'build progress {invalid\n' + json.dumps([{'Env': [SECRET]}]) + json.dumps(expected)
    assert diagnostic.settings_objects(log) == [expected]


def test_logged_source_except_clause_is_not_an_import_failure():
    pattern = diagnostic.LOG_CATEGORIES['python_import_error']
    assert not re.search(pattern, '            except ImportError:\n                pass')
    assert re.search(pattern, "ImportError: cannot import name 'model_validator'\n")


def test_full_report_only_runs_readonly_commands_and_redacts(tmp_path):
    old, new = configs()
    for config in (old, new):
        config['services']['app'].update({'volumes': [], 'networks': ['appnet']})
        config['networks'] = {'appnet': {'name': 'synthetic'}}
    (tmp_path / 'previous-app.json').write_text(json.dumps(
        {'Mounts': [], 'NetworkSettings': {'Networks': {'synthetic': {}}}}))
    (tmp_path / 'previous-settings.json').write_text(json.dumps({'database_url': SECRET, 'openai_model': 'synthetic'}))
    (tmp_path / 'previous-override.json').write_text(json.dumps(old))
    (tmp_path / 'candidate-override.json').write_text(json.dumps(new))
    (tmp_path / 'private.log').write_text('Error: invalid mount config ' + SECRET)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[:2] == ['docker', 'inspect']:
            body = [{'Image': 'sha256:f65fd1b449e2e107d2137158fc05bdbcc376c301b74c97e4a56c57a73b576a21',
                     'State': {'Running': True, 'Health': {'Status': 'healthy'}}}]
        else:
            assert args[:2] == ['docker', 'compose']
            assert args[-3:] == ['config', '--format', 'json']
            body = new if 'candidate-override.json' in args[-4] else old
        return SimpleNamespace(returncode=0, stdout=json.dumps(body).encode())

    output = io.StringIO()
    with mock.patch.object(diagnostic, 'RELEASE', tmp_path), \
         mock.patch.object(diagnostic.subprocess, 'run', side_effect=run), \
         contextlib.redirect_stdout(output):
        diagnostic.main()
    result = json.loads(output.getvalue())
    assert result['previous_app_still_running'] and result['previous_app_healthy']
    assert not result['cutover_started']
    assert result['log_categories'] == ['mount_error']
    assert result['candidate_compose']['normalized_config_matches']
    assert SECRET not in output.getvalue()
    assert len(calls) == 3


def test_failed_command_never_returns_stderr():
    with mock.patch.object(diagnostic.subprocess, 'run', return_value=SimpleNamespace(
            returncode=1, stdout=SECRET.encode(), stderr=SECRET.encode())):
        assert diagnostic.command(['docker', 'inspect', 'ai-voice-app']) is None
