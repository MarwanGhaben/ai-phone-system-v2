import importlib.util
import json
from pathlib import Path

source = Path(__file__).resolve().parents[2] / 'scripts/inspect-observation-cutover.py'
spec = importlib.util.spec_from_file_location('cutover_diagnostic', source)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_reordered_mounts_are_distinguished_from_actual_drift_without_values():
    old = {'Image': diagnostic.OLD_IMAGE, 'State': {'Running': True, 'Health': {'Status': 'healthy'}},
           'Mounts': [{'Source': 'PRIVATE_A', 'RW': False}, {'Source': 'PRIVATE_B', 'RW': True}],
           'NetworkSettings': {'Networks': {}}, 'HostConfig': {}, 'Config': {'Env': ['SECRET=PRIVATE']}}
    current = json.loads(json.dumps(old))
    current['Mounts'].reverse()
    report = diagnostic.app_checks(old, current, {'secret': 'PRIVATE'}, {'secret': 'PRIVATE'})
    assert report['mount_list_order_matches'] is False
    assert report['mount_contents_match'] is True
    assert report['effective_settings_match'] is True
    assert 'PRIVATE' not in json.dumps(report)
    current['Mounts'][0]['RW'] = False
    assert diagnostic.app_checks(old, current, {}, {})['mount_contents_match'] is False


def test_db_snapshot_comes_from_inspect_object_not_logged_source_or_later_state():
    expected = {'Name': '/ai-voice-db', 'Image': 'synthetic', 'Config': {'Env': ['SECRET=PRIVATE']}}
    log = 'build source {invalid\n' + json.dumps([expected]) + json.dumps([dict(expected, Image='changed')])
    assert diagnostic.first_db_snapshot(log) == expected
    assert diagnostic.first_db_snapshot('source except ImportError:') is None
