"""Provider-gate behavior with synthetic transport; never uses a real API key."""
import importlib.util
import json
from pathlib import Path
import sys

import httpx
import pytest
from openai import AuthenticationError

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('luna_probe', ROOT / 'scripts/probe-luna-compatibility.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.asyncio
async def test_six_offline_calls_use_actual_sdk_and_linked_tool_response():
    assert await probe.run(False) == 6


def test_default_cli_is_offline(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['probe'])
    modes = []
    async def run(live):
        modes.append(live)
        return 6
    monkeypatch.setattr(probe, 'run', run)
    assert probe.main() == 0
    assert modes == [False]
    assert json.loads(capsys.readouterr().out)['mode'] == 'offline'


@pytest.mark.parametrize('failure', ['api', 'timeout', 'invalid'])
def test_live_failure_output_is_redacted_and_nonzero(monkeypatch, capsys, failure):
    monkeypatch.setattr(sys, 'argv', ['probe', '--live'])
    response = httpx.Response(401, request=httpx.Request('POST', 'https://example.invalid'))
    errors = {'api': AuthenticationError('secret canary', response=response, body={'secret': 'canary'}),
              'timeout': TimeoutError('secret canary'), 'invalid': ValueError('secret canary')}
    async def run(live):
        assert live
        raise errors[failure]
    monkeypatch.setattr(probe, 'run', run)
    assert probe.main() == 1
    output = capsys.readouterr()
    assert 'secret' not in output.out + output.err and 'canary' not in output.out + output.err
    assert json.loads(output.out)['status'] == 'LUNA_COMPATIBILITY_FAILED'
