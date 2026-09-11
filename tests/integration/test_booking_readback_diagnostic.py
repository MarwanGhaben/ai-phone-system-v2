"""Read-only diagnostic errors must identify the stage without leaking response data."""
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import asyncpg
import httpx
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['success', 'legacy_fields', 'missing_token', 'unknown_timezone'])
async def test_readback_reports_safe_shape_and_stage(case, monkeypatch, capsys):
    path = Path(__file__).resolve().parents[2] / 'scripts/check-latest-booking.py'
    spec = importlib.util.spec_from_file_location('readback_diagnostic', path)
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    import config.settings as settings_module
    monkeypatch.setattr(settings_module, 'settings', SimpleNamespace(
        database_url='PRIVATE_DSN', ms_bookings_tenant_id='synthetic-tenant',
        ms_bookings_client_id='synthetic-client', ms_bookings_client_secret='PRIVATE_SECRET',
        ms_bookings_business_id='synthetic-business'))

    class Connection:
        def transaction(self, **kwargs):
            assert kwargs == {'readonly': True}
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def fetchrow(self, sql):
            assert sql.strip().startswith('SELECT')
            return {'id': 17, 'call_sid': 'PRIVATE_CALL', 'ms_booking_id': 'PRIVATE_ID',
                    'appointment_time': datetime(2026, 9, 14, 11),
                    'created_at': datetime(2026, 9, 11, 16)}

        async def fetchval(self, sql, *args):
            assert sql.startswith('SELECT')
            return 1

        async def close(self, **kwargs):
            pass

    async def connect(*args, **kwargs):
        return Connection()

    monkeypatch.setattr(asyncpg, 'connect', connect)
    appointment = {
        'id': 'PRIVATE_ID', 'customerName': 'PRIVATE_NAME', 'PRIVATE_KEY': 'PRIVATE_VALUE',
        'start': {'dateTime': '2026-09-14T15:00:00', 'timeZone': 'UTC'},
        'end': {'dateTime': '2026-09-14T15:30:00', 'timeZone': 'UTC'},
        'staffMemberIds': ['93ee7133-8b0c-42c4-a886-a368b998de4b'],
        'serviceId': '357dc857-4360-4801-8bc4-12d3ed63afa3', 'isLocationOnline': False,
    }
    if case == 'legacy_fields':
        appointment['startDateTime'] = appointment.pop('start')
        appointment['endDateTime'] = appointment.pop('end')
    if case == 'unknown_timezone':
        appointment['start']['timeZone'] = 'PRIVATE_ZONE'

    requests = []

    def respond(request):
        requests.append((request.method, request.url.host))
        if request.url.host == 'login.microsoftonline.com':
            return httpx.Response(200, json={} if case == 'missing_token' else {'access_token': 'PRIVATE_TOKEN'})
        assert request.method == 'GET' and request.url.host == 'graph.microsoft.com'
        return httpx.Response(200, json=appointment)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: real_client(
        transport=httpx.MockTransport(respond), **kw))
    await diagnostic.report()
    output = capsys.readouterr().out
    assert 'PRIVATE' not in output
    report = json.loads(output)
    if case == 'success':
        assert report['status'] == 'READBACK_COMPLETE'
        assert report['matches_expected_time_staff_duration'] is True
    else:
        assert report['status'] == 'CHECK_FAILED'
        if case == 'missing_token':
            assert report['stage'] == 'authenticate'
            assert report['missing_expected_field'] == 'access_token'
            assert requests == [('POST', 'login.microsoftonline.com')]
        else:
            assert report['stage'] == 'validate_appointment_fields'
            shape = report['response_shape']
            if case == 'legacy_fields':
                assert report['missing_expected_field'] == 'start'
                assert shape['fields_present']['start'] is False
                assert shape['time_fields']['startDateTime']['parsed_datetime'] == '2026-09-14T15:00:00'
            else:
                assert shape['time_fields']['start']['timezone'] == 'unrecognized_or_missing'
