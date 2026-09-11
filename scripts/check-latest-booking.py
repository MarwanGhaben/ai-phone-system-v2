import asyncio, json, logging
from datetime import datetime, timezone
from urllib.parse import quote
from zoneinfo import ZoneInfo

EXPECTED_FIELDS = ('access_token', 'start', 'end', 'dateTime', 'timeZone')
APPOINTMENT_FIELDS = ('id', 'start', 'end', 'startDateTime', 'endDateTime',
                      'staffMemberIds', 'serviceId', 'isLocationOnline', 'value', 'error')


def response_shape(appointment):
    """Describe only known fields, never arbitrary keys or customer content."""
    if not isinstance(appointment, dict):
        return {'is_object': False}
    shape = {'is_object': True,
             'fields_present': {field: field in appointment for field in APPOINTMENT_FIELDS}}
    times = {}
    for field in ('start', 'end', 'startDateTime', 'endDateTime'):
        value = appointment.get(field)
        if not isinstance(value, dict):
            continue
        zone = value.get('timeZone')
        details = {'has_dateTime': 'dateTime' in value,
                   'timezone': zone if zone in ('UTC', 'Eastern Standard Time', 'America/Toronto') else 'unrecognized_or_missing'}
        try:
            timestamp = value.get('dateTime')
            if isinstance(timestamp, str) and len(timestamp) <= 64:
                details['parsed_datetime'] = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).isoformat()
        except ValueError:
            details['dateTime_parseable'] = False
        times[field] = details
    shape['time_fields'] = times
    return shape


async def check(diagnostics):
    diagnostics['stage'] = 'load_settings'
    import asyncpg, httpx
    from config.settings import settings as s
    diagnostics['stage'] = 'database_read'
    connection = await asyncpg.connect(s.database_url, timeout=10, command_timeout=10)
    try:
        async with connection.transaction(readonly=True):
            row = await connection.fetchrow('''
                SELECT id, call_sid, ms_booking_id, appointment_time, created_at
                FROM public.bookings ORDER BY id DESC LIMIT 1
            ''')
            if not row or not row['ms_booking_id']:
                print(json.dumps({'status': 'NO_PROVIDER_ID_ON_LATEST_LOCAL_BOOKING'}))
                return
            count = await connection.fetchval(
                'SELECT count(*) FROM public.bookings WHERE call_sid=$1', row['call_sid']
            ) if row['call_sid'] else None
    finally:
        await connection.close(timeout=5)
    async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
        diagnostics['stage'] = 'authenticate'
        token = await client.post(
            'https://login.microsoftonline.com/' + quote(s.ms_bookings_tenant_id, safe='') + '/oauth2/v2.0/token',
            data={'grant_type': 'client_credentials', 'client_id': s.ms_bookings_client_id,
                  'client_secret': s.ms_bookings_client_secret, 'scope': 'https://graph.microsoft.com/.default'})
        token.raise_for_status()
        access_token = token.json()['access_token']
        url = ('https://graph.microsoft.com/v1.0/solutions/bookingBusinesses/'
               + quote(s.ms_bookings_business_id, safe='') + '/appointments/'
               + quote(row['ms_booking_id'], safe=''))
        diagnostics['stage'] = 'appointment_read'
        response = await client.get(url, headers={'Authorization': 'Bearer ' + access_token})
        response.raise_for_status()
        appointment = response.json()
    diagnostics['stage'] = 'validate_appointment_fields'
    diagnostics['response_shape'] = response_shape(appointment)
    def utc(field):
        value = appointment[field]
        if value.get('timeZone') != 'UTC':
            raise ValueError('unexpected_provider_timezone')
        result = datetime.fromisoformat(value['dateTime'].replace('Z', '+00:00'))
        if result.utcoffset() not in (None, timezone.utc.utcoffset(None)):
            raise ValueError('conflicting_provider_offset')
        return result.replace(tzinfo=timezone.utc)
    start, end = utc('start'), utc('end')
    toronto = ZoneInfo('America/Toronto')
    expected = datetime(2026, 9, 14, 11, 0, tzinfo=toronto)
    staff_match = appointment.get('staffMemberIds') == ['93ee7133-8b0c-42c4-a886-a368b998de4b']
    duration = (end - start).total_seconds() / 60
    print(json.dumps({
        'status': 'READBACK_COMPLETE',
        'local_row_id': row['id'],
        'local_created_at_raw': str(row['created_at']),
        'local_appointment_time_raw': str(row['appointment_time']),
        'local_records_same_call': count,
        'provider_id_matches_local': appointment.get('id') == row['ms_booking_id'],
        'provider_start_toronto': start.astimezone(toronto).isoformat(),
        'provider_end_toronto': end.astimezone(toronto).isoformat(),
        'duration_minutes': duration,
        'consultant_is_hussam': staff_match,
        'service_matches': appointment.get('serviceId') == '357dc857-4360-4801-8bc4-12d3ed63afa3',
        'in_person': appointment.get('isLocationOnline') is False,
        'matches_expected_time_staff_duration': start == expected and staff_match and duration == 30
    }, indent=2))

async def report():
    diagnostics = {'stage': 'starting'}
    try:
        await asyncio.wait_for(check(diagnostics), timeout=70)
    except Exception as error:
        failure = {'status': 'CHECK_FAILED', 'error_type': type(error).__name__,
                   'http_status': getattr(getattr(error, 'response', None), 'status_code', None),
                   **diagnostics}
        if isinstance(error, KeyError):
            key = error.args[0] if error.args else None
            failure['missing_expected_field'] = key if key in EXPECTED_FIELDS else 'other'
        print(json.dumps(failure, indent=2))


if __name__ == '__main__':
    from loguru import logger
    logger.remove()
    logging.disable(logging.CRITICAL)
    asyncio.run(report())
