"""Offline caller-flow probe; synthetic calendar, no provider or database contact."""
import asyncio
import logging
import os
from datetime import datetime, timedelta
from unittest import mock
from zoneinfo import ZoneInfo

os.environ.update({
    'SECRET_KEY': 'synthetic', 'DATABASE_URL': 'postgresql://synthetic:synthetic@localhost/test',
    'TWILIO_ACCOUNT_SID': 'ACsynthetic', 'TWILIO_AUTH_TOKEN': 'synthetic',
    'TWILIO_PHONE_NUMBER': '+14165550100', 'DEEPGRAM_API_KEY': 'synthetic',
    'ELEVENLABS_API_KEY': 'synthetic', 'OPENAI_API_KEY': 'synthetic',
})
from loguru import logger
logger.remove()
logging.disable(logging.CRITICAL)
from services.calendar.ms_bookings_service import MSBookingsService
from services.calendar.calendar_base import BookingResult
from services.conversation import orchestrator as module

NOW = datetime(2030, 1, 7, 9, tzinfo=ZoneInfo('America/Toronto'))


class FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is None else NOW.astimezone(tz)


async def journey(language, zero_status, chosen):
    staff = [
        {'staff_id': 'hussam', 'service_id': 'service', 'name': 'Hussam', 'name_ar': 'حسام'},
        {'staff_id': 'rami', 'service_id': 'service', 'name': 'Rami', 'name_ar': 'رامي'},
        {'staff_id': 'abdul', 'service_id': 'service', 'name': 'Abdul', 'name_ar': 'عبدول'},
    ]
    accountants = mock.Mock()
    accountants.get_all_accountants.return_value = staff
    accountants.get_accountant_by_name.side_effect = lambda name: next(row for row in staff if row['name'] == name)
    calendar = MSBookingsService.__new__(MSBookingsService)
    calendar.business_id = 'business'
    calendar._staff_cache = {}
    calendar.is_available = mock.AsyncMock(return_value=True)
    calendar.get_staff_members = mock.AsyncMock(return_value=[])
    calendar.get_customer_appointments = mock.AsyncMock(return_value=[])
    calendar.create_booking = mock.AsyncMock(return_value=BookingResult(
        success=True, appointment_id='synthetic-provider-id', staff_name=chosen))

    async def request(method, endpoint, *, json_data):
        assert method == 'POST' and endpoint.endswith('/getStaffAvailability')
        staff_id = json_data['staffIds'][0]
        return {'value': [{'staffId': staff_id, 'availabilityItems': [{
            'status': 'busy' if staff_id == 'hussam' else 'available',
            'startDateTime': {'dateTime': '2030-01-08T10:00:00-05:00'},
            'endDateTime': {'dateTime': '2030-01-08T11:00:00-05:00'},
        }, {
            'status': zero_status,
            'startDateTime': {'dateTime': '2030-01-08T10:00:00-05:00'},
            'endDateTime': {'dateTime': '2030-01-08T10:00:00-05:00'},
        }]}]}

    calendar._make_request = mock.AsyncMock(side_effect=request)
    context = module.ConversationContext(call_sid='synthetic', phone_number='+14165550100', language=language)
    context.caller_name = 'Synthetic Caller'
    agent = module.ConversationOrchestrator.__new__(module.ConversationOrchestrator)
    agent._conversations = {'synthetic': context}
    agent._speak_to_caller = mock.AsyncMock()
    persist = mock.AsyncMock()
    settings = mock.Mock(automatic_notifications_enabled=True, ms_bookings_tenant_id='tenant', ms_bookings_business_id='business')
    arguments = {'accountant_name': 'Hussam', 'date_time': '2030-01-08 10:00', 'customer_name': 'Synthetic Caller'}
    with (mock.patch.object(module, 'datetime', FrozenDateTime),
          mock.patch.object(module, 'get_settings', return_value=settings),
          mock.patch.object(module, 'persist_booking_record', persist),
          mock.patch('services.database.get_db_pool', new=mock.AsyncMock()),
          mock.patch('services.calendar.ms_bookings_service.get_calendar_service', return_value=calendar),
          mock.patch('services.config.accountants_service.get_accountants_service', return_value=accountants)):
        blocked = await agent._check_booking('synthetic', arguments)
        assert blocked.startswith('STAFF_UNAVAILABLE:'), blocked.split(':', 1)[0]
        assert ('رامي' if language == 'ar' else 'Rami') in blocked
        assert context.pending_booking is None
        assert (await agent._confirm_booking('synthetic', {'confirm': True})).startswith('BOOKING_NEEDS_RECHECK:')
        calendar.create_booking.assert_not_awaited()
        arguments.update(accountant_name=chosen, date_time='2030-01-09 10:00')
        assert (await agent._check_booking('synthetic', arguments)).startswith('SCHEDULE_FULL:')
        assert context.pending_booking is None
        arguments['date_time'] = '2030-01-08 10:00'
        assert (await agent._check_booking('synthetic', arguments)).startswith('SLOT_AVAILABLE:')
        assert context.pending_booking['staff_id'] == chosen.lower()
        assert (await agent._confirm_booking('synthetic', {'confirm': False})).startswith('BOOKING_CANCELLED:')
        calendar.create_booking.assert_not_awaited()
        assert (await agent._check_booking('synthetic', arguments)).startswith('SLOT_AVAILABLE:')
        assert (await agent._confirm_booking('synthetic', {'confirm': True})).startswith('BOOKING_SUCCESS:')
        calendar.create_booking.assert_awaited_once()
        persist.assert_awaited_once()
        assert persist.await_args.kwargs['provider_appointment_id'] == 'synthetic-provider-id'
        assert persist.await_args.kwargs['notifications_enabled'] is True
        assert calendar.create_booking.await_args.kwargs['staff_id'] == chosen.lower()
        assert calendar.create_booking.await_args.kwargs['start_time'] == NOW + timedelta(days=1, hours=1)
        calendar._make_request.side_effect = None
        calendar._make_request.return_value = None
        failed = await agent._check_booking('synthetic', arguments)
        assert failed.startswith('AVAILABILITY_UNVERIFIED:')
        assert context.pending_booking is None
        calendar.create_booking.assert_awaited_once()


async def main():
    assert not hasattr(module.ConversationOrchestrator, '_process_verified_turn')
    for language in ('en', 'ar'):
        for zero_status in ('available', 'busy', 'outOfOffice'):
            for chosen in ('Rami', 'Abdul'):
                await journey(language, zero_status, chosen)


if __name__ == '__main__':
    asyncio.run(main())
    print('ZERO_INTERVAL_OFFLINE_OK')
