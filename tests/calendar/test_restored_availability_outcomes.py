"""Exercise the restored phone handler against the real legacy calendar reader."""
import ast
import asyncio
import importlib.util
from pathlib import Path
import subprocess
import sys
import types
from unittest import mock

import pytest

from services.calendar.ms_bookings_service import MSBookingsService
from services.calendar.legacy_availability import AvailabilityReadError

ROOT = Path(__file__).resolve().parents[2]
RESTORED = '792952b2965e85513bf0971a41bfaff7bd3377c4'


def restored_source():
    source = subprocess.check_output(['git', 'show', RESTORED + ':services/conversation/orchestrator.py'], cwd=ROOT)
    release_path = ROOT / 'scripts/deploy-restored-availability.py'
    if release_path.exists():
        spec = importlib.util.spec_from_file_location('availability_release', release_path)
        release = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(release)
        source = release.patch_orchestrator(source)
    return source


@pytest.fixture
def restored_guard():
    module = types.ModuleType('restored_availability_test_orchestrator')
    sys.modules[module.__name__] = module
    exec(compile(restored_source(), 'restored_orchestrator', 'exec'), module.__dict__)
    tree = ast.parse((ROOT / 'scripts/probe-restored-call-flow.py').read_text(encoding='utf-8'))
    fixture = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FIXTURE' for t in node.targets))
    fixture = fixture.replace('from services.conversation import orchestrator as orchestrator_module',
                              'import restored_availability_test_orchestrator as orchestrator_module')
    scope = {'__name__': 'restored_guard_fixture'}
    exec(compile(fixture, 'restored_guard_fixture', 'exec'), scope)
    guard = scope['BookingSafetyGuardsTests']()
    # This fixture borrows synchronous setUp only, not unittest's async runner.
    # Its mock.patch cleanups must still execute when pytest tears it down.
    guard._callCleanup = lambda function, *args, **kwargs: function(*args, **kwargs)
    guard.setUp()
    try:
        yield guard
    finally:
        guard.doCleanups()
        sys.modules.pop(module.__name__, None)


def wire(status='busy'):
    return {'value': [{'staffId': 'staff-1', 'availabilityItems': [{
        'status': status,
        'startDateTime': {'dateTime': '2030-01-08T10:00:00-05:00'},
        'endDateTime': {'dateTime': '2030-01-08T11:00:00-05:00'},
    }]}]}


def calendar(response):
    service = MSBookingsService.__new__(MSBookingsService)
    service.business_id = 'synthetic'
    service._staff_cache = {}
    service.is_available = mock.AsyncMock(return_value=True)
    service.get_staff_members = mock.AsyncMock(return_value=[])
    service._make_request = mock.AsyncMock(return_value=response)
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize('language', ['en', 'ar'])
@pytest.mark.parametrize('status', ['busy', 'outOfOffice'])
async def test_blocked_staff_offers_other_consultants_without_authorizing_booking(restored_guard, language, status):
    guard = restored_guard
    guard.context.language = language
    guard.context.pending_booking = {'stale': True}
    guard.accountants.get_all_accountants.return_value = [
        {'staff_id': 'staff-1', 'name': 'Hussam', 'name_ar': 'حسام'},
        {'staff_id': 'staff-2', 'name': 'Rami', 'name_ar': 'رامي'},
        {'staff_id': 'staff-3', 'name': 'Abdul', 'name_ar': 'عبدول'},
    ]
    guard.calendar.get_available_slots = calendar(wire(status)).get_available_slots
    reply = await guard.orchestrator._check_booking('call-a', guard._arguments())
    assert reply.startswith('STAFF_UNAVAILABLE:'), reply
    assert ('رامي' if language == 'ar' else 'Rami') in reply
    assert ('حسام' if language == 'ar' else 'Hussam') in reply
    assert 'AVAILABILITY_UNVERIFIED' not in reply
    assert guard.context.pending_booking is None
    await guard.orchestrator._confirm_booking('call-a', {'confirm': True})
    guard.calendar.create_booking.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('response', [None, {}, {'value': []}, {'value': [{'staffId': 'other', 'availabilityItems': []}]},
                                    {'value': [{'staffId': 'staff-1'}]}, wire('unknownFutureValue'),
                                    {**wire(), '@odata.nextLink': 'private-continuation'}])
async def test_failed_or_incomplete_read_is_not_reported_as_busy(restored_guard, response):
    service = calendar(response)
    with pytest.raises(AvailabilityReadError):
        await service.get_available_slots('service-1', 'staff-1')
    guard = restored_guard
    guard.calendar.get_available_slots = service.get_available_slots
    reply = await guard.orchestrator._check_booking('call-a', guard._arguments())
    assert reply.startswith('AVAILABILITY_UNVERIFIED:'), reply
    assert guard.context.pending_booking is None
    guard.calendar.create_booking.assert_not_awaited()


@pytest.mark.asyncio
async def test_available_slots_preserve_exact_pending_booking(restored_guard):
    guard = restored_guard
    guard.calendar.get_available_slots = calendar(wire('available')).get_available_slots
    reply = await guard.orchestrator._check_booking('call-a', guard._arguments())
    assert reply.startswith('SLOT_AVAILABLE:'), reply
    assert guard.context.pending_booking['staff_id'] == 'staff-1'
    guard.calendar.create_booking.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancellation_propagates():
    service = calendar(wire())
    service._make_request.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await service.get_available_slots('service-1', 'staff-1')


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['nested-page', 'invalid-date', 'reversed-time', 'duplicate-staff'])
async def test_incomplete_busy_evidence_never_becomes_unavailability(fault):
    response = wire()
    staff = response['value'][0]
    if fault == 'nested-page':
        staff['availabilityItems@odata.nextLink'] = 'private'
    elif fault == 'invalid-date':
        staff['availabilityItems'][0]['startDateTime']['dateTime'] = 'invalid'
    elif fault == 'reversed-time':
        staff['availabilityItems'][0]['endDateTime']['dateTime'] = '2030-01-08T09:00:00-05:00'
    else:
        response['value'].append(staff)
    with pytest.raises(AvailabilityReadError):
        await calendar(response).get_available_slots('service-1', 'staff-1')


@pytest.mark.asyncio
async def test_selected_staff_empty_items_is_a_completed_empty_search():
    response = {'staffAvailabilityItem': [{'staffId': 'staff-1', 'availabilityItems': []}]}
    assert await calendar(response).get_available_slots('service-1', 'staff-1') == []


@pytest.mark.asyncio
async def test_unconfigured_calendar_cannot_report_busy():
    service = calendar(wire())
    service.is_available.return_value = False
    with pytest.raises(AvailabilityReadError):
        await service.get_available_slots('service-1', 'staff-1')
    service._make_request.assert_not_awaited()


def test_only_restored_check_method_changes():
    before = ast.parse(subprocess.check_output(['git', 'show', RESTORED + ':services/conversation/orchestrator.py'], cwd=ROOT))
    after = ast.parse(restored_source())
    def methods(tree):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ConversationOrchestrator')
        return {n.name: ast.dump(n) for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    previous, candidate = methods(before), methods(after)
    assert previous.keys() == candidate.keys()
    assert [name for name in previous if previous[name] != candidate[name]] == ['_check_booking']


@pytest.mark.asyncio
@pytest.mark.parametrize('language', ['en', 'ar'])
@pytest.mark.parametrize('zero_status', ['available', 'busy', 'outOfOffice'])
async def test_zero_duration_entry_preserves_real_slots_and_pending_offer(restored_guard, language, zero_status):
    """Live Rami/Abdul responses each include an equal-endpoint entry."""
    guard = restored_guard
    guard.context.language = language
    response = wire('available')
    zero = {
        'status': zero_status,
        'startDateTime': {'dateTime': '2030-01-08T10:00:00-05:00'},
        'endDateTime': {'dateTime': '2030-01-08T10:00:00-05:00'},
    }
    response['value'][0]['availabilityItems'].insert(0, zero)
    service = calendar(response)
    slots = await service.get_available_slots('service-1', 'staff-1')
    assert len(slots) == 2
    assert all(slot.end_time > slot.start_time for slot in slots)
    guard.calendar.get_available_slots = service.get_available_slots
    reply = await guard.orchestrator._check_booking('call-a', guard._arguments())
    assert reply.startswith('SLOT_AVAILABLE:'), reply
    assert guard.context.pending_booking['staff_id'] == 'staff-1'
    guard.calendar.create_booking.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['available', 'busy', 'outOfOffice'])
async def test_only_zero_duration_entries_never_create_a_slot(status):
    response = wire(status)
    item = response['value'][0]['availabilityItems'][0]
    item['endDateTime'] = dict(item['startDateTime'])
    assert await calendar(response).get_available_slots('service-1', 'staff-1') == []
