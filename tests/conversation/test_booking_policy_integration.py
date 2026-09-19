"""Actual caller check through the bounded Graph readers and approved policy."""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import httpx
import pytest
import yaml
from zoneinfo import ZoneInfo

from services.calendar.ms_bookings_service import MSBookingsService
from services.calendar.service_facts import decode_service_facts
from services.conversation import orchestrator as module
from services.scheduling.booking_check import (
    load_booking_policy, make_booking_query, parse_requested_time,
)
from services.scheduling import booking_check
from services.scheduling.models import (
    AvailabilityFailure, AvailabilityFailureCategory, AvailabilityResult,
    AvailabilityStatus, CalendarScope, FreeInterval, TimeInterval,
    AvailabilityQuery,
)


UTC = timezone.utc
TORONTO = ZoneInfo("America/Toronto")
NOW = datetime(2026, 9, 18, 13, 0, tzinfo=UTC)
STAFF = "93ee7133-8b0c-42c4-a886-a368b998de4b"
SERVICE = "357dc857-4360-4801-8bc4-12d3ed63afa3"
START = datetime(2026, 9, 18, 10, 0, tzinfo=TORONTO)


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is None else NOW.astimezone(tz)


def service_wire(staff=STAFF):
    return {
        "id": SERVICE, "defaultDuration": "PT30M", "preBuffer": "PT0S",
        "postBuffer": "PT0S", "staffMemberIds": [staff],
        "isLocationOnline": False, "maximumAttendeesCount": 1,
        "isHiddenFromCustomers": False,
        "schedulingPolicy": {
            "allowStaffSelection": True, "timeSlotInterval": "PT30M",
            "minimumLeadTime": "PT0S", "maximumAdvance": "P365D",
            "generalAvailability": {"availabilityType": "bookWhenStaffAreFree",
                                    "businessHours": None},
            "customAvailabilities": [],
        },
    }


def availability_wire(staff=STAFF, start=START):
    end = start + timedelta(minutes=30)
    def field(value):
        return {"dateTime": value.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "timeZone": "UTC"}
    return {"value": [{"staffId": staff, "availabilityItems": [
        {"status": "available", "startDateTime": field(start),
         "endDateTime": field(end)}]}]}


@pytest.mark.asyncio
async def test_real_transport_decoder_policy_and_handler_exact_selection():
    observed = []

    def handler(request):
        observed.append((request.method, str(request.url)))
        if request.url.host == "login.microsoftonline.com":
            return httpx.Response(200, json={"access_token": "canary", "expires_in": 600})
        if request.method == "GET":
            return httpx.Response(200, json=service_wire())
        return httpx.Response(200, json=availability_wire())

    calendar = MSBookingsService({
        "tenant_id": "tenant", "business_id": "business",
        "client_id": "client", "client_secret": "secret"})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    calendar._http_client = client
    calendar.get_customer_appointments = mock.AsyncMock(return_value=[])
    calendar.create_booking = mock.AsyncMock()
    orchestrator = module.ConversationOrchestrator.__new__(module.ConversationOrchestrator)
    orchestrator._conversations = {"call": module.ConversationContext(
        call_sid="call", phone_number="+14165550100")}
    orchestrator._speak_to_caller = mock.AsyncMock()
    context = orchestrator._conversations["call"]
    try:
        with mock.patch("services.calendar.ms_bookings_service.get_calendar_service",
                        return_value=calendar), mock.patch.object(
                            module, "business_now", return_value=NOW), mock.patch(
                            "services.calendar.ms_bookings_service.datetime", Clock):
            result = await orchestrator._check_booking("call", {
                "date_time": "2026-09-18 10:00", "accountant_name": "Hussam",
                "customer_name": "Caller"})
        assert result.startswith("SLOT_AVAILABLE:"), result
        assert "1901 Banff Ave" in result
        assert context.pending_booking["staff_id"] == STAFF
        assert context.pending_booking["service_id"] == SERVICE
        assert context.pending_booking["appointment_time"] == START
        assert observed[-2][0] == "GET"
        assert observed[-1][0] == "POST"
        calendar.create_booking.assert_not_awaited()
    finally:
        await client.aclose()


def fake_check(*, service=None, free=True, availability_status=None, observed=NOW,
               availability_observed=None, slot=START):
    calendar = mock.Mock()
    calendar.tenant_id = "tenant"
    calendar.business_id = "business"
    calendar.is_available = mock.AsyncMock(return_value=True)
    calendar.get_staff_members = mock.AsyncMock()
    calendar.get_services = mock.AsyncMock()
    calendar.get_available_slots = mock.AsyncMock()
    calendar.get_customer_appointments = mock.AsyncMock(return_value=[])
    calendar.create_booking = mock.AsyncMock()
    calendar.get_service_facts = mock.AsyncMock(side_effect=lambda scope:
        decode_service_facts(service if service is not None else service_wire(),
                             scope, observed))

    async def availability(query):
        read_at = availability_observed or observed
        if availability_status:
            return AvailabilityResult(query, availability_status, read_at,
                failure=AvailabilityFailure(
                    AvailabilityFailureCategory.PROVIDER_ERROR))
        intervals = (FreeInterval(query.scope, query.staff_ids[0],
                                  TimeInterval(slot, slot + timedelta(minutes=30))),) if free else ()
        return AvailabilityResult(query, AvailabilityStatus.AVAILABLE if intervals else
                                  AvailabilityStatus.NO_AVAILABILITY, read_at, intervals)

    calendar.get_availability = mock.AsyncMock(side_effect=availability)
    orchestrator = module.ConversationOrchestrator.__new__(module.ConversationOrchestrator)
    orchestrator._conversations = {"call": module.ConversationContext(
        call_sid="call", phone_number="+14165550100")}
    orchestrator._speak_to_caller = mock.AsyncMock()
    return orchestrator, calendar


async def checked(orchestrator, calendar, when="2026-09-18 10:00", sid="call",
                  now=NOW):
    with mock.patch("services.calendar.ms_bookings_service.get_calendar_service",
                    return_value=calendar), mock.patch.object(
                    module, "business_now", return_value=now):
        return await orchestrator._check_booking(sid, {
            "date_time": when, "accountant_name": "Hussam",
            "customer_name": "Caller"})


@pytest.mark.asyncio
async def test_invalid_offset_minutes_cannot_normalize_into_an_available_slot():
    orchestrator, calendar = fake_check()
    context = orchestrator._conversations["call"]
    context.pending_booking = {"appointment_time": "stale"}
    response = await checked(orchestrator, calendar, "2026-09-18T15:00:00+00:60")
    assert response.startswith("INVALID_DATE_TIME:"), response
    assert context.pending_booking is None
    calendar.get_service_facts.assert_not_awaited()
    calendar.get_availability.assert_not_awaited()
    with mock.patch("services.calendar.ms_bookings_service.get_calendar_service",
                    return_value=calendar):
        await orchestrator._confirm_booking("call", {"confirm": True})
    calendar.create_booking.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("change, expected", [
    ("empty", "NO_ELIGIBLE_SLOT:"),
    ("provider_error", "AVAILABILITY_UNVERIFIED:"),
    ("removed_staff", "POLICY_UNVERIFIED:"),
    ("duration", "POLICY_UNVERIFIED:"),
    ("grid", "POLICY_UNVERIFIED:"),
    ("buffer", "POLICY_UNVERIFIED:"),
    ("online", "POLICY_UNVERIFIED:"),
    ("capacity", "POLICY_UNVERIFIED:"),
    ("closed", "NO_ELIGIBLE_SLOT:"),
    ("general_closed", "NO_ELIGIBLE_SLOT:"),
    ("unsupported", "POLICY_UNVERIFIED:"),
    ("custom_unsupported", "POLICY_UNVERIFIED:"),
    ("lead", "NO_ELIGIBLE_SLOT:"),
    ("advance", "NO_ELIGIBLE_SLOT:"),
    ("stale", "AVAILABILITY_UNVERIFIED:"),
    ("future", "AVAILABILITY_UNVERIFIED:"),
    ("stale_availability", "AVAILABILITY_UNVERIFIED:"),
    ("future_availability", "AVAILABILITY_UNVERIFIED:"),
])
async def test_rejected_evidence_clears_old_pending_and_cannot_confirm(change, expected):
    wire = service_wire()
    free = True
    availability_status = None
    observed = NOW
    availability_observed = None
    if change == "empty":
        free = False
    elif change == "provider_error":
        availability_status = AvailabilityStatus.UNAVAILABLE
    elif change == "removed_staff":
        wire["staffMemberIds"] = ["someone-else"]
    elif change == "duration":
        wire["defaultDuration"] = "PT45M"
    elif change == "grid":
        wire["schedulingPolicy"]["timeSlotInterval"] = "PT15M"
    elif change == "buffer":
        wire["preBuffer"] = "PT5M"
    elif change == "online":
        wire["isLocationOnline"] = True
    elif change == "capacity":
        wire["maximumAttendeesCount"] = 2
    elif change == "closed":
        wire["schedulingPolicy"]["customAvailabilities"] = [{
            "startDate": "2026-09-18", "endDate": "2026-09-18",
            "availabilityType": "notBookable", "businessHours": None}]
    elif change == "general_closed":
        wire["schedulingPolicy"]["generalAvailability"]["availabilityType"] = "notBookable"
    elif change == "unsupported":
        wire["schedulingPolicy"]["generalAvailability"]["availabilityType"] = "customWeeklyHours"
    elif change == "custom_unsupported":
        wire["schedulingPolicy"]["customAvailabilities"] = [{
            "startDate": "2026-09-18", "endDate": "2026-09-18",
            "availabilityType": "customWeeklyHours", "businessHours": []}]
    elif change == "lead":
        wire["schedulingPolicy"]["minimumLeadTime"] = "PT2H"
    elif change == "advance":
        wire["schedulingPolicy"]["maximumAdvance"] = "PT30M"
    elif change == "stale":
        observed -= timedelta(seconds=61)
    elif change == "future":
        observed += timedelta(seconds=1)
    elif change == "stale_availability":
        availability_observed = NOW - timedelta(seconds=61)
    elif change == "future_availability":
        availability_observed = NOW + timedelta(seconds=1)
    orchestrator, calendar = fake_check(service=wire, free=free,
        availability_status=availability_status, observed=observed,
        availability_observed=availability_observed)
    context = orchestrator._conversations["call"]
    context.pending_booking = {"appointment_time": "stale"}
    response = await checked(orchestrator, calendar)
    assert response.startswith(expected), response
    assert context.pending_booking is None
    with mock.patch("services.calendar.ms_bookings_service.get_calendar_service",
                    return_value=calendar):
        confirmed = await orchestrator._confirm_booking("call", {"confirm": True})
    assert confirmed.startswith("BOOKING_NEEDS_RECHECK:")
    calendar.create_booking.assert_not_awaited()
    calendar.get_staff_members.assert_not_awaited()
    calendar.get_services.assert_not_awaited()
    calendar.get_available_slots.assert_not_awaited()


@pytest.mark.asyncio
async def test_old_service_exception_does_not_close_current_window():
    wire = service_wire()
    wire["schedulingPolicy"]["customAvailabilities"] = [{
        "startDate": "2023-12-22", "endDate": "2024-01-01",
        "availabilityType": "notBookable", "businessHours": None}]
    orchestrator, calendar = fake_check(service=wire)
    response = await checked(orchestrator, calendar)
    assert response.startswith("SLOT_AVAILABLE:"), response


@pytest.mark.asyncio
async def test_availability_from_another_query_cannot_authorize():
    orchestrator, calendar = fake_check()
    context = orchestrator._conversations["call"]
    context.pending_booking = {"appointment_time": "stale"}

    async def wrong_query(query):
        other = AvailabilityQuery(query.scope, query.window, query.staff_ids,
                                  "different-request")
        return AvailabilityResult(other, AvailabilityStatus.NO_AVAILABILITY, NOW)

    calendar.get_availability.side_effect = wrong_query
    response = await checked(orchestrator, calendar)
    assert response.startswith("AVAILABILITY_UNVERIFIED:"), response
    assert context.pending_booking is None
    with mock.patch("services.calendar.ms_bookings_service.get_calendar_service",
                    return_value=calendar):
        await orchestrator._confirm_booking("call", {"confirm": True})
    calendar.create_booking.assert_not_awaited()


def test_explicit_dates_holiday_horizon_dst_and_coverage():
    scope = CalendarScope("tenant", "business", SERVICE)
    snapshot = load_booking_policy(scope, STAFF, datetime(2026, 12, 24, 14, tzinfo=UTC))
    assert snapshot.horizon_end == date(2026, 12, 29)
    assert date(2026, 12, 28) not in snapshot.policy.closure_calendar.closed_dates
    snapshot_2027 = load_booking_policy(scope, STAFF, datetime(2027, 12, 24, 14, tzinfo=UTC))
    assert snapshot_2027.horizon_end == date(2027, 12, 28)
    assert load_booking_policy(scope, STAFF, datetime(2027, 12, 30, 14, tzinfo=UTC)) is None
    assert load_booking_policy(scope, STAFF, datetime(2028, 1, 1, tzinfo=UTC)) is None
    assert parse_requested_time("2026-03-08 02:30") is None
    assert parse_requested_time("2026-11-01 01:30") is None
    assert parse_requested_time("2026-09-18") is None
    assert parse_requested_time("2026-09-18 10:00") == parse_requested_time(
        "2026-09-18T14:00:00Z")
    query = make_booking_query(snapshot, STAFF, datetime(2026, 12, 24, 14, tzinfo=UTC))
    assert query.staff_ids == (STAFF,)
    assert query.window.start.astimezone(TORONTO).date() == date(2026, 12, 24)
    assert query.window.end.astimezone(TORONTO).date() == date(2026, 12, 30)


def test_friday_and_monday_holidays_extend_only_the_approved_horizon():
    scope = CalendarScope("tenant", "business", SERVICE)
    original = yaml.safe_load(Path("clients/booking-policy.yaml").read_text(encoding="utf-8"))
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "policy.yaml"
        original["closures"]["dates"].append("2026-04-06")
        path.write_text(yaml.safe_dump(original), encoding="utf-8")
        snapshot = load_booking_policy(scope, STAFF,
            datetime(2026, 4, 2, 13, tzinfo=UTC), path)
        assert snapshot.horizon_end == date(2026, 4, 8)
        original["closures"]["dates"].remove("2026-04-06")
        path.write_text(yaml.safe_dump(original), encoding="utf-8")
        snapshot = load_booking_policy(scope, STAFF,
            datetime(2026, 9, 4, 13, tzinfo=UTC), path)
        assert snapshot.horizon_end == date(2026, 9, 9)


def test_production_policy_rejects_missing_approved_holiday():
    scope = CalendarScope("tenant", "business", SERVICE)
    data = yaml.safe_load(Path("clients/booking-policy.yaml").read_text(encoding="utf-8"))
    data["closures"]["dates"].remove("2026-09-07")
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "policy.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        with mock.patch.object(booking_check, "DEFAULT_POLICY_PATH", path):
            assert load_booking_policy(scope, STAFF, NOW) is None
        path.write_text("closures: [", encoding="utf-8")
        with mock.patch.object(booking_check, "DEFAULT_POLICY_PATH", path):
            assert load_booking_policy(scope, STAFF, NOW) is None


@pytest.mark.asyncio
async def test_exact_grid_close_boundary_offset_and_past_do_not_round_or_roll():
    for when, slot, expected in (
        ("2026-09-18 16:30", datetime(2026, 9, 18, 16, 30, tzinfo=TORONTO),
         "SLOT_AVAILABLE:"),
        ("2026-09-18 17:00", datetime(2026, 9, 18, 17, 0, tzinfo=TORONTO),
         "NO_ELIGIBLE_SLOT:"),
        ("2026-09-18T14:00:00Z", START, "SLOT_AVAILABLE:"),
        ("2026-09-17 10:00", START, "NO_ELIGIBLE_SLOT:"),
    ):
        orchestrator, calendar = fake_check(slot=slot)
        context = orchestrator._conversations["call"]
        context.pending_booking = {"appointment_time": "stale"}
        response = await checked(orchestrator, calendar, when)
        assert response.startswith(expected), response
        if expected != "SLOT_AVAILABLE:":
            assert context.pending_booking is None
            with mock.patch("services.calendar.ms_bookings_service.get_calendar_service",
                            return_value=calendar):
                await orchestrator._confirm_booking("call", {"confirm": True})
            calendar.create_booking.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("old_failure", [False, True])
async def test_late_old_check_cannot_replace_or_clear_new_pending(old_failure):
    orchestrator, calendar = fake_check()
    entered, release = asyncio.Event(), asyncio.Event()
    reads = 0
    original = calendar.get_service_facts.side_effect

    async def delayed(scope):
        nonlocal reads
        reads += 1
        if reads == 1:
            entered.set()
            await release.wait()
            if old_failure:
                raise RuntimeError("private old failure")
        return original(scope)

    calendar.get_service_facts.side_effect = delayed
    args = {"date_time": "2026-09-18 10:00", "accountant_name": "Hussam"}
    with mock.patch("services.calendar.ms_bookings_service.get_calendar_service",
                    return_value=calendar), mock.patch.object(
                    module, "business_now", return_value=NOW):
        old = asyncio.create_task(orchestrator._check_booking("call", args))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            newest = await orchestrator._check_booking("call", args)
            assert newest.startswith("SLOT_AVAILABLE:"), newest
            pending = orchestrator._conversations["call"].pending_booking
            release.set()
            await asyncio.wait_for(old, 1)
            assert orchestrator._conversations["call"].pending_booking is pending
        finally:
            release.set()
            if not old.done():
                old.cancel()
            await asyncio.gather(old, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("replace", [False, True])
async def test_teardown_or_replacement_while_waiting_cannot_install_pending(replace):
    orchestrator, calendar = fake_check()
    entered, release = asyncio.Event(), asyncio.Event()
    original = calendar.get_service_facts.side_effect

    async def delayed(scope):
        entered.set()
        await release.wait()
        return original(scope)

    calendar.get_service_facts.side_effect = delayed
    context = orchestrator._conversations["call"]
    with mock.patch("services.calendar.ms_bookings_service.get_calendar_service",
                    return_value=calendar), mock.patch.object(
                    module, "business_now", return_value=NOW):
        task = asyncio.create_task(orchestrator._check_booking("call", {
            "date_time": "2026-09-18 10:00", "accountant_name": "Hussam"}))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            if replace:
                orchestrator._conversations["call"] = module.ConversationContext(
                    call_sid="call", phone_number="+14165550100")
            else:
                context.state = module.ConversationState.ENDED
            release.set()
            response = await asyncio.wait_for(task, 1)
            assert response.startswith("BOOKING_CHECK_SUPERSEDED:"), response
            assert context.pending_booking is None
            assert orchestrator._conversations["call"].pending_booking is None
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_notice_is_rechecked_after_provider_wait():
    initial = datetime(2026, 9, 18, 13, 30, tzinfo=UTC)  # 09:30 Toronto
    later = initial + timedelta(minutes=1)
    orchestrator, calendar = fake_check(observed=initial)
    entered, release = asyncio.Event(), asyncio.Event()
    original = calendar.get_availability.side_effect
    current_now = initial

    async def delayed(query):
        entered.set()
        await release.wait()
        return await original(query)

    calendar.get_availability.side_effect = delayed
    context = orchestrator._conversations["call"]
    context.pending_booking = {"appointment_time": "stale"}
    with mock.patch("services.calendar.ms_bookings_service.get_calendar_service",
                    return_value=calendar), mock.patch.object(
                    module, "business_now", side_effect=lambda: current_now):
        task = asyncio.create_task(orchestrator._check_booking("call", {
            "date_time": "2026-09-18 10:00", "accountant_name": "Hussam"}))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            current_now = later
            release.set()
            response = await asyncio.wait_for(task, 1)
            assert response.startswith("NO_ELIGIBLE_SLOT:"), response
            assert context.pending_booking is None
            confirmed = await orchestrator._confirm_booking("call", {"confirm": True})
            assert confirmed.startswith("BOOKING_NEEDS_RECHECK:"), confirmed
            calendar.create_booking.assert_not_awaited()
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
