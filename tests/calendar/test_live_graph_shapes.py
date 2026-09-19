"""Regressions for redacted provider shapes observed on 2026-09-19."""
from datetime import datetime, timezone
from unittest.mock import patch

import httpx
import pytest

from services.calendar.contracts import decode_availability
from services.calendar.ms_bookings_service import MSBookingsService
from services.calendar.service_facts import decode_service_facts
from services.scheduling.booking_check import (
    assess_booking, load_booking_policy, make_booking_query,
)
from services.scheduling.models import CalendarScope
from tests.conversation.test_booking_policy_integration import (
    Clock, NOW, SERVICE, STAFF, START, availability_wire, service_wire,
)


UTC_LABEL = "(UTC) Coordinated Universal Time"
SCOPE = CalendarScope("tenant", "business", SERVICE)


def hours():
    # Synthetic contents: the owner report established only a seven-item list.
    # Even apparently open hours must never reopen a notBookable period.
    return [{"day": day, "timeSlots": [{"start": "00:00:00", "end": "23:59:00"}]}
            for day in ("monday", "tuesday", "wednesday", "thursday", "friday",
                        "saturday", "sunday")]


def observed_service(start="2023-12-22", end="2024-01-01"):
    data = service_wire()
    policy = data["schedulingPolicy"]
    policy["generalAvailability"]["businessHours"] = []
    policy["customAvailabilities"] = [{
        "startDate": start, "endDate": end, "availabilityType": "notBookable",
        "businessHours": hours(),
    }]
    return data


def observed_availability(status="available"):
    data = availability_wire()
    data["staffAvailabilityItem"] = data.pop("value")
    item = data["staffAvailabilityItem"][0]["availabilityItems"][0]
    item["status"] = status
    for key in ("startDateTime", "endDateTime"):
        item[key]["timeZone"] = UTC_LABEL
    return data


@pytest.mark.parametrize("hours_value", [None, [], hours()])
def test_closed_period_accepts_ancillary_hours_without_opening_them(hours_value):
    data = observed_service()
    data["schedulingPolicy"]["customAvailabilities"][0]["businessHours"] = hours_value
    result = decode_service_facts(data, SCOPE, NOW)
    assert result.status == "verified"
    assert result.facts.general_mode == "bookWhenStaffAreFree"
    assert result.facts.custom_windows[0].mode == "notBookable"


@pytest.mark.parametrize("value", [{}, "", False, 0, hours()])
def test_free_mode_does_not_accept_populated_or_malformed_hours(value):
    data = service_wire()
    data["schedulingPolicy"]["generalAvailability"]["businessHours"] = value
    result = decode_service_facts(data, SCOPE, NOW)
    assert result.status == "unverified"
    assert result.facts is None


@pytest.mark.parametrize("value", [{}, "", False, 0])
def test_closed_period_still_rejects_nonlist_hours(value):
    data = observed_service()
    data["schedulingPolicy"]["customAvailabilities"][0]["businessHours"] = value
    assert decode_service_facts(data, SCOPE, NOW).status == "unverified"


@pytest.mark.parametrize("month", [1, 7])
@pytest.mark.parametrize("offset", ["", "Z", "+00:00"])
def test_exact_utc_display_alias_keeps_instants_in_both_seasons(month, offset):
    now = datetime(2026, month, 12, 13, tzinfo=timezone.utc)
    snapshot = load_booking_policy(SCOPE, STAFF, now)
    query = make_booking_query(snapshot, STAFF, now)
    data = observed_availability()
    item = data["staffAvailabilityItem"][0]["availabilityItems"][0]
    for field, hour in (("startDateTime", 14), ("endDateTime", 15)):
        item[field]["dateTime"] = f"2026-{month:02d}-12T{hour}:00:00{offset}"
    result = decode_availability(data, query, now)
    assert result.status.value == "available"
    assert result.intervals[0].interval.start == now.replace(hour=14)
    assert result.intervals[0].interval.end == now.replace(hour=15)


@pytest.mark.parametrize("zone,offset", [
    (UTC_LABEL, "+01:00"), ("(UTC) Unrecognized", "Z"),
    ("(UTC+01:00) Coordinated Universal Time", ""), (UTC_LABEL + " ", "Z"),
])
def test_alias_does_not_guess_unknown_zones_or_override_conflicting_offset(zone, offset):
    data = observed_availability()
    item = data["staffAvailabilityItem"][0]["availabilityItems"][0]
    item["startDateTime"] = {"dateTime": "2026-09-18T14:00:00" + offset, "timeZone": zone}
    snapshot = load_booking_policy(SCOPE, STAFF, NOW)
    result = decode_availability(data, make_booking_query(snapshot, STAFF, NOW), NOW)
    assert result.status.value == "invalid_response"
    assert not result.intervals


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario,expected", [
    ("historical_closure", "available"), ("closed_today", "no_eligible"),
    ("closure_starts_today", "no_eligible"), ("closure_ends_today", "no_eligible"),
    ("general_closed", "no_eligible"), ("busy", "no_eligible"),
])
async def test_actual_readers_and_policy_preserve_closure_and_busy_semantics(scenario, expected):
    service = observed_service()
    window = service["schedulingPolicy"]["customAvailabilities"][0]
    if scenario == "closed_today":
        window.update(startDate="2026-09-18", endDate="2026-09-18")
    elif scenario == "closure_starts_today":
        window.update(startDate="2026-09-18", endDate="2026-09-21")
    elif scenario == "closure_ends_today":
        window.update(startDate="2026-09-17", endDate="2026-09-18")
    elif scenario == "general_closed":
        service["schedulingPolicy"]["generalAvailability"] = {
            "availabilityType": "notBookable", "businessHours": hours()}
    contacted = []

    def handler(request):
        contacted.append((request.method, request.url.path))
        if request.url.host == "login.microsoftonline.com":
            return httpx.Response(200, json={"access_token": "synthetic", "expires_in": 600})
        if request.method == "GET" and request.url.path.endswith("/services/" + SERVICE):
            return httpx.Response(200, json=service)
        assert request.method == "POST" and request.url.path.endswith("/getStaffAvailability")
        return httpx.Response(200, json=observed_availability("Busy" if scenario == "busy" else "Available"))

    calendar = MSBookingsService({"tenant_id": "tenant", "business_id": "business",
                                 "client_id": "client", "client_secret": "synthetic"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        calendar._http_client = client
        snapshot = load_booking_policy(SCOPE, STAFF, NOW)
        query = make_booking_query(snapshot, STAFF, NOW)
        with patch("services.calendar.ms_bookings_service.datetime", Clock):
            facts = await calendar.get_service_facts(SCOPE)
            availability = await calendar.get_availability(query)
        assert facts.status == "verified"
        assessment = assess_booking(snapshot, query, facts, availability, NOW)
        assert assessment.status == expected
        if expected == "available":
            assert len(assessment.candidates) == 1
            assert assessment.candidates[0].interval.start == START
        else:
            assert not assessment.candidates
        assert len(contacted) == 3
