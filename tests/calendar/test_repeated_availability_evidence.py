"""Reproduce the duplicate-evidence rejection identified by the live incident.

Times and identities are synthetic; the server diagnostic established the
duplicate rejection, not the original private response body.
"""
from copy import deepcopy
from unittest.mock import patch

import httpx
import pytest

from services.calendar.contracts import MAX_AVAILABILITY_ITEMS
from services.calendar.ms_bookings_service import MSBookingsService
from services.scheduling.booking_check import (
    assess_booking, load_booking_policy, make_booking_query,
)
from tests.calendar.test_graph_availability_contracts import (
    decode, item, make_query, payload, staff_entry,
)
from tests.calendar.test_live_graph_shapes import (
    Clock, NOW, SCOPE, SERVICE, STAFF, START, observed_availability, observed_service,
)


@pytest.mark.parametrize("status", ["available", "Available", "busy", "Busy",
                                   "outOfOffice", "OutOfOffice"])
def test_repeating_valid_evidence_does_not_change_availability(status):
    evidence = item(status=status)
    original = decode(payload([staff_entry(items=[evidence])]))
    repeated = decode(payload([staff_entry(items=[evidence, deepcopy(evidence)])]))
    assert repeated == original


@pytest.mark.parametrize("statuses,expected", [
    (["available", "Available"], "available"),
    (["busy", "outOfOffice", "Busy"], "no_availability"),
    (["available", "available", "busy"], "invalid_response"),
    (["busy", "busy", "available"], "invalid_response"),
    (["unknownFutureValue", "unknownFutureValue"], "incomplete"),
    (["slotsAvailable", "slotsAvailable", "available"], "incomplete"),
])
def test_duplicates_do_not_hide_contradictions_or_incomplete_evidence(statuses, expected):
    result = decode(payload([staff_entry(items=[item(status=s) for s in statuses])]))
    assert result.status.value == expected
    if expected != "available":
        assert not result.intervals


def test_identical_times_for_different_staff_stay_separate():
    result = decode(payload([staff_entry(s, [item(), item()]) for s in ("a", "b")]),
                    make_query(staff_ids=("a", "b")))
    assert result.status.value == "available"
    assert [e.staff_id for e in result.intervals] == ["a", "b"]


def test_duplicate_staff_envelopes_remain_invalid():
    result = decode(payload([staff_entry(items=[item()]), staff_entry(items=[item()])]))
    assert result.status.value == "invalid_response"


def test_duplicates_count_toward_response_limit():
    result = decode(payload([staff_entry(items=[item()] * (MAX_AVAILABILITY_ITEMS + 1))]))
    assert result.status.value == "incomplete"
    assert not result.intervals


@pytest.mark.asyncio
@pytest.mark.parametrize("duplicate_status", ["Available", "Busy", "OutOfOffice"])
async def test_duplicate_provider_evidence_still_reaches_caller_policy(duplicate_status):
    wire = observed_availability()
    items = wire["staffAvailabilityItem"][0]["availabilityItems"]
    repeated = deepcopy(items[0])
    repeated["status"] = duplicate_status
    if duplicate_status != "Available":
        # Non-free evidence is adjacent, so deduplication must not erase the
        # genuinely available interval or invent availability in blocked time.
        repeated["startDateTime"]["dateTime"] = "2026-09-18T15:00:00Z"
        repeated["endDateTime"]["dateTime"] = "2026-09-18T15:30:00Z"
    items.extend([repeated, deepcopy(repeated)])
    contacted = []

    def handler(request):
        contacted.append(request.url.path)
        if request.url.host == "login.microsoftonline.com":
            return httpx.Response(200, json={"access_token": "synthetic", "expires_in": 600})
        if request.method == "GET" and request.url.path.endswith("/services/" + SERVICE):
            return httpx.Response(200, json=observed_service())
        assert request.method == "POST" and request.url.path.endswith("/getStaffAvailability")
        return httpx.Response(200, json=wire)

    calendar = MSBookingsService({"tenant_id": "tenant", "business_id": "business",
                                 "client_id": "client", "client_secret": "synthetic"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        calendar._http_client = client
        snapshot = load_booking_policy(SCOPE, STAFF, NOW)
        query = make_booking_query(snapshot, STAFF, NOW)
        with patch("services.calendar.ms_bookings_service.datetime", Clock):
            facts = await calendar.get_service_facts(SCOPE)
            availability = await calendar.get_availability(query)
        assessment = assess_booking(snapshot, query, facts, availability, NOW)
        assert assessment.status == "available"
        assert len(assessment.candidates) == 1
        assert assessment.candidates[0].interval.start == START
        assert len(contacted) == 3
