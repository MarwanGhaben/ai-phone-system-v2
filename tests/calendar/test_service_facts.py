"""T014-C service facts must come from one exact, bounded Graph read."""
import asyncio
from datetime import datetime, timezone

import httpx

import pytest

from services.calendar.service_facts import decode_service_facts
from services.calendar.ms_bookings_service import MSBookingsService
from services.scheduling.models import AvailabilityContractError, CalendarScope


SCOPE = CalendarScope("tenant", "business", "service")
NOW = datetime(2026, 9, 18, tzinfo=timezone.utc)


def wire():
    return {
        "id": "service", "defaultDuration": "PT30M", "preBuffer": "PT0S",
        "postBuffer": "PT0S", "staffMemberIds": ["staff"],
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


def test_decodes_approved_service_without_guessing_defaults():
    result = decode_service_facts(wire(), SCOPE, NOW)
    assert result.status == "verified"
    assert result.facts.staff_ids == ("staff",)
    assert result.facts.maximum_attendees == 1


@pytest.mark.parametrize("duration", [
    "PT1800.0000000000000000000000000001S",
    "PT1799.9999999999999999999999999999S",
    "PT1800.0000001S",
])
def test_submicrosecond_duration_never_rounds_to_approved_service(duration):
    data = wire()
    data["defaultDuration"] = duration
    result = decode_service_facts(data, SCOPE, NOW)
    assert result.status == "unverified"
    assert result.facts is None


def test_exact_duration_with_extra_zero_precision_is_representable():
    data = wire()
    data["defaultDuration"] = "PT1800.0000000000000000000000000000S"
    assert decode_service_facts(data, SCOPE, NOW).facts.duration.total_seconds() == 1800


def test_naive_observation_cannot_acquire_the_machine_timezone():
    with pytest.raises(ValueError, match="invalid service facts observation"):
        decode_service_facts(wire(), SCOPE, NOW.replace(tzinfo=None))


@pytest.mark.parametrize("field,value", [
    ("defaultDuration", "P1M"), ("defaultDuration", "P1Y"),
    ("defaultDuration", "P1DT"),
    ("defaultDuration", "PT-1M"), ("defaultDuration", "PTNaNS"),
    ("staffMemberIds", []), ("isLocationOnline", "false"),
    ("maximumAttendeesCount", True), ("schedulingPolicy", None),
])
def test_malformed_facts_never_authorize(field, value):
    data = wire()
    data[field] = value
    result = decode_service_facts(data, SCOPE, NOW)
    assert result.status == "unverified"
    assert result.facts is None


def test_mismatched_id_and_nested_continuation_fail_without_facts():
    data = wire()
    data["id"] = "other-service"
    result = decode_service_facts(data, SCOPE, NOW)
    assert result.status == "unverified" and result.facts is None
    data = wire()
    data["schedulingPolicy"]["customAvailabilities"] = [
        {"startDate": "2023-12-22", "endDate": "2024-01-01",
         "availabilityType": "notBookable", "@odata.nextLink": "private"}]
    result = decode_service_facts(data, SCOPE, NOW)
    assert result.status == "unverified" and result.facts is None


@pytest.mark.asyncio
async def test_scope_mismatch_never_contacts_transport():
    contacted = []
    def handler(request):
        contacted.append(request)
        return httpx.Response(200, json=wire())
    service = MSBookingsService({"tenant_id": "tenant", "business_id": "business",
                                 "client_id": "client", "client_secret": "secret"})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service._http_client = client
    try:
        with pytest.raises(AvailabilityContractError):
            await service.get_service_facts(CalendarScope("other", "business", "service"))
        assert not contacted
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_real_adapter_exact_get_and_failure_classes():
    paths = []
    status = 200

    def handler(request):
        nonlocal status
        paths.append((request.method, str(request.url)))
        if request.url.host == "login.microsoftonline.com":
            return httpx.Response(200, json={"access_token": "canary", "expires_in": 600})
        if status != 200:
            return httpx.Response(status, json={"error": "private body"})
        return httpx.Response(200, json=wire())

    service = MSBookingsService({"tenant_id": "tenant", "business_id": "business",
                                 "client_id": "client", "client_secret": "secret"})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                               follow_redirects=True)
    service._http_client = client
    try:
        result = await service.get_service_facts(SCOPE)
        assert result.status == "verified"
        assert paths[-1] == (
            "GET", "https://graph.microsoft.com/v1.0/solutions/bookingBusinesses/business/services/service")
        for status, category in ((401, "authentication"), (403, "permission_denied"),
                                 (429, "throttled"), (503, "provider_error")):
            result = await service.get_service_facts(SCOPE)
            assert result.status == "unverified"
            assert result.failure == category
            assert result.facts is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_service_read_timeout_and_cancellation_release_shared_slot():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request):
        if request.url.host == "login.microsoftonline.com":
            return httpx.Response(200, json={"access_token": "canary", "expires_in": 600})
        entered.set()
        await release.wait()
        return httpx.Response(200, json=wire())

    service = MSBookingsService({"tenant_id": "tenant", "business_id": "business",
                                 "client_id": "client", "client_secret": "secret"})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service._http_client = client
    service.AVAILABILITY_TIMEOUT_SECONDS = 0.1
    try:
        task = asyncio.create_task(service.get_service_facts(SCOPE))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await asyncio.sleep(0)
        assert not service._availability_cleanup_tasks
    finally:
        release.set()
        await client.aclose()


@pytest.mark.asyncio
async def test_body_and_close_overrun_returns_timeout_and_retains_slot_until_cleanup():
    body_entered, close_entered = asyncio.Event(), asyncio.Event()
    release_body, release_close = asyncio.Event(), asyncio.Event()

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            body_entered.set()
            try:
                await release_body.wait()
                yield b"{}"
            finally:
                close_entered.set()
                await release_close.wait()

        async def aclose(self):
            close_entered.set()
            await release_close.wait()

    def handler(request):
        if request.url.host == "login.microsoftonline.com":
            return httpx.Response(200, json={"access_token": "canary", "expires_in": 600})
        return httpx.Response(200, stream=SlowStream())

    service = MSBookingsService({"tenant_id": "tenant", "business_id": "business",
                                 "client_id": "client", "client_secret": "secret"})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service._http_client = client
    service.AVAILABILITY_TIMEOUT_SECONDS = 0.05
    service.AVAILABILITY_CLEANUP_GRACE_SECONDS = 0.05
    task = asyncio.create_task(service.get_service_facts(SCOPE))
    try:
        await asyncio.wait_for(body_entered.wait(), 1)
        result = await asyncio.wait_for(task, 1)
        assert result.status == "unverified" and result.failure == "timeout"
        assert service._availability_cleanup_tasks
    finally:
        release_body.set()
        release_close.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        workers = tuple(service._availability_cleanup_tasks)
        if workers:
            # The transport is deliberately cancelled at its deadline. Joining
            # owned cleanup must consume that terminal cancellation, not treat
            # it as a failure of the already-asserted public timeout contract.
            done, pending = await asyncio.wait(workers, timeout=1)
            assert not pending, "owned transport cleanup did not finish"
            await asyncio.gather(*done, return_exceptions=True)
        await asyncio.sleep(0)
        assert not service._availability_cleanup_tasks
        await client.aclose()
