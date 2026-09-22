from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import AsyncIterator

import httpx
import pytest

from services.calendar.contracts import (
    MAX_AVAILABILITY_ITEMS,
    decode_availability,
)
from services.calendar.ms_bookings_service import MSBookingsService
from services.scheduling.models import (
    AvailabilityContractError,
    AvailabilityFailureCategory,
    AvailabilityQuery,
    AvailabilityStatus,
    CalendarScope,
    TimeInterval,
)


UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[2]
T007_HASHES = {
    "services/scheduling/models.py":
        "576d6627208d2a621edf2a41c01493da04284d6a94f16174347a48002f1f931d",
    "tests/scheduling/test_booking_result_contracts.py":
        "ceaefbc1b4b8372b6aab614800a1acb4f360ab9dca687be2b5c48f06f4e260ad",
    "docs/delegation/T007-A-result.md":
        "f6ff3d65f12d40a559b47f7313989d4409b63887bd93c3b4a5a4bd8ba9020e3b",
}


def make_query(
    *,
    tenant_id: str = "tenant/opaque",
    business_id: str = "business/opaque",
    service_id: str = "service-context",
    staff_ids: tuple[str, ...] = ("staff-a",),
    start: datetime = datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
    end: datetime = datetime(2027, 1, 1, 0, 0, tzinfo=UTC),
    request_id: str = "request-canary",
) -> AvailabilityQuery:
    return AvailabilityQuery(
        CalendarScope(tenant_id, business_id, service_id),
        TimeInterval(start, end),
        staff_ids,
        request_id,
    )


def date_time(
    value: str,
    zone: str | None = "UTC",
) -> dict[str, object]:
    result: dict[str, object] = {"dateTime": value}
    if zone is not None:
        result["timeZone"] = zone
    return result


def item(
    *,
    status: object = "available",
    start: object = None,
    end: object = None,
) -> dict[str, object]:
    return {
        "status": status,
        "startDateTime": start or date_time("2026-06-18T14:00:00Z"),
        "endDateTime": end or date_time("2026-06-18T15:00:00Z"),
    }


def staff_entry(
    staff_id: object = "staff-a",
    items: object = None,
) -> dict[str, object]:
    return {
        "staffId": staff_id,
        "availabilityItems": [] if items is None else items,
    }


def payload(
    entries: object,
    *,
    envelope: str = "value",
    **extra: object,
) -> dict[str, object]:
    return {envelope: entries, **extra}


def decode(
    wire: object,
    query: AvailabilityQuery | None = None,
) -> object:
    return decode_availability(
        wire,
        query or make_query(),
        datetime(2026, 6, 18, 15, 0, 1, tzinfo=UTC),
    )


def assert_failure(result: object, status: AvailabilityStatus,
                   category: AvailabilityFailureCategory) -> None:
    assert result.status is status
    assert result.intervals == ()
    assert result.failure is not None
    assert result.failure.category is category


def config(**overrides: str) -> dict[str, str]:
    values = {
        "tenant_id": "tenant/opaque",
        "client_id": "client-id",
        "client_secret": "client-secret-canary",
        "business_id": "business/opaque",
    }
    values.update(overrides)
    return values


async def service_with_handler(
    handler: object,
    *,
    settings: dict[str, str] | None = None,
    budget: float = 1.0,
    cleanup_grace: float = 0.1,
) -> tuple[MSBookingsService, httpx.AsyncClient]:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    )
    service = MSBookingsService(settings or config())
    service._http_client = client
    service.AVAILABILITY_TIMEOUT_SECONDS = budget
    service.AVAILABILITY_CLEANUP_GRACE_SECONDS = cleanup_grace
    return service, client


def token_response() -> httpx.Response:
    return httpx.Response(200, json={"access_token": "synthetic-token", "expires_in": 600})


def test_t007_contract_files_remain_byte_for_byte_frozen() -> None:
    observed = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in T007_HASHES
    }
    assert observed == T007_HASHES


@pytest.mark.parametrize("envelope", ["value", "staffAvailabilityItem"])
@pytest.mark.parametrize("available_status", ["available", "Available"])
def test_both_documented_envelopes_and_status_spellings_produce_free_evidence(
    envelope: str,
    available_status: str,
) -> None:
    result = decode(payload([
        staff_entry(items=[item(status=available_status)])
    ], envelope=envelope))

    assert result.status is AvailabilityStatus.AVAILABLE
    assert result.failure is None
    assert len(result.intervals) == 1
    assert result.intervals[0].staff_id == "staff-a"
    assert result.intervals[0].interval.start == datetime(
        2026, 6, 18, 14, 0, tzinfo=UTC
    )


@pytest.mark.parametrize("status", ["busy", "Busy", "outOfOffice", "OutOfOffice"])
def test_complete_non_free_or_explicit_empty_evidence_is_no_availability(
    status: str,
) -> None:
    busy = decode(payload([staff_entry(items=[item(status=status)])]))
    empty = decode(payload([staff_entry(items=[])]))

    assert busy.status is AvailabilityStatus.NO_AVAILABILITY
    assert busy.failure is None
    assert empty.status is AvailabilityStatus.NO_AVAILABILITY
    assert empty.failure is None


def test_dual_missing_null_wrong_type_and_top_level_empty_envelopes_fail_closed() -> None:
    dual = payload([], staffAvailabilityItem=[])
    assert_failure(decode(dual), AvailabilityStatus.INVALID_RESPONSE,
                   AvailabilityFailureCategory.INVALID_RESPONSE)
    for wire in ({}, {"value": None}, {"value": {}}, None):
        assert_failure(decode(wire), AvailabilityStatus.INVALID_RESPONSE,
                       AvailabilityFailureCategory.INVALID_RESPONSE)
    assert_failure(decode({"value": []}), AvailabilityStatus.INCOMPLETE,
                   AvailabilityFailureCategory.INCOMPLETE)


def test_exact_staff_coverage_distinguishes_missing_from_malformed_identity() -> None:
    requested = make_query(staff_ids=("staff-a", "staff-b"))
    missing = decode(payload([staff_entry("staff-a")]), requested)
    assert_failure(missing, AvailabilityStatus.INCOMPLETE,
                   AvailabilityFailureCategory.INCOMPLETE)

    for entries in (
        [staff_entry("staff-a"), staff_entry("staff-a")],
        [staff_entry("staff-a"), staff_entry("staff-b"), staff_entry("staff-c")],
        [staff_entry(None), staff_entry("staff-b")],
        [{"availabilityItems": []}, staff_entry("staff-b")],
    ):
        assert_failure(decode(payload(entries), requested),
                       AvailabilityStatus.INVALID_RESPONSE,
                       AvailabilityFailureCategory.INVALID_RESPONSE)


def test_missing_or_malformed_item_collection_is_not_empty_success() -> None:
    for entry in (
        {"staffId": "staff-a"},
        {"staffId": "staff-a", "availabilityItems": None},
        {"staffId": "staff-a", "availabilityItems": {}},
    ):
        assert_failure(decode(payload([entry])),
                       AvailabilityStatus.INVALID_RESPONSE,
                       AvailabilityFailureCategory.INVALID_RESPONSE)


@pytest.mark.parametrize(
    "status",
    ["slotsAvailable", "SlotsAvailable", "unknownFutureValue",
     "UnknownFutureValue", "futureProviderStatus"],
)
def test_group_capacity_and_unknown_statuses_are_incomplete(status: str) -> None:
    result = decode(payload([staff_entry(items=[item(status=status)])]))
    assert_failure(result, AvailabilityStatus.INCOMPLETE,
                   AvailabilityFailureCategory.INCOMPLETE)


def test_unknown_status_still_requires_valid_interval_fields() -> None:
    malformed = item(status="futureProviderStatus")
    malformed["endDateTime"] = None
    result = decode(payload([staff_entry(items=[malformed])]))
    assert_failure(result, AvailabilityStatus.INVALID_RESPONSE,
                   AvailabilityFailureCategory.INVALID_RESPONSE)


def test_late_malformed_item_discards_earlier_valid_free_evidence() -> None:
    late = item(status="busy")
    late["startDateTime"] = {"dateTime": "private-malformed-canary", "timeZone": "UTC"}
    result = decode(payload([staff_entry(items=[item(), late])]))

    assert_failure(result, AvailabilityStatus.INVALID_RESPONSE,
                   AvailabilityFailureCategory.INVALID_RESPONSE)


def test_continuation_marker_is_incomplete_and_never_followed() -> None:
    result = decode(payload(
        [staff_entry(items=[item()])],
        **{"@odata.nextLink": "https://evil.invalid/private-canary"},
    ))
    assert_failure(result, AvailabilityStatus.INCOMPLETE,
                   AvailabilityFailureCategory.INCOMPLETE)


@pytest.mark.parametrize("envelope", ["value", "staffAvailabilityItem"])
@pytest.mark.parametrize("marker_value", [None, {}, "https://evil.invalid/page"])
def test_collection_qualified_top_level_continuation_is_incomplete(
    envelope: str,
    marker_value: object,
) -> None:
    wire = payload([staff_entry(items=[item()])], envelope=envelope)
    wire[f"{envelope}@odata.nextLink"] = marker_value

    assert_failure(decode(wire), AvailabilityStatus.INCOMPLETE,
                   AvailabilityFailureCategory.INCOMPLETE)


@pytest.mark.parametrize("marker_value", [None, {}, "https://evil.invalid/page"])
def test_nested_availability_collection_continuation_discards_free_evidence(
    marker_value: object,
) -> None:
    entry = staff_entry(items=[item()])
    entry["availabilityItems@odata.nextLink"] = marker_value

    assert_failure(decode(payload([entry])), AvailabilityStatus.INCOMPLETE,
                   AvailabilityFailureCategory.INCOMPLETE)


def test_unrelated_metadata_does_not_invent_continuation_or_free_time() -> None:
    free = payload([staff_entry(items=[item()])], **{
        "@odata.context": "synthetic-context",
        "unrelated": {"@odata.nextLink": "https://evil.invalid/not-a-collection"},
    })
    empty_entry = staff_entry(items=[])
    empty_entry["availabilityItems@odata.count"] = 0

    assert decode(free).status is AvailabilityStatus.AVAILABLE
    assert decode(payload([empty_entry])).status is AvailabilityStatus.NO_AVAILABILITY


@pytest.mark.parametrize(
    ("start", "end", "zone", "expected_start"),
    [
        ("2026-06-18T14:00:00Z", "2026-06-18T15:00:00Z", "UTC",
         datetime(2026, 6, 18, 14, 0, tzinfo=UTC)),
        ("2026-06-18T10:00:00", "2026-06-18T11:00:00", "Eastern Standard Time",
         datetime(2026, 6, 18, 14, 0, tzinfo=UTC)),
        ("2026-06-18T07:00:00", "2026-06-18T08:00:00", "Pacific Standard Time",
         datetime(2026, 6, 18, 14, 0, tzinfo=UTC)),
        ("2026-06-18T07:00:00", "2026-06-18T08:00:00",
         "Pacific Time (US & Canada)", datetime(2026, 6, 18, 14, 0, tzinfo=UTC)),
        ("2026-06-18T10:00:00", "2026-06-18T11:00:00", "America/Toronto",
         datetime(2026, 6, 18, 14, 0, tzinfo=UTC)),
        ("2026-06-18T14:00:00Z", "2026-06-18T15:00:00Z", "GMT",
         datetime(2026, 6, 18, 14, 0, tzinfo=UTC)),
        ("2026-06-18T10:00:00-04:00", "2026-06-18T11:00:00-04:00",
         "Eastern Standard Time", datetime(2026, 6, 18, 14, 0, tzinfo=UTC)),
        ("2026-01-15T10:00:00", "2026-01-15T11:00:00", "Eastern Standard Time",
         datetime(2026, 1, 15, 15, 0, tzinfo=UTC)),
        ("2026-01-15T07:00:00", "2026-01-15T08:00:00", "America/Los_Angeles",
         datetime(2026, 1, 15, 15, 0, tzinfo=UTC)),
    ],
)
def test_supported_utc_windows_and_iana_zones_normalize_with_dst(
    start: str,
    end: str,
    zone: str,
    expected_start: datetime,
) -> None:
    wire_item = item(
        start=date_time(start, zone),
        end=date_time(end, zone),
    )
    result = decode(payload([staff_entry(items=[wire_item])]))
    assert result.status is AvailabilityStatus.AVAILABLE
    assert result.intervals[0].interval.start == expected_start


@pytest.mark.parametrize(
    ("start", "end", "zone"),
    [
        ("2026-03-08T02:30:00", "2026-03-08T03:30:00", "America/Toronto"),
        ("2026-11-01T01:30:00", "2026-11-01T02:30:00", "Eastern Standard Time"),
        ("2026-06-18T10:00:00-05:00", "2026-06-18T11:00:00-05:00",
         "Eastern Standard Time"),
        ("2026-06-18T14:00:00-04:00", "2026-06-18T15:00:00-04:00", "UTC"),
    ],
)
def test_dst_gap_fold_and_conflicting_offsets_fail_closed(
    start: str,
    end: str,
    zone: str,
) -> None:
    result = decode(payload([staff_entry(items=[item(
        start=date_time(start, zone), end=date_time(end, zone)
    )])]))
    assert_failure(result, AvailabilityStatus.INVALID_RESPONSE,
                   AvailabilityFailureCategory.INVALID_RESPONSE)


@pytest.mark.parametrize("zone", [None, "", "Central Standard Time", "Mars/Base"])
def test_missing_or_unsupported_zones_fail_closed(zone: str | None) -> None:
    result = decode(payload([staff_entry(items=[item(
        start=date_time("2026-06-18T10:00:00", zone),
        end=date_time("2026-06-18T11:00:00", zone),
    )])]))
    assert_failure(result, AvailabilityStatus.INVALID_RESPONSE,
                   AvailabilityFailureCategory.INVALID_RESPONSE)


def test_seventh_fractional_digit_is_accepted_only_when_representable() -> None:
    accepted = decode(payload([staff_entry(items=[item(
        start=date_time("2026-06-18T14:00:00.1234560Z"),
        end=date_time("2026-06-18T15:00:00.1234560Z"),
    )])]))
    assert accepted.status is AvailabilityStatus.AVAILABLE
    assert accepted.intervals[0].interval.start.microsecond == 123456

    rejected = decode(payload([staff_entry(items=[item(
        start=date_time("2026-06-18T14:00:00.1234567Z"),
        end=date_time("2026-06-18T15:00:00.1234567Z"),
    )])]))
    assert_failure(rejected, AvailabilityStatus.INVALID_RESPONSE,
                   AvailabilityFailureCategory.INVALID_RESPONSE)


def test_equal_reversed_and_out_of_window_intervals_are_invalid() -> None:
    query = make_query(
        start=datetime(2026, 6, 18, 14, 0, tzinfo=UTC),
        end=datetime(2026, 6, 18, 16, 0, tzinfo=UTC),
    )
    cases = (
        item(start=date_time("2026-06-18T14:00:00Z"),
             end=date_time("2026-06-18T14:00:00Z")),
        item(start=date_time("2026-06-18T15:00:00Z"),
             end=date_time("2026-06-18T14:00:00Z")),
        item(start=date_time("2026-06-18T13:59:59Z"),
             end=date_time("2026-06-18T15:00:00Z")),
        item(start=date_time("2026-06-18T15:00:00Z"),
             end=date_time("2026-06-18T16:00:01Z")),
    )
    for wire_item in cases:
        assert_failure(decode(payload([staff_entry(items=[wire_item])]), query),
                       AvailabilityStatus.INVALID_RESPONSE,
                       AvailabilityFailureCategory.INVALID_RESPONSE)


def test_exact_bounds_are_allowed_and_duplicate_items_are_idempotent() -> None:
    query = make_query(
        start=datetime(2026, 6, 18, 14, 0, tzinfo=UTC),
        end=datetime(2026, 6, 18, 15, 0, tzinfo=UTC),
    )
    exact = item()
    accepted = decode(payload([staff_entry(items=[exact])]), query)
    assert accepted.status is AvailabilityStatus.AVAILABLE

    duplicate = decode(payload([staff_entry(items=[exact, dict(exact)])]), query)
    assert duplicate == accepted


def test_free_non_free_overlap_is_invalid_but_adjacent_and_free_overlap_are_valid() -> None:
    free = item(start=date_time("2026-06-18T14:00:00Z"),
                end=date_time("2026-06-18T15:00:00Z"))
    overlapping_busy = item(
        status="busy",
        start=date_time("2026-06-18T14:30:00Z"),
        end=date_time("2026-06-18T15:30:00Z"),
    )
    contradiction = decode(payload([
        staff_entry(items=[free, overlapping_busy])
    ]))
    assert_failure(contradiction, AvailabilityStatus.INVALID_RESPONSE,
                   AvailabilityFailureCategory.INVALID_RESPONSE)

    adjacent_busy = item(
        status="busy",
        start=date_time("2026-06-18T15:00:00Z"),
        end=date_time("2026-06-18T15:30:00Z"),
    )
    adjacent = decode(payload([staff_entry(items=[free, adjacent_busy])]))
    assert adjacent.status is AvailabilityStatus.AVAILABLE

    overlapping_free = item(
        start=date_time("2026-06-18T14:30:00Z"),
        end=date_time("2026-06-18T15:30:00Z"),
    )
    retained = decode(payload([staff_entry(items=[free, overlapping_free])]))
    assert retained.status is AvailabilityStatus.AVAILABLE
    assert len(retained.intervals) == 2


def test_item_limit_is_incomplete_without_partial_intervals() -> None:
    repeated = [item(status="busy") for _ in range(MAX_AVAILABILITY_ITEMS + 1)]
    result = decode(payload([staff_entry(items=repeated)]))
    assert_failure(result, AvailabilityStatus.INCOMPLETE,
                   AvailabilityFailureCategory.INCOMPLETE)


def test_decoder_requires_accepted_query_and_aware_observation_time() -> None:
    with pytest.raises(AvailabilityContractError):
        decode_availability({"value": []}, object(), datetime.now(UTC))
    with pytest.raises(AvailabilityContractError):
        decode_availability(
            payload([staff_entry()]), make_query(), datetime(2026, 1, 1)
        )


@pytest.mark.asyncio
async def test_adapter_sends_exact_scoped_url_body_and_uses_no_mutation_route() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "login.microsoftonline.com":
            return token_response()
        return httpx.Response(200, json=payload([
            staff_entry("staff-a", [item(
                status="busy",
                start=date_time("2026-06-18T14:00:00.123456Z"),
                end=date_time("2026-06-18T15:00:00.123456Z"),
            )]),
            staff_entry("staff b/+", []),
        ]))

    query = make_query(
        staff_ids=("staff-a", "staff b/+"),
        start=datetime(2026, 6, 18, 14, 0, 0, 123456, tzinfo=UTC),
        end=datetime(2026, 6, 18, 16, 0, tzinfo=UTC),
    )
    service, client = await service_with_handler(handler)
    try:
        result = await service.get_availability(query)
    finally:
        await client.aclose()

    assert result.status is AvailabilityStatus.NO_AVAILABILITY
    assert len(requests) == 2
    assert requests[0].method == "POST"
    assert requests[0].url == httpx.URL(
        "https://login.microsoftonline.com/tenant%2Fopaque/oauth2/v2.0/token"
    )
    graph = requests[1]
    assert graph.method == "POST"
    assert graph.url == httpx.URL(
        "https://graph.microsoft.com/v1.0/solutions/bookingBusinesses/"
        "business%2Fopaque/getStaffAvailability"
    )
    assert json.loads(graph.content) == {
        "staffIds": ["staff-a", "staff b/+"],
        "startDateTime": {
            "dateTime": "2026-06-18T14:00:00.123456Z",
            "timeZone": "UTC",
        },
        "endDateTime": {
            "dateTime": "2026-06-18T16:00:00Z",
            "timeZone": "UTC",
        },
    }
    assert query.scope.service_id not in graph.content.decode()
    assert "/appointments" not in graph.url.path


@pytest.mark.asyncio
async def test_invalid_query_and_configured_scope_mismatch_never_contact_transport() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return token_response()

    service, client = await service_with_handler(handler)
    try:
        with pytest.raises(AvailabilityContractError):
            await service.get_availability(object())
        with pytest.raises(AvailabilityContractError):
            await service.get_availability(make_query(tenant_id="wrong-tenant"))
        with pytest.raises(AvailabilityContractError):
            await service.get_availability(make_query(business_id="wrong-business"))
    finally:
        await client.aclose()
    assert calls == 0


@pytest.mark.asyncio
async def test_missing_configuration_is_explicit_unavailable_without_network() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return token_response()

    service, client = await service_with_handler(
        handler,
        settings=config(client_secret=""),
    )
    try:
        result = await service.get_availability(make_query())
    finally:
        await client.aclose()
    assert_failure(result, AvailabilityStatus.UNAVAILABLE,
                   AvailabilityFailureCategory.NOT_CONFIGURED)
    assert calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "category"),
    [
        (401, AvailabilityFailureCategory.AUTHENTICATION),
        (403, AvailabilityFailureCategory.PERMISSION_DENIED),
        (404, AvailabilityFailureCategory.PROVIDER_ERROR),
        (429, AvailabilityFailureCategory.THROTTLED),
        (500, AvailabilityFailureCategory.PROVIDER_ERROR),
    ],
)
async def test_graph_http_failures_are_classified_without_retry(
    status_code: int,
    category: AvailabilityFailureCategory,
) -> None:
    graph_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal graph_calls
        if request.url.host == "login.microsoftonline.com":
            return token_response()
        graph_calls += 1
        return httpx.Response(status_code, text="private-response-canary")

    service, client = await service_with_handler(handler)
    try:
        result = await service.get_availability(make_query())
    finally:
        await client.aclose()
    assert_failure(result, AvailabilityStatus.UNAVAILABLE, category)
    assert result.failure.http_status == status_code
    assert graph_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "category"),
    [
        (400, AvailabilityFailureCategory.AUTHENTICATION),
        (401, AvailabilityFailureCategory.AUTHENTICATION),
        (403, AvailabilityFailureCategory.PERMISSION_DENIED),
        (429, AvailabilityFailureCategory.THROTTLED),
        (500, AvailabilityFailureCategory.PROVIDER_ERROR),
    ],
)
async def test_token_http_failures_are_classified_without_graph_contact(
    status_code: int,
    category: AvailabilityFailureCategory,
) -> None:
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        return httpx.Response(status_code, text="private-token-canary")

    service, client = await service_with_handler(handler)
    try:
        result = await service.get_availability(make_query())
    finally:
        await client.aclose()
    assert_failure(result, AvailabilityStatus.UNAVAILABLE, category)
    assert result.failure.http_status == status_code
    assert calls == ["login.microsoftonline.com"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [204, 206])
async def test_unexpected_success_status_is_invalid_response(status_code: int) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "login.microsoftonline.com":
            return token_response()
        return httpx.Response(status_code)

    service, client = await service_with_handler(handler)
    try:
        result = await service.get_availability(make_query())
    finally:
        await client.aclose()
    assert_failure(result, AvailabilityStatus.INVALID_RESPONSE,
                   AvailabilityFailureCategory.INVALID_RESPONSE)


@pytest.mark.asyncio
async def test_redirect_is_not_followed() -> None:
    graph_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal graph_calls
        if request.url.host == "login.microsoftonline.com":
            return token_response()
        graph_calls += 1
        return httpx.Response(302, headers={"Location": "https://evil.invalid/private"})

    service, client = await service_with_handler(handler)
    try:
        result = await service.get_availability(make_query())
    finally:
        await client.aclose()
    assert_failure(result, AvailabilityStatus.UNAVAILABLE,
                   AvailabilityFailureCategory.PROVIDER_ERROR)
    assert graph_calls == 1


@pytest.mark.asyncio
async def test_invalid_json_and_token_json_fail_safely() -> None:
    async def graph_invalid(request: httpx.Request) -> httpx.Response:
        if request.url.host == "login.microsoftonline.com":
            return token_response()
        return httpx.Response(200, content=b"not-json-private-canary")

    service, client = await service_with_handler(graph_invalid)
    try:
        graph_result = await service.get_availability(make_query())
    finally:
        await client.aclose()
    assert_failure(graph_result, AvailabilityStatus.INVALID_RESPONSE,
                   AvailabilityFailureCategory.INVALID_RESPONSE)

    async def token_invalid(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json-private-token-canary")

    service, client = await service_with_handler(token_invalid)
    try:
        token_result = await service.get_availability(make_query())
    finally:
        await client.aclose()
    assert_failure(token_result, AvailabilityStatus.INVALID_RESPONSE,
                   AvailabilityFailureCategory.INVALID_RESPONSE)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_lifetime",
    ["NaN", "Infinity", "-Infinity", "1e999", "true", "null", '"600"', "[]"],
)
async def test_malformed_token_lifetime_never_mutates_cache_or_contacts_graph(
    raw_lifetime: str,
) -> None:
    hosts: list[str] = []
    raw = (
        '{"access_token":"replacement-token","expires_in":'
        + raw_lifetime
        + "}"
    ).encode()

    async def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(200, content=raw)

    service, client = await service_with_handler(handler)
    previous_expiry = datetime.now(UTC) - timedelta(seconds=1)
    service._access_token = "previous-token"
    service._token_expires_at = previous_expiry
    try:
        result = await service.get_availability(make_query())
    finally:
        await client.aclose()

    assert_failure(result, AvailabilityStatus.INVALID_RESPONSE,
                   AvailabilityFailureCategory.INVALID_RESPONSE)
    assert hosts == ["login.microsoftonline.com"]
    assert service._access_token == "previous-token"
    assert service._token_expires_at is previous_expiry


@pytest.mark.asyncio
async def test_short_lived_token_is_used_once_but_not_cached_past_its_lifetime() -> None:
    token_calls = 0
    graph_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_calls, graph_calls
        if request.url.host == "login.microsoftonline.com":
            token_calls += 1
            return httpx.Response(200, json={
                "access_token": f"short-token-{token_calls}",
                "expires_in": 0.05,
            })
        graph_calls += 1
        body = json.loads(request.content)
        return httpx.Response(200, json=payload([
            staff_entry(body["staffIds"][0], [])
        ]))

    service, client = await service_with_handler(handler)
    try:
        first = await service.get_availability(make_query(request_id="short-one"))
        second = await service.get_availability(make_query(request_id="short-two"))
    finally:
        await client.aclose()

    assert first.status is AvailabilityStatus.NO_AVAILABILITY
    assert second.status is AvailabilityStatus.NO_AVAILABILITY
    assert token_calls == 2
    assert graph_calls == 2
    assert service._access_token is None
    assert service._token_expires_at is None


class TrackingStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class BlockingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.entered.set()
        await self.release.wait()
        yield b"{}"

    async def aclose(self) -> None:
        self.closed = True


class CancellationResistantCloseStream(httpx.AsyncByteStream):
    def __init__(self, *, close_error: bool = False) -> None:
        self.body_entered = asyncio.Event()
        self.closing = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False
        self.close_error = close_error
        self.close_cancellations = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.body_entered.set()
        await asyncio.Event().wait()
        yield b"{}"

    async def aclose(self) -> None:
        self.closing.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.close_cancellations += 1
        if self.close_error:
            raise RuntimeError("private-late-close-canary")
        self.closed = True


async def wait_for_availability_cleanup(
    service: MSBookingsService,
    timeout: float = 1.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while getattr(service, "_availability_cleanup_tasks", ()):
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("availability cleanup did not finish")
        await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_graph_and_token_response_size_limits_close_streams() -> None:
    graph_stream = TrackingStream([b"x" * (1024 * 1024 + 1)])

    async def graph_oversize(request: httpx.Request) -> httpx.Response:
        if request.url.host == "login.microsoftonline.com":
            return token_response()
        return httpx.Response(200, stream=graph_stream)

    service, client = await service_with_handler(graph_oversize)
    try:
        result = await service.get_availability(make_query())
    finally:
        await client.aclose()
    assert_failure(result, AvailabilityStatus.INCOMPLETE,
                   AvailabilityFailureCategory.INCOMPLETE)
    assert graph_stream.closed

    token_stream = TrackingStream([b"x" * (64 * 1024 + 1)])

    async def token_oversize(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=token_stream)

    service, client = await service_with_handler(token_oversize)
    try:
        result = await service.get_availability(make_query())
    finally:
        await client.aclose()
    assert_failure(result, AvailabilityStatus.INCOMPLETE,
                   AvailabilityFailureCategory.INCOMPLETE)
    assert token_stream.closed


@pytest.mark.asyncio
async def test_overall_budget_covers_token_and_body_stages_and_closes_body() -> None:
    never = asyncio.Event()
    token_entered = asyncio.Event()

    async def token_timeout(request: httpx.Request) -> httpx.Response:
        token_entered.set()
        await never.wait()
        return token_response()

    service, client = await service_with_handler(token_timeout, budget=0.2)
    token_task = asyncio.create_task(service.get_availability(make_query()))
    try:
        await asyncio.wait_for(token_entered.wait(), 1.0)
        token_result = await asyncio.wait_for(token_task, 1.0)
    finally:
        never.set()
        await asyncio.gather(token_task, return_exceptions=True)
        await client.aclose()
    assert_failure(token_result, AvailabilityStatus.UNAVAILABLE,
                   AvailabilityFailureCategory.TIMEOUT)

    body_stream = BlockingStream()

    async def body_timeout(request: httpx.Request) -> httpx.Response:
        if request.url.host == "login.microsoftonline.com":
            return token_response()
        return httpx.Response(200, stream=body_stream)

    service, client = await service_with_handler(body_timeout, budget=0.2)
    body_task = asyncio.create_task(service.get_availability(make_query()))
    try:
        await asyncio.wait_for(body_stream.entered.wait(), 1.0)
        body_result = await asyncio.wait_for(body_task, 1.0)
    finally:
        body_stream.release.set()
        await asyncio.gather(body_task, return_exceptions=True)
        await client.aclose()
    assert_failure(body_result, AvailabilityStatus.UNAVAILABLE,
                   AvailabilityFailureCategory.TIMEOUT)
    assert body_stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["token", "graph"])
async def test_public_timeout_is_bounded_when_response_close_stalls(
    boundary: str,
) -> None:
    stream = CancellationResistantCloseStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        if boundary == "graph" and request.url.host == "login.microsoftonline.com":
            return token_response()
        return httpx.Response(200, stream=stream)

    service, client = await service_with_handler(
        handler,
        budget=0.05,
        cleanup_grace=0.05,
    )
    task = asyncio.create_task(service.get_availability(make_query()))
    returned = False
    try:
        await asyncio.wait_for(stream.closing.wait(), 1.0)
        done, _ = await asyncio.wait({task}, timeout=0.4)
        returned = bool(done)
        if returned:
            result = task.result()
            assert_failure(result, AvailabilityStatus.UNAVAILABLE,
                           AvailabilityFailureCategory.TIMEOUT)
            assert len(service._availability_cleanup_tasks) == 1
            assert not stream.closed
    finally:
        stream.release.set()
        await asyncio.wait_for(task, 1.0)
        await wait_for_availability_cleanup(service)
        await client.aclose()
    assert returned
    assert stream.closed


@pytest.mark.asyncio
async def test_repeated_owner_cancellation_propagates_and_cleanup_stays_owned() -> None:
    stream = CancellationResistantCloseStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "login.microsoftonline.com":
            return token_response()
        return httpx.Response(200, stream=stream)

    service, client = await service_with_handler(
        handler,
        budget=5.0,
        cleanup_grace=0.2,
    )
    task = asyncio.create_task(service.get_availability(make_query()))
    try:
        await asyncio.wait_for(stream.body_entered.wait(), 1.0)
        task.cancel()
        await asyncio.wait_for(stream.closing.wait(), 1.0)
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=0.5)
        assert done == {task}
        with pytest.raises(asyncio.CancelledError):
            task.result()
        assert len(service._availability_cleanup_tasks) == 1
    finally:
        stream.release.set()
        await wait_for_availability_cleanup(service)
        await client.aclose()
    assert stream.closed


@pytest.mark.asyncio
async def test_late_cleanup_exception_is_consumed_and_releases_admission() -> None:
    stream = CancellationResistantCloseStream(close_error=True)
    loop = asyncio.get_running_loop()
    observed: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: observed.append(context))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    service, client = await service_with_handler(
        handler,
        budget=0.05,
        cleanup_grace=0.02,
    )
    task = asyncio.create_task(service.get_availability(make_query()))
    returned = False
    try:
        await asyncio.wait_for(stream.closing.wait(), 1.0)
        done, _ = await asyncio.wait({task}, timeout=0.4)
        returned = bool(done)
        if returned:
            assert_failure(task.result(), AvailabilityStatus.UNAVAILABLE,
                           AvailabilityFailureCategory.TIMEOUT)
            assert len(service._availability_cleanup_tasks) == 1
            stream.release.set()
            await wait_for_availability_cleanup(service)
            await asyncio.sleep(0)
            assert observed == []
    finally:
        stream.release.set()
        await asyncio.gather(task, return_exceptions=True)
        await wait_for_availability_cleanup(service)
        loop.set_exception_handler(previous_handler)
        await client.aclose()
    assert returned


@pytest.mark.asyncio
async def test_hostile_cleanup_retention_is_bounded_and_blocks_only_new_reads() -> None:
    streams: list[CancellationResistantCloseStream] = []
    request_hosts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        request_hosts.append(request.url.host)
        stream = CancellationResistantCloseStream()
        streams.append(stream)
        return httpx.Response(200, stream=stream)

    service, client = await service_with_handler(
        handler,
        budget=0.05,
        cleanup_grace=0.02,
    )
    tasks = [
        asyncio.create_task(service.get_availability(
            make_query(request_id=f"bounded-{index}")
        ))
        for index in range(service.AVAILABILITY_CLEANUP_CAPACITY + 1)
    ]
    try:
        done, pending = await asyncio.wait(set(tasks), timeout=0.8)
        assert not pending
        assert len(done) == len(tasks)
        assert len(streams) == service.AVAILABILITY_CLEANUP_CAPACITY
        assert len(service._availability_cleanup_tasks) == len(streams)
        assert request_hosts == ["login.microsoftonline.com"] * len(streams)
        for completed in done:
            assert_failure(completed.result(), AvailabilityStatus.UNAVAILABLE,
                           AvailabilityFailureCategory.TIMEOUT)
    finally:
        for stream in streams:
            stream.release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await wait_for_availability_cleanup(service)
        await client.aclose()


@pytest.mark.asyncio
async def test_service_close_does_not_multiply_or_wait_forever_on_cleanup() -> None:
    stream = CancellationResistantCloseStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    service, client = await service_with_handler(
        handler,
        budget=0.05,
        cleanup_grace=0.02,
    )
    request_task = asyncio.create_task(service.get_availability(make_query()))
    try:
        await asyncio.wait_for(stream.closing.wait(), 1.0)
        result = await asyncio.wait_for(request_task, 1.0)
        assert_failure(result, AvailabilityStatus.UNAVAILABLE,
                       AvailabilityFailureCategory.TIMEOUT)
        retained = tuple(service._availability_cleanup_tasks)
        assert len(retained) == 1

        await asyncio.wait_for(service.close(), 0.5)
        await asyncio.wait_for(service.close(), 0.5)
        assert tuple(service._availability_cleanup_tasks) == retained
        assert service._http_client is None
    finally:
        stream.release.set()
        await asyncio.gather(request_task, return_exceptions=True)
        await wait_for_availability_cleanup(service)
        if service._http_client is not None:
            await client.aclose()


@pytest.mark.asyncio
async def test_stalled_availability_cleanup_does_not_block_legacy_lookup() -> None:
    stream = CancellationResistantCloseStream()
    token_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_calls
        if request.url.host == "login.microsoftonline.com":
            token_calls += 1
            if token_calls == 1:
                return httpx.Response(200, stream=stream)
            return token_response()
        return httpx.Response(200, json={"value": []})

    service, client = await service_with_handler(
        handler,
        budget=0.05,
        cleanup_grace=0.02,
    )
    availability_task = asyncio.create_task(
        service.get_availability(make_query())
    )
    returned = False
    try:
        await asyncio.wait_for(stream.closing.wait(), 1.0)
        done, _ = await asyncio.wait({availability_task}, timeout=0.4)
        returned = bool(done)
        if returned:
            assert_failure(availability_task.result(), AvailabilityStatus.UNAVAILABLE,
                           AvailabilityFailureCategory.TIMEOUT)
            lookup = await asyncio.wait_for(
                service.get_customer_appointments("6135550100"),
                1.0,
            )
            assert lookup == []
            assert token_calls == 2
    finally:
        stream.release.set()
        await asyncio.gather(availability_task, return_exceptions=True)
        await wait_for_availability_cleanup(service)
        await client.aclose()
    assert returned


@pytest.mark.asyncio
async def test_cancellation_propagates_and_closes_streamed_response() -> None:
    stream = BlockingStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "login.microsoftonline.com":
            return token_response()
        return httpx.Response(200, stream=stream)

    service, client = await service_with_handler(handler, budget=10)
    task = asyncio.create_task(service.get_availability(make_query()))
    await stream.entered.wait()
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await client.aclose()
    assert stream.closed


@pytest.mark.asyncio
async def test_timeout_and_transport_exception_text_never_escape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = "PRIVATE-TRANSPORT-CANARY-918"

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "login.microsoftonline.com":
            return token_response()
        raise httpx.ConnectError(canary, request=request)

    service, client = await service_with_handler(handler)
    try:
        result = await service.get_availability(make_query(request_id=canary))
    finally:
        await client.aclose()
    assert_failure(result, AvailabilityStatus.UNAVAILABLE,
                   AvailabilityFailureCategory.TRANSPORT)
    assert canary not in repr(result)
    assert canary not in str(result.failure)
    assert canary not in caplog.text


@pytest.mark.asyncio
async def test_concurrent_queries_share_client_without_scope_or_window_crossover() -> None:
    seen: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "login.microsoftonline.com":
            return token_response()
        body = json.loads(request.content)
        seen.append(body)
        staff_id = body["staffIds"][0]
        return httpx.Response(200, json=payload([
            staff_entry(staff_id, [item(
                start=date_time(body["startDateTime"]["dateTime"]),
                end=date_time(body["endDateTime"]["dateTime"]),
            )])
        ]))

    service, client = await service_with_handler(handler)
    first = make_query(
        service_id="service-one",
        staff_ids=("staff-one",),
        start=datetime(2026, 6, 18, 14, 0, tzinfo=UTC),
        end=datetime(2026, 6, 18, 15, 0, tzinfo=UTC),
        request_id="request-one",
    )
    second = make_query(
        service_id="service-two",
        staff_ids=("staff-two",),
        start=datetime(2026, 6, 19, 16, 0, tzinfo=UTC),
        end=datetime(2026, 6, 19, 17, 0, tzinfo=UTC),
        request_id="request-two",
    )
    try:
        first_result, second_result = await asyncio.gather(
            service.get_availability(first),
            service.get_availability(second),
        )
    finally:
        await client.aclose()

    assert first_result.query is first
    assert second_result.query is second
    assert first_result.intervals[0].staff_id == "staff-one"
    assert second_result.intervals[0].staff_id == "staff-two"
    assert first_result.intervals[0].interval == first.window
    assert second_result.intervals[0].interval == second.window
    assert {tuple(body["staffIds"]) for body in seen} == {
        ("staff-one",), ("staff-two",)
    }


@pytest.mark.asyncio
async def test_successful_token_is_cached_across_availability_queries() -> None:
    token_calls = 0
    graph_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_calls, graph_calls
        if request.url.host == "login.microsoftonline.com":
            token_calls += 1
            return token_response()
        graph_calls += 1
        body = json.loads(request.content)
        return httpx.Response(200, json=payload([
            staff_entry(body["staffIds"][0], [])
        ]))

    service, client = await service_with_handler(handler)
    try:
        await service.get_availability(make_query(request_id="request-one"))
        await service.get_availability(make_query(request_id="request-two"))
    finally:
        await client.aclose()
    assert token_calls == 1
    assert graph_calls == 2
