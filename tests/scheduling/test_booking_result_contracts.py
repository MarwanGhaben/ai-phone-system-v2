from __future__ import annotations

import builtins
import importlib
import sys
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from services.scheduling.models import (
    AvailabilityContractError,
    AvailabilityFailure,
    AvailabilityFailureCategory,
    AvailabilityQuery,
    AvailabilityResult,
    AvailabilityStatus,
    CalendarScope,
    FreeInterval,
    TimeInterval,
)


UTC = timezone.utc


def scope(
    *,
    tenant_id: str = "tenant-canary",
    business_id: str = "business-canary",
    service_id: str = "service-canary",
) -> CalendarScope:
    return CalendarScope(tenant_id, business_id, service_id)


def window() -> TimeInterval:
    return TimeInterval(
        datetime(2030, 1, 8, 14, 0, tzinfo=UTC),
        datetime(2030, 1, 8, 18, 0, tzinfo=UTC),
    )


def query(
    *,
    calendar_scope: CalendarScope | None = None,
    time_window: TimeInterval | None = None,
    staff_ids: tuple[str, ...] = ("staff-a-canary", "staff-b-canary"),
    request_id: str = "request-canary",
) -> AvailabilityQuery:
    return AvailabilityQuery(
        calendar_scope or scope(),
        time_window or window(),
        staff_ids,
        request_id,
    )


def free_interval(
    *,
    calendar_scope: CalendarScope | None = None,
    staff_id: str = "staff-a-canary",
    start: datetime | None = None,
    end: datetime | None = None,
) -> FreeInterval:
    return FreeInterval(
        calendar_scope or scope(),
        staff_id,
        TimeInterval(
            start or datetime(2030, 1, 8, 15, 0, tzinfo=UTC),
            end or datetime(2030, 1, 8, 15, 30, tzinfo=UTC),
        ),
    )


def observed_at() -> datetime:
    return datetime(2030, 1, 8, 13, 59, tzinfo=UTC)


def failure(category: AvailabilityFailureCategory) -> AvailabilityFailure:
    return AvailabilityFailure(category, 503)


def test_available_empty_and_all_failure_categories_remain_distinct() -> None:
    request = query()
    interval = free_interval()
    results = [
        AvailabilityResult(
            request,
            AvailabilityStatus.AVAILABLE,
            observed_at(),
            (interval,),
        ),
        AvailabilityResult(
            request,
            AvailabilityStatus.NO_AVAILABILITY,
            observed_at(),
        ),
    ]

    for category in AvailabilityFailureCategory:
        status = {
            AvailabilityFailureCategory.INVALID_RESPONSE:
                AvailabilityStatus.INVALID_RESPONSE,
            AvailabilityFailureCategory.INCOMPLETE:
                AvailabilityStatus.INCOMPLETE,
        }.get(category, AvailabilityStatus.UNAVAILABLE)
        results.append(
            AvailabilityResult(
                request,
                status,
                observed_at(),
                failure=failure(category),
            )
        )

    assert len(results) == 2 + len(AvailabilityFailureCategory)
    assert results[0].intervals == (interval,)
    assert results[1].intervals == ()
    assert results[1].failure is None
    assert {
        result.failure.category
        for result in results
        if result.failure is not None
    } == set(AvailabilityFailureCategory)


@pytest.mark.parametrize("status", list(AvailabilityStatus))
@pytest.mark.parametrize(
    "failure_category",
    [None, *list(AvailabilityFailureCategory)],
)
@pytest.mark.parametrize("has_intervals", [False, True])
def test_only_valid_status_interval_failure_combinations_are_accepted(
    status: AvailabilityStatus,
    failure_category: AvailabilityFailureCategory | None,
    has_intervals: bool,
) -> None:
    supplied_failure = failure(failure_category) if failure_category else None
    intervals = (free_interval(),) if has_intervals else ()
    expected_valid = (
        status is AvailabilityStatus.AVAILABLE
        and has_intervals
        and supplied_failure is None
    ) or (
        status is AvailabilityStatus.NO_AVAILABILITY
        and not has_intervals
        and supplied_failure is None
    ) or (
        status is AvailabilityStatus.UNAVAILABLE
        and not has_intervals
        and supplied_failure is not None
        and failure_category not in {
            AvailabilityFailureCategory.INVALID_RESPONSE,
            AvailabilityFailureCategory.INCOMPLETE,
        }
    ) or (
        status is AvailabilityStatus.INVALID_RESPONSE
        and not has_intervals
        and failure_category is AvailabilityFailureCategory.INVALID_RESPONSE
    ) or (
        status is AvailabilityStatus.INCOMPLETE
        and not has_intervals
        and failure_category is AvailabilityFailureCategory.INCOMPLETE
    )

    if expected_valid:
        result = AvailabilityResult(
            query(), status, observed_at(), intervals, supplied_failure
        )
        assert result.status is status
    else:
        with pytest.raises(AvailabilityContractError):
            AvailabilityResult(
                query(), status, observed_at(), intervals, supplied_failure
            )


@pytest.mark.parametrize(
    ("factory", "arguments"),
    [
        (CalendarScope, (None, "business", "service")),
        (CalendarScope, ("", "business", "service")),
        (CalendarScope, (" tenant", "business", "service")),
        (CalendarScope, ("tenant ", "business", "service")),
        (CalendarScope, (True, "business", "service")),
        (AvailabilityFailure, ("timeout", None)),
        (AvailabilityFailure, (None, None)),
        (AvailabilityFailure, (AvailabilityFailureCategory.TIMEOUT, True)),
        (AvailabilityFailure, (AvailabilityFailureCategory.TIMEOUT, 99)),
        (AvailabilityFailure, (AvailabilityFailureCategory.TIMEOUT, 600)),
    ],
)
def test_none_empty_boolean_raw_enum_and_invalid_http_values_are_rejected(
    factory: object,
    arguments: tuple[object, ...],
) -> None:
    with pytest.raises(AvailabilityContractError):
        factory(*arguments)


@pytest.mark.parametrize(
    "invalid_status",
    [None, "AVAILABLE", "available", True, 1, object()],
)
def test_invalid_status_values_cannot_become_success(invalid_status: object) -> None:
    with pytest.raises(AvailabilityContractError):
        AvailabilityResult(
            query(),
            invalid_status,
            observed_at(),
            (free_interval(),),
        )


def test_aware_offsets_normalize_to_the_same_utc_instants() -> None:
    eastern = timezone(timedelta(hours=-5))
    first = TimeInterval(
        datetime(2030, 1, 8, 10, 0, tzinfo=eastern),
        datetime(2030, 1, 8, 10, 30, tzinfo=eastern),
    )
    second = TimeInterval(
        datetime(2030, 1, 8, 15, 0, tzinfo=UTC),
        datetime(2030, 1, 8, 15, 30, tzinfo=UTC),
    )

    assert first == second
    assert first.start.tzinfo is UTC
    assert first.end.tzinfo is UTC


def test_ordering_uses_absolute_instants_including_an_explicit_dst_fold() -> None:
    first_fold = datetime(
        2030, 11, 3, 1, 30,
        tzinfo=timezone(timedelta(hours=-4)),
        fold=0,
    )
    second_fold = datetime(
        2030, 11, 3, 1, 30,
        tzinfo=timezone(timedelta(hours=-5)),
        fold=1,
    )

    interval = TimeInterval(first_fold, second_fold)

    assert interval.end - interval.start == timedelta(hours=1)
    with pytest.raises(AvailabilityContractError):
        TimeInterval(second_fold, first_fold)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (
            datetime(2030, 1, 8, 15, 0),
            datetime(2030, 1, 8, 15, 30, tzinfo=UTC),
        ),
        (
            datetime(2030, 1, 8, 15, 0, tzinfo=UTC),
            datetime(2030, 1, 8, 15, 30),
        ),
        ("2030-01-08T15:00:00Z", datetime(2030, 1, 8, 15, 30, tzinfo=UTC)),
        (
            datetime(2030, 1, 8, 15, 0, tzinfo=UTC),
            datetime(2030, 1, 8, 15, 0, tzinfo=UTC),
        ),
        (
            datetime(2030, 1, 8, 15, 30, tzinfo=UTC),
            datetime(2030, 1, 8, 15, 0, tzinfo=UTC),
        ),
    ],
)
def test_invalid_equal_reversed_or_naive_intervals_fail(
    start: object,
    end: object,
) -> None:
    with pytest.raises(AvailabilityContractError):
        TimeInterval(start, end)


def test_half_open_point_and_interval_containment_has_exact_boundaries() -> None:
    outer = window()
    exact = TimeInterval(outer.start, outer.end)
    inside = TimeInterval(
        outer.start + timedelta(minutes=1),
        outer.end - timedelta(minutes=1),
    )

    assert outer.contains(outer.start)
    assert not outer.contains(outer.end)
    assert outer.contains_interval(exact)
    assert outer.contains_interval(inside)
    assert not outer.contains(outer.start - timedelta(microseconds=1))
    assert not outer.contains(outer.end + timedelta(microseconds=1))


@pytest.mark.parametrize(
    "outside_interval",
    [
        TimeInterval(
            datetime(2030, 1, 8, 13, 59, 59, tzinfo=UTC),
            datetime(2030, 1, 8, 15, 0, tzinfo=UTC),
        ),
        TimeInterval(
            datetime(2030, 1, 8, 17, 0, tzinfo=UTC),
            datetime(2030, 1, 8, 18, 0, 0, 1, tzinfo=UTC),
        ),
    ],
)
def test_result_rejects_an_interval_outside_either_query_bound(
    outside_interval: TimeInterval,
) -> None:
    request = query()
    item = FreeInterval(request.scope, request.staff_ids[0], outside_interval)

    with pytest.raises(AvailabilityContractError):
        AvailabilityResult(
            request,
            AvailabilityStatus.AVAILABLE,
            observed_at(),
            (item,),
        )


def test_result_accepts_an_interval_at_exact_query_bounds() -> None:
    request = query()
    item = FreeInterval(request.scope, request.staff_ids[0], request.window)

    result = AvailabilityResult(
        request,
        AvailabilityStatus.AVAILABLE,
        observed_at(),
        (item,),
    )

    assert result.intervals == (item,)


@pytest.mark.parametrize(
    "staff_ids",
    [(), ("",), (" staff-a",), ("staff-a", "staff-a"), ["staff-a"]],
)
def test_query_rejects_empty_invalid_duplicate_or_mutable_staff_collections(
    staff_ids: object,
) -> None:
    with pytest.raises(AvailabilityContractError):
        AvailabilityQuery(scope(), window(), staff_ids, "request")


@pytest.mark.parametrize(
    "request_id",
    [None, "", " request", "request ", True],
)
def test_query_rejects_invalid_request_ids(request_id: object) -> None:
    with pytest.raises(AvailabilityContractError):
        AvailabilityQuery(scope(), window(), ("staff-a",), request_id)


@pytest.mark.parametrize(
    "wrong_item",
    [
        free_interval(calendar_scope=scope(tenant_id="wrong-tenant")),
        free_interval(calendar_scope=scope(business_id="wrong-business")),
        free_interval(calendar_scope=scope(service_id="wrong-service")),
        free_interval(staff_id="wrong-staff"),
    ],
)
def test_result_rejects_wrong_scope_or_staff(wrong_item: FreeInterval) -> None:
    with pytest.raises(AvailabilityContractError):
        AvailabilityResult(
            query(),
            AvailabilityStatus.AVAILABLE,
            observed_at(),
            (wrong_item,),
        )


def test_result_rejects_duplicate_intervals_but_preserves_staff_distinctions() -> None:
    request = query()
    staff_a = free_interval(staff_id=request.staff_ids[0])
    staff_b = free_interval(staff_id=request.staff_ids[1])

    with pytest.raises(AvailabilityContractError):
        AvailabilityResult(
            request,
            AvailabilityStatus.AVAILABLE,
            observed_at(),
            (staff_a, staff_a),
        )

    result = AvailabilityResult(
        request,
        AvailabilityStatus.AVAILABLE,
        observed_at(),
        (staff_a, staff_b),
    )
    assert result.intervals == (staff_a, staff_b)


def test_overlapping_distinct_intervals_are_contract_valid() -> None:
    request = query(staff_ids=("staff-a-canary",))
    first = free_interval(
        start=datetime(2030, 1, 8, 15, 0, tzinfo=UTC),
        end=datetime(2030, 1, 8, 16, 0, tzinfo=UTC),
    )
    second = free_interval(
        start=datetime(2030, 1, 8, 15, 30, tzinfo=UTC),
        end=datetime(2030, 1, 8, 16, 30, tzinfo=UTC),
    )

    result = AvailabilityResult(
        request,
        AvailabilityStatus.AVAILABLE,
        observed_at(),
        (first, second),
    )

    assert result.intervals == (first, second)


def test_objects_and_nested_containers_are_immutable() -> None:
    requested_staff = ["staff-a-canary"]
    with pytest.raises(AvailabilityContractError):
        AvailabilityQuery(scope(), window(), requested_staff, "request-canary")

    request = query(staff_ids=("staff-a-canary",))
    supplied_intervals = [free_interval()]
    with pytest.raises(AvailabilityContractError):
        AvailabilityResult(
            request,
            AvailabilityStatus.AVAILABLE,
            observed_at(),
            supplied_intervals,
        )

    result = AvailabilityResult(
        request,
        AvailabilityStatus.AVAILABLE,
        observed_at(),
        tuple(supplied_intervals),
    )
    supplied_intervals.clear()
    requested_staff.append("staff-b-canary")

    assert len(result.intervals) == 1
    assert result.query.staff_ids == ("staff-a-canary",)
    with pytest.raises(FrozenInstanceError):
        result.status = AvailabilityStatus.NO_AVAILABILITY
    with pytest.raises(TypeError):
        result.intervals[0] = free_interval()


def test_observation_time_is_aware_and_normalized_to_utc() -> None:
    offset_time = datetime(
        2030,
        1,
        8,
        8,
        59,
        tzinfo=timezone(timedelta(hours=-5)),
    )
    result = AvailabilityResult(
        query(), AvailabilityStatus.NO_AVAILABILITY, offset_time
    )

    assert result.observed_at == observed_at()
    assert result.observed_at.tzinfo is UTC
    with pytest.raises(AvailabilityContractError):
        AvailabilityResult(
            query(),
            AvailabilityStatus.NO_AVAILABILITY,
            datetime(2030, 1, 8, 13, 59),
        )


def test_availability_result_has_no_truth_value() -> None:
    result = AvailabilityResult(
        query(), AvailabilityStatus.NO_AVAILABILITY, observed_at()
    )

    with pytest.raises(
        TypeError,
        match="AvailabilityResult has no truth value; check status explicitly",
    ):
        bool(result)


def test_repr_str_and_validation_errors_do_not_reveal_identifiers() -> None:
    canaries = (
        "TENANT-SENSITIVE-CANARY",
        "BUSINESS-SENSITIVE-CANARY",
        "SERVICE-SENSITIVE-CANARY",
        "STAFF-SENSITIVE-CANARY",
        "REQUEST-SENSITIVE-CANARY",
    )
    sensitive_scope = CalendarScope(*canaries[:3])
    request = AvailabilityQuery(
        sensitive_scope,
        window(),
        (canaries[3],),
        canaries[4],
    )
    item = FreeInterval(sensitive_scope, canaries[3], window())
    result = AvailabilityResult(
        request,
        AvailabilityStatus.AVAILABLE,
        observed_at(),
        (item,),
    )
    objects = (
        sensitive_scope,
        request.window,
        request,
        item,
        AvailabilityFailure(AvailabilityFailureCategory.TIMEOUT, 504),
        result,
    )

    for value in objects:
        rendered = repr(value) + str(value)
        assert all(canary not in rendered for canary in canaries)

    wrong_scope_item = FreeInterval(
        CalendarScope("WRONG-TENANT-CANARY", *canaries[1:3]),
        canaries[3],
        window(),
    )
    with pytest.raises(AvailabilityContractError) as captured:
        AvailabilityResult(
            request,
            AvailabilityStatus.AVAILABLE,
            observed_at(),
            (wrong_scope_item,),
        )
    rendered_error = repr(captured.value) + str(captured.value)
    assert all(canary not in rendered_error for canary in canaries)
    assert "WRONG-TENANT-CANARY" not in rendered_error


def test_import_is_pure_and_does_not_load_settings_or_provider_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "services.scheduling.models"
    blocked_roots = (
        "config.settings",
        "httpx",
        "asyncpg",
        "services.calendar.ms_bookings_service",
    )
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if any(name == root or name.startswith(root + ".") for root in blocked_roots):
            raise AssertionError("pure contract import crossed an I/O boundary")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    existing = sys.modules.pop(module_name, None)
    try:
        imported = importlib.import_module(module_name)
        assert imported.AvailabilityStatus.AVAILABLE.name == "AVAILABLE"
    finally:
        sys.modules.pop(module_name, None)
        if existing is not None:
            sys.modules[module_name] = existing
