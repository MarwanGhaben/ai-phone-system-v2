from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from datetime import date, datetime, time, timedelta, timezone
import hashlib
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from services.scheduling.models import (
    AvailabilityFailure,
    AvailabilityFailureCategory,
    AvailabilityQuery,
    AvailabilityResult,
    AvailabilityStatus,
    CalendarScope,
    FreeInterval,
    TimeInterval,
)
from services.scheduling.policy import (
    AppointmentCandidate,
    AppointmentFormat,
    ClosureCalendar,
    PolicyEvaluation,
    PolicyFailureReason,
    PolicyResultStatus,
    SchedulingPolicy,
    SchedulingPolicyContractError,
    evaluate_scheduling_policy,
)


UTC = timezone.utc
TORONTO = ZoneInfo("America/Toronto")
ROOT = Path(__file__).resolve().parents[2]
T007_HASHES = {
    "services/scheduling/models.py":
        "576d6627208d2a621edf2a41c01493da04284d6a94f16174347a48002f1f931d",
    "tests/scheduling/test_booking_result_contracts.py":
        "ceaefbc1b4b8372b6aab614800a1acb4f360ab9dca687be2b5c48f06f4e260ad",
    "docs/delegation/T007-A-result.md":
        "f6ff3d65f12d40a559b47f7313989d4409b63887bd93c3b4a5a4bd8ba9020e3b",
}
T013_NORMALIZED_HASHES = {
    "services/calendar/contracts.py":
        "79ab6a4ecaeb42b81160c8b873a6025684bbb235a857d358dcaa91c85dd74aee",
    "tests/calendar/test_graph_availability_contracts.py":
        "02061cb6c9d0fcb8bfe0ae6ff54e4c04c4dc9e9dd4d0aef1fbefa5d55f1d9ee6",
    "docs/delegation/T013-A-result.md":
        "adc3ff91b3810db7b4eeed6a09d6d7632ddcdc234456c6a58b80fa1b79cdc082",
}

# Pre-edit T013-A AST spans and protected constants, verified against its
# accepted whole-file normalized hash before the additive T014-C method.
T013_ADAPTER_SPANS = {
    '_TOKEN_RESPONSE_LIMIT': '4d7a500fffd55159ac087f95539acd101942197d3b878b55523ae00905686de0',
    '_AVAILABILITY_RESPONSE_LIMIT': '784c8dc6ed0d6a95bfb1edad4f08f0b9a7e6d277f88b3a488e2e8c456be64ebc',
    '_CONTRACT_ERROR': '817bc40b5c6075ef2d1dc777550cee4d1427eff130d0d859ea49a0331d959844',
    '_AvailabilityReadFailure.__init__': 'b6c928e03932315d8d32c2d71e7eaf10e33393e27694bad1f8d32c0dbd93833f',
    '_AvailabilityPermit.__init__': 'e46f12cb85c90eb3bfa369f6352eb68b4344ee4a88cfe60d5a0499d190ebdc74',
    '_reject_json_constant': '5bbb7f686b49bd37c3bcbc5d63a540d06fb6ec17c20eee356199aed809ae22c8',
    '_availability_http_failure': 'df356ee4f344ca71f0310468eb83cdb6c417df2a99f169415fb7dc3f4b337203',
    '_availability_token_http_failure': '395dd11263e2c9522189886af0f4d420171e0967e3c40cb29ea3bab75808edf7',
    'normalize_customer_phone': 'd2206e83b3a5b58e968e7be549082619e235d33bf98376ce0ca01e30db5997d8',
    'appointment_customer_phone': '95f9d01d654e5f9d021c1186caed0d8408de8f5315a1d007e10474b08d859892',
    'MSBookingsService.AVAILABILITY_TIMEOUT_SECONDS': '6cf56df95ef0480084be81a5075f14f6b96886f18aab3e23fae3e899a20cf7f7',
    'MSBookingsService.AVAILABILITY_CLEANUP_GRACE_SECONDS': 'edf1de8bf3936db04872369149ac24ead412b9a9efaa8a2efe8d6810a887d590',
    'MSBookingsService.AVAILABILITY_CLEANUP_CAPACITY': '442a5f5215e6fdfeaa9f8125bc64fd11ed871ba381d54c2146bf2a4d5808f85b',
    'MSBookingsService.__init__': '634dba4f6c6c8ed4852cb0e1dd4fa769a3e04da08f4e16dab19d783de4499705',
    'MSBookingsService.is_available': 'c3ba14f643287c5e11539f6479dee345e038c8fe9b8e8720ad876bae455a80f4',
    'MSBookingsService._get_client': '1104c942f712a196d5a4ab23ab6cd3383b1ae957f20c67e648d0778f1d0a80fb',
    'MSBookingsService._read_availability_body': 'f1471d8d0857976cd395817c3f4e8b7dba8f5840ead06a60279d5a13272fc12f',
    'MSBookingsService._read_streamed_availability_response': 'da1a07e5e9bc36e714dcae907d2367eb1304d52a00f26bed2be966aa3f1b1e9b',
    'MSBookingsService._availability_cleanup_finished': '65ad816fded7a2f500836a018cf7038c443d740f6c2bb2d402a462bb23bf457a',
    'MSBookingsService._retain_availability_cleanup': 'dae596d86e76a60cbf2064758252747f2a415e73b00d543b61bc952b54146411',
    'MSBookingsService._run_availability_io': '67448d8cf367dc36c9d8879a4d10dbecdbb933c6345a3d69cf719bd9d8267624',
    'MSBookingsService._get_availability_access_token': '9736fdad17c5205e28897b8ca0a5a468874085530c6e14da9553f8332da3b2cd',
    'MSBookingsService._availability_failure_result': 'a2e9bbd280e0c8b6a78b3473f0beee122733c0ed03ecfa7182b34882b615050c',
    'MSBookingsService.get_availability': '0acc1e0ba2ad5b90b11496797337db6019edda620c6ab1230793697bf11775c3',
    'MSBookingsService._availability_wire_datetime': '40a7b7cc9318ab7b914f9b7d766924dba17521e66ce3d190c9e1b1397ed68d78',
    'MSBookingsService._get_access_token': '952c1d1b5dab174da261c28bac75ac38e09810b9feb2693e8bee0dea05b72be9',
    'MSBookingsService._make_request': '8549895b064bd2b8c399221e935bf13ae35231ddda983e001608b8eb979867be',
    'MSBookingsService.get_staff_members': '37bf7ba83f970cfdf0c5a3de13549baac51cbd7dd43878f407b25715dc97332f',
    'MSBookingsService.get_staff_by_name': 'dda46e28901dcbcef39332e306405d77f7650e31d1c03cfc68f62b56f59545ee',
    'MSBookingsService.get_services': '47de32850353bb3d0cdd2e03d71e3d16c9e39c55ad97b7d76bdb5716f72eb0aa',
    'MSBookingsService.get_available_slots': 'c3e31549249b513e361a02c51bd58009b7d450210389a3c830016a2107150984',
    'MSBookingsService.create_booking': 'ddd085b3702eec6ee5497730008b23d19900e8b25a9553d00cbda15369c00450',
    'MSBookingsService.get_customer_appointments': '07f0ceae072ee6410fda38a44eb473086074673f5351be405754206e7f96760d',
    'MSBookingsService.cancel_customer_appointment': 'da8a46d583201576ced965684cb38e5b27a6ffcd1939dd215f6d96712aaff87f',
    'MSBookingsService.close': '1341351dd2f9e00e660dfc9171667170fc0b2aa3c7de6f35deb45b2581d7e40a',
    'get_calendar_service': '7d6f00244cbcd4243458716622a62af1b7e17a271adcbd30f3f3bb4adb5ee5ef',
    'create_calendar_service': 'bac32e698fc38000171a4c6e86300790044a10cb61a2a611b3de6fc424527716',
}



def scope(
    tenant_id: str = "tenant-canary",
    business_id: str = "business-canary",
    service_id: str = "service-canary",
) -> CalendarScope:
    return CalendarScope(tenant_id, business_id, service_id)


def local(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), TORONTO).astimezone(UTC)


def closure(
    start: date = date(2026, 1, 1),
    end: date = date(2027, 12, 31),
    closed: frozenset[date] = frozenset(),
) -> ClosureCalendar:
    return ClosureCalendar(start, end, closed)


def policy(**overrides: object) -> SchedulingPolicy:
    values: dict[str, object] = {
        "policy_version": "policy-v1-canary",
        "scope": scope(),
        "approved_staff_ids": ("staff-a", "staff-b"),
        "service_duration": timedelta(minutes=30),
        "slot_interval": timedelta(minutes=30),
        "pre_buffer": timedelta(0),
        "post_buffer": timedelta(0),
        "business_timezone": "America/Toronto",
        "open_weekdays": (0, 1, 2, 3, 4),
        "opens_at": time(10, 0),
        "closes_at": time(17, 0),
        "minimum_notice": timedelta(minutes=30),
        "business_day_horizon": 2,
        "closure_calendar": closure(),
        "appointment_format": AppointmentFormat.IN_PERSON,
        "maximum_attendees": 1,
        "seasonal_overrides": (),
    }
    values.update(overrides)
    return SchedulingPolicy(**values)


def result(
    *,
    status: AvailabilityStatus = AvailabilityStatus.AVAILABLE,
    intervals: tuple[tuple[str, datetime, datetime], ...] = (),
    staff_ids: tuple[str, ...] = ("staff-a",),
    query_start: datetime | None = None,
    query_end: datetime | None = None,
    observed_at: datetime | None = None,
    query_scope: CalendarScope | None = None,
) -> AvailabilityResult:
    selected_scope = query_scope or scope()
    starts = [entry[1] for entry in intervals]
    ends = [entry[2] for entry in intervals]
    start = query_start or (min(starts) if starts else local(date(2026, 6, 18), 10))
    end = query_end or (max(ends) if ends else local(date(2026, 6, 18), 17))
    query = AvailabilityQuery(
        selected_scope,
        TimeInterval(start, end),
        staff_ids,
        "request-canary",
    )
    free = tuple(
        FreeInterval(selected_scope, staff_id, TimeInterval(item_start, item_end))
        for staff_id, item_start, item_end in intervals
    )
    failure = None
    if status is AvailabilityStatus.UNAVAILABLE:
        failure = AvailabilityFailure(AvailabilityFailureCategory.PROVIDER_ERROR)
    elif status is AvailabilityStatus.INVALID_RESPONSE:
        failure = AvailabilityFailure(AvailabilityFailureCategory.INVALID_RESPONSE)
    elif status is AvailabilityStatus.INCOMPLETE:
        failure = AvailabilityFailure(AvailabilityFailureCategory.INCOMPLETE)
    return AvailabilityResult(
        query,
        status,
        observed_at or datetime(2026, 6, 18, 13, 0, tzinfo=UTC),
        free,
        failure,
    )


def evaluate(
    source: AvailabilityResult,
    *,
    snapshot: SchedulingPolicy | None = None,
    now: datetime = datetime(2026, 6, 18, 13, 0, tzinfo=UTC),
    maximum_age: timedelta = timedelta(minutes=5),
) -> PolicyEvaluation:
    return evaluate_scheduling_policy(
        snapshot or policy(),
        source,
        now,
        maximum_age,
    )


def candidate_starts(evaluation: PolicyEvaluation) -> list[datetime]:
    return [item.interval.start for item in evaluation.candidates]


def normalized_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def adapter_spans(path: Path) -> dict[str, str]:
    source = path.read_bytes().replace(b"\r\n", b"\n").decode()
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    def digest(node: ast.AST) -> str:
        return hashlib.sha256(
            "".join(lines[node.lineno - 1:node.end_lineno]).encode()
        ).hexdigest()

    observed: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            observed[node.name] = digest(node)
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == "MSBookingsService" and item.name == "get_service_facts":
                        continue
                    observed[node.name + "." + item.name] = digest(item)
                if isinstance(item, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id.startswith("AVAILABILITY_")
                    for target in item.targets
                ):
                    observed[node.name + "." + item.targets[0].id] = digest(item)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in {"_TOKEN_RESPONSE_LIMIT", "_AVAILABILITY_RESPONSE_LIMIT",
                              "_CONTRACT_ERROR"}
            for target in node.targets
        ):
            observed[node.targets[0].id] = digest(node)
    return observed



def test_accepted_t007_and_t013_files_remain_frozen() -> None:
    assert {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in T007_HASHES
    } == T007_HASHES
    assert {
        name: normalized_hash(ROOT / name)
        for name in T013_NORMALIZED_HASHES
    } == T013_NORMALIZED_HASHES
    assert adapter_spans(ROOT / "services/calendar/ms_bookings_service.py") == T013_ADAPTER_SPANS


def test_exact_notice_boundary_and_past_slots() -> None:
    day = date(2026, 6, 18)
    source = result(
        intervals=(("staff-a", local(day, 10), local(day, 12)),),
        observed_at=local(day, 10),
    )

    exact = evaluate(source, now=local(day, 10))
    short = evaluate(source, now=local(day, 10, 0) + timedelta(seconds=1))

    assert candidate_starts(exact) == [
        local(day, 10, 30), local(day, 11), local(day, 11, 30)
    ]
    assert candidate_starts(short) == [local(day, 11), local(day, 11, 30)]
    assert all(item.interval.start >= local(day, 10, 30) for item in exact.candidates)


def test_utc_date_crossover_uses_the_toronto_local_date() -> None:
    local_today = date(2026, 6, 17)
    next_local_day = local_today + timedelta(days=1)
    now = datetime(2026, 6, 18, 3, 45, tzinfo=UTC)
    source = result(
        intervals=((
            "staff-a",
            local(next_local_day, 10),
            local(next_local_day, 11),
        ),),
        observed_at=now,
    )
    evaluation = evaluate(source, now=now)

    assert candidate_starts(evaluation) == [
        local(next_local_day, 10),
        local(next_local_day, 10, 30),
    ]


def test_open_close_edges_duration_and_grid_alignment() -> None:
    day = date(2026, 6, 18)
    source = result(
        intervals=(("staff-a", local(day, 9), local(day, 18)),),
        observed_at=local(day, 8),
    )
    evaluation = evaluate(source, now=local(day, 8))

    starts = candidate_starts(evaluation)
    assert starts[0] == local(day, 10)
    assert starts[-1] == local(day, 16, 30)
    assert local(day, 17) not in starts
    assert all(item.interval.end - item.interval.start == timedelta(minutes=30)
               for item in evaluation.candidates)


def test_free_time_beginning_at_1010_first_allows_1030() -> None:
    day = date(2026, 6, 18)
    source = result(intervals=(("staff-a", local(day, 10, 10), local(day, 11, 30)),))
    evaluation = evaluate(source, now=local(day, 9))
    assert candidate_starts(evaluation) == [local(day, 10, 30), local(day, 11)]


@pytest.mark.parametrize(
    ("today", "closed", "expected_latest"),
    [
        (date(2026, 6, 19), frozenset(), date(2026, 6, 23)),
        (date(2026, 6, 19), frozenset({date(2026, 6, 22)}), date(2026, 6, 24)),
        (date(2026, 6, 20), frozenset(), date(2026, 6, 23)),
        (date(2026, 6, 18), frozenset({date(2026, 6, 18)}), date(2026, 6, 22)),
        (date(2026, 12, 31), frozenset(), date(2027, 1, 4)),
    ],
)
def test_two_open_business_day_horizon(
    today: date,
    closed: frozenset[date],
    expected_latest: date,
) -> None:
    end = expected_latest + timedelta(days=1)
    source = result(
        intervals=(("staff-a", local(today, 10), local(end, 17)),),
        query_start=local(today, 0),
        query_end=local(end, 23),
        observed_at=local(today, 8),
    )
    snapshot = policy(closure_calendar=closure(
        today - timedelta(days=1), end + timedelta(days=1), closed
    ))
    evaluation = evaluate(source, snapshot=snapshot, now=local(today, 8))

    local_dates = {
        item.interval.start.astimezone(TORONTO).date()
        for item in evaluation.candidates
    }
    assert max(local_dates) == expected_latest
    assert not local_dates.intersection(closed)
    assert all(day.weekday() < 5 for day in local_dates)


@pytest.mark.parametrize(
    ("today", "expected_latest", "expected_latest_utc_hour"),
    [
        (date(2026, 3, 6), date(2026, 3, 10), 20),
        (date(2026, 10, 30), date(2026, 11, 3), 21),
    ],
)
def test_dst_weekends_use_toronto_rules(
    today: date,
    expected_latest: date,
    expected_latest_utc_hour: int,
) -> None:
    query_end_day = expected_latest + timedelta(days=1)
    source = result(
        intervals=(("staff-a", local(today, 10), local(query_end_day, 17)),),
        query_start=local(today, 0),
        query_end=local(query_end_day, 23),
        observed_at=local(today, 8),
    )
    snapshot = policy(closure_calendar=closure(today, query_end_day))
    evaluation = evaluate(source, snapshot=snapshot, now=local(today, 8))
    final = evaluation.candidates[-1].interval.start

    assert final.astimezone(TORONTO).date() == expected_latest
    assert final.astimezone(TORONTO).time() == time(16, 30)
    assert final.hour == expected_latest_utc_hour


@pytest.mark.parametrize(
    "calendar",
    [
        None,
        closure(date(2026, 6, 19), date(2026, 6, 30)),
        closure(date(2026, 1, 1), date(2026, 6, 19)),
        closure(
            date(2026, 6, 18),
            date(2028, 1, 1),
            frozenset(
                date(2026, 6, 19) + timedelta(days=index)
                for index in range(500)
                if (date(2026, 6, 19) + timedelta(days=index)).weekday() < 5
            ),
        ),
    ],
)
def test_missing_narrow_or_defensively_long_closure_coverage_is_unverified(
    calendar: ClosureCalendar | None,
) -> None:
    day = date(2026, 6, 18)
    source = result(intervals=(("staff-a", local(day, 10), local(day, 17)),))
    evaluation = evaluate(
        source,
        snapshot=policy(closure_calendar=calendar),
        now=local(day, 8),
    )
    assert evaluation.status is PolicyResultStatus.POLICY_UNVERIFIED
    assert evaluation.reason is PolicyFailureReason.CLOSURE_COVERAGE
    assert evaluation.candidates == ()


def test_known_empty_closure_calendar_does_not_guess_holidays() -> None:
    today = date(2026, 6, 18)
    source = result(
        intervals=(("staff-a", local(today, 10), local(today, 17)),),
        observed_at=local(today, 8),
    )
    snapshot = policy(closure_calendar=closure(today, date(2026, 6, 30)))
    evaluation = evaluate(source, snapshot=snapshot, now=local(today, 8))
    assert evaluation.status is PolicyResultStatus.CANDIDATES


def test_scope_mismatch_and_invalid_public_inputs_raise_fixed_error() -> None:
    source = result(status=AvailabilityStatus.NO_AVAILABILITY)
    wrong_scope_policy = policy(scope=scope(tenant_id="private-wrong-tenant"))
    calls = (
        lambda: evaluate_scheduling_policy(None, source, datetime.now(UTC),
                                           timedelta(minutes=5)),
        lambda: evaluate_scheduling_policy(policy(), object(), datetime.now(UTC),
                                           timedelta(minutes=5)),
        lambda: evaluate(source, snapshot=wrong_scope_policy),
        lambda: evaluate(source, now=datetime(2026, 6, 18, 9)),
        lambda: evaluate(source, maximum_age=timedelta(0)),
    )
    for call in calls:
        with pytest.raises(
            SchedulingPolicyContractError,
            match="^invalid scheduling policy contract$",
        ):
            call()


@pytest.mark.parametrize(
    "wrong_scope",
    [
        scope(tenant_id="private-wrong-tenant"),
        scope(business_id="private-wrong-business"),
        scope(service_id="private-wrong-service"),
    ],
)
def test_every_scope_component_must_match(wrong_scope: CalendarScope) -> None:
    source = result(
        status=AvailabilityStatus.NO_AVAILABILITY,
        query_scope=wrong_scope,
    )
    with pytest.raises(
        SchedulingPolicyContractError,
        match="^invalid scheduling policy contract$",
    ):
        evaluate(source)


@pytest.mark.parametrize(
    "overrides",
    [
        {"service_duration": timedelta(minutes=45)},
        {"slot_interval": timedelta(minutes=15)},
        {"pre_buffer": timedelta(minutes=5)},
        {"post_buffer": timedelta(minutes=5)},
        {"business_timezone": "UTC"},
        {"open_weekdays": (0, 1, 2, 3, 4, 5)},
        {"opens_at": time(9)},
        {"closes_at": time(22)},
        {"minimum_notice": timedelta(minutes=10)},
        {"business_day_horizon": 3},
        {"appointment_format": AppointmentFormat.ONLINE},
        {"maximum_attendees": 2},
        {"seasonal_overrides": ("private-override",)},
    ],
)
def test_conflicting_service_or_seasonal_settings_are_unverified(
    overrides: dict[str, object],
) -> None:
    source = result(status=AvailabilityStatus.NO_AVAILABILITY)
    evaluation = evaluate(source, snapshot=policy(**overrides))
    assert evaluation.status is PolicyResultStatus.POLICY_UNVERIFIED
    assert evaluation.reason is PolicyFailureReason.UNSUPPORTED_POLICY


def test_unsupported_requested_staff_is_unverified_and_never_dropped() -> None:
    source = result(
        status=AvailabilityStatus.NO_AVAILABILITY,
        staff_ids=("staff-a", "private-unapproved-staff"),
    )
    evaluation = evaluate(source)
    assert evaluation.status is PolicyResultStatus.POLICY_UNVERIFIED
    assert evaluation.reason is PolicyFailureReason.UNSUPPORTED_STAFF
    assert evaluation.candidates == ()


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (
            AvailabilityStatus.UNAVAILABLE,
            PolicyFailureReason.PROVIDER_UNAVAILABLE,
        ),
        (
            AvailabilityStatus.INVALID_RESPONSE,
            PolicyFailureReason.INVALID_RESPONSE,
        ),
        (
            AvailabilityStatus.INCOMPLETE,
            PolicyFailureReason.INCOMPLETE_RESPONSE,
        ),
    ],
)
def test_provider_failure_states_are_availability_unverified(
    status: AvailabilityStatus,
    reason: PolicyFailureReason,
) -> None:
    evaluation = evaluate(result(status=status))
    assert evaluation.status is PolicyResultStatus.AVAILABILITY_UNVERIFIED
    assert evaluation.reason is reason
    assert evaluation.candidates == ()


def test_stale_future_and_exact_age_evidence() -> None:
    day = date(2026, 6, 18)
    now = local(day, 10)
    free = (("staff-a", local(day, 10, 30), local(day, 11)),)
    stale = evaluate(result(intervals=free, observed_at=now - timedelta(minutes=5, microseconds=1)), now=now)
    future = evaluate(result(intervals=free, observed_at=now + timedelta(microseconds=1)), now=now)
    exact = evaluate(result(intervals=free, observed_at=now - timedelta(minutes=5)), now=now)

    assert stale.status is PolicyResultStatus.AVAILABILITY_UNVERIFIED
    assert stale.reason is PolicyFailureReason.STALE_EVIDENCE
    assert future.status is PolicyResultStatus.AVAILABILITY_UNVERIFIED
    assert future.reason is PolicyFailureReason.FUTURE_EVIDENCE
    assert exact.status is PolicyResultStatus.CANDIDATES


def test_complete_provider_empty_is_no_candidates_after_policy_checks() -> None:
    source = result(status=AvailabilityStatus.NO_AVAILABILITY)
    evaluation = evaluate(source)
    assert evaluation.status is PolicyResultStatus.NO_CANDIDATES
    assert evaluation.reason is None
    assert evaluation.source_query is source.query
    assert evaluation.source_observed_at == source.observed_at


def test_missing_timezone_capability_is_policy_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import services.scheduling.policy as policy_module

    monkeypatch.setattr(
        policy_module,
        "ZoneInfo",
        lambda _name: (_ for _ in ()).throw(policy_module.ZoneInfoNotFoundError),
    )
    evaluation = evaluate(result(status=AvailabilityStatus.NO_AVAILABILITY))
    assert evaluation.status is PolicyResultStatus.POLICY_UNVERIFIED
    assert evaluation.reason is PolicyFailureReason.TIMEZONE_UNAVAILABLE


def test_overlap_and_adjacency_merge_deduplicate_but_positive_gap_is_preserved() -> None:
    day = date(2026, 6, 18)
    source = result(intervals=(
        ("staff-a", local(day, 10), local(day, 11)),
        ("staff-a", local(day, 10, 30), local(day, 11, 30)),
        ("staff-a", local(day, 11, 30), local(day, 12)),
        ("staff-a", local(day, 12), local(day, 12, 15)),
        ("staff-a", local(day, 12, 15) + timedelta(microseconds=1), local(day, 12, 30)),
    ))
    evaluation = evaluate(source, now=local(day, 9))
    assert candidate_starts(evaluation) == [
        local(day, 10), local(day, 10, 30), local(day, 11), local(day, 11, 30)
    ]


def test_two_staff_multiple_periods_have_stable_order_and_keep_staff_identity() -> None:
    day = date(2026, 6, 18)
    source = result(
        staff_ids=("staff-b", "staff-a"),
        intervals=(
            ("staff-b", local(day, 11), local(day, 11, 30)),
            ("staff-a", local(day, 10, 30), local(day, 11, 30)),
            ("staff-b", local(day, 10, 30), local(day, 11)),
            ("staff-a", local(day, 15), local(day, 15, 30)),
        ),
    )
    evaluation = evaluate(source, now=local(day, 9))
    assert [
        (item.interval.start, item.staff_id) for item in evaluation.candidates
    ] == [
        (local(day, 10, 30), "staff-a"),
        (local(day, 10, 30), "staff-b"),
        (local(day, 11), "staff-a"),
        (local(day, 11), "staff-b"),
        (local(day, 15), "staff-a"),
    ]


def test_narrow_query_empty_result_remains_explicitly_query_scoped() -> None:
    day = date(2026, 6, 18)
    query_start = local(day, 10, 5)
    query_end = local(day, 10, 25)
    source = result(
        status=AvailabilityStatus.NO_AVAILABILITY,
        query_start=query_start,
        query_end=query_end,
    )
    evaluation = evaluate(source, now=local(day, 9))
    assert evaluation.status is PolicyResultStatus.NO_CANDIDATES
    assert evaluation.source_query.window == TimeInterval(query_start, query_end)


def test_generated_candidate_invariants_and_inputs_unchanged() -> None:
    today = date(2026, 6, 18)
    intervals: list[tuple[str, datetime, datetime]] = []
    for staff_id in ("staff-a", "staff-b"):
        for index in range(24):
            start = local(today, 9) + timedelta(minutes=17 * index)
            end = start + timedelta(minutes=35 + (index % 5) * 11)
            intervals.append((staff_id, start, end))
    source = result(
        staff_ids=("staff-a", "staff-b"),
        intervals=tuple(intervals),
        query_start=local(today, 8),
        query_end=local(today + timedelta(days=5), 18),
    )
    snapshot = policy(closure_calendar=closure(
        today,
        today + timedelta(days=10),
        frozenset({date(2026, 6, 19)}),
    ))
    source_before = repr(source), source.intervals
    policy_before = repr(snapshot), snapshot.closure_calendar
    evaluation = evaluate(source, snapshot=snapshot, now=local(today, 8))

    merged_by_staff: dict[str, list[TimeInterval]] = {"staff-a": [], "staff-b": []}
    for free in source.intervals:
        merged_by_staff[free.staff_id].append(free.interval)
    for item in evaluation.candidates:
        local_start = item.interval.start.astimezone(TORONTO)
        local_end = item.interval.end.astimezone(TORONTO)
        assert item.interval.end - item.interval.start == timedelta(minutes=30)
        assert local_start.minute in {0, 30}
        assert time(10) <= local_start.time() <= time(16, 30)
        assert local_end.time() <= time(17)
        assert local_start.weekday() < 5
        assert local_start.date() != date(2026, 6, 19)
        assert item.staff_id in source.query.staff_ids
        assert item.scope == source.query.scope == snapshot.scope
        assert source.query.window.contains_interval(item.interval)
        assert any(interval.contains_interval(item.interval)
                   for interval in merged_by_staff[item.staff_id])
    assert (repr(source), source.intervals) == source_before
    assert (repr(snapshot), snapshot.closure_calendar) == policy_before


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: ClosureCalendar(date(2026, 1, 2), date(2026, 1, 1), frozenset()),
        lambda: ClosureCalendar(date(2026, 1, 1), date(2026, 1, 2),
                                frozenset({date(2026, 1, 3)})),
        lambda: ClosureCalendar(date(2026, 1, 1), date(2026, 1, 2), set()),
        lambda: ClosureCalendar(datetime(2026, 1, 1), date(2026, 1, 2),
                                frozenset()),
        lambda: policy(approved_staff_ids=["staff-a"]),
        lambda: policy(approved_staff_ids=("staff-a", "staff-a")),
        lambda: policy(maximum_attendees=True),
        lambda: policy(open_weekdays=[0, 1, 2, 3, 4]),
        lambda: policy(seasonal_overrides=[]),
        lambda: policy(seasonal_overrides=(["mutable"],)),
    ],
)
def test_invalid_or_mutable_policy_construction_is_rejected_safely(
    constructor: object,
) -> None:
    with pytest.raises(
        SchedulingPolicyContractError,
        match="^invalid scheduling policy contract$",
    ):
        constructor()


def test_nested_inputs_outputs_are_frozen_and_reprs_are_sanitized() -> None:
    day = date(2026, 6, 18)
    snapshot = policy(policy_version="PRIVATE-POLICY", approved_staff_ids=(
        "PRIVATE-STAFF",
    ))
    source = result(
        staff_ids=("PRIVATE-STAFF",),
        intervals=(("PRIVATE-STAFF", local(day, 10), local(day, 11)),),
        query_scope=snapshot.scope,
    )
    evaluation = evaluate(source, snapshot=snapshot, now=local(day, 9))
    candidate = evaluation.candidates[0]

    frozen_attributes = (
        (snapshot, "policy_version"),
        (snapshot.closure_calendar, "coverage_start"),
        (evaluation, "policy_version"),
        (candidate, "policy_version"),
    )
    for value, attribute in frozen_attributes:
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(value, attribute, "changed")
    rendered = " ".join(map(repr, (
        snapshot, snapshot.closure_calendar, evaluation, candidate
    )))
    assert "PRIVATE-POLICY" not in rendered
    assert "PRIVATE-STAFF" not in rendered
    with pytest.raises(TypeError, match="PolicyEvaluation has no truth value"):
        bool(evaluation)


def test_public_result_constructor_rejects_inconsistent_outcomes() -> None:
    source = result(status=AvailabilityStatus.NO_AVAILABILITY)
    candidate = AppointmentCandidate(
        source.query.scope,
        "staff-a",
        TimeInterval(
            source.query.window.start,
            source.query.window.start + timedelta(minutes=30),
        ),
        "policy-v1",
        source.query,
        source.observed_at,
    )
    with pytest.raises(SchedulingPolicyContractError):
        PolicyEvaluation(
            PolicyResultStatus.NO_CANDIDATES,
            "policy-v1",
            source.query,
            source.observed_at,
            (candidate,),
        )


def test_public_candidate_rejects_nonstandard_duration() -> None:
    day = date(2026, 6, 18)
    source = result(
        intervals=(("staff-a", local(day, 10), local(day, 11)),),
    )
    with pytest.raises(
        SchedulingPolicyContractError,
        match="^invalid scheduling policy contract$",
    ):
        AppointmentCandidate(
            source.query.scope,
            "staff-a",
            TimeInterval(local(day, 10), local(day, 10, 45)),
            "policy-v1",
            source.query,
            source.observed_at,
        )
