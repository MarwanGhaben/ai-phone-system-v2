# T007-A implementation result

Implemented the bounded pure availability-result contract on
`codex/phase-5-booking-reliability` at unchanged HEAD
`792952b2965e85513bf0971a41bfaff7bd3377c4`.

This is a domain boundary only. These contracts are **not wired into live
calendar calls**. `CalendarServiceBase` and `MSBookingsService` are unchanged;
the deployed adapter therefore still returns `[]` for both successful empty
availability and multiple failure cases until the reviewed T010/T013 integration.

## Exact public API

`services.scheduling.models` exports:

```text
AvailabilityContractError(ValueError)

CalendarScope(
    tenant_id: str,
    business_id: str,
    service_id: str,
)

TimeInterval(start: datetime, end: datetime)
TimeInterval.contains(self, instant: datetime) -> bool
TimeInterval.contains_interval(self, interval: TimeInterval) -> bool

AvailabilityQuery(
    scope: CalendarScope,
    window: TimeInterval,
    staff_ids: tuple[str, ...],
    request_id: str,
)

FreeInterval(
    scope: CalendarScope,
    staff_id: str,
    interval: TimeInterval,
)

AvailabilityStatus:
    AVAILABLE = "available"
    NO_AVAILABILITY = "no_availability"
    UNAVAILABLE = "unavailable"
    INVALID_RESPONSE = "invalid_response"
    INCOMPLETE = "incomplete"

AvailabilityFailureCategory:
    NOT_CONFIGURED = "not_configured"
    AUTHENTICATION = "authentication"
    PERMISSION_DENIED = "permission_denied"
    THROTTLED = "throttled"
    TIMEOUT = "timeout"
    TRANSPORT = "transport"
    PROVIDER_ERROR = "provider_error"
    INVALID_RESPONSE = "invalid_response"
    INCOMPLETE = "incomplete"

AvailabilityFailure(
    category: AvailabilityFailureCategory,
    http_status: int | None = None,
)

AvailabilityResult(
    query: AvailabilityQuery,
    status: AvailabilityStatus,
    observed_at: datetime,
    intervals: tuple[FreeInterval, ...] = (),
    failure: AvailabilityFailure | None = None,
)
AvailabilityResult.__bool__(self) -> bool
```

`AvailabilityResult.__bool__` always raises `TypeError` with the fixed message
`AvailabilityResult has no truth value; check status explicitly`.

## Enforced invariants

- All dataclasses are frozen and slotted. Staff and result collections must be
  immutable tuples; mutable lists are rejected rather than copied or retained.
- Tenant, business, service, staff, and request IDs are nonempty, case-preserved,
  opaque strings with no leading or trailing whitespace. No UUID syntax,
  rewriting, trimming, or case folding is applied.
- `TimeInterval` accepts only aware datetimes, normalizes them to UTC, and requires
  `start < end` by absolute instant. Naive values, strings, equal/reversed times,
  and timezone implementations returning no offset are invalid.
- Intervals use half-open `[start, end)` semantics. Point containment includes
  `start` and excludes `end`; interval containment accepts exact query bounds.
- Each query explicitly names a nonempty, unique tuple of staff IDs. There is no
  implicit all-staff lookup or staff selection.
- `FreeInterval` records provider free-time evidence only. It does not imply
  duration, slot splitting, policy eligibility, or permission to book.
- Status and failure-category inputs must be their exact enums. Raw strings,
  `None`, booleans, and unknown values cannot be coerced into success.
- `AVAILABLE` requires at least one interval and no failure.
  `NO_AVAILABILITY` requires no intervals and no failure.
  `UNAVAILABLE` requires no intervals and a failure other than
  `invalid_response`/`incomplete`. `INVALID_RESPONSE` and `INCOMPLETE` require
  their matching failure categories and no intervals.
- Failure results cannot retain partial usable intervals. HTTP status, when
  present, must be an integer from 100 through 599 and cannot be `bool`.
- Every returned interval must match the exact query scope, name requested staff,
  and fit within the query window. Duplicate staff/start/end entries are rejected;
  equal instants for different staff and overlapping distinct intervals remain
  valid evidence for later policy processing.
- Observation time is aware and normalized to UTC without consulting a clock or
  applying a freshness rule.
- Validation raises only the fixed, identifier-safe
  `AvailabilityContractError("invalid availability contract")`. Object
  representations omit tenant, business, service, staff, and request identifiers.
- The module uses only standard-library pure data types. Import performs no
  settings/credential read, network/database access, clock read, async work, or
  provider registration.

Construction of `NO_AVAILABILITY` does not establish that a provider read was
complete. T010/T013 must validate coverage of every requested staff member and
map authoritative Graph responses into these variants.

## Behavioral evidence

Tests were written before the implementation. The host's default Python 3.14
does not contain pytest (`C:\Python314\python.exe: No module named pytest`). The
requested suites were then run with the available bundled Python 3.12.14 and the
already-cached project wheels; no dependencies were installed.

```text
python -B -m pytest tests/scheduling/test_booking_result_contracts.py -q
149 passed in 1.26s

python -B -m pytest tests/calendar tests/conversation/test_booking_safety_guards.py tests/integration/test_booking_persistence.py -q
20 passed, 3 skipped, 1 warning in 13.17s

python -B -m pytest -q
751 passed, 53 skipped, 7 failed, 8 warnings in 68.07s
```

Six full-suite failures are the documented historical release-manifest guards:

- four in `tests/integration/test_barge_in_diagnostics_release.py`
- two in `tests/integration/test_voice_release.py`

The seventh failure is unrelated to these new files:
`tests/stt/test_reset_preservation.py::test_connection_and_close_timeouts_are_bounded_and_logs_are_safe`.
Its synthetic first connection exceeded the monkeypatched 0.01-second deadline.
An isolated rerun reproduced that timeout (`1 failed in 5.39s`) and emitted the
test's pending-cleanup warnings. No scheduling module is imported by that test.

Additional local validation:

```text
ruff check services/scheduling/models.py tests/scheduling/test_booking_result_contracts.py
clean

git diff --check
exit 0, no output

git status --short --branch
exit 0; branch codex/phase-5-booking-reliability
```

The focused suite covers all 100 status/failure/interval-presence combinations,
all failure categories, aware-offset equivalence, explicit DST-fold instants,
half-open boundaries, scope/staff/window/duplicate validation, nested immutability,
truthiness rejection, sanitized representations/errors, and a guarded pure import.

## Unresolved work and preserved scope

There are no new policy decisions in this slice. Duration, slot generation,
business hours, holidays, notice/horizon, eligibility, completeness evidence,
Graph parsing/error classification, retry behavior, proposal/confirmation,
operations, migrations, and runtime integration remain later reviewed tasks.

Only these three new files belong to T007-A:

- `services/scheduling/models.py`
- `tests/scheduling/test_booking_result_contracts.py`
- `docs/delegation/T007-A-result.md`

Scoped final status lists those three paths as untracked. `git diff --name-status`
is empty, confirming no tracked-file edit; the repository's numerous pre-existing
untracked artifacts remain present. Final HEAD remains
`792952b2965e85513bf0971a41bfaff7bd3377c4`.

No existing file, task checkbox, dependency, setting, database, provider, branch,
commit, deployment, or live service was changed.
