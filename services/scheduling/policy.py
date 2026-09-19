"""Pure scheduling-policy evaluation over verified availability evidence.

This module turns one scoped :class:`AvailabilityResult` into candidate times.
Candidates are proposals only: they do not authorize confirmation or mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from services.scheduling.models import (
    AvailabilityQuery,
    AvailabilityResult,
    AvailabilityStatus,
    CalendarScope,
    FreeInterval,
    TimeInterval,
)


__all__ = (
    "AppointmentCandidate",
    "AppointmentFormat",
    "ClosureCalendar",
    "PolicyEvaluation",
    "PolicyFailureReason",
    "PolicyResultStatus",
    "SchedulingPolicy",
    "SchedulingPolicyContractError",
    "evaluate_scheduling_policy",
)


_VALIDATION_MESSAGE = "invalid scheduling policy contract"
_NO_TRUTH_VALUE_MESSAGE = (
    "PolicyEvaluation has no truth value; check status explicitly"
)
_APPROVED_TIMEZONE = "America/Toronto"
_APPROVED_WEEKDAYS = (0, 1, 2, 3, 4)
_APPROVED_OPENS_AT = time(10, 0)
_APPROVED_CLOSES_AT = time(17, 0)
_APPROVED_DURATION = timedelta(minutes=30)
_APPROVED_SLOT_INTERVAL = timedelta(minutes=30)
_APPROVED_NOTICE = timedelta(minutes=30)
_APPROVED_HORIZON = 2
_MAX_HORIZON_SCAN_DAYS = 366


class SchedulingPolicyContractError(ValueError):
    """Fixed, identifier-safe validation failure for policy contracts."""


def _invalid() -> SchedulingPolicyContractError:
    return SchedulingPolicyContractError(_VALIDATION_MESSAGE)


def _validate_identifier(value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise _invalid()
    return value


def _as_utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise _invalid()
    try:
        if value.tzinfo is None or value.utcoffset() is None:
            raise _invalid()
        return value.astimezone(timezone.utc)
    except SchedulingPolicyContractError:
        raise
    except Exception:
        raise _invalid() from None


def _is_date(value: object) -> bool:
    return type(value) is date


def _is_time(value: object) -> bool:
    return type(value) is time and value.tzinfo is None


class AppointmentFormat(Enum):
    """Supported appointment delivery formats."""

    IN_PERSON = "in_person"
    ONLINE = "online"


class PolicyResultStatus(Enum):
    """Explicit outcomes of scheduling-policy evaluation."""

    CANDIDATES = "candidates"
    NO_CANDIDATES = "no_candidates"
    AVAILABILITY_UNVERIFIED = "availability_unverified"
    POLICY_UNVERIFIED = "policy_unverified"


class PolicyFailureReason(Enum):
    """Fixed, sanitized reasons for a non-complete evaluation."""

    CLOSURE_COVERAGE = "closure_coverage"
    UNSUPPORTED_POLICY = "unsupported_policy"
    UNSUPPORTED_STAFF = "unsupported_staff"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INVALID_RESPONSE = "invalid_response"
    INCOMPLETE_RESPONSE = "incomplete_response"
    STALE_EVIDENCE = "stale_evidence"
    FUTURE_EVIDENCE = "future_evidence"
    TIMEZONE_UNAVAILABLE = "timezone_unavailable"


@dataclass(frozen=True, slots=True, repr=False)
class ClosureCalendar:
    """Authoritative local closed dates within an inclusive coverage range."""

    coverage_start: date
    coverage_end: date
    closed_dates: frozenset[date]

    def __post_init__(self) -> None:
        if not _is_date(self.coverage_start) or not _is_date(self.coverage_end):
            raise _invalid()
        if self.coverage_start > self.coverage_end:
            raise _invalid()
        if type(self.closed_dates) is not frozenset:
            raise _invalid()
        for closed_date in self.closed_dates:
            if (
                not _is_date(closed_date)
                or closed_date < self.coverage_start
                or closed_date > self.coverage_end
            ):
                raise _invalid()

    def __repr__(self) -> str:
        return f"ClosureCalendar(closed_count={len(self.closed_dates)})"


@dataclass(frozen=True, slots=True, repr=False)
class SchedulingPolicy:
    """One immutable snapshot of scoped service and booking rules."""

    policy_version: str
    scope: CalendarScope
    approved_staff_ids: tuple[str, ...]
    service_duration: timedelta
    slot_interval: timedelta
    pre_buffer: timedelta
    post_buffer: timedelta
    business_timezone: str
    open_weekdays: tuple[int, ...]
    opens_at: time
    closes_at: time
    minimum_notice: timedelta
    business_day_horizon: int
    closure_calendar: ClosureCalendar | None
    appointment_format: AppointmentFormat
    maximum_attendees: int
    seasonal_overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.policy_version)
        if type(self.scope) is not CalendarScope:
            raise _invalid()
        if type(self.approved_staff_ids) is not tuple or not self.approved_staff_ids:
            raise _invalid()
        for staff_id in self.approved_staff_ids:
            _validate_identifier(staff_id)
        if len(set(self.approved_staff_ids)) != len(self.approved_staff_ids):
            raise _invalid()

        durations = (
            self.service_duration,
            self.slot_interval,
            self.pre_buffer,
            self.post_buffer,
            self.minimum_notice,
        )
        if any(type(value) is not timedelta for value in durations):
            raise _invalid()
        if self.service_duration <= timedelta(0):
            raise _invalid()
        if self.slot_interval <= timedelta(0):
            raise _invalid()
        if self.pre_buffer < timedelta(0) or self.post_buffer < timedelta(0):
            raise _invalid()
        if self.minimum_notice < timedelta(0):
            raise _invalid()

        _validate_identifier(self.business_timezone)
        if type(self.open_weekdays) is not tuple or not self.open_weekdays:
            raise _invalid()
        for weekday in self.open_weekdays:
            if (
                not isinstance(weekday, int)
                or isinstance(weekday, bool)
                or not 0 <= weekday <= 6
            ):
                raise _invalid()
        if len(set(self.open_weekdays)) != len(self.open_weekdays):
            raise _invalid()
        if not _is_time(self.opens_at) or not _is_time(self.closes_at):
            raise _invalid()
        if self.opens_at >= self.closes_at:
            raise _invalid()
        if (
            not isinstance(self.business_day_horizon, int)
            or isinstance(self.business_day_horizon, bool)
            or self.business_day_horizon <= 0
        ):
            raise _invalid()
        if self.closure_calendar is not None and (
            type(self.closure_calendar) is not ClosureCalendar
        ):
            raise _invalid()
        if type(self.appointment_format) is not AppointmentFormat:
            raise _invalid()
        if (
            not isinstance(self.maximum_attendees, int)
            or isinstance(self.maximum_attendees, bool)
            or self.maximum_attendees <= 0
        ):
            raise _invalid()
        if type(self.seasonal_overrides) is not tuple:
            raise _invalid()
        for override in self.seasonal_overrides:
            _validate_identifier(override)

    def __repr__(self) -> str:
        return (
            "SchedulingPolicy("
            f"staff_count={len(self.approved_staff_ids)}, "
            f"has_closure_calendar={self.closure_calendar is not None})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class AppointmentCandidate:
    """A policy-compliant candidate, without mutation authority."""

    scope: CalendarScope
    staff_id: str
    interval: TimeInterval
    policy_version: str
    source_query: AvailabilityQuery
    source_observed_at: datetime

    def __post_init__(self) -> None:
        if type(self.scope) is not CalendarScope:
            raise _invalid()
        _validate_identifier(self.staff_id)
        if type(self.interval) is not TimeInterval:
            raise _invalid()
        if self.interval.end - self.interval.start != _APPROVED_DURATION:
            raise _invalid()
        _validate_identifier(self.policy_version)
        if type(self.source_query) is not AvailabilityQuery:
            raise _invalid()
        observed_at = _as_utc(self.source_observed_at)
        object.__setattr__(self, "source_observed_at", observed_at)
        if (
            self.scope != self.source_query.scope
            or self.staff_id not in self.source_query.staff_ids
            or not self.source_query.window.contains_interval(self.interval)
        ):
            raise _invalid()

    def __repr__(self) -> str:
        return "AppointmentCandidate()"


@dataclass(frozen=True, slots=True, repr=False)
class PolicyEvaluation:
    """Explicit policy result whose status must be inspected."""

    status: PolicyResultStatus
    policy_version: str
    source_query: AvailabilityQuery
    source_observed_at: datetime
    candidates: tuple[AppointmentCandidate, ...] = ()
    reason: PolicyFailureReason | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not PolicyResultStatus:
            raise _invalid()
        _validate_identifier(self.policy_version)
        if type(self.source_query) is not AvailabilityQuery:
            raise _invalid()
        observed_at = _as_utc(self.source_observed_at)
        object.__setattr__(self, "source_observed_at", observed_at)
        if type(self.candidates) is not tuple:
            raise _invalid()
        if self.reason is not None and type(self.reason) is not PolicyFailureReason:
            raise _invalid()
        identities: set[tuple[str, datetime, datetime]] = set()
        for candidate in self.candidates:
            if type(candidate) is not AppointmentCandidate:
                raise _invalid()
            if (
                candidate.policy_version != self.policy_version
                or candidate.source_query != self.source_query
                or candidate.source_observed_at != self.source_observed_at
            ):
                raise _invalid()
            identity = (
                candidate.staff_id,
                candidate.interval.start,
                candidate.interval.end,
            )
            if identity in identities:
                raise _invalid()
            identities.add(identity)

        availability_reasons = {
            PolicyFailureReason.PROVIDER_UNAVAILABLE,
            PolicyFailureReason.INVALID_RESPONSE,
            PolicyFailureReason.INCOMPLETE_RESPONSE,
            PolicyFailureReason.STALE_EVIDENCE,
            PolicyFailureReason.FUTURE_EVIDENCE,
        }
        policy_reasons = {
            PolicyFailureReason.CLOSURE_COVERAGE,
            PolicyFailureReason.UNSUPPORTED_POLICY,
            PolicyFailureReason.UNSUPPORTED_STAFF,
            PolicyFailureReason.TIMEZONE_UNAVAILABLE,
        }
        if self.status is PolicyResultStatus.CANDIDATES:
            valid = bool(self.candidates) and self.reason is None
        elif self.status is PolicyResultStatus.NO_CANDIDATES:
            valid = not self.candidates and self.reason is None
        elif self.status is PolicyResultStatus.AVAILABILITY_UNVERIFIED:
            valid = not self.candidates and self.reason in availability_reasons
        else:
            valid = not self.candidates and self.reason in policy_reasons
        if not valid:
            raise _invalid()

    def __bool__(self) -> bool:
        raise TypeError(_NO_TRUTH_VALUE_MESSAGE)

    def __repr__(self) -> str:
        return (
            "PolicyEvaluation("
            f"status={self.status.name}, candidate_count={len(self.candidates)})"
        )


def _evaluation(
    status: PolicyResultStatus,
    policy: SchedulingPolicy,
    availability: AvailabilityResult,
    *,
    candidates: tuple[AppointmentCandidate, ...] = (),
    reason: PolicyFailureReason | None = None,
) -> PolicyEvaluation:
    return PolicyEvaluation(
        status,
        policy.policy_version,
        availability.query,
        availability.observed_at,
        candidates,
        reason,
    )


def _uses_approved_rules(policy: SchedulingPolicy) -> bool:
    return (
        policy.service_duration == _APPROVED_DURATION
        and policy.slot_interval == _APPROVED_SLOT_INTERVAL
        and policy.pre_buffer == timedelta(0)
        and policy.post_buffer == timedelta(0)
        and policy.business_timezone == _APPROVED_TIMEZONE
        and policy.open_weekdays == _APPROVED_WEEKDAYS
        and policy.opens_at == _APPROVED_OPENS_AT
        and policy.closes_at == _APPROVED_CLOSES_AT
        and policy.minimum_notice == _APPROVED_NOTICE
        and policy.business_day_horizon == _APPROVED_HORIZON
        and policy.appointment_format is AppointmentFormat.IN_PERSON
        and policy.maximum_attendees == 1
        and not policy.seasonal_overrides
    )


def _horizon_end(
    today: date,
    policy: SchedulingPolicy,
) -> date | None:
    calendar = policy.closure_calendar
    if (
        calendar is None
        or calendar.coverage_start > today
        or calendar.coverage_end < today
    ):
        return None

    open_days = 0
    current = today
    for _ in range(_MAX_HORIZON_SCAN_DAYS):
        current += timedelta(days=1)
        if current > calendar.coverage_end:
            return None
        if (
            current.weekday() in policy.open_weekdays
            and current not in calendar.closed_dates
        ):
            open_days += 1
            if open_days == policy.business_day_horizon:
                return current
    return None


def _merge_intervals(
    intervals: tuple[FreeInterval, ...],
    staff_ids: tuple[str, ...],
) -> dict[str, tuple[TimeInterval, ...]]:
    grouped: dict[str, list[TimeInterval]] = {staff_id: [] for staff_id in staff_ids}
    for free in intervals:
        grouped[free.staff_id].append(free.interval)

    merged_by_staff: dict[str, tuple[TimeInterval, ...]] = {}
    for staff_id, staff_intervals in grouped.items():
        ordered = sorted(staff_intervals, key=lambda item: (item.start, item.end))
        merged: list[TimeInterval] = []
        for interval in ordered:
            if not merged or interval.start > merged[-1].end:
                merged.append(interval)
                continue
            if interval.end > merged[-1].end:
                merged[-1] = TimeInterval(merged[-1].start, interval.end)
        merged_by_staff[staff_id] = tuple(merged)
    return merged_by_staff


def _local_instant(day: date, clock_time: time, zone: ZoneInfo) -> datetime:
    return datetime.combine(day, clock_time, tzinfo=zone).astimezone(timezone.utc)


def _generate_candidates(
    policy: SchedulingPolicy,
    availability: AvailabilityResult,
    now: datetime,
    zone: ZoneInfo,
    today: date,
    horizon_end: date,
) -> tuple[AppointmentCandidate, ...]:
    calendar = policy.closure_calendar
    if calendar is None:  # Kept defensive for static narrowing.
        raise _invalid()
    merged = _merge_intervals(availability.intervals, availability.query.staff_ids)
    earliest_start = now + policy.minimum_notice
    generated: list[AppointmentCandidate] = []
    current = today
    while current <= horizon_end:
        if (
            current.weekday() in policy.open_weekdays
            and current not in calendar.closed_dates
        ):
            business_open = _local_instant(current, policy.opens_at, zone)
            business_close = _local_instant(current, policy.closes_at, zone)
            start = business_open
            while start + policy.service_duration <= business_close:
                interval = TimeInterval(start, start + policy.service_duration)
                if (
                    start >= earliest_start
                    and availability.query.window.contains_interval(interval)
                ):
                    for staff_id in availability.query.staff_ids:
                        if any(
                            free.contains_interval(interval)
                            for free in merged[staff_id]
                        ):
                            generated.append(AppointmentCandidate(
                                policy.scope,
                                staff_id,
                                interval,
                                policy.policy_version,
                                availability.query,
                                availability.observed_at,
                            ))
                start += policy.slot_interval
        current += timedelta(days=1)
    generated.sort(key=lambda item: (item.interval.start, item.staff_id))
    return tuple(generated)


def evaluate_scheduling_policy(
    policy: SchedulingPolicy,
    availability: AvailabilityResult,
    now: datetime,
    maximum_evidence_age: timedelta,
) -> PolicyEvaluation:
    """Apply the approved policy to one accepted provider result.

    The caller remains responsible for supplying validated live service facts,
    approved opaque IDs, authoritative closure coverage, and an approved evidence
    age. This pure function performs no I/O and grants no write authority.
    """
    if type(policy) is not SchedulingPolicy:
        raise _invalid()
    if type(availability) is not AvailabilityResult:
        raise _invalid()
    normalized_now = _as_utc(now)
    if (
        type(maximum_evidence_age) is not timedelta
        or maximum_evidence_age <= timedelta(0)
    ):
        raise _invalid()
    if policy.scope != availability.query.scope:
        raise _invalid()

    if not _uses_approved_rules(policy):
        return _evaluation(
            PolicyResultStatus.POLICY_UNVERIFIED,
            policy,
            availability,
            reason=PolicyFailureReason.UNSUPPORTED_POLICY,
        )
    if any(
        staff_id not in policy.approved_staff_ids
        for staff_id in availability.query.staff_ids
    ):
        return _evaluation(
            PolicyResultStatus.POLICY_UNVERIFIED,
            policy,
            availability,
            reason=PolicyFailureReason.UNSUPPORTED_STAFF,
        )

    try:
        zone = ZoneInfo(policy.business_timezone)
        today = normalized_now.astimezone(zone).date()
    except (ZoneInfoNotFoundError, OSError, ValueError):
        return _evaluation(
            PolicyResultStatus.POLICY_UNVERIFIED,
            policy,
            availability,
            reason=PolicyFailureReason.TIMEZONE_UNAVAILABLE,
        )

    horizon_end = _horizon_end(today, policy)
    if horizon_end is None:
        return _evaluation(
            PolicyResultStatus.POLICY_UNVERIFIED,
            policy,
            availability,
            reason=PolicyFailureReason.CLOSURE_COVERAGE,
        )

    if availability.observed_at > normalized_now:
        return _evaluation(
            PolicyResultStatus.AVAILABILITY_UNVERIFIED,
            policy,
            availability,
            reason=PolicyFailureReason.FUTURE_EVIDENCE,
        )
    if normalized_now - availability.observed_at > maximum_evidence_age:
        return _evaluation(
            PolicyResultStatus.AVAILABILITY_UNVERIFIED,
            policy,
            availability,
            reason=PolicyFailureReason.STALE_EVIDENCE,
        )
    provider_failure_reasons = {
        AvailabilityStatus.UNAVAILABLE: PolicyFailureReason.PROVIDER_UNAVAILABLE,
        AvailabilityStatus.INVALID_RESPONSE: PolicyFailureReason.INVALID_RESPONSE,
        AvailabilityStatus.INCOMPLETE: PolicyFailureReason.INCOMPLETE_RESPONSE,
    }
    if availability.status in provider_failure_reasons:
        return _evaluation(
            PolicyResultStatus.AVAILABILITY_UNVERIFIED,
            policy,
            availability,
            reason=provider_failure_reasons[availability.status],
        )
    if availability.status is AvailabilityStatus.NO_AVAILABILITY:
        return _evaluation(
            PolicyResultStatus.NO_CANDIDATES,
            policy,
            availability,
        )

    candidates = _generate_candidates(
        policy,
        availability,
        normalized_now,
        zone,
        today,
        horizon_end,
    )
    if not candidates:
        return _evaluation(
            PolicyResultStatus.NO_CANDIDATES,
            policy,
            availability,
        )
    return _evaluation(
        PolicyResultStatus.CANDIDATES,
        policy,
        availability,
        candidates=candidates,
    )
