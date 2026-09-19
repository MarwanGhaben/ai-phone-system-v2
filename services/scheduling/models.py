"""Pure availability contracts for the scheduling domain.

Intervals are normalized to UTC and use half-open ``[start, end)`` semantics.
These types describe provider availability evidence only; they do not apply
business policy, create bookable slots, or authorize an appointment mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


__all__ = (
    "AvailabilityContractError",
    "AvailabilityFailure",
    "AvailabilityFailureCategory",
    "AvailabilityQuery",
    "AvailabilityResult",
    "AvailabilityStatus",
    "CalendarScope",
    "FreeInterval",
    "TimeInterval",
)


_VALIDATION_MESSAGE = "invalid availability contract"
_NO_TRUTH_VALUE_MESSAGE = (
    "AvailabilityResult has no truth value; check status explicitly"
)


class AvailabilityContractError(ValueError):
    """Fixed, identifier-safe validation failure for availability contracts."""


def _invalid() -> AvailabilityContractError:
    return AvailabilityContractError(_VALIDATION_MESSAGE)


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
    except AvailabilityContractError:
        raise
    except Exception:
        raise _invalid() from None


@dataclass(frozen=True, slots=True, repr=False)
class CalendarScope:
    """Opaque, case-preserved tenant, business, and service identifiers."""

    tenant_id: str
    business_id: str
    service_id: str

    def __post_init__(self) -> None:
        _validate_identifier(self.tenant_id)
        _validate_identifier(self.business_id)
        _validate_identifier(self.service_id)

    def __repr__(self) -> str:
        return "CalendarScope()"


@dataclass(frozen=True, slots=True, repr=False)
class TimeInterval:
    """A nonempty half-open ``[start, end)`` interval of UTC instants."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        start = _as_utc(self.start)
        end = _as_utc(self.end)
        if start >= end:
            raise _invalid()
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def contains(self, instant: datetime) -> bool:
        """Return whether an aware instant is in ``[start, end)``."""
        normalized = _as_utc(instant)
        return self.start <= normalized < self.end

    def contains_interval(self, interval: TimeInterval) -> bool:
        """Return whether an interval lies within these inclusive outer bounds."""
        if type(interval) is not TimeInterval:
            raise _invalid()
        return self.start <= interval.start and interval.end <= self.end

    def __repr__(self) -> str:
        return "TimeInterval()"


@dataclass(frozen=True, slots=True, repr=False)
class AvailabilityQuery:
    """One scoped, bounded request for explicitly selected staff members."""

    scope: CalendarScope
    window: TimeInterval
    staff_ids: tuple[str, ...]
    request_id: str

    def __post_init__(self) -> None:
        if type(self.scope) is not CalendarScope:
            raise _invalid()
        if type(self.window) is not TimeInterval:
            raise _invalid()
        if type(self.staff_ids) is not tuple or not self.staff_ids:
            raise _invalid()
        for staff_id in self.staff_ids:
            _validate_identifier(staff_id)
        if len(set(self.staff_ids)) != len(self.staff_ids):
            raise _invalid()
        _validate_identifier(self.request_id)

    def __repr__(self) -> str:
        return f"AvailabilityQuery(staff_count={len(self.staff_ids)})"


@dataclass(frozen=True, slots=True, repr=False)
class FreeInterval:
    """Provider free-time evidence, not a policy-approved bookable slot."""

    scope: CalendarScope
    staff_id: str
    interval: TimeInterval

    def __post_init__(self) -> None:
        if type(self.scope) is not CalendarScope:
            raise _invalid()
        _validate_identifier(self.staff_id)
        if type(self.interval) is not TimeInterval:
            raise _invalid()

    def __repr__(self) -> str:
        return "FreeInterval()"


class AvailabilityStatus(Enum):
    """Explicit outcomes of a scoped availability query."""

    AVAILABLE = "available"
    NO_AVAILABILITY = "no_availability"
    UNAVAILABLE = "unavailable"
    INVALID_RESPONSE = "invalid_response"
    INCOMPLETE = "incomplete"


class AvailabilityFailureCategory(Enum):
    """Safe fixed classifications for availability failures."""

    NOT_CONFIGURED = "not_configured"
    AUTHENTICATION = "authentication"
    PERMISSION_DENIED = "permission_denied"
    THROTTLED = "throttled"
    TIMEOUT = "timeout"
    TRANSPORT = "transport"
    PROVIDER_ERROR = "provider_error"
    INVALID_RESPONSE = "invalid_response"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True, repr=False)
class AvailabilityFailure:
    """Sanitized failure classification without raw provider material."""

    category: AvailabilityFailureCategory
    http_status: int | None = None

    def __post_init__(self) -> None:
        if type(self.category) is not AvailabilityFailureCategory:
            raise _invalid()
        if self.http_status is not None and (
            not isinstance(self.http_status, int)
            or isinstance(self.http_status, bool)
            or not 100 <= self.http_status <= 599
        ):
            raise _invalid()

    def __repr__(self) -> str:
        status = "none" if self.http_status is None else str(self.http_status)
        return (
            "AvailabilityFailure("
            f"category={self.category.name}, http_status={status})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class AvailabilityResult:
    """Validated result whose status must always be inspected explicitly.

    ``NO_AVAILABILITY`` is only the domain representation of a complete,
    successful empty read. Construction cannot prove that a provider response was
    complete; the future adapter is responsible for establishing that evidence.
    """

    query: AvailabilityQuery
    status: AvailabilityStatus
    observed_at: datetime
    intervals: tuple[FreeInterval, ...] = ()
    failure: AvailabilityFailure | None = None

    def __post_init__(self) -> None:
        if type(self.query) is not AvailabilityQuery:
            raise _invalid()
        if type(self.status) is not AvailabilityStatus:
            raise _invalid()
        observed_at = _as_utc(self.observed_at)
        object.__setattr__(self, "observed_at", observed_at)
        if type(self.intervals) is not tuple:
            raise _invalid()
        if self.failure is not None and type(self.failure) is not AvailabilityFailure:
            raise _invalid()

        seen: set[tuple[str, datetime, datetime]] = set()
        for item in self.intervals:
            if type(item) is not FreeInterval:
                raise _invalid()
            if item.scope != self.query.scope:
                raise _invalid()
            if item.staff_id not in self.query.staff_ids:
                raise _invalid()
            if not self.query.window.contains_interval(item.interval):
                raise _invalid()
            identity = (item.staff_id, item.interval.start, item.interval.end)
            if identity in seen:
                raise _invalid()
            seen.add(identity)

        self._validate_outcome()

    def _validate_outcome(self) -> None:
        has_intervals = bool(self.intervals)
        category = self.failure.category if self.failure is not None else None

        if self.status is AvailabilityStatus.AVAILABLE:
            valid = has_intervals and self.failure is None
        elif self.status is AvailabilityStatus.NO_AVAILABILITY:
            valid = not has_intervals and self.failure is None
        elif self.status is AvailabilityStatus.UNAVAILABLE:
            valid = (
                not has_intervals
                and self.failure is not None
                and category not in {
                    AvailabilityFailureCategory.INVALID_RESPONSE,
                    AvailabilityFailureCategory.INCOMPLETE,
                }
            )
        elif self.status is AvailabilityStatus.INVALID_RESPONSE:
            valid = (
                not has_intervals
                and category is AvailabilityFailureCategory.INVALID_RESPONSE
            )
        else:
            valid = (
                not has_intervals
                and category is AvailabilityFailureCategory.INCOMPLETE
            )

        if not valid:
            raise _invalid()

    def __bool__(self) -> bool:
        raise TypeError(_NO_TRUTH_VALUE_MESSAGE)

    def __repr__(self) -> str:
        return (
            "AvailabilityResult("
            f"status={self.status.name}, interval_count={len(self.intervals)})"
        )
