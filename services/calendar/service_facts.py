"""Strict, scoped facts from one Microsoft Bookings service-by-ID response."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import re

from services.scheduling.models import CalendarScope


_DURATION = re.compile(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?\Z")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")


@dataclass(frozen=True, slots=True, repr=False)
class ServiceWindow:
    start: date
    end: date
    mode: str


@dataclass(frozen=True, slots=True, repr=False)
class ServiceFacts:
    scope: CalendarScope
    duration: timedelta
    pre_buffer: timedelta
    post_buffer: timedelta
    staff_ids: tuple[str, ...]
    online: bool
    maximum_attendees: int
    hidden: bool
    slot_interval: timedelta
    minimum_notice: timedelta
    maximum_advance: timedelta
    allow_staff_selection: bool
    general_mode: str
    custom_windows: tuple[ServiceWindow, ...]


@dataclass(frozen=True, slots=True, repr=False)
class ServiceFactsRead:
    status: str
    scope: CalendarScope
    observed_at: datetime
    facts: ServiceFacts | None = None
    failure: str | None = None

    def __post_init__(self) -> None:
        if type(self.scope) is not CalendarScope or type(self.observed_at) is not datetime:
            raise ValueError("invalid service facts result")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("invalid service facts result")
        if self.status == "verified":
            if type(self.facts) is not ServiceFacts or self.facts.scope != self.scope or self.failure:
                raise ValueError("invalid service facts result")
        elif self.status == "unverified":
            if self.facts is not None or not isinstance(self.failure, str):
                raise ValueError("invalid service facts result")
        else:
            raise ValueError("invalid service facts result")


def _duration(value: object) -> timedelta:
    if not isinstance(value, str):
        raise ValueError
    match = _DURATION.fullmatch(value)
    if match is None or not any(part is not None for part in match.groups()):
        raise ValueError
    if "T" in value and not any(part is not None for part in match.groups()[1:]):
        raise ValueError
    try:
        days, hours, minutes, seconds = match.groups()
        whole, _, fraction = (seconds or "0").partition(".")
        # Reject unrepresentable precision before arithmetic; Decimal's ambient
        # precision can otherwise round a different duration to exactly 30 min.
        if any(digit != "0" for digit in fraction[6:]):
            raise ValueError
        return timedelta(days=int(days or 0), hours=int(hours or 0),
                         minutes=int(minutes or 0), seconds=int(whole),
                         microseconds=int(fraction[:6].ljust(6, "0")))
    except (OverflowError, TypeError):
        raise ValueError from None


def _date(value: object) -> date:
    if not isinstance(value, str) or not _DATE.fullmatch(value):
        raise ValueError
    return date.fromisoformat(value)


def _no_continuation(value: object) -> bool:
    if isinstance(value, dict):
        if any(str(key).lower().endswith(("nextlink", "deltalink")) for key in value):
            return False
        return all(_no_continuation(item) for item in value.values())
    if isinstance(value, list):
        return all(_no_continuation(item) for item in value)
    return True


def decode_service_facts(wire: object, scope: CalendarScope,
                         observed_at: datetime) -> ServiceFactsRead:
    """Return no usable facts on malformed, incomplete or unsupported evidence."""
    if type(scope) is not CalendarScope:
        raise ValueError("invalid service scope")
    if (type(observed_at) is not datetime or observed_at.tzinfo is None
            or observed_at.utcoffset() is None):
        raise ValueError("invalid service facts observation")
    observed_at = observed_at.astimezone(timezone.utc)
    try:
        if not isinstance(wire, dict) or not _no_continuation(wire):
            raise ValueError
        if wire.get("id") != scope.service_id:
            raise ValueError
        staff = wire["staffMemberIds"]
        if (type(staff) is not list or not staff
                or any(not isinstance(item, str) or not item or item != item.strip()
                       for item in staff) or len(staff) != len(set(staff))):
            raise ValueError
        online = wire["isLocationOnline"]
        hidden = wire["isHiddenFromCustomers"]
        capacity = wire["maximumAttendeesCount"]
        if (type(online) is not bool or type(hidden) is not bool
                or type(capacity) is not int or capacity <= 0):
            raise ValueError
        policy = wire["schedulingPolicy"]
        if not isinstance(policy, dict):
            raise ValueError
        allow_staff = policy["allowStaffSelection"]
        if type(allow_staff) is not bool:
            raise ValueError
        general = policy["generalAvailability"]
        if (not isinstance(general, dict)
                or not isinstance(general.get("availabilityType"), str)
                or general.get("businessHours") is not None):
            raise ValueError
        if general["availabilityType"] not in ("bookWhenStaffAreFree", "notBookable"):
            return ServiceFactsRead("unverified", scope, observed_at,
                                    failure="unsupported_policy")
        windows = policy["customAvailabilities"]
        if type(windows) is not list:
            raise ValueError
        parsed = []
        for item in windows:
            if not isinstance(item, dict) or not isinstance(item.get("availabilityType"), str):
                raise ValueError
            start, end = _date(item["startDate"]), _date(item["endDate"])
            if start > end:
                raise ValueError
            mode = item["availabilityType"]
            if mode == "notBookable" and item.get("businessHours") is not None:
                raise ValueError
            parsed.append(ServiceWindow(start, end, mode))
        facts = ServiceFacts(
            scope, _duration(wire["defaultDuration"]),
            _duration(wire["preBuffer"]), _duration(wire["postBuffer"]),
            tuple(staff), online, capacity, hidden,
            _duration(policy["timeSlotInterval"]),
            _duration(policy["minimumLeadTime"]),
            _duration(policy["maximumAdvance"]), allow_staff,
            general["availabilityType"], tuple(parsed),
        )
        return ServiceFactsRead("verified", scope, observed_at, facts)
    except (KeyError, ValueError, TypeError, OverflowError):
        return ServiceFactsRead("unverified", scope, observed_at,
                                failure="invalid_response")
