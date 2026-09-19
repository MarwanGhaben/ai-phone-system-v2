"""Caller booking read-side composition of approved policy and fresh Graph facts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
import re
from uuid import uuid4
from zoneinfo import ZoneInfo

import yaml

from services.calendar.service_facts import ServiceFactsRead
from services.scheduling.models import AvailabilityQuery, AvailabilityResult, CalendarScope, TimeInterval
from services.scheduling.policy import (
    AppointmentCandidate, AppointmentFormat, ClosureCalendar,
    PolicyResultStatus, SchedulingPolicy, evaluate_scheduling_policy,
)


TORONTO = ZoneInfo("America/Toronto")
EVIDENCE_AGE = timedelta(seconds=60)
DEFAULT_POLICY_PATH = Path(__file__).parents[2] / "clients" / "booking-policy.yaml"
_LOCAL = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}\Z")
_OFFSET = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)\Z"
)
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_APPROVED_HOLIDAYS = frozenset({
    date(2026, 1, 1), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 18), date(2026, 7, 1), date(2026, 9, 7),
    date(2026, 10, 12), date(2026, 12, 25), date(2026, 12, 26),
    date(2027, 1, 1), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 24), date(2027, 7, 1), date(2027, 9, 6),
    date(2027, 10, 11), date(2027, 12, 25), date(2027, 12, 26),
})
_EXPECTED = {
    "timezone": "America/Toronto", "open_weekdays": [0, 1, 2, 3, 4],
    "opens_at": "10:00", "closes_at": "17:00", "duration_minutes": 30,
    "slot_minutes": 30, "pre_buffer_minutes": 0, "post_buffer_minutes": 0,
    "minimum_notice_minutes": 30, "business_day_horizon": 2,
    "appointment_format": "in_person", "maximum_attendees": 1,
    "office_location": "1901 Banff Ave, Ottawa, ON K1V 7W9",
    "seasonal_overrides": [],
}


@dataclass(frozen=True, slots=True, repr=False)
class BookingPolicySnapshot:
    policy: SchedulingPolicy
    horizon_end: date
    location: str


@dataclass(frozen=True, slots=True, repr=False)
class BookingAssessment:
    status: str
    candidates: tuple[AppointmentCandidate, ...] = ()


def _date(value: object) -> date:
    if type(value) is not str or not _DATE.fullmatch(value):
        raise ValueError
    return date.fromisoformat(value)


def parse_requested_time(value: object) -> datetime | None:
    """Accept only complete local minute or explicit-offset ISO forms."""
    if type(value) is not str:
        return None
    try:
        if _LOCAL.fullmatch(value):
            naive = datetime.strptime(value, "%Y-%m-%d %H:%M")
            first = naive.replace(tzinfo=TORONTO, fold=0)
            second = naive.replace(tzinfo=TORONTO, fold=1)
            if first.utcoffset() != second.utcoffset():
                return None
            if first.astimezone(timezone.utc).astimezone(TORONTO).replace(
                    tzinfo=None) != naive:
                return None
            return first
        if _OFFSET.fullmatch(value):
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(TORONTO)
    except (ValueError, OverflowError):
        pass
    return None


def load_booking_policy(scope: CalendarScope, staff_id: str, now: datetime,
                        path: Path | None = None) -> BookingPolicySnapshot | None:
    """Load and validate explicit closure coverage; never infer later holidays."""
    if (type(scope) is not CalendarScope or type(now) is not datetime
            or now.tzinfo is None or now.utcoffset() is None):
        return None
    try:
        with open(path or DEFAULT_POLICY_PATH, encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if (type(data) is not dict or type(data.get("version")) is not str
                or not data["version"] or data["version"] != data["version"].strip()):
            raise ValueError
        for key, expected in _EXPECTED.items():
            if type(data.get(key)) is not type(expected) or data[key] != expected:
                raise ValueError
        closure = data["closures"]
        if type(closure) is not dict or type(closure.get("dates")) is not list:
            raise ValueError
        start, end = _date(closure["coverage_start"]), _date(closure["coverage_end"])
        days = tuple(_date(item) for item in closure["dates"])
        if len(days) != len(set(days)):
            raise ValueError
        if path is None and (
            data["version"] != "2026-09-18-owner-approved-v1"
            or start != date(2026, 1, 1) or end != date(2027, 12, 31)
            or frozenset(days) != _APPROVED_HOLIDAYS
        ):
            raise ValueError
        calendar = ClosureCalendar(start, end, frozenset(days))
        today = now.astimezone(TORONTO).date()
        if not start <= today <= end:
            raise ValueError
        current, open_days = today, 0
        for _ in range(366):
            current += timedelta(days=1)
            if current > end:
                raise ValueError
            if current.weekday() < 5 and current not in calendar.closed_dates:
                open_days += 1
                if open_days == 2:
                    break
        else:
            raise ValueError
        policy = SchedulingPolicy(
            policy_version=data["version"], scope=scope,
            approved_staff_ids=(staff_id,), service_duration=timedelta(minutes=30),
            slot_interval=timedelta(minutes=30), pre_buffer=timedelta(0),
            post_buffer=timedelta(0), business_timezone="America/Toronto",
            open_weekdays=(0, 1, 2, 3, 4), opens_at=time(10), closes_at=time(17),
            minimum_notice=timedelta(minutes=30), business_day_horizon=2,
            closure_calendar=calendar, appointment_format=AppointmentFormat.IN_PERSON,
            maximum_attendees=1,
        )
        return BookingPolicySnapshot(policy, current, data["office_location"])
    except (OSError, ValueError, KeyError, TypeError, OverflowError, yaml.YAMLError):
        return None


def make_booking_query(snapshot: BookingPolicySnapshot, staff_id: str,
                       now: datetime) -> AvailabilityQuery:
    today = now.astimezone(TORONTO).date()
    start = datetime.combine(today, time.min, TORONTO)
    end = datetime.combine(snapshot.horizon_end + timedelta(days=1), time.min, TORONTO)
    return AvailabilityQuery(snapshot.policy.scope, TimeInterval(start, end),
                             (staff_id,), uuid4().hex)


def assess_booking(snapshot: BookingPolicySnapshot, query: AvailabilityQuery,
                   facts_read: ServiceFactsRead, availability: AvailabilityResult,
                   now: datetime) -> BookingAssessment:
    """Require two fresh scoped reads; return only policy-approved candidates."""
    now_utc = now.astimezone(timezone.utc)
    if query.window.start.astimezone(TORONTO).date() != now.astimezone(TORONTO).date():
        return BookingAssessment("policy_unverified")
    if (type(facts_read) is not ServiceFactsRead
            or facts_read.scope != query.scope or facts_read.status != "verified"
            or facts_read.facts is None):
        return BookingAssessment("availability_unverified")
    if (facts_read.observed_at > now_utc
            or now_utc - facts_read.observed_at > EVIDENCE_AGE):
        return BookingAssessment("availability_unverified")
    if (type(availability) is not AvailabilityResult
            or availability.query != query):
        return BookingAssessment("availability_unverified")
    facts = facts_read.facts
    if (query.staff_ids[0] not in facts.staff_ids
            or facts.duration != timedelta(minutes=30)
            or facts.slot_interval != timedelta(minutes=30)
            or facts.pre_buffer != timedelta(0) or facts.post_buffer != timedelta(0)
            or facts.online or facts.maximum_attendees != 1 or facts.hidden
            or not facts.allow_staff_selection):
        return BookingAssessment("policy_unverified")
    evaluation = evaluate_scheduling_policy(snapshot.policy, availability, now_utc,
                                            EVIDENCE_AGE)
    if evaluation.status is PolicyResultStatus.AVAILABILITY_UNVERIFIED:
        return BookingAssessment("availability_unverified")
    if evaluation.status is PolicyResultStatus.POLICY_UNVERIFIED:
        return BookingAssessment("policy_unverified")
    for window in facts.custom_windows:
        if window.end >= now.astimezone(TORONTO).date() and window.start <= snapshot.horizon_end:
            if window.mode != "notBookable":
                return BookingAssessment("policy_unverified")
    if facts.general_mode == "notBookable":
        return BookingAssessment("no_eligible")
    candidates = tuple(candidate for candidate in evaluation.candidates
                       if candidate.interval.start >= now_utc + max(
                           timedelta(minutes=30), facts.minimum_notice)
                       and candidate.interval.start <= now_utc + facts.maximum_advance
                       and not any(window.start <= candidate.interval.start.astimezone(TORONTO).date()
                                   <= window.end for window in facts.custom_windows))
    return BookingAssessment("available" if candidates else "no_eligible", candidates)
