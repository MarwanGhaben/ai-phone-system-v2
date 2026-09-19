"""Pure Microsoft Bookings availability wire decoding.

This module validates structural staff coverage and provider evidence. It does
not apply service eligibility, duration, business hours, buffers, or booking
policy, and it performs no I/O or clock reads.
"""
from __future__ import annotations

from datetime import datetime, timezone
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from services.scheduling.models import (
    AvailabilityContractError,
    AvailabilityFailure,
    AvailabilityFailureCategory,
    AvailabilityQuery,
    AvailabilityResult,
    AvailabilityStatus,
    FreeInterval,
    TimeInterval,
)


MAX_AVAILABILITY_ITEMS = 10_000

_DATETIME = re.compile(
    r"^(?P<base>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,7}))?"
    r"(?P<offset>Z|[+-]\d{2}:\d{2})?$"
)
_WINDOWS_ZONES = {
    "Eastern Standard Time": "America/Toronto",
    "Pacific Standard Time": "America/Los_Angeles",
    "Pacific Time (US & Canada)": "America/Los_Angeles",
    "(UTC-08:00) Pacific Time (US & Canada)": "America/Los_Angeles",
}
_STATUS = {
    "available": "free",
    "Available": "free",
    "busy": "non_free",
    "Busy": "non_free",
    "outOfOffice": "non_free",
    "OutOfOffice": "non_free",
    "slotsAvailable": "incomplete",
    "SlotsAvailable": "incomplete",
    "unknownFutureValue": "incomplete",
    "UnknownFutureValue": "incomplete",
}
_CONTINUATION_KEYS = frozenset({"@odata.nextLink", "odata.nextLink", "nextLink"})
_VALIDATION_MESSAGE = "invalid availability contract"


class _DecodeFailure(Exception):
    def __init__(self, category: AvailabilityFailureCategory):
        super().__init__(category.value)
        self.category = category


def _invalid() -> None:
    raise _DecodeFailure(AvailabilityFailureCategory.INVALID_RESPONSE)


def _incomplete() -> None:
    raise _DecodeFailure(AvailabilityFailureCategory.INCOMPLETE)


def _failure_result(
    query: AvailabilityQuery,
    observed_at: datetime,
    category: AvailabilityFailureCategory,
) -> AvailabilityResult:
    status = (
        AvailabilityStatus.INVALID_RESPONSE
        if category is AvailabilityFailureCategory.INVALID_RESPONSE
        else AvailabilityStatus.INCOMPLETE
    )
    return AvailabilityResult(
        query,
        status,
        observed_at,
        failure=AvailabilityFailure(category),
    )


def _require_contract(query: object, observed_at: object) -> None:
    if type(query) is not AvailabilityQuery or not isinstance(observed_at, datetime):
        raise AvailabilityContractError(_VALIDATION_MESSAGE)
    try:
        aware = observed_at.tzinfo is not None and observed_at.utcoffset() is not None
    except Exception:
        aware = False
    if not aware:
        raise AvailabilityContractError(_VALIDATION_MESSAGE)


def _zone(value: object) -> timezone | ZoneInfo:
    if not isinstance(value, str) or not value:
        _invalid()
    if value in ("UTC", "(UTC) Coordinated Universal Time"):
        return timezone.utc
    iana_name = _WINDOWS_ZONES.get(value, value)
    try:
        return ZoneInfo(iana_name)
    except ValueError:
        _invalid()
    except ZoneInfoNotFoundError:
        if value in _WINDOWS_ZONES:
            _incomplete()
        try:
            ZoneInfo("America/Toronto")
        except ZoneInfoNotFoundError:
            _incomplete()
        _invalid()


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, dict):
        _invalid()
    text = value.get("dateTime")
    match = _DATETIME.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        _invalid()
    fraction = match.group("fraction")
    if fraction is not None and len(fraction) == 7:
        if fraction[-1] != "0":
            _invalid()
        fraction = fraction[:6]
    normalized_text = match.group("base")
    if fraction is not None:
        normalized_text += "." + fraction
    offset_text = match.group("offset")
    if offset_text == "Z":
        normalized_text += "+00:00"
    elif offset_text is not None:
        normalized_text += offset_text
    try:
        parsed = datetime.fromisoformat(normalized_text)
    except (TypeError, ValueError, OverflowError):
        _invalid()
    zone = _zone(value.get("timeZone"))

    if parsed.tzinfo is not None:
        instant = parsed.astimezone(timezone.utc)
        local = instant.astimezone(zone)
        if (
            local.replace(tzinfo=None) != parsed.replace(tzinfo=None)
            or local.utcoffset() != parsed.utcoffset()
        ):
            _invalid()
        return instant

    candidates: dict[datetime, datetime] = {}
    for fold in (0, 1):
        aware = parsed.replace(tzinfo=zone, fold=fold)
        instant = aware.astimezone(timezone.utc)
        if instant.astimezone(zone).replace(tzinfo=None) == parsed:
            candidates[instant] = aware
    if len(candidates) != 1:
        _invalid()
    return next(iter(candidates))


def _overlap(first: TimeInterval, second: TimeInterval) -> bool:
    return first.start < second.end and second.start < first.end


def _has_collection_continuation(
    owner: dict[object, object],
    collection_name: str,
) -> bool:
    return (
        any(key in owner for key in _CONTINUATION_KEYS)
        or f"{collection_name}@odata.nextLink" in owner
    )


def decode_availability(
    payload: object,
    query: AvailabilityQuery,
    observed_at: datetime,
) -> AvailabilityResult:
    """Decode one complete availability response into the accepted domain types.

    Distinct overlapping free intervals are retained in provider order. Exact
    duplicates and free/non-free contradictions are rejected; adjacent evidence
    is not overlapping.
    """
    _require_contract(query, observed_at)
    try:
        if not isinstance(payload, dict):
            _invalid()
        envelopes = [
            key for key in ("value", "staffAvailabilityItem") if key in payload
        ]
        if len(envelopes) != 1:
            _invalid()
        envelope = envelopes[0]
        if _has_collection_continuation(payload, envelope):
            _incomplete()
        collection = payload[envelope]
        if not isinstance(collection, list):
            _invalid()
        if not collection:
            _incomplete()

        requested_staff = set(query.staff_ids)
        returned_staff: set[str] = set()
        free_intervals: list[FreeInterval] = []
        evidence: dict[str, list[tuple[str, TimeInterval]]] = {
            staff_id: [] for staff_id in query.staff_ids
        }
        seen_items: set[tuple[str, str, datetime, datetime]] = set()
        item_count = 0
        response_incomplete = False

        for staff_payload in collection:
            if not isinstance(staff_payload, dict):
                _invalid()
            staff_id = staff_payload.get("staffId")
            if (
                not isinstance(staff_id, str)
                or not staff_id
                or staff_id not in requested_staff
                or staff_id in returned_staff
            ):
                _invalid()
            returned_staff.add(staff_id)
            if _has_collection_continuation(
                staff_payload,
                "availabilityItems",
            ):
                _incomplete()
            items = staff_payload.get("availabilityItems")
            if not isinstance(items, list):
                _invalid()
            item_count += len(items)
            if item_count > MAX_AVAILABILITY_ITEMS:
                _incomplete()

            for availability_payload in items:
                if not isinstance(availability_payload, dict):
                    _invalid()
                status = availability_payload.get("status")
                if not isinstance(status, str) or not status:
                    _invalid()
                kind = _STATUS.get(status, "incomplete")
                start = _parse_datetime(availability_payload.get("startDateTime"))
                end = _parse_datetime(availability_payload.get("endDateTime"))
                try:
                    interval = TimeInterval(start, end)
                except AvailabilityContractError:
                    _invalid()
                if not query.window.contains_interval(interval):
                    _invalid()
                identity = (staff_id, kind, interval.start, interval.end)
                if identity in seen_items:
                    _invalid()
                seen_items.add(identity)
                if kind == "incomplete":
                    response_incomplete = True
                    continue
                for other_kind, other_interval in evidence[staff_id]:
                    if other_kind != kind and _overlap(interval, other_interval):
                        _invalid()
                evidence[staff_id].append((kind, interval))
                if kind == "free":
                    free_intervals.append(
                        FreeInterval(query.scope, staff_id, interval)
                    )

        if returned_staff != requested_staff or response_incomplete:
            _incomplete()
        if free_intervals:
            return AvailabilityResult(
                query,
                AvailabilityStatus.AVAILABLE,
                observed_at,
                tuple(free_intervals),
            )
        return AvailabilityResult(
            query,
            AvailabilityStatus.NO_AVAILABILITY,
            observed_at,
        )
    except _DecodeFailure as error:
        return _failure_result(query, observed_at, error.category)


__all__ = ("MAX_AVAILABILITY_ITEMS", "decode_availability")
