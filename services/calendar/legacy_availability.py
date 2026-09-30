"""Keep a failed legacy calendar read distinct from a completed empty search."""
from services.calendar.business_time import parse_graph_datetime


class AvailabilityReadError(RuntimeError):
    """A calendar read did not establish availability; never means staff are busy."""

    def __init__(self):
        super().__init__('Calendar availability could not be verified')


def _collection(owner, names):
    if not isinstance(owner, dict):
        raise AvailabilityReadError()
    if any('nextlink' in key.lower() for key in owner if isinstance(key, str)):
        raise AvailabilityReadError()
    present = [name for name in names if name in owner]
    if len(present) != 1 or not isinstance(owner[present[0]], list):
        raise AvailabilityReadError()
    return owner[present[0]]


def completed_staff_availability(payload, staff_ids):
    """Require the requested staff and complete, recognized availability items."""
    staff_rows = _collection(payload, ('value', 'staffAvailabilityItem'))
    if not staff_ids or not staff_rows:
        raise AvailabilityReadError()
    seen = set()
    for row in staff_rows:
        items = _collection(row, ('availabilityItems',))
        staff_id = row.get('staffId')
        if not isinstance(staff_id, str) or staff_id not in staff_ids or staff_id in seen:
            raise AvailabilityReadError()
        seen.add(staff_id)
        for item in items:
            _validate_item(item)
    if seen != set(staff_ids):
        raise AvailabilityReadError()
    return staff_rows


def _validate_item(item):
    if not isinstance(item, dict) or item.get('status') not in ('available', 'busy', 'outOfOffice'):
        raise AvailabilityReadError()
    try:
        start = parse_graph_datetime(item['startDateTime']['dateTime'])
        end = parse_graph_datetime(item['endDateTime']['dateTime'])
        # Graph can include equal endpoints alongside usable intervals. They
        # occupy no time and the slot generator emits nothing for them.
        if end < start:
            raise AvailabilityReadError()
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        raise AvailabilityReadError() from None
