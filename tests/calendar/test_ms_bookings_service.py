from datetime import datetime
from types import MethodType
from zoneinfo import ZoneInfo

import pytest

from services.calendar.ms_bookings_service import MSBookingsService


def make_calendar_service() -> MSBookingsService:
    service = MSBookingsService.__new__(MSBookingsService)
    service.business_id = "business-1"
    service._staff_cache = {}
    return service


@pytest.mark.asyncio
async def test_available_slots_convert_utc_to_toronto() -> None:
    service = make_calendar_service()

    async def is_available(self) -> bool:
        return True

    async def get_staff_members(self):
        return []

    async def make_request(self, method, endpoint, **kwargs):
        return {
            "value": [
                {
                    "staffId": "staff-1",
                    "availabilityItems": [
                        {
                            "status": "available",
                            "startDateTime": {"dateTime": "2026-06-18T14:00:00Z"},
                            "endDateTime": {"dateTime": "2026-06-18T15:00:00Z"},
                        }
                    ],
                }
            ]
        }

    service.is_available = MethodType(is_available, service)
    service.get_staff_members = MethodType(get_staff_members, service)
    service._make_request = MethodType(make_request, service)

    slots = await service.get_available_slots("service-1", "staff-1")

    assert slots[0].start_time == datetime(
        2026, 6, 18, 10, 0, tzinfo=ZoneInfo("America/Toronto")
    )


@pytest.mark.asyncio
async def test_customer_appointment_keeps_toronto_timezone() -> None:
    service = make_calendar_service()

    async def is_available(self) -> bool:
        return True

    async def make_request(self, method, endpoint, **kwargs):
        return {
            "value": [
                {
                    "id": "appointment-1",
                    "customerPhone": "+1 416 555 0100",
                    "startDateTime": {"dateTime": "2026-11-02T15:00:00Z"},
                    "staffMemberIds": [],
                }
            ]
        }

    service.is_available = MethodType(is_available, service)
    service._make_request = MethodType(make_request, service)

    appointments = await service.get_customer_appointments("+14165550100")

    assert appointments[0]["start_time"] == datetime(
        2026, 11, 2, 10, 0, tzinfo=ZoneInfo("America/Toronto")
    )


@pytest.mark.asyncio
async def test_invalid_customer_phone_does_not_query_appointments() -> None:
    service = make_calendar_service()
    request_count = 0

    async def is_available(self) -> bool:
        return True

    async def make_request(self, method, endpoint, **kwargs):
        nonlocal request_count
        request_count += 1
        return {"value": []}

    service.is_available = MethodType(is_available, service)
    service._make_request = MethodType(make_request, service)

    assert await service.get_customer_appointments("unknown") == []
    assert request_count == 0


@pytest.mark.asyncio
async def test_cancellation_rejects_appointment_owned_by_another_caller() -> None:
    service = make_calendar_service()
    methods: list[str] = []

    async def is_available(self) -> bool:
        return True

    async def make_request(self, method, endpoint, **kwargs):
        methods.append(method)
        if method == "GET":
            return {"id": "appointment-1", "customerPhone": "+1 647 555 0100"}
        return {"deleted": True}

    service.is_available = MethodType(is_available, service)
    service._make_request = MethodType(make_request, service)

    cancelled = await service.cancel_customer_appointment(
        "appointment-1", "+1 416 555 0100"
    )

    assert cancelled is False
    assert methods == ["GET"]


@pytest.mark.asyncio
async def test_cancellation_deletes_appointment_owned_by_caller() -> None:
    service = make_calendar_service()
    methods: list[str] = []

    async def is_available(self) -> bool:
        return True

    async def make_request(self, method, endpoint, **kwargs):
        methods.append(method)
        if method == "GET":
            return {"id": "appointment-1", "customerPhone": "+1 416 555 0100"}
        return {"deleted": True}

    service.is_available = MethodType(is_available, service)
    service._make_request = MethodType(make_request, service)

    cancelled = await service.cancel_customer_appointment(
        "appointment-1", "+1 416 555 0100"
    )

    assert cancelled is True
    assert methods == ["GET", "DELETE"]
