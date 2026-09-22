"""Offline duplicate-availability and single-attempt create diagnostic probe."""
import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from unittest.mock import patch

os.environ.update({"SECRET_KEY": "synthetic", "DATABASE_URL": "postgresql://synthetic:synthetic@localhost/test",
    "TWILIO_ACCOUNT_SID": "ACsynthetic", "TWILIO_AUTH_TOKEN": "synthetic", "TWILIO_PHONE_NUMBER": "+14165550100",
    "DEEPGRAM_API_KEY": "synthetic", "ELEVENLABS_API_KEY": "synthetic", "OPENAI_API_KEY": "synthetic"})

import httpx
from loguru import logger

from services.calendar import ms_bookings_service as calendar_module
from services.calendar.booking_mutations import BookingMutationRequest, GraphBookingMutations
from services.calendar.contracts import decode_availability
from services.scheduling.booking_check import assess_booking, load_booking_policy, make_booking_query
from services.scheduling.models import CalendarScope

logger.remove()
logging.disable(logging.CRITICAL)
NOW = datetime(2026, 9, 22, 13, tzinfo=timezone.utc)
START = NOW + timedelta(hours=1)
SCOPE = CalendarScope("tenant", "business", "service")


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is None else NOW.astimezone(tz)


def interval(status, start=START):
    return {"status": status,
        "startDateTime": {"dateTime": start.isoformat(), "timeZone": "(UTC) Coordinated Universal Time"},
        "endDateTime": {"dateTime": (start + timedelta(minutes=30)).isoformat(),
                        "timeZone": "(UTC) Coordinated Universal Time"}}


def service_wire():
    return {"id": "service", "defaultDuration": "PT30M", "preBuffer": "PT0S", "postBuffer": "PT0S",
        "staffMemberIds": ["staff"], "isLocationOnline": False, "maximumAttendeesCount": 1,
        "isHiddenFromCustomers": False, "schedulingPolicy": {
            "allowStaffSelection": True, "timeSlotInterval": "PT30M", "minimumLeadTime": "PT0S",
            "maximumAdvance": "P365D", "generalAvailability": {
                "availabilityType": "bookWhenStaffAreFree", "businessHours": []},
            "customAvailabilities": []}}


async def availability_checks():
    items = [interval("available"), interval("busy", START + timedelta(hours=1)),
             interval("busy", START + timedelta(hours=1))]
    wire = {"staffAvailabilityItem": [{"staffId": "staff", "availabilityItems": items}]}
    calls = []
    def respond(request):
        calls.append((request.method, request.url.path))
        if request.url.host == "login.microsoftonline.com":
            return httpx.Response(200, json={"access_token": "synthetic", "expires_in": 600})
        if request.method == "GET" and request.url.path.endswith("/services/service"):
            return httpx.Response(200, json=service_wire())
        assert request.method == "POST" and request.url.path.endswith("/getStaffAvailability")
        return httpx.Response(200, json=wire)
    calendar = calendar_module.MSBookingsService({"tenant_id": "tenant", "business_id": "business",
        "client_id": "client", "client_secret": "synthetic"})
    snapshot = load_booking_policy(SCOPE, "staff", NOW)
    query = make_booking_query(snapshot, "staff", NOW)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        calendar._http_client = client
        with patch.object(calendar_module, "datetime", Clock):
            facts = await calendar.get_service_facts(SCOPE)
            evidence = await calendar.get_availability(query)
        assessment = assess_booking(snapshot, query, facts, evidence, NOW)
        assert assessment.status == "available"
        assert len(assessment.candidates) == 1
        assert assessment.candidates[0].interval.start == START
    assert len(calls) == 3
    items.append(interval("busy"))
    assert decode_availability(wire, query, NOW).status.value == "invalid_response"
    items[:] = [interval("outOfOffice"), interval("outOfOffice")]
    assert decode_availability(wire, query, NOW).status.value == "no_availability"
    items[:] = [interval("available"), interval("available")]
    assert len(decode_availability(wire, query, NOW).intervals) == 1
    wire["@odata.nextLink"] = "https://invalid.example/continuation"
    assert decode_availability(wire, query, NOW).status.value == "incomplete"


async def create_checks():
    request = BookingMutationRequest("tenant", "business", "service", "staff", START,
        START + timedelta(minutes=30), "PRIVATE-CUSTOMER", "+14165550100",
        "synthetic@example.invalid", "Synthetic office", 0, 0)
    for status in (201, 400, 403, 429, 503):
        messages, methods = [], []
        def respond(req):
            if req.url.host == "login.microsoftonline.com":
                return httpx.Response(200, json={"access_token": "PRIVATE-TOKEN"})
            assert req.method == "POST" and req.url.path.endswith("/appointments")
            methods.append(req.method)
            assert json.loads(req.content)["staffMemberIds"] == ["staff"]
            return httpx.Response(status, json={"id": "PRIVATE-PROVIDER-ID"} if status == 201 else {
                "error": {"code": "BadRequest", "message": "PRIVATE-MESSAGE"}})
        sink = logger.add(lambda entry: messages.append(str(entry)), format="{message}")
        try:
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                adapter = GraphBookingMutations(tenant_id="tenant", business_id="business",
                    client_id="client", client_secret="PRIVATE-SECRET", client=client)
                prepared = await adapter.prepare(request)
                result = await adapter.create_once(prepared)
                assert result.status == ("receipt" if status == 201 else "uncertain")
                async def deny():
                    return False
                assert (await adapter.create_once(prepared, before_send=deny)).status == "not_sent"
        finally:
            logger.remove(sink)
        assert methods == ["POST"]
        output = "".join(messages)
        assert "PRIVATE-" not in output
        assert f'"http_status": {status}' in output
        assert '"reason": "not_sent"' in output


async def main():
    await availability_checks()
    await create_checks()
    print("BOOKING_INCIDENT_OFFLINE_AVAILABILITY_DIAGNOSTICS_OK")


if __name__ == "__main__":
    asyncio.run(main())
