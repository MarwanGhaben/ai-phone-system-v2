"""Offline synthetic Graph transport/policy probe; no provider or database calls."""
import asyncio
from datetime import datetime, timezone
import os
from unittest.mock import patch

os.environ.update({
    "SECRET_KEY": "synthetic", "DATABASE_URL": "postgresql://synthetic:synthetic@localhost/test",
    "TWILIO_ACCOUNT_SID": "ACsynthetic", "TWILIO_AUTH_TOKEN": "synthetic",
    "TWILIO_PHONE_NUMBER": "+14165550100", "DEEPGRAM_API_KEY": "synthetic",
    "ELEVENLABS_API_KEY": "synthetic", "OPENAI_API_KEY": "synthetic",
})

import httpx
from services.calendar.ms_bookings_service import MSBookingsService
from services.scheduling.booking_check import load_booking_policy, make_booking_query, assess_booking
from services.scheduling.models import CalendarScope

NOW = datetime(2026, 9, 21, 13, tzinfo=timezone.utc)
SCOPE = CalendarScope("tenant", "business", "service")


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is None else NOW.astimezone(tz)


def service_wire(closed):
    return {
        "id": "service", "defaultDuration": "PT30M", "preBuffer": "PT0S", "postBuffer": "PT0S",
        "staffMemberIds": ["staff"], "isLocationOnline": False,
        "maximumAttendeesCount": 1, "isHiddenFromCustomers": False,
        "schedulingPolicy": {"allowStaffSelection": True, "timeSlotInterval": "PT30M",
            "minimumLeadTime": "PT0S", "maximumAdvance": "P365D",
            "generalAvailability": {"availabilityType": "bookWhenStaffAreFree", "businessHours": []},
            "customAvailabilities": [{"availabilityType": "notBookable",
                "startDate": "2026-09-21" if closed else "2023-12-22",
                "endDate": "2026-09-21" if closed else "2024-01-01",
                "businessHours": [{"day": day, "timeSlots": []} for day in
                    ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")]}]},
    }


async def main():
    for scenario, expected in (("free", "available"), ("busy", "no_eligible"),
                               ("closed", "no_eligible"), ("bad_zone", "availability_unverified")):
        seen = []

        def respond(request):
            seen.append(request.url.path)
            if request.url.host == "login.microsoftonline.com":
                return httpx.Response(200, json={"access_token": "synthetic", "expires_in": 600})
            if request.method == "GET" and request.url.path.endswith("/services/service"):
                return httpx.Response(200, json=service_wire(scenario == "closed"))
            assert request.method == "POST" and request.url.path.endswith("/getStaffAvailability")
            zone = "invalid-zone" if scenario == "bad_zone" else "(UTC) Coordinated Universal Time"
            return httpx.Response(200, json={"staffAvailabilityItem": [{"staffId": "staff",
                "availabilityItems": [{"status": "Busy" if scenario == "busy" else "Available",
                    "startDateTime": {"dateTime": "2026-09-21T14:00:00", "timeZone": zone},
                    "endDateTime": {"dateTime": "2026-09-21T14:30:00", "timeZone": zone}}]}]})

        calendar = MSBookingsService({"tenant_id": "tenant", "business_id": "business",
                                     "client_id": "client", "client_secret": "synthetic"})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            calendar._http_client = client
            snapshot = load_booking_policy(SCOPE, "staff", NOW)
            assert snapshot is not None
            query = make_booking_query(snapshot, "staff", NOW)
            with patch("services.calendar.ms_bookings_service.datetime", Clock):
                facts = await calendar.get_service_facts(SCOPE)
                availability = await calendar.get_availability(query)
            assessment = assess_booking(snapshot, query, facts, availability, NOW)
            assert facts.status == "verified"
            assert assessment.status == expected
            assert len(assessment.candidates) == (1 if scenario == "free" else 0)
            assert len(seen) == 3
    print("GRAPH_PARSER_OFFLINE_TRANSPORT_POLICY_OK")


if __name__ == "__main__":
    asyncio.run(main())
