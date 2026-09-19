"""Exercise the actual proposal and caller paths, not just a formatter."""
import asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from unittest import mock

import pytest

from services.scheduling.proposals import CustomerSnapshot
from tests.scheduling.test_proposals import session, offer, NOW
from tests.conversation.test_booking_policy_integration import fake_check, checked
from tests.conversation.test_safe_booking_flow import setup, final, take_pending
from services.llm.llm_base import LLMResponse
from services.scheduling import spoken_arabic


def test_actual_arabic_proposal_has_spoken_date_clock_service_and_zone():
    state = session()
    offered = offer(state, customer=CustomerSnapshot("مروان", "+14165550100", ""),
                    service_display="appointment")
    started = state.begin_presentation(offered.token, "ar", "playback", "mark", NOW)
    for forbidden in ("appointment", "America/Toronto", "2030-01-08", "10:00", "10:30"):
        assert forbidden not in started.text
    for required in ("الثلاثاء", "يناير", "العاشرة صباحاً", "العاشرة والنصف صباحاً", "حسام", "نصف ساعة"):
        assert required in started.text
    assert state.current is offered.proposal
    assert "1901 Banff Ave" in started.text
    assert "+14165550100" in started.text


def test_actual_arabic_availability_alternatives_are_not_english_dates():
    async def run():
        orchestrator, calendar = fake_check()
        orchestrator._conversations["call"].language = "ar"
        response = await checked(orchestrator, calendar, "2026-09-18 11:00")
        assert "Verified alternatives:" in response  # internal tool marker stays stable
        alternatives = response.split("Verified alternatives:", 1)[1].split(". Ask", 1)[0]
        assert "الجمعة" in alternatives and "سبتمبر" in alternatives and "العاشرة صباحاً" in alternatives
        assert "Friday" not in alternatives and "AM" not in alternatives and "10:00" not in alternatives
        assert orchestrator._conversations["call"].pending_booking is None
    asyncio.run(run())


@pytest.mark.parametrize("hour,minute,expected", [
    (0, 0, "الثانية عشرة بعد منتصف الليل"), (12, 0, "الثانية عشرة ظهراً"),
    (10, 0, "العاشرة صباحاً"), (10, 15, "العاشرة والربع صباحاً"),
    (10, 30, "العاشرة والنصف صباحاً"), (10, 45, "العاشرة وخمس وأربعون دقيقة صباحاً"),
    (11, 5, "الحادية عشرة وخمس دقائق صباحاً"), (15, 30, "الثالثة والنصف عصراً"),
    (23, 59, "الحادية عشرة وتسع وخمسون دقيقة مساءً"),
])
def test_spoken_clock_boundaries(hour, minute, expected):
    assert spoken_arabic.clock(datetime(2026, 9, 21, hour, minute)) == expected


@pytest.mark.parametrize("utc,expected", [
    (datetime(2026, 9, 21, 14, tzinfo=timezone.utc), "العاشرة صباحاً"),
    (datetime(2026, 1, 21, 15, tzinfo=timezone.utc), "العاشرة صباحاً"),
    (datetime(2026, 9, 22, 1, tzinfo=timezone.utc), "التاسعة مساءً"),
])
def test_conversion_happens_before_speech(utc, expected):
    local = utc.astimezone(ZoneInfo("America/Toronto"))
    assert expected in spoken_arabic.slot(local)
    assert spoken_arabic.date(local) in spoken_arabic.slot(local)


def test_speech_preserves_seconds_and_unknown_names():
    assert "واحد فاصلة صفر صفر واحد ثانية" in spoken_arabic.clock(datetime(2026, 9, 21, 10, 0, 1, 1000))
    assert spoken_arabic.consultant("Unconfigured Name") == "Unconfigured Name"
    assert spoken_arabic.service("Special service") == "Special service"


@pytest.mark.parametrize("start,expected", [
    (datetime(2026, 9, 21, 14, tzinfo=timezone.utc), "العاشرة صباحاً"),
    (datetime(2026, 9, 21, 14), "لم أتمكن من التحقق"),
    (None, "لم أتمكن من التحقق"),
])
def test_actual_arabic_lookup_uses_aware_time_not_english_display(start, expected):
    async def run():
        orchestrator, context, booking = setup()
        context.language = "ar"
        context.found_appointments = [{"staff_name": "Rami Kahwaji", "start_time": start,
                                       "formatted_time": "Monday at 10:00 AM"}]
        assert booking.admit(final(1, "ما هي مواعيدي"))
        batch = take_pending(booking)
        orchestrator._lookup_my_bookings = mock.AsyncMock(return_value="APPOINTMENTS_FOUND: synthetic")
        orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
            "", finish_reason="tool_calls", tool_calls=[{"id": "lookup", "name": "lookup_my_bookings",
                "arguments": '{}'}]))
        reply = await orchestrator._get_verified_response("call", booking, "ما هي مواعيدي", batch.token)
        assert expected in reply and "رامي قهوجي" in reply
        assert "Monday" not in reply and "AM" not in reply
    asyncio.run(run())


def test_actual_arabic_success_does_not_read_machine_date_or_english_service():
    async def run():
        orchestrator, context, booking = setup()
        context.language = "ar"
        state = session()
        proposal = offer(state, service_display="appointment").proposal
        booking.approval = mock.Mock(proposal=proposal)
        assert booking.admit(final(1, "نعم"))
        batch = take_pending(booking)
        orchestrator._confirm_booking = mock.AsyncMock(return_value="BOOKING_SUCCESS")
        orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
            "", finish_reason="tool_calls", tool_calls=[{"id": "confirm", "name": "confirm_appointment",
                "arguments": '{"confirm":true}'}]))
        reply = await orchestrator._get_verified_response("call", booking, "نعم", batch.token)
        assert "الثلاثاء" in reply and "العاشرة صباحاً" in reply and "حسام" in reply
        assert "appointment" not in reply and "2030-01-08" not in reply and "10:00" not in reply
    asyncio.run(run())
