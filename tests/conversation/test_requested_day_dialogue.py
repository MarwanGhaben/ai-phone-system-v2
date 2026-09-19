"""Owner regressions: requested weekdays and Arabic continuity in the enabled route."""
import asyncio
from datetime import datetime
from datetime import date
import json
from unittest import mock
from zoneinfo import ZoneInfo
import pytest

from services.llm.llm_base import LLMResponse
from services.conversation import orchestrator as module
from tests.conversation.test_safe_booking_flow import setup, final, take_pending
from tests.conversation.test_booking_policy_integration import fake_check, checked
from tests.conversation.test_booking_policy_integration import NOW, service_wire
from services.conversation.booking_dialogue import requested_day, turn_language, has_clock


def test_monday_check_does_not_offer_tuesday_without_request():
    async def run():
        orchestrator, calendar = fake_check(slot=datetime(2026, 9, 22, 10, tzinfo=ZoneInfo("America/Toronto")))
        response = await checked(orchestrator, calendar, "2026-09-21 11:00")
        assert "Tuesday" not in response, response
        assert "Verified alternatives:" not in response, response
    asyncio.run(run())


@pytest.mark.parametrize("text,day", [
    ("Abdul Monday", date(2026, 9, 21)), ("عبدول يوم الإثنين", date(2026, 9, 21)),
    ("يوم الاتنين", date(2026, 9, 21)), ("الثلاثاء", date(2026, 9, 22)),
    ("tomorrow", date(2026, 9, 20)), ("بكرة", date(2026, 9, 20)),
    ("بعد بكرة", date(2026, 9, 21)), ("day after tomorrow", date(2026, 9, 21)),
    ("2026-09-21", date(2026, 9, 21)), ("٢٠٢٦-٠٩-٢١", date(2026, 9, 21)),
    ("at ten", None), ("Abdul", None),
    ("May I check Monday", date(2026, 9, 21)),
    ("شو عنده بالاثنين", date(2026, 9, 21)), ("موعد للإثنين", date(2026, 9, 21)),
    ("Monday October 12", date(2026, 10, 12)), ("الاثنين 21 سبتمبر", date(2026, 9, 21)),
    ("الاثنين الحادي والعشرين من سبتمبر", date(2026, 9, 21)),
])
def test_caller_day_reference(text, day):
    result = requested_day(text, datetime(2026, 9, 19, 12, tzinfo=ZoneInfo("America/Toronto")))
    assert result.day == day and not result.ambiguous


@pytest.mark.parametrize("text", ["Monday or Tuesday", "الاثنين ولا الثلاثاء", "2026-02-30", "Monday next week",
    "Tuesday October 12", "not Monday", "غير الاثنين", "last Monday", "21/09"])
def test_unclear_days_are_not_guessed(text):
    result = requested_day(text, NOW)
    assert result.ambiguous and result.day is None


def test_day_resolution_uses_toronto_even_at_utc_midnight():
    instant = datetime(2026, 9, 20, 1, tzinfo=ZoneInfo("UTC"))
    assert requested_day("today", instant).day == date(2026, 9, 19)


@pytest.mark.parametrize("text,expected", [
    ("Monday at Abdul's office", False), ("الاثنين الساعة المناسبة", False),
    ("Monday at ten", True), ("Monday 10 AM", True), ("Monday at noon", True),
    ("الاثنين الساعة عشرة", True), ("الاثنين الساعة العاشرة", True),
    ("2026-09-21", False), ("مواعيد الاثنين", False),
])
def test_day_request_does_not_invent_a_clock(text, expected):
    assert has_clock(text) is expected


@pytest.mark.parametrize("text,reported,current,expected", [
    ("Abdul", "en", "ar", "ar"), ("Monday", "en", "ar", "ar"),
    ("10 AM", "en", "ar", "ar"), ("yes", "en", "ar", "ar"),
    ("test@example.com", "en", "ar", "ar"),
    ("Please speak English", "en", "ar", "en"),
    ("What times are available", "en", "ar", "en"),
    ("شو عندكم مواعيد", "en", "en", "ar"),
    ("احكي انجليزي", "ar", "ar", "en"),
])
def test_language_changes_require_a_request_or_substantive_speech(text, reported, current, expected):
    assert turn_language(text, reported, current) == expected


@pytest.mark.parametrize("name,staff", [
    ("Abdul", "91142a93-3c60-4127-bdbd-efde7fd61b75"),
    ("Rami", "d3d0fa56-b0ef-4267-a15d-85e3af42db38"),
])
def test_actual_day_search_then_time_choice_preserves_staff_day_and_readonly(name, staff):
    async def run():
        monday = datetime(2026, 9, 21, 10, tzinfo=ZoneInfo("America/Toronto"))
        _, calendar = fake_check(slot=monday, service=service_wire(staff))
        orchestrator, context, session = setup()
        context.language = "ar"
        session.clock = lambda: NOW
        session.present = mock.AsyncMock(return_value=False)
        context.caller_name = "مروان"
        async def invoke(sequence, text, tool, arguments):
            context.add_user_message(text)
            assert session.admit(final(sequence, text, when=NOW))
            batch = take_pending(session)
            session.active_token = batch.token
            orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
                "", finish_reason="tool_calls", tool_calls=[{"id": f"call{sequence}", "name": tool,
                                                           "arguments": json.dumps(arguments)}]))
            return await orchestrator._get_verified_response("call", session, text, batch.token)
        with mock.patch("services.calendar.ms_bookings_service.get_calendar_service", return_value=calendar), \
                mock.patch.object(module, "business_now", return_value=NOW):
            response = await invoke(1, "شو مواعيد يوم الاثنين", "search_appointments",
                                    {"accountant_name": name, "date": "2026-09-22"})
            assert "الاثنين" in response and "العاشرة صباحاً" in response
            assert "الثلاثاء" not in response
            assert session.proposals.current is None and context.pending_booking is None
            calendar.create_booking.assert_not_called()
            assert calendar.get_availability.await_args.args[0].staff_ids == (staff,)
            await invoke(2, "الساعة العاشرة", "check_appointment",
                         {"accountant_name": name, "date_time": "2026-09-22 10:00"})
            assert session.proposals.current is None
            await invoke(3, "الاثنين الساعة العاشرة", "check_appointment",
                         {"accountant_name": name, "date_time": "2026-09-21 10:00"})
            proposal = session.proposals.current
            assert proposal is not None
            assert proposal.candidate.interval.start == monday
            assert proposal.candidate.staff_id == staff
            calendar.create_booking.assert_not_called()
            await invoke(4, "لا شكرا", "confirm_appointment", {"confirm": False})
            assert session.proposals.current is None
            calendar.create_booking.assert_not_called()
    asyncio.run(run())


def test_model_tuesday_cannot_override_callers_explicit_monday():
    async def run():
        orchestrator, context, session = setup()
        text = "شو المواعيد المتاحة عند عبدول يوم الاثنين"
        context.language = "ar"
        context.add_user_message(text)
        assert session.admit(final(1, text))
        token = take_pending(session).token
        orchestrator._check_booking = mock.AsyncMock(return_value="NO_ELIGIBLE_SLOT: none")
        orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
            "", finish_reason="tool_calls", tool_calls=[{"id": "check", "name": "check_appointment",
                "arguments": '{"accountant_name":"Abdul","date_time":"2026-09-22 10:00"}'}]))
        with mock.patch.object(module, "business_now", return_value=datetime(2026, 9, 19, 12, tzinfo=ZoneInfo("America/Toronto"))):
            reply = await orchestrator._get_verified_response("call", session, text, token)
        orchestrator._check_booking.assert_awaited_once_with(
            "call", {"accountant_name": "Abdul", "search_date": "2026-09-21"})
        assert "الثلاثاء" not in reply
    asyncio.run(run())


def test_latin_consultant_name_cannot_switch_arabic_call_to_english():
    async def run():
        orchestrator, context, session = setup()
        context.language = "ar"
        utterance = final(1, "Abdul")
        assert session.admit(utterance)
        token = take_pending(session).token
        orchestrator._get_verified_response = mock.AsyncMock(return_value="حاضر، سأتحقق من المواعيد.")
        orchestrator._speak_to_caller = mock.AsyncMock()
        await orchestrator._process_verified_turn("call", session, utterance, token)
        assert context.language == "ar"
        assert orchestrator._speak_to_caller.await_args.args[2] == "ar"
    asyncio.run(run())


def test_arabic_call_does_not_speak_english_model_holding_response():
    async def run():
        orchestrator, context, session = setup()
        context.language = "ar"
        text = "شوف عبدول الاثنين"
        context.add_user_message(text)
        assert session.admit(final(1, text))
        token = take_pending(session).token
        orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
            "I will check with the staff member.", finish_reason="stop"))
        reply = await orchestrator._get_verified_response("call", session, text, token)
        assert "staff member" not in reply and "I will" not in reply
    asyncio.run(run())


def test_invalid_language_reply_gets_one_bounded_correction_then_real_tool():
    async def run():
        orchestrator, context, session = setup()
        context.language = "ar"
        text = "مواعيد الاثنين"
        assert session.admit(final(1, text))
        token = take_pending(session).token
        orchestrator._check_booking = mock.AsyncMock(return_value="DAY_AVAILABILITY: الاثنين الساعة العاشرة صباحاً")
        orchestrator.llm.chat_with_tools = mock.AsyncMock(side_effect=[
            LLMResponse("I will check with the staff member.", finish_reason="stop"),
            LLMResponse("", finish_reason="tool_calls", tool_calls=[{"id": "search", "name": "search_appointments",
                "arguments": '{"accountant_name":"Abdul","date":"2026-09-21"}'}])])
        with mock.patch.object(module, "business_now", return_value=NOW):
            reply = await orchestrator._get_verified_response("call", session, text, token)
        assert "العاشرة صباحاً" in reply and "staff" not in reply
        assert orchestrator.llm.chat_with_tools.await_count == 2
        orchestrator._check_booking.assert_awaited_once()
        prompt = orchestrator.llm.chat_with_tools.await_args.args[0].messages[0].content
        assert "Current response language: ar" in prompt
        assert "Monday" in prompt and "2026-09-21" in prompt
    asyncio.run(run())


def test_ambiguous_correction_clears_old_day_until_caller_clarifies():
    async def run():
        orchestrator, context, session = setup()
        session.requested_day = date(2026, 9, 21)
        orchestrator._check_booking = mock.AsyncMock()
        for sequence, text in enumerate(("Monday or Tuesday", "at ten"), 1):
            assert session.admit(final(sequence, text))
            token = take_pending(session).token
            orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
                "", finish_reason="tool_calls", tool_calls=[{"id": f"x{sequence}", "name": "check_appointment",
                    "arguments": '{"accountant_name":"Abdul","date_time":"2026-09-21 10:00"}'}]))
            await orchestrator._get_verified_response("call", session, text, token)
        orchestrator._check_booking.assert_not_awaited()
        assert session.requested_day is None and session.requested_day_unclear
    asyncio.run(run())


@pytest.mark.parametrize("text,expected", [("نعم", True), ("yes", True), ("no", False),
    ("لا", False), ("yes but Tuesday", None), ("نعم بس الثلاثاء", None)])
def test_short_bilingual_answers_are_not_language_switches_or_qualified_approval(text, expected):
    from services.llm.tool_protocol import caller_approval
    assert caller_approval(text, "ar") is expected
