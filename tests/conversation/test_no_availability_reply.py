"""Known absence of eligible slots must not sound like a provider failure."""
import asyncio
from unittest import mock

import pytest

from services.llm.llm_base import LLMResponse
from tests.conversation.test_safe_booking_flow import setup, final, take_pending


@pytest.mark.parametrize("language", ["en", "ar"])
@pytest.mark.parametrize("known_empty", [True, False])
def test_no_slots_and_unverified_read_have_distinct_caller_replies(language, known_empty):
    async def run():
        orchestrator, context, session = setup()
        context.language = language
        assert session.admit(final(1, "Check Hussam availability"))
        batch = take_pending(session)
        orchestrator._check_booking = mock.AsyncMock(return_value=(
            "NO_ELIGIBLE_SLOT: No eligible appointment was verified for this consultant."
            if known_empty else "AVAILABILITY_UNVERIFIED: Unable to verify availability."))
        orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
            "", finish_reason="tool_calls", tool_calls=[{
                "id": "check1", "name": "check_appointment",
                "arguments": '{"accountant_name":"Hussam Saadaldin","date_time":"2026-09-21 10:00"}'}]))
        reply = await orchestrator._get_verified_response(
            "call", session, batch.inputs[-1].text, batch.token)
        if known_empty:
            assert ("no available appointments" if language == "en" else "لا توجد مواعيد متاحة") in reply
            assert ("another accountant" if language == "en" else "محاسب آخر") in reply
            assert ("period I checked" if language == "en" else "الفترة التي تحققت منها") in reply
        else:
            assert ("couldn't read the calendar" if language == "en" else "لم أتمكن من قراءة التقويم") in reply
        assert "October" not in reply and "أكتوبر" not in reply and "13" not in reply
        assert context.pending_booking is None
    asyncio.run(run())
