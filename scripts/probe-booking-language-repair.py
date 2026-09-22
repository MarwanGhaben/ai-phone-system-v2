"""Offline canonical STT/language-switch and actual phone availability probe. No database, audio or provider traffic."""
import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from types import SimpleNamespace
from unittest import mock
from zoneinfo import ZoneInfo

os.environ.update({"SECRET_KEY": "synthetic", "DATABASE_URL": "postgresql://synthetic:synthetic@localhost/test",
    "TWILIO_ACCOUNT_SID": "ACsynthetic", "TWILIO_AUTH_TOKEN": "synthetic", "TWILIO_PHONE_NUMBER": "+14165550100",
    "DEEPGRAM_API_KEY": "synthetic", "ELEVENLABS_API_KEY": "synthetic", "OPENAI_API_KEY": "synthetic"})
from loguru import logger
logger.remove()
logging.disable(logging.CRITICAL)

from services.conversation import orchestrator as module
from services.conversation.booking_session import BookingSession
from services.stt.elevenlabs_stt_service import ElevenLabsSTT
from services.config.accountants_service import get_accountants_service
from services.calendar.service_facts import ServiceFacts, ServiceFactsRead
from services.scheduling.models import AvailabilityResult, AvailabilityStatus, FreeInterval, TimeInterval
from services.llm.llm_base import LLMResponse

NOW = datetime(2026, 9, 19, 13, tzinfo=timezone.utc)
MONDAY = datetime(2026, 9, 21, 10, tzinfo=ZoneInfo("America/Toronto"))


class Calendar:
    tenant_id, business_id = "synthetic-tenant", "synthetic-business"

    def __init__(self, starts):
        self.starts = starts
        self.staff_seen = []

    async def is_available(self):
        return True

    async def get_customer_appointments(self, phone):
        return []

    async def get_service_facts(self, scope):
        staff = tuple(get_accountants_service().resolve_booking_accountant(name).selection.staff_id
                      for name in ("Abdul", "Rami", "Hussam"))
        facts = ServiceFacts(scope, timedelta(minutes=30), timedelta(0), timedelta(0), staff,
            False, 1, False, timedelta(minutes=30), timedelta(0), timedelta(days=365),
            True, "bookWhenStaffAreFree", ())
        return ServiceFactsRead("verified", scope, NOW, facts)

    async def get_availability(self, query):
        self.staff_seen.append(query.staff_ids[0])
        intervals = tuple(FreeInterval(query.scope, query.staff_ids[0], TimeInterval(start, start + timedelta(minutes=30)))
                          for start in self.starts)
        return AvailabilityResult(query, AvailabilityStatus.AVAILABLE, NOW, intervals)


async def main():
    orchestrator = module.ConversationOrchestrator.__new__(module.ConversationOrchestrator)
    context = module.ConversationContext("synthetic-call", "+14165550100", language="ar")
    context.caller_name = "Synthetic Customer"
    handler = SimpleNamespace()
    orchestrator._verified_phone_booking_enabled = True
    orchestrator._conversations = {context.call_sid: context}
    orchestrator._twilio_handlers = {context.call_sid: handler}
    orchestrator._system_prompt = "Synthetic approved company context."
    orchestrator.llm = mock.Mock()
    orchestrator._speak_to_caller = mock.AsyncMock()
    session = BookingSession(context, handler, None, None,
        SimpleNamespace(tenant_id=Calendar.tenant_id, business_id=Calendar.business_id), clock=lambda: NOW)
    context.booking_session = session
    session.present = mock.AsyncMock(return_value=False)

    stt = ElevenLabsSTT(api_key="synthetic", language="auto")
    async def invoke(sequence, text, tool, arguments):
        class Socket:
            def __init__(self):
                self.events = [
                    {"message_type": "committed_transcript", "text": text},
                    {"message_type": "session_ended"},
                ]
            async def recv(self):
                return json.dumps(self.events.pop(0))
        stt._websocket = Socket()
        stt._transcript_queue = asyncio.Queue()
        stt._is_listening = True
        with mock.patch("datetime.datetime") as clock:
            clock.now.return_value = NOW
            await stt._receive_loop()
        event = stt._transcript_queue.get_nowait()
        assert event.language == "auto"
        assert session.admit(event)
        batch = session.turn.take_pending()
        session.active_token = batch.token
        session.current_text = text
        session.current_receipt = session.receipts[batch.inputs[-1].identity]
        context.add_user_message(text)
        orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse("", finish_reason="tool_calls",
            tool_calls=[{"id": f"call{sequence}", "name": tool, "arguments": json.dumps(arguments)}]))
        await orchestrator._process_verified_turn(context.call_sid, session, event, batch.token)
        return orchestrator._speak_to_caller.await_args.args[1]

    calendar = Calendar((MONDAY, MONDAY + timedelta(days=1)))
    with mock.patch.object(module, "business_now", return_value=NOW), mock.patch(
            "services.calendar.ms_bookings_service.get_calendar_service", return_value=calendar):
        reply = await invoke(1, "شو مواعيد عبدول يوم الاثنين", "check_appointment",
            {"accountant_name": "Abdul", "date_time": "2026-09-22 10:00"})
        assert "الاثنين" in reply and "الثلاثاء" not in reply and "العاشرة صباحاً" in reply
        assert session.proposals.current is None
        assert calendar.staff_seen[-1] == get_accountants_service().resolve_booking_accountant("Abdul").selection.staff_id
        reply = await invoke(2, "Rami", "search_appointments", {"accountant_name": "Rami", "date": "2026-09-22"})
        assert context.language == "ar" and "رامي" in reply and "الاثنين" in reply and "الثلاثاء" not in reply
        assert session.proposals.current is None
        calendar.starts = (MONDAY + timedelta(days=1),)
        reply = await invoke(3, "الاثنين", "search_appointments", {"accountant_name": "Abdul", "date": "2026-09-21"})
        assert "الاثنين" in reply and "الثلاثاء" not in reply and "لم أجد" in reply
        calendar.starts = (MONDAY,)
        await invoke(4, "الاثنين الساعة العاشرة", "check_appointment",
            {"accountant_name": "Abdul", "date_time": "2026-09-21 10:00"})
        assert session.proposals.current.candidate.interval.start == MONDAY
        assert session.proposals._presentation_completed is None
        await invoke(5, "لا شكرا", "confirm_appointment", {"confirm": False})
        assert session.proposals.current is None
        # New English speech must override Arabic history without metadata.
        reply = await invoke(6, "What times is Rami available Monday?", "search_appointments",
            {"accountant_name": "Rami", "date": "2026-09-21"})
        assert context.language == "en" and "Monday" in reply and "10:00" in reply
        assert not any("\u0621" <= c <= "\u064a" for c in reply)
        assert orchestrator._speak_to_caller.await_args.args[2] == "en"
        # A wrong model choice must never trap explicit requests in clarification.
        for sequence, text, language in (
            (7, "Please speak Arabic", "ar"),
            (8, "Can you please switch to English?", "en"),
        ):
            reply = await invoke(sequence, text, "save_language_preference", {"language": "ar"})
            assert context.language == language and "?" not in reply and "؟" not in reply
            assert orchestrator._speak_to_caller.await_args.args[2] == language
            orchestrator.llm.chat_with_tools.assert_not_awaited()
        reply = await invoke(9, "Rami", "search_appointments",
            {"accountant_name": "Rami", "date": "2026-09-21"})
        assert context.language == "en" and "Monday" in reply
        assert session.proposals.current is None and session.turn.unsettled_operation is None
    session.close()
    print("BOOKING_LANGUAGE_OFFLINE_CANONICAL_SWITCH_AVAILABILITY_OK")


if __name__ == "__main__":
    asyncio.run(main())
