"""Actual enabled orchestrator methods with synthetic caller and model traffic."""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

from services.conversation.events import UtteranceIdentity
from services.conversation.orchestrator import ConversationContext, ConversationOrchestrator
from services.conversation.booking_session import BookingSession
from services.llm.llm_base import LLMResponse
from services.scheduling.models import AvailabilityQuery, CalendarScope, TimeInterval
from services.scheduling.policy import AppointmentCandidate
from services.scheduling.proposals import CustomerSnapshot
from services.scheduling.booking_service import BookingOutcome, BookingResult


def setup():
    orchestrator = ConversationOrchestrator.__new__(ConversationOrchestrator)
    orchestrator._verified_phone_booking_enabled = True
    orchestrator._conversations = {}
    orchestrator._twilio_handlers = {}
    orchestrator._active_speech = {}
    orchestrator._system_prompt = "Approved Flexible Accounting company context."
    orchestrator.llm = mock.Mock()
    orchestrator.caller_service = mock.Mock()
    context = ConversationContext("call", "+14165550100", language="en")
    context.caller_name = "Synthetic Customer"
    handler = SimpleNamespace()
    orchestrator._conversations["call"] = context
    orchestrator._twilio_handlers["call"] = handler
    mutation = SimpleNamespace(tenant_id="tenant", business_id="business")
    session = BookingSession(context, handler, object(), object(), mutation,
        owner_check=lambda: orchestrator._conversations.get("call") is context)
    context.booking_session = session
    return orchestrator, context, session


def final(sequence, text, *, when=None, epoch="epoch"):
    return SimpleNamespace(text=text, language="en", is_final=True,
        utterance_id=UtteranceIdentity("stt", epoch, sequence),
        received_at=when or datetime.now(timezone.utc))


def take_pending(session):
    """Mirror the worker's receipt/text binding for direct session tests."""
    batch = session.turn.take_pending()
    if batch.inputs:
        latest = batch.inputs[-1]
        session.current_receipt = session.receipts.get(latest.identity)
        session.current_text = latest.text
    return batch


def candidate():
    now = datetime.now(timezone.utc)
    start = now + timedelta(hours=3)
    scope = CalendarScope("tenant", "business", "service")
    query = AvailabilityQuery(scope, TimeInterval(now, start + timedelta(days=1)),
                              ("staff",), "request")
    return AppointmentCandidate(scope, "staff", TimeInterval(
        start, start + timedelta(minutes=30)), "policy", query, now)


def test_canonical_identity_and_commit_time_are_required_before_processing():
    orchestrator, context, session = setup()
    missing = final(1, "yes")
    missing.received_at = None
    assert not session.admit(missing)
    assert session.turn.generation == 0
    session.unresolved = False
    assert session.admit(final(2, "same answer"))
    assert session.admit(final(3, "same answer"))
    assert not session.admit(final(3, "same answer"))
    assert session.turn.generation == 2
    assert len(take_pending(session).inputs) == 2
    assert orchestrator._conversations["call"] is context


def test_early_yes_and_interrupted_playback_cannot_authorize_create():
    async def run():
        orchestrator, _, session = setup()
        assert session.admit(final(1, "check Tuesday"))
        token = take_pending(session).token
        offered = session.offer(candidate(), CustomerSnapshot(
            "Synthetic Customer", "+14165550100", ""),
            "Synthetic Consultant", "appointment", "1901 Banff Ave, Ottawa",
            "America/Toronto", timedelta(0), timedelta(0))
        assert offered.token is not None
        early = final(2, "yes")
        assert session.admit(early)
        new_token = take_pending(session).token

        async def failed_speech(call_sid, text, language, *, on_booking_playback=None, on_booking_playback_complete=None):
            assert on_booking_playback(SimpleNamespace(session_id="stream", generation=1))
            return False

        orchestrator._speak_to_caller = failed_speech
        assert not await session.present(orchestrator, "call", "en", new_token)
        assert await session.confirm("yes", "en", new_token) is None
        assert session.turn.unsettled_operation is None
    asyncio.run(run())


def test_later_presented_yes_dispatches_once_and_replays_are_rejected():
    async def run():
        orchestrator, _, session = setup()
        assert session.admit(final(1, "check Tuesday"))
        first = take_pending(session).token
        offered = session.offer(candidate(), CustomerSnapshot(
            "Synthetic Customer", "+14165550100", ""),
            "Synthetic Consultant", "appointment", "1901 Banff Ave, Ottawa",
            "America/Toronto", timedelta(0), timedelta(0))
        assert offered.token is not None

        async def speech(call_sid, text, language, *, on_booking_playback=None, on_booking_playback_complete=None):
            readback = on_booking_playback(SimpleNamespace(session_id="stream", generation=1))
            assert "Synthetic Customer" in readback
            assert "Synthetic Consultant" in readback
            assert "+14165550100" in readback
            on_booking_playback_complete(SimpleNamespace(session_id="stream", generation=1), True)
            return True

        orchestrator._speak_to_caller = speech
        assert await session.present(orchestrator, "call", "en", first)
        await asyncio.sleep(0.02)
        assert session.admit(final(2, "yes", when=datetime.now(timezone.utc)))
        second = take_pending(session).token
        with mock.patch("services.conversation.booking_session.BookingService") as service:
            service.return_value.create = mock.AsyncMock(return_value=
                BookingResult(BookingOutcome.VERIFIED))
            assert await session.confirm("yes", "en", second) is BookingOutcome.VERIFIED
            assert await session.confirm("yes", "en", second) is None
            service.return_value.create.assert_awaited_once()
        assert not session.admit(final(2, "yes"))
    asyncio.run(run())


def test_orchestrator_rejects_partial_tool_and_preserves_linked_result():
    async def run():
        orchestrator, context, session = setup()
        assert session.admit(final(1, "check Tuesday"))
        batch = take_pending(session)
        session.active_token = batch.token
        context.add_user_message("check Tuesday")
        orchestrator._check_booking = mock.AsyncMock(return_value="NO_ELIGIBLE_SLOT")
        orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
            "Booked!", tool_calls=[{"id": "call_1", "name": "check_appointment",
                                    "arguments": '{"accountant_name":'}]))
        answer = await orchestrator._get_verified_response(
            "call", session, "check Tuesday", batch.token)
        assert "clarify" in answer
        orchestrator._check_booking.assert_not_awaited()
        orchestrator.llm.chat_with_tools.return_value = LLMResponse(
            "Booked!", tool_calls=[{"id": "call_2", "name": "check_appointment",
                "arguments": '{"accountant_name":"Rami Kahwaji","date_time":"2026-09-21 10:00"}'}])
        await orchestrator._get_verified_response(
            "call", session, "check Tuesday", batch.token)
        assert context.conversation_history[-2]["tool_calls"][0]["id"] == "call_2"
        assert context.conversation_history[-1]["role"] == "tool"
        assert context.conversation_history[-1]["tool_call_id"] == "call_2"
    asyncio.run(run())


def test_explicitly_incomplete_completion_cannot_dispatch_a_tool():
    async def run():
        orchestrator, context, session = setup()
        assert session.admit(final(1, "Cancel my appointment"))
        batch = take_pending(session)
        session.active_token = batch.token
        context.add_user_message(session.current_text)
        orchestrator._cancel_booking = mock.AsyncMock(return_value="CANCEL_ABORTED: synthetic")
        orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
            "", finish_reason="length", tool_calls=[{
                "id": "cancel1", "name": "cancel_booking",
                "arguments": '{"confirm_cancel":true}'
            }]))

        answer = await orchestrator._get_verified_response(
            "call", session, session.current_text, batch.token)

        assert "clarify" in answer.lower() or "repeat" in answer.lower()
        orchestrator._cancel_booking.assert_not_awaited()

    asyncio.run(run())


def test_enabled_prompt_keeps_company_caller_consultant_and_fresh_date_context():
    async def run():
        orchestrator, context, session = setup()
        orchestrator._system_prompt = "Approved company fact: synthetic-office-reference-47."
        context.caller_name = "Current Caller"
        assert session.admit(final(1, "Where is your office?"))
        batch = take_pending(session)
        context.add_user_message(batch.inputs[-1].text)
        orchestrator.llm.chat_with_tools = mock.AsyncMock(
            return_value=LLMResponse("The office information is in our approved context."))

        await orchestrator._get_verified_response(
            "call", session, batch.inputs[-1].text, batch.token)

        prompt = orchestrator.llm.chat_with_tools.call_args.args[0].messages[0].content
        assert "synthetic-office-reference-47" in prompt
        assert "Current Caller" in prompt
        assert "Toronto local date" in prompt
        assert "Rami Kahwaji" in prompt
        assert "1901 Banff Ave, Ottawa, ON K1V 7W9" in prompt
        assert "NO minimum-notice rule" not in prompt

    asyncio.run(run())


def test_arabic_tool_outcomes_remain_arabic_and_keep_verified_alternatives():
    async def run():
        orchestrator, context, session = setup()
        orchestrator._system_prompt = "Approved company context."
        context.language = "ar"
        assert session.admit(final(1, "اسمي مروان"))
        batch = take_pending(session)
        context.add_user_message(batch.inputs[-1].text)
        orchestrator.caller_service.register_caller = mock.AsyncMock()
        orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
            "", finish_reason="tool_calls", tool_calls=[{
                "id": "name1", "name": "register_caller_name",
                "arguments": '{"caller_name":"مروان"}'
            }]))
        name_reply = await orchestrator._get_verified_response(
            "call", session, batch.inputs[-1].text, batch.token)
        assert "شكراً" in name_reply and "Thank you" not in name_reply

        assert session.admit(final(2, "افحص موعداً آخر"))
        second = take_pending(session)
        context.add_user_message(second.inputs[-1].text)
        orchestrator._check_booking = mock.AsyncMock(return_value=(
            "NO_ELIGIBLE_SLOT: The requested time is not verified as eligible. "
            "Verified alternatives: Monday at 10:00, Monday at 10:30."
        ))
        orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
            "", finish_reason="tool_calls", tool_calls=[{
                "id": "check1", "name": "check_appointment",
                "arguments": '{"accountant_name":"Rami Kahwaji",'
                             '"date_time":"2026-09-21 10:00"}'
            }]))
        slot_reply = await orchestrator._get_verified_response(
            "call", session, second.inputs[-1].text, second.token)
        assert "Monday at 10:00" in slot_reply
        assert any("\u0600" <= char <= "\u06ff" for char in slot_reply)

    asyncio.run(run())


def test_enabled_faq_keeps_approved_facts_and_caller_language():
    async def invoke(language, question, answer):
        orchestrator, context, session = setup()
        context.language = language
        orchestrator._system_prompt = (
            "Approved company/contact/accountant fact: synthetic-office-reference-47.")
        assert session.admit(final(1, question))
        batch = take_pending(session)
        context.add_user_message(batch.inputs[-1].text)
        orchestrator.llm.chat_with_tools = mock.AsyncMock(
            return_value=LLMResponse(answer, finish_reason="stop"))

        reply = await orchestrator._get_verified_response(
            "call", session, question, batch.token)

        prompt = orchestrator.llm.chat_with_tools.call_args.args[0].messages[0].content
        assert "synthetic-office-reference-47" in prompt
        assert "Toronto local date and time" in prompt
        assert reply == answer

    async def run():
        await invoke(
            "en",
            "What is tomorrow's date, and how can your accountants help me?",
            "Our approved company and contact information can help answer that.",
        )
        await invoke(
            "ar",
            "ما تاريخ الغد، وكيف يمكن للمحاسبين مساعدتي؟",
            "يمكنني الإجابة من معلومات الشركة والاتصال المعتمدة.",
        )

    asyncio.run(run())


def test_free_model_booking_success_claims_remain_blocked_in_both_languages():
    async def invoke(language, question, claim):
        orchestrator, context, session = setup()
        context.language = language
        assert session.admit(final(1, question))
        batch = take_pending(session)
        context.add_user_message(batch.inputs[-1].text)
        orchestrator.llm.chat_with_tools = mock.AsyncMock(
            return_value=LLMResponse(claim, finish_reason="stop"))
        reply = await orchestrator._get_verified_response(
            "call", session, question, batch.token)
        assert reply != claim

    async def run():
        await invoke("en", "Did that work?", "Your appointment is booked.")
        await invoke("ar", "هل تم ذلك؟", "تم حجز الموعد.")

    asyncio.run(run())


def test_arabic_lookup_cancel_and_incomplete_outcomes_are_localized():
    async def invoke(tool, result, target):
        orchestrator, context, session = setup()
        context.language = "ar"
        assert session.admit(final(1, "طلب عربي"))
        batch = take_pending(session)
        context.add_user_message(batch.inputs[-1].text)
        setattr(orchestrator, target, mock.AsyncMock(return_value=result))
        orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
            "", finish_reason="tool_calls", tool_calls=[tool]))
        reply = await orchestrator._get_verified_response(
            "call", session, batch.inputs[-1].text, batch.token)
        assert any("\u0600" <= char <= "\u06ff" for char in reply)
        return reply

    async def run():
        lookup = await invoke(
            {"id": "lookup1", "name": "lookup_my_bookings", "arguments": "{}"},
            "NO_APPOINTMENTS_FOUND: none",
            "_lookup_my_bookings",
        )
        assert "upcoming appointment" not in lookup

        cancelled = await invoke(
            {"id": "cancel1", "name": "cancel_booking",
             "arguments": '{"confirm_cancel":false}'},
            "CANCEL_ABORTED: synthetic",
            "_cancel_booking",
        )
        assert "haven't cancelled" not in cancelled

        orchestrator, context, session = setup()
        context.language = "ar"
        assert session.admit(final(1, "طلب غير مكتمل"))
        batch = take_pending(session)
        context.add_user_message(batch.inputs[-1].text)
        orchestrator._cancel_booking = mock.AsyncMock()
        orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
            "", finish_reason="length", tool_calls=[{
                "id": "cancel2", "name": "cancel_booking",
                "arguments": '{"confirm_cancel":true}'
            }]))
        incomplete = await orchestrator._get_verified_response(
            "call", session, batch.inputs[-1].text, batch.token)
        assert any("\u0600" <= char <= "\u06ff" for char in incomplete)
        orchestrator._cancel_booking.assert_not_awaited()

    asyncio.run(run())


def test_enabled_availability_check_keeps_owned_acknowledgement():
    async def run():
        orchestrator, context, session = setup()
        assert session.admit(final(1, "Check Rami Monday at ten"))
        batch = take_pending(session)
        context.add_user_message(batch.inputs[-1].text)
        orchestrator._call_state_locks = {"call": asyncio.Lock()}
        orchestrator._acknowledge_availability_search = mock.AsyncMock(return_value=True)
        orchestrator._check_booking = mock.AsyncMock(return_value="NO_ELIGIBLE_SLOT: none")
        orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
            "", finish_reason="tool_calls", tool_calls=[{
                "id": "check1", "name": "check_appointment",
                "arguments": '{"accountant_name":"Rami Kahwaji",'
                             '"date_time":"2026-09-21 10:00"}'
            }]))

        with mock.patch("services.conversation.orchestrator.business_now",
                        return_value=datetime(2026, 9, 19, 13, tzinfo=timezone.utc)):
            await orchestrator._get_verified_response(
                "call", session, batch.inputs[-1].text, batch.token)

        orchestrator._acknowledge_availability_search.assert_awaited_once_with(
            "call", context, session.handler,
            orchestrator._call_state_locks["call"])
        orchestrator._check_booking.assert_awaited_once()

    asyncio.run(run())


def test_delayed_model_result_is_rejected_after_new_caller_final():
    async def run():
        orchestrator, context, session = setup()
        assert session.admit(final(1, "check Tuesday"))
        first = take_pending(session).token
        context.add_user_message("check Tuesday")
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed(_request):
            entered.set()
            await release.wait()
            return LLMResponse("The appointment is booked", tool_calls=[{
                "id": "call_1", "name": "check_appointment",
                "arguments": '{"accountant_name":"Rami Kahwaji","date_time":"2026-09-21 10:00"}'
            }])

        orchestrator.llm.chat_with_tools = delayed
        orchestrator._check_booking = mock.AsyncMock()
        task = asyncio.create_task(orchestrator._get_verified_response(
            "call", session, "check Tuesday", first))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert session.admit(final(2, "No, change the time"))
            release.set()
            await asyncio.wait_for(task, 2)
            orchestrator._check_booking.assert_not_awaited()
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


def test_bounded_overload_and_reused_call_sid_revoke_old_authority():
    orchestrator, context, session = setup()
    for number in range(1, 17):
        assert session.admit(final(number, "same text"))
    assert not session.admit(final(17, "same text"))
    assert session.turn.overloaded and session.unresolved
    replacement = ConversationContext("call", "+14165550100", language="en")
    orchestrator._conversations["call"] = replacement
    assert not session.owns(context, session.handler)
    assert session.scope.session_id != BookingSession(
        replacement, SimpleNamespace(), object(), object(),
        SimpleNamespace(tenant_id="tenant", business_id="business")).scope.session_id


def test_unknown_create_outcome_blocks_another_dispatch():
    async def run():
        orchestrator, _, session = setup()
        assert session.admit(final(1, "check Tuesday"))
        first = take_pending(session).token
        session.offer(candidate(), CustomerSnapshot(
            "Synthetic Customer", "+14165550100", ""),
            "Synthetic Consultant", "appointment", "1901 Banff Ave, Ottawa",
            "America/Toronto", timedelta(0), timedelta(0))

        async def speech(call_sid, text, language, *, on_booking_playback=None, on_booking_playback_complete=None):
            assert on_booking_playback(SimpleNamespace(session_id="stream", generation=1))
            on_booking_playback_complete(SimpleNamespace(session_id="stream", generation=1), True)
            return True

        orchestrator._speak_to_caller = speech
        assert await session.present(orchestrator, "call", "en", first)
        await asyncio.sleep(0.02)
        assert session.admit(final(2, "yes"))
        second = take_pending(session).token
        with mock.patch("services.conversation.booking_session.BookingService") as service:
            service.return_value.create = mock.AsyncMock(return_value=
                BookingResult(BookingOutcome.PENDING))
            assert await session.confirm("yes", "en", second) is BookingOutcome.PENDING
            assert session.unresolved
            assert await session.confirm("yes", "en", second) is None
            service.return_value.create.assert_awaited_once()
    asyncio.run(run())


def test_unknown_outcome_keeps_safe_followup_dialogue_available():
    async def run():
        orchestrator, _, session = setup()
        processed = []
        helper_called = asyncio.Event()

        async def process(_call_sid, _session, utterance, _token):
            processed.append(utterance.text)

        async def helper(*_args):
            helper_called.set()

        orchestrator._process_verified_turn = process
        orchestrator._verified_help = helper
        session.unresolved = True
        worker = asyncio.create_task(session.run(orchestrator, "call"))
        try:
            assert session.admit(final(1, "Please transfer me to a person"))
            await asyncio.wait_for(helper_called.wait(), 2)
            assert processed == ["Please transfer me to a person"]
        finally:
            session.close()
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(run())


def test_unknown_outcome_blocks_repeat_mutation_but_allows_transfer_and_goodbye():
    async def invoke(text, tool):
        orchestrator, context, session = setup()
        session.unresolved = True
        assert session.admit(final(1, text))
        batch = take_pending(session)
        context.add_user_message(text)
        orchestrator._confirm_booking = mock.AsyncMock(return_value="BOOKING_SUCCESS")
        orchestrator._handle_transfer = mock.AsyncMock()
        orchestrator.end_call = mock.AsyncMock()
        orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
            "", finish_reason="tool_calls", tool_calls=[tool]))
        answer = await orchestrator._get_verified_response(
            "call", session, text, batch.token)
        return orchestrator, answer

    async def run():
        repeat, repeat_answer = await invoke("Yes, book it again", {
            "id": "confirm1", "name": "confirm_appointment",
            "arguments": '{"confirm":true}'})
        repeat._confirm_booking.assert_not_awaited()
        assert "confirmed" not in repeat_answer.casefold()

        transfer, transfer_answer = await invoke("Please transfer me", {
            "id": "transfer1", "name": "transfer_to_human",
            "arguments": '{"reason":"caller requested human help"}'})
        transfer._handle_transfer.assert_awaited_once()
        assert transfer_answer is not None

        goodbye, goodbye_answer = await invoke("Goodbye", {
            "id": "end1", "name": "end_call",
            "arguments": '{"reason":"caller finished"}'})
        goodbye.end_call.assert_awaited_once()
        assert goodbye_answer is not None

    asyncio.run(run())


def test_correction_revokes_presented_tuple_before_model_work():
    async def run():
        orchestrator, _, session = setup()
        assert session.admit(final(1, "check Tuesday"))
        first = take_pending(session).token
        session.offer(candidate(), CustomerSnapshot(
            "Synthetic Customer", "+14165550100", ""),
            "Synthetic Consultant", "appointment", "1901 Banff Ave, Ottawa",
            "America/Toronto", timedelta(0), timedelta(0))

        async def speech(call_sid, text, language, *, on_booking_playback=None, on_booking_playback_complete=None):
            assert on_booking_playback(SimpleNamespace(session_id="stream", generation=1))
            on_booking_playback_complete(SimpleNamespace(session_id="stream", generation=1), True)
            return True

        orchestrator._speak_to_caller = speech
        assert await session.present(orchestrator, "call", "en", first)
        assert session.admit(final(2, "yes, but change the time"))
        assert session.proposals.current is None
        second = take_pending(session).token
        with mock.patch("services.conversation.booking_session.BookingService") as service:
            assert await session.confirm("yes, but change the time", "en", second) is None
            service.assert_not_called()
    asyncio.run(run())


def test_queued_approval_before_playback_finishes_cannot_become_later_consent():
    async def run():
        orchestrator, _, session = setup()
        assert session.admit(final(1, "check Tuesday"))
        first = take_pending(session).token
        session.offer(candidate(), CustomerSnapshot(
            "Synthetic Customer", "+14165550100", ""),
            "Synthetic Consultant", "appointment", "1901 Banff Ave, Ottawa",
            "America/Toronto", timedelta(0), timedelta(0))
        started, release = asyncio.Event(), asyncio.Event()

        async def speech(call_sid, text, language, *, on_booking_playback=None, on_booking_playback_complete=None):
            assert on_booking_playback(SimpleNamespace(session_id="stream", generation=1))
            started.set()
            await release.wait()
            on_booking_playback_complete(SimpleNamespace(session_id="stream", generation=1), True)
            return True

        orchestrator._speak_to_caller = speech
        task = asyncio.create_task(session.present(orchestrator, "call", "en", first))
        try:
            await asyncio.wait_for(started.wait(), 2)
            assert session.admit(final(2, "yes"))
            second = take_pending(session).token
            release.set()
            assert not await asyncio.wait_for(task, 2)
            with mock.patch("services.conversation.booking_session.BookingService") as service:
                assert await session.confirm("yes", "en", second) is None
                service.assert_not_called()
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


def test_expired_presented_proposal_cannot_dispatch():
    async def run():
        orchestrator, _, session = setup()
        assert session.admit(final(1, "check Tuesday"))
        first = take_pending(session).token
        offered = session.offer(candidate(), CustomerSnapshot(
            "Synthetic Customer", "+14165550100", ""),
            "Synthetic Consultant", "appointment", "1901 Banff Ave, Ottawa",
            "America/Toronto", timedelta(0), timedelta(0))

        async def speech(call_sid, text, language, *, on_booking_playback=None, on_booking_playback_complete=None):
            assert on_booking_playback(SimpleNamespace(session_id="stream", generation=1))
            on_booking_playback_complete(SimpleNamespace(session_id="stream", generation=1), True)
            return True

        orchestrator._speak_to_caller = speech
        assert await session.present(orchestrator, "call", "en", first)
        expired = offered.proposal.expires_at + timedelta(seconds=1)
        session.clock = lambda: expired
        assert session.admit(final(2, "yes", when=expired))
        second = take_pending(session).token
        with mock.patch("services.conversation.booking_session.BookingService") as service:
            assert await session.confirm("yes", "en", second) is None
            service.assert_not_called()
    asyncio.run(run())
