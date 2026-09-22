"""Caller language must survive canonical STT fallback and an Arabic history."""
import asyncio
import json
from datetime import datetime
from unittest import mock
from zoneinfo import ZoneInfo

import pytest

from services.conversation.booking_dialogue import plain_reply_ok, turn_language
from services.conversation.language_policy import explicit_language_request
from services.conversation import orchestrator as module
from services.llm.llm_base import LLMResponse
from services.stt.elevenlabs_stt_service import ElevenLabsSTT
from tests.conversation.test_safe_booking_flow import setup, take_pending
from tests.stt.test_utterance_events import commit, run_events
from tests.conversation.test_booking_policy_integration import fake_check, NOW, service_wire


@pytest.mark.parametrize("reported", [None, "auto", "ar", "en"])
@pytest.mark.parametrize("text", [
    "What times is Rami available tomorrow?",
    "I would like an appointment with Rami",
    "My name is Marwan",
])
def test_substantive_english_does_not_depend_on_provider_language_label(text, reported):
    assert turn_language(text, reported, "ar") == "en"


@pytest.mark.parametrize("text", [
    "Can you please switch to English?", "Please switch to English",
    "Could you speak in English please?", "I want you to speak English",
    "Please respond to me in English", "No, English please",
])
def test_unambiguous_conversational_switch_requests(text):
    assert explicit_language_request(text) == "en"


def test_english_reply_guard_rejects_arabic_sentences():
    assert not plain_reply_ok("هل تفضل اللغة الإنجليزية أم العربية؟", "en")


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", ["auto", "ar"])
async def test_actual_canonical_adapter_to_enabled_turn_uses_english(configured):
    stt = ElevenLabsSTT(api_key="synthetic", language=configured)
    results = await run_events(stt, [commit("What times is Rami available tomorrow?")])
    assert len(results) == 1
    assert results[0].language in ("auto", "ar")
    orchestrator, context, session = setup()
    context.language = "ar"
    orchestrator._get_verified_response = mock.AsyncMock(return_value="I can check Rami's available times.")
    orchestrator._speak_to_caller = mock.AsyncMock()
    assert session.admit(results[0])
    batch = take_pending(session)
    await orchestrator._process_verified_turn("call", session, batch.inputs[-1], batch.token)
    assert context.language == "en"
    assert orchestrator._speak_to_caller.await_args.args[2] == "en"
    session.close()


@pytest.mark.parametrize("text", [
    "Do not switch to English", "Don't speak Arabic", "He said speak English",
    'She said "English please"', "Can Rami speak English?", "Arabic or English?",
    "Please switch to English or Arabic", "I speak Arabic at home",
])
def test_ambiguous_quoted_and_negated_requests_are_not_explicit_switches(text):
    assert explicit_language_request(text) is None


@pytest.mark.asyncio
async def test_canonical_english_reaches_real_availability_and_english_playback():
    staff = "d3d0fa56-b0ef-4267-a15d-85e3af42db38"
    monday = datetime(2026, 9, 21, 10, tzinfo=ZoneInfo("America/Toronto"))
    _, calendar = fake_check(slot=monday, service=service_wire(staff))
    orchestrator, context, session = setup()
    context.language = "ar"
    session.clock = lambda: NOW
    orchestrator._speak_to_caller = mock.AsyncMock()
    text = "What times is Rami available Monday?"
    context.add_user_message(text)
    stt = ElevenLabsSTT(api_key="synthetic", language="auto")
    with mock.patch("datetime.datetime") as clock:
        clock.now.return_value = NOW
        results = await run_events(stt, [commit(text)])
    orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
        "", finish_reason="tool_calls", tool_calls=[{
            "id": "search", "name": "search_appointments", "arguments": json.dumps({
                "accountant_name": "Rami", "date": "2026-09-21"})}]))
    assert session.admit(results[0])
    batch = take_pending(session)
    session.active_token = batch.token
    try:
        with mock.patch("services.calendar.ms_bookings_service.get_calendar_service", return_value=calendar), \
                mock.patch.object(module, "business_now", return_value=NOW):
            await orchestrator._process_verified_turn("call", session, batch.inputs[-1], batch.token)
        spoken = orchestrator._speak_to_caller.await_args_list
        assert spoken and all(call.args[2] == "en" for call in spoken)
        assert "Monday" in spoken[-1].args[1] and "10:00" in spoken[-1].args[1]
        assert not any("\u0621" <= char <= "\u064a" for char in spoken[-1].args[1])
        assert calendar.get_availability.await_args.args[0].staff_ids == (staff,)
        assert session.proposals.current is None
        calendar.create_booking.assert_not_called()
    finally:
        session.close()


@pytest.mark.asyncio
async def test_both_directions_and_short_answers_preserve_call_language():
    orchestrator, context, session = setup()
    other_orchestrator, other_context, other_session = setup()
    other_context.language = "ar"
    stt = ElevenLabsSTT(api_key="synthetic", language="auto")
    texts = ["Please speak Arabic", "Rami", "Could you please switch to English?", "yes"]
    results = await run_events(stt, [commit(text) for text in texts])
    orchestrator._speak_to_caller = mock.AsyncMock()
    orchestrator.caller_service.update_caller_language = mock.AsyncMock()
    try:
        for utterance, language in zip(results, ["ar", "ar", "en", "en"]):
            orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
                "كيف أساعدك؟" if language == "ar" else "How can I help?", finish_reason="stop"))
            context.add_user_message(utterance.text)
            assert session.admit(utterance)
            batch = take_pending(session)
            await orchestrator._process_verified_turn("call", session, batch.inputs[-1], batch.token)
            assert context.language == language
            assert orchestrator._speak_to_caller.await_args.args[2] == language
        assert other_context.language == "ar"
        orchestrator.caller_service.update_caller_language.assert_not_awaited()
    finally:
        session.close()
        other_session.close()


@pytest.mark.asyncio
async def test_explicit_persistent_preference_keeps_existing_save_route():
    orchestrator, context, session = setup()
    context.language = "ar"
    text = "Always speak to me in English"
    stt = ElevenLabsSTT(api_key="synthetic", language="auto")
    utterance = (await run_events(stt, [commit(text)]))[0]
    context.add_user_message(text)
    orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
        "", finish_reason="tool_calls", tool_calls=[{"id": "save", "name": "save_language_preference",
            "arguments": '{"language":"en"}'}]))
    orchestrator.caller_service.update_caller_language = mock.AsyncMock()
    orchestrator._speak_to_caller = mock.AsyncMock()
    try:
        assert session.admit(utterance)
        batch = take_pending(session)
        await orchestrator._process_verified_turn("call", session, batch.inputs[-1], batch.token)
        orchestrator.caller_service.update_caller_language.assert_awaited_once_with(context.phone_number, "en")
        assert context.language == "en"
    finally:
        session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("corrected", [True, False])
async def test_arabic_model_reply_is_not_spoken_on_an_english_turn(corrected):
    orchestrator, context, session = setup()
    context.language = "ar"
    stt = ElevenLabsSTT(api_key="synthetic", language="auto")
    utterance = (await run_events(stt, [commit("What is the company name?")]))[0]
    context.add_user_message(utterance.text)
    arabic = LLMResponse("اسم الشركة فليكسيبل للمحاسبة", finish_reason="stop")
    orchestrator.llm.chat_with_tools = mock.AsyncMock(side_effect=[arabic,
        LLMResponse("Our company is Flexible Accounting.", finish_reason="stop") if corrected else arabic])
    orchestrator._speak_to_caller = mock.AsyncMock()
    try:
        assert session.admit(utterance)
        batch = take_pending(session)
        await orchestrator._process_verified_turn("call", session, batch.inputs[-1], batch.token)
        spoken = orchestrator._speak_to_caller.await_args.args
        assert spoken[2] == "en"
        assert not any("\u0621" <= char <= "\u064a" for char in spoken[1])
        assert orchestrator.llm.chat_with_tools.await_count == 2
    finally:
        session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["stale", "closed", "replaced"])
async def test_invalid_owner_cannot_change_language_or_speak(state):
    orchestrator, context, session = setup()
    context.language = "ar"
    stt = ElevenLabsSTT(api_key="synthetic", language="auto")
    results = await run_events(stt, [commit("English please"), commit("مرحبا")])
    assert session.admit(results[0])
    batch = take_pending(session)
    if state == "stale":
        assert session.admit(results[1])
    elif state == "closed":
        session.close()
    else:
        orchestrator._conversations["call"] = object()
    orchestrator._speak_to_caller = mock.AsyncMock()
    orchestrator.llm.chat_with_tools = mock.AsyncMock()
    try:
        await orchestrator._process_verified_turn("call", session, batch.inputs[-1], batch.token)
        assert context.language == "ar"
        orchestrator._speak_to_caller.assert_not_awaited()
        orchestrator.llm.chat_with_tools.assert_not_awaited()
    finally:
        session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Can you please switch to English?", "switch to English", "English"])
async def test_switch_is_acknowledged_without_model_language_question_or_database_write(text):
    stt = ElevenLabsSTT(api_key="synthetic", language="ar")
    results = await run_events(stt, [commit(text)])
    orchestrator, context, session = setup()
    context.language = "ar"
    context.add_assistant_message("هل تفضل اللغة الإنجليزية أم العربية؟")
    context.add_user_message(text)
    orchestrator.llm.chat_with_tools = mock.AsyncMock(return_value=LLMResponse(
        "", finish_reason="tool_calls", tool_calls=[{
            "id": "language", "name": "save_language_preference",
            "arguments": '{"language":"ar"}'}]))
    orchestrator.caller_service.update_caller_language = mock.AsyncMock()
    orchestrator._speak_to_caller = mock.AsyncMock()
    assert session.admit(results[0])
    batch = take_pending(session)
    await orchestrator._process_verified_turn("call", session, batch.inputs[-1], batch.token)
    assert context.language == "en"
    spoken = orchestrator._speak_to_caller.await_args.args
    assert spoken[2] == "en"
    assert "English" in spoken[1] and "?" not in spoken[1]
    assert "Arabic" not in spoken[1]
    orchestrator.llm.chat_with_tools.assert_not_awaited()
    orchestrator.caller_service.update_caller_language.assert_not_awaited()
    assert session.proposals.current is None
    assert session.turn.unsettled_operation is None
    session.close()
