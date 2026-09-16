import asyncio
import json
from unittest import mock

import pytest

from services.conversation.orchestrator import (
    ConversationContext,
    ConversationOrchestrator,
    ConversationState,
)
from services.llm.llm_base import LLMChunk, LLMResponse
from services.telephony.twilio_service import TwilioMediaStreamHandler


class FakeSTT:
    def __init__(self, language="en"):
        self.language = language
        self.reconnects = []
        self.resets = 0

    async def reconnect_with_language(self, language):
        self.reconnects.append(language)
        self.language = language
        return True

    async def reset_for_listening(self):
        self.resets += 1


class OneChunkStream:
    def __init__(self):
        self.sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.sent:
            raise StopAsyncIteration
        self.sent = True
        return b"A" * 160

    def cancel(self):
        pass

    async def aclose(self):
        pass


class FakeTTS:
    def create_stream(self, _request):
        return OneChunkStream()


class AutoMarkWebSocket:
    def __init__(self):
        self.handler = None

    async def send_json(self, message):
        if message["event"] == "mark":
            self.handler._on_mark(message)


class ToolLLM:
    def __init__(self, tool_language):
        self.tool_language = tool_language
        self.stream_calls = 0

    async def chat_with_tools(self, _request):
        return LLMResponse(
            content="I saved that preference.",
            tool_calls=[
                {
                    "name": "save_language_preference",
                    "arguments": json.dumps({"language": self.tool_language}),
                }
            ],
        )

    async def chat_stream(self, _request):
        self.stream_calls += 1
        yield LLMChunk(
            delta="Preference confirmed.",
            content="Preference confirmed.",
            is_final=True,
        )


def make_transcript_orchestrator(call_sid="call", language="en"):
    orchestrator = ConversationOrchestrator.__new__(ConversationOrchestrator)
    context = ConversationContext(call_sid, "+10000000000", language=language)
    context.state = ConversationState.LISTENING
    stt = FakeSTT(language)
    orchestrator._conversations = {call_sid: context}
    orchestrator._call_stt_instances = {call_sid: stt}
    orchestrator._call_state_locks = {call_sid: asyncio.Lock()}
    orchestrator._garbled_drop_count = {}
    orchestrator._echo_guard_until = {}
    orchestrator._barge_in_speech_start = {}
    orchestrator._get_ai_response = mock.AsyncMock(return_value="Understood")
    orchestrator._speak_to_caller = mock.AsyncMock()
    return orchestrator, context, stt


def make_tool_orchestrator(user_language="en", tool_language="ar"):
    orchestrator = ConversationOrchestrator.__new__(ConversationOrchestrator)
    context = ConversationContext("call", "+10000000000", language=user_language)
    orchestrator._conversations = {"call": context}
    orchestrator._system_prompt = "system"
    orchestrator.llm = ToolLLM(tool_language)
    orchestrator.caller_service = mock.Mock()
    orchestrator.caller_service.update_caller_language = mock.AsyncMock()
    return orchestrator, context


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "utterance",
    [
        "My email is naam@example.com",
        "Yalla, I need an appointment tomorrow",
        "Please ask Habibi Accounting about my file",
    ],
)
async def test_romanized_keyword_reaches_dialogue_unchanged_once(utterance):
    orchestrator, context, stt = make_transcript_orchestrator()

    await orchestrator.process_transcript("call", utterance, "en", True)

    orchestrator._get_ai_response.assert_awaited_once_with("call", utterance)
    assert context.conversation_history[0]["content"] == utterance
    assert context.language == "en"
    assert stt.reconnects == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caller_language", "stt_language", "reply", "render_language"),
    [
        ("auto", "", "The office is at 123 Main Street.", "en"),
        ("auto", "", "يمكنني مساعدتك اليوم.", "ar"),
        ("auto", "", "The address is Main Street، ثم نكمل.", "en"),
        ("ar", "ar", "The office is at 123 Main Street.", "en"),
        ("en", "en", "اسم المستشار هو حسام.", "ar"),
    ],
)
async def test_assistant_reply_language_never_changes_listening_language(
    caller_language, stt_language, reply, render_language
):
    orchestrator, context, stt = make_transcript_orchestrator(
        language=caller_language
    )
    del orchestrator._speak_to_caller
    stt.language = stt_language
    context.state = ConversationState.SPEAKING
    websocket = AutoMarkWebSocket()
    handler = TwilioMediaStreamHandler("call", "stream-call", websocket)
    websocket.handler = handler
    handler._is_streaming = True
    handler._is_connected = True
    orchestrator.tts = FakeTTS()
    orchestrator._twilio_handlers = {"call": handler}
    orchestrator._speech_setup_locks = {}
    orchestrator._active_speech = {}
    orchestrator._barge_in_reset_done = {"call": False}

    await orchestrator._speak_to_caller("call", reply, render_language)

    assert context.language == caller_language
    assert stt.language == stt_language
    assert stt.resets == 1
    await handler.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_input", "tool_language", "history"),
    [
        ("I need an English-speaking accountant", "ar", []),
        ("Tell me about tax documents", "ar", [{"role": "user", "content": "Arabic please"}]),
        ("Hello", "ar", [{"role": "assistant", "content": "سأتحدث بالعربية"}]),
        ("Arabic please", "fr", []),
        ("Arabic please", "en", []),
        ('"please speak Arabic"', "ar", []),
    ],
)
async def test_preference_tool_without_matching_current_request_is_rejected(
    user_input, tool_language, history
):
    orchestrator, context = make_tool_orchestrator(tool_language=tool_language)
    context.conversation_history.extend(history)

    response = await orchestrator._get_ai_response("call", user_input)

    orchestrator.caller_service.update_caller_language.assert_not_awaited()
    assert context.language == "en"
    assert "saved" not in response.casefold()
    assert response


@pytest.mark.asyncio
async def test_matching_current_request_authorizes_only_the_same_preference():
    orchestrator, context = make_tool_orchestrator(tool_language="ar")

    response = await orchestrator._get_ai_response("call", "Arabic please!")

    orchestrator.caller_service.update_caller_language.assert_awaited_once_with(
        "+10000000000", "ar"
    )
    assert context.language == "ar"
    assert response == "Preference confirmed."
    assert orchestrator.llm.stream_calls == 1


@pytest.mark.asyncio
async def test_non_string_tool_language_fails_closed_without_another_model_call():
    orchestrator, context = make_tool_orchestrator(tool_language=None)

    response = await orchestrator._get_ai_response("call", "Arabic please")

    orchestrator.caller_service.update_caller_language.assert_not_awaited()
    assert context.language == "en"
    assert response == "عذراً، هل تفضل العربية أم الإنجليزية؟"
    assert orchestrator.llm.stream_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_language", "stt_language", "utterance", "expected", "reconnects"),
    [
        ("en", "en", "ARABIC PLEASE!", "ar", ["ar"]),
        ("ar", "ar", "بالإنجليزي من فضلك؟", "en", ["en"]),
        ("ar", "ar", "عربي.", "ar", []),
        ("en", "en", "English please", "en", []),
    ],
)
async def test_explicit_request_updates_context_and_listening_hint_together(
    initial_language, stt_language, utterance, expected, reconnects
):
    orchestrator, context, stt = make_transcript_orchestrator(
        language=initial_language
    )
    stt.language = stt_language

    await orchestrator.process_transcript("call", utterance, stt_language, True)

    assert context.language == expected
    assert stt.language == expected
    assert stt.reconnects == reconnects
    orchestrator._get_ai_response.assert_awaited_once_with("call", utterance)


@pytest.mark.asyncio
async def test_two_calls_keep_independent_language_context_and_hints():
    orchestrator, context_a, stt_a = make_transcript_orchestrator(
        call_sid="A", language="en"
    )
    context_b = ConversationContext("B", "+10000000001", language="ar")
    context_b.state = ConversationState.LISTENING
    stt_b = FakeSTT("ar")
    orchestrator._conversations["B"] = context_b
    orchestrator._call_stt_instances["B"] = stt_b
    orchestrator._call_state_locks["B"] = asyncio.Lock()

    await orchestrator.process_transcript("A", "Arabic please", "en", True)
    await orchestrator.process_transcript("B", "English please", "ar", True)

    assert (context_a.language, stt_a.language) == ("ar", "ar")
    assert (context_b.language, stt_b.language) == ("en", "en")
    assert stt_a.reconnects == ["ar"]
    assert stt_b.reconnects == ["en"]
