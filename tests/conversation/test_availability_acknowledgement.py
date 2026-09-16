"""T034-C regressions for owned availability-search acknowledgement speech."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from services.conversation import orchestrator as orchestrator_module
from services.conversation.orchestrator import (
    ConversationContext,
    ConversationOrchestrator,
    ConversationState,
)
from services.llm.llm_base import LLMChunk, LLMResponse
from services.telephony.twilio_service import TwilioMediaStreamHandler


ENGLISH_ACK = "Certainly, I'll check availability."
ARABIC_ACK = "حاضر، سأتحقق من المواعيد المتاحة."
VALID_ARGUMENTS = {
    "client_type": "individual",
    "accountant_name": "Rami Kahwaji",
    "date_time": "2099-01-08 10:00",
    "customer_name": "Synthetic Caller",
    "customer_email": "",
}
_DEFAULT_ARGUMENTS = object()


@pytest.mark.asyncio
async def test_interrupted_ack_invalidates_previous_booking_proposal():
    orchestrator, context = make_orchestrator()
    context.pending_booking = {"staff_id": "previous-staff", "confirmed": False}
    orchestrator._speak_to_caller = mock.AsyncMock(return_value=False)

    response = await orchestrator._get_ai_response("call", "Use Rami instead")

    assert response is orchestrator_module._ABORTED_RESPONSE
    assert context.pending_booking is None
    orchestrator._check_booking.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_end_while_fallback_awaits_eof_aborts_response():
    orchestrator, context = make_orchestrator()
    orchestrator._speak_to_caller = mock.AsyncMock(return_value=True)
    waiting_for_eof = asyncio.Event()
    release_eof = asyncio.Event()
    requests = 0

    async def response_stream(_request):
        nonlocal requests
        requests += 1
        if requests == 1:
            raise RuntimeError("synthetic provider failure")
        yield LLMChunk(delta="Old response", content="Old response", is_final=False)
        waiting_for_eof.set()
        await release_eof.wait()

    orchestrator.llm.chat_stream = response_stream
    history_before = list(context.conversation_history)
    task = asyncio.create_task(orchestrator._get_ai_response("call", "check"))
    try:
        await asyncio.wait_for(waiting_for_eof.wait(), 1)
        context.state = ConversationState.ENDED
        orchestrator._twilio_handlers["call"] = object()
        release_eof.set()
        assert await asyncio.wait_for(task, 1) is orchestrator_module._ABORTED_RESPONSE
        assert context.conversation_history == history_before
        assert context.state == ConversationState.ENDED
    finally:
        release_eof.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_ack_exception_after_call_end_does_not_enter_fallback():
    orchestrator, context = make_orchestrator()

    async def failing_speech(*_args):
        context.state = ConversationState.ENDED
        raise RuntimeError("synthetic acknowledgement failure")

    orchestrator._speak_to_caller = failing_speech
    response = await orchestrator._get_ai_response("call", "check")
    assert response is orchestrator_module._ABORTED_RESPONSE
    assert not orchestrator.llm.stream_requests
    orchestrator._check_booking.assert_not_awaited()


class ToolLLM:
    def __init__(
        self, arguments=_DEFAULT_ARGUMENTS, *, results=None,
        final="Synthetic final response", tool_name="check_appointment",
    ):
        self.arguments = (
            VALID_ARGUMENTS if arguments is _DEFAULT_ARGUMENTS else arguments
        )
        self.results = list(results or [])
        self.final = final
        self.tool_name = tool_name
        self.tool_requests = []
        self.stream_requests = []

    async def chat_with_tools(self, request):
        self.tool_requests.append(request)
        arguments = self.results.pop(0) if self.results else self.arguments
        return LLMResponse(
            content="",
            tool_calls=[
                {
                    "name": self.tool_name,
                    "arguments": json.dumps(arguments),
                }
            ],
        )

    async def chat_stream(self, request):
        self.stream_requests.append(request)
        yield LLMChunk(delta=self.final, content=self.final, is_final=True)


class AutoMarkWebSocket:
    def __init__(self):
        self.handler = None
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)
        if message["event"] == "mark":
            self.handler._on_mark(message)


class GatedClearWebSocket(AutoMarkWebSocket):
    def __init__(self):
        super().__init__()
        self.clear_started = asyncio.Event()
        self.clear_release = asyncio.Event()

    async def send_json(self, message):
        if message["event"] == "clear" and not self.clear_started.is_set():
            self.clear_started.set()
            await self.clear_release.wait()
        await super().send_json(message)


class FakeSTT:
    def __init__(self):
        self.resets = 0

    async def reset_for_listening(self):
        self.resets += 1


class QueueSTT(FakeSTT):
    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()

    async def get_transcript(self):
        while True:
            yield await self.queue.get()


class OneChunkStream:
    def __init__(self, payload=b"A" * 1600):
        self.payload = payload
        self.sent = False
        self.cancelled = False
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.sent:
            raise StopAsyncIteration
        self.sent = True
        return self.payload

    def cancel(self):
        self.cancelled = True

    async def aclose(self):
        self.cancel()
        self.closed = True


@pytest.mark.asyncio
async def test_real_audio_interrupt_aborts_ack_and_consumes_queued_correction():
    first = dict(VALID_ARGUMENTS, accountant_name="Hussam Saadaldin")
    second = dict(VALID_ARGUMENTS, accountant_name="Rami Kahwaji")
    orchestrator, _ = make_orchestrator()
    context, handler = add_playback_call(orchestrator, "call", "en")
    context.state = ConversationState.LISTENING
    orchestrator.llm = ToolLLM(results=[first, second])
    orchestrator._GARBLED_AUTO_SWITCH_THRESHOLD = 3
    waiting = asyncio.Event()
    second_lookup = asyncio.Event()
    streams = []
    lookups = []

    class InterruptibleStream(OneChunkStream):
        def __init__(self):
            super().__init__(b"A" * 160)
            self.released = asyncio.Event()

        async def __anext__(self):
            if not self.sent:
                return await super().__anext__()
            waiting.set()
            await self.released.wait()
            raise StopAsyncIteration

        def cancel(self):
            super().cancel()
            self.released.set()

    class FreshTTS:
        def create_stream(self, _request):
            stream = InterruptibleStream() if not streams else OneChunkStream(b"B" * 160)
            streams.append(stream)
            return stream

    correction = SimpleNamespace(
        text="Use Rami instead", language="en", is_final=True,
        utterance_id="real-audio-correction",
    )

    class AudioQueueSTT(QueueSTT):
        async def stream_audio(self, _chunk):
            await self.queue.put(correction)

    stt = AudioQueueSTT()
    orchestrator._call_stt_instances["call"] = stt
    orchestrator.tts = FreshTTS()

    async def lookup(_call_sid, arguments):
        lookups.append(dict(arguments))
        second_lookup.set()
        return "SLOT_BUSY: synthetic"

    orchestrator._check_booking = mock.AsyncMock(side_effect=lookup)
    await stt.queue.put(SimpleNamespace(
        text="Check Hussam", language="en", is_final=True,
        utterance_id="real-audio-first",
    ))
    consumer = asyncio.create_task(orchestrator._consume_stt_transcripts("call"))
    try:
        await asyncio.wait_for(waiting.wait(), 2)
        # Exercise real decoding/detection beyond its documented grace period.
        orchestrator._barge_in_speech_start["call"] = 0
        callback = orchestrator._handle_incoming_audio("call")
        for _ in range(8):
            await asyncio.wait_for(callback(b"\x00" * 160), 2)
        await asyncio.wait_for(second_lookup.wait(), 2)
        assert lookups == [second]
        assert len(orchestrator.llm.tool_requests) == 2
        assert stt.queue.empty()
        assert streams[0].cancelled and streams[0].closed
    finally:
        for stream in streams:
            stream.cancel()
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        for stream in streams:
            await stream.aclose()
        await handler.cleanup()
    assert not orchestrator._active_speech
    assert handler._active_playback is None


class SlowCloseStream(OneChunkStream):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closing = asyncio.Event()
        self.close_release = asyncio.Event()

    async def __anext__(self):
        if not self.sent:
            self.sent = True
            self.started.set()
            return self.payload
        await self.release.wait()
        raise StopAsyncIteration

    def cancel(self):
        self.cancelled = True
        self.release.set()

    async def aclose(self):
        self.cancel()
        self.closing.set()
        await self.close_release.wait()
        self.closed = True


class FakeTTS:
    def __init__(self, streams):
        self.streams = streams
        self.requests = []

    def create_stream(self, request):
        self.requests.append((request.text, request.language))
        return self.streams[request.text]


def make_orchestrator(language="en", *, arguments=_DEFAULT_ARGUMENTS, call_sid="call"):
    orchestrator = ConversationOrchestrator.__new__(ConversationOrchestrator)
    context = ConversationContext(call_sid, "+14165550100", language=language)
    context.state = ConversationState.THINKING
    orchestrator._conversations = {call_sid: context}
    orchestrator._twilio_handlers = {call_sid: object()}
    orchestrator._call_state_locks = {call_sid: asyncio.Lock()}
    orchestrator._speech_setup_locks = {}
    orchestrator._active_speech = {}
    orchestrator._call_stt_instances = {}
    orchestrator._barge_in_reset_done = {call_sid: False}
    orchestrator._barge_in_speech_start = {}
    orchestrator._barge_in_consecutive = {}
    orchestrator._echo_guard_until = {}
    orchestrator._garbled_drop_count = {}
    orchestrator._system_prompt = "synthetic system"
    orchestrator.tts = object()
    orchestrator.llm = ToolLLM(arguments)
    orchestrator.caller_service = mock.Mock()
    orchestrator._check_booking = mock.AsyncMock(return_value="SLOT_BUSY: synthetic")
    orchestrator._confirm_booking = mock.AsyncMock()
    orchestrator._lookup_my_bookings = mock.AsyncMock()
    orchestrator._cancel_booking = mock.AsyncMock()
    return orchestrator, context


def add_playback_call(orchestrator, call_sid, language, websocket=None):
    context = ConversationContext(call_sid, "+14165550100", language=language)
    context.state = ConversationState.THINKING
    websocket = websocket or AutoMarkWebSocket()
    handler = TwilioMediaStreamHandler(call_sid, "stream-" + call_sid, websocket)
    websocket.handler = handler
    handler._is_streaming = True
    handler._is_connected = True
    orchestrator._conversations[call_sid] = context
    orchestrator._twilio_handlers[call_sid] = handler
    orchestrator._call_state_locks[call_sid] = asyncio.Lock()
    orchestrator._barge_in_reset_done[call_sid] = False
    orchestrator._call_stt_instances[call_sid] = FakeSTT()
    return context, handler


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("language", "expected"),
    [("en", ENGLISH_ACK), ("ar", ARABIC_ACK)],
)
async def test_acknowledgement_precedes_one_lookup_in_current_language(language, expected):
    orchestrator, context = make_orchestrator(language)
    events = []

    async def speak(call_sid, text, spoken_language):
        events.append(("speak", call_sid, text, spoken_language, context.state))
        return True

    async def lookup(call_sid, arguments):
        events.append(("lookup", call_sid, dict(arguments), context.state))
        return "SLOT_BUSY: synthetic"

    orchestrator._speak_to_caller = speak
    orchestrator._check_booking = mock.AsyncMock(side_effect=lookup)

    response = await orchestrator._get_ai_response("call", "Please check Rami")

    assert events[0] == (
        "speak", "call", expected, language, ConversationState.SPEAKING
    )
    assert events[1][0] == "lookup"
    assert events[1][2] == VALID_ARGUMENTS
    assert events[1][3] == ConversationState.THINKING
    assert orchestrator._barge_in_speech_start["call"] > 0
    assert response == "Synthetic final response"
    assert len(orchestrator.llm.tool_requests) == 1
    assert len(orchestrator.llm.stream_requests) == 1
    orchestrator._check_booking.assert_awaited_once_with("call", VALID_ARGUMENTS)
    orchestrator._confirm_booking.assert_not_awaited()
    orchestrator._lookup_my_bookings.assert_not_awaited()
    orchestrator._cancel_booking.assert_not_awaited()


@pytest.mark.asyncio
async def test_corrected_consultant_arguments_are_unchanged_and_never_spoken():
    corrected = dict(
        VALID_ARGUMENTS,
        accountant_name="Rami Kahwaji",
        date_time="2099-01-08 11:30",
    )
    orchestrator, _context = make_orchestrator("en", arguments=corrected)
    spoken = []

    async def speak(_call_sid, text, _language):
        spoken.append(text)
        return True

    orchestrator._speak_to_caller = speak
    await orchestrator._get_ai_response(
        "call", "Use Rami instead of Hussam at eleven thirty"
    )

    orchestrator._check_booking.assert_awaited_once_with("call", corrected)
    assert spoken == [ENGLISH_ACK]
    assert "Rami" not in spoken[0]
    assert "Hussam" not in spoken[0]
    assert "11" not in spoken[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments", "method_name"),
    [
        ("confirm_appointment", {"confirm": True}, "_confirm_booking"),
        ("lookup_my_bookings", {}, "_lookup_my_bookings"),
        ("cancel_booking", {"confirm_cancel": False}, "_cancel_booking"),
    ],
)
async def test_non_availability_tools_do_not_receive_new_availability_ack(
    tool_name, arguments, method_name
):
    orchestrator, _context = make_orchestrator("en")
    orchestrator.llm = ToolLLM(arguments, tool_name=tool_name)
    orchestrator._speak_to_caller = mock.AsyncMock(return_value=True)
    selected = getattr(orchestrator, method_name)
    selected.return_value = "BOOKING_CANCELLED: synthetic"

    await orchestrator._get_ai_response("call", "synthetic tool request")

    orchestrator._speak_to_caller.assert_not_awaited()
    selected.assert_awaited_once_with("call", arguments)
    orchestrator._check_booking.assert_not_awaited()


@pytest.mark.asyncio
async def test_interrupted_ack_aborts_old_lookup_and_queued_correction_is_consumed():
    first = dict(VALID_ARGUMENTS, accountant_name="Hussam Saadaldin")
    second = dict(VALID_ARGUMENTS, accountant_name="Rami Kahwaji")
    orchestrator, context = make_orchestrator("en")
    orchestrator.llm = ToolLLM(results=[first, second])
    stt = QueueSTT()
    orchestrator._call_stt_instances["call"] = stt
    context.state = ConversationState.LISTENING
    first_ack_started = asyncio.Event()
    interrupt_release = asyncio.Event()
    correction_finished = asyncio.Event()
    acknowledgement_count = 0
    lookups = []

    async def speak(call_sid, text, _language):
        nonlocal acknowledgement_count
        if text == ENGLISH_ACK:
            acknowledgement_count += 1
            orchestrator._barge_in_reset_done[call_sid] = False
            if acknowledgement_count == 1:
                first_ack_started.set()
                await interrupt_release.wait()
                orchestrator._barge_in_reset_done[call_sid] = True
                context.state = ConversationState.LISTENING
                return False
            return True
        context.state = ConversationState.LISTENING
        correction_finished.set()
        return True

    async def lookup(_call_sid, arguments):
        lookups.append(dict(arguments))
        return "SLOT_BUSY: synthetic"

    orchestrator._speak_to_caller = speak
    orchestrator._check_booking = mock.AsyncMock(side_effect=lookup)
    first_result = SimpleNamespace(
        text="Check Hussam", language="en", is_final=True, utterance_id="first"
    )
    correction = SimpleNamespace(
        text="Use Rami instead", language="en", is_final=True,
        utterance_id="correction",
    )
    await stt.queue.put(first_result)
    consumer = asyncio.create_task(
        orchestrator._consume_stt_transcripts("call"),
        name="t034c-consumer",
    )
    try:
        await asyncio.wait_for(first_ack_started.wait(), timeout=0.5)
        await stt.queue.put(correction)
        assert stt.queue.qsize() == 1
        interrupt_release.set()
        await asyncio.wait_for(correction_finished.wait(), timeout=1)
    finally:
        interrupt_release.set()
        consumer.cancel()
        await asyncio.wait_for(consumer, timeout=0.5)

    assert lookups == [second]
    assert acknowledgement_count == 2
    assert len(orchestrator.llm.tool_requests) == 2
    assert len(orchestrator.llm.stream_requests) == 1
    assert consumer.done()
    assert not [task for task in asyncio.all_tasks() if task.get_name() == "t034c-consumer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["ended", "context", "handler"])
async def test_session_change_at_ack_boundary_prevents_old_lookup_and_final(replacement):
    orchestrator, old_context = make_orchestrator("en")
    started = asyncio.Event()
    release = asyncio.Event()
    old_handler = orchestrator._twilio_handlers["call"]
    replacement_context = None

    async def speak(_call_sid, _text, _language):
        started.set()
        await release.wait()
        return True

    orchestrator._speak_to_caller = speak
    task = asyncio.create_task(orchestrator._get_ai_response("call", "check"))
    await asyncio.wait_for(started.wait(), timeout=0.5)
    if replacement == "ended":
        old_context.state = ConversationState.ENDED
    elif replacement == "context":
        replacement_context = ConversationContext("call", "+14165550101", language="ar")
        replacement_context.state = ConversationState.LISTENING
        orchestrator._conversations["call"] = replacement_context
    else:
        orchestrator._twilio_handlers["call"] = object()
        old_context.state = ConversationState.GREETING
    release.set()
    response = await asyncio.wait_for(task, timeout=0.5)

    assert response is orchestrator_module._ABORTED_RESPONSE
    orchestrator._check_booking.assert_not_awaited()
    assert len(orchestrator.llm.stream_requests) == 0
    if replacement == "ended":
        assert old_context.state == ConversationState.ENDED
    if replacement_context is not None:
        assert replacement_context.state == ConversationState.LISTENING
    if replacement == "handler":
        assert orchestrator._twilio_handlers["call"] is not old_handler
        assert old_context.state == ConversationState.GREETING


@pytest.mark.asyncio
async def test_process_transcript_does_not_write_or_speak_an_aborted_response():
    orchestrator, old_context = make_orchestrator("en")
    old_context.state = ConversationState.LISTENING
    replacement = ConversationContext("call", "+14165550101", language="ar")
    replacement.state = ConversationState.GREETING
    final_speech = mock.AsyncMock()

    async def abort(_call_sid, _text):
        orchestrator._conversations["call"] = replacement
        return orchestrator_module._ABORTED_RESPONSE

    orchestrator._get_ai_response = mock.AsyncMock(side_effect=abort)
    orchestrator._speak_to_caller = final_speech
    orchestrator._call_stt_instances["call"] = FakeSTT()
    orchestrator._GARBLED_AUTO_SWITCH_THRESHOLD = 3

    await orchestrator.process_transcript("call", "check availability", "en", True)

    assert replacement.state == ConversationState.GREETING
    assert replacement.ai_response == ""
    assert replacement.turn_count == 0
    final_speech.assert_not_awaited()


@pytest.mark.asyncio
async def test_actual_handler_replacement_during_setup_cannot_reset_new_session_state():
    orchestrator, context = make_orchestrator("en")
    websocket = GatedClearWebSocket()
    context, old_handler = add_playback_call(
        orchestrator, "call", "en", websocket=websocket
    )
    orchestrator.tts = FakeTTS({ENGLISH_ACK: OneChunkStream()})
    task = asyncio.create_task(orchestrator._get_ai_response("call", "check"))
    await asyncio.wait_for(websocket.clear_started.wait(), timeout=0.5)

    new_websocket = AutoMarkWebSocket()
    new_handler = TwilioMediaStreamHandler("call", "new-stream", new_websocket)
    new_websocket.handler = new_handler
    new_handler._is_streaming = True
    new_handler._is_connected = True
    orchestrator._twilio_handlers["call"] = new_handler
    context.state = ConversationState.SPEAKING
    websocket.clear_release.set()
    response = await asyncio.wait_for(task, timeout=0.5)

    assert response is orchestrator_module._ABORTED_RESPONSE
    assert context.state == ConversationState.SPEAKING
    orchestrator._check_booking.assert_not_awaited()
    assert old_handler._active_playback is None
    await old_handler.cleanup()
    await new_handler.cleanup()


@pytest.mark.asyncio
async def test_repeated_cancellation_propagates_and_retires_ack_playback():
    orchestrator, _context = make_orchestrator("en")
    stream = SlowCloseStream()
    orchestrator.tts = FakeTTS({ENGLISH_ACK: stream})
    _context, handler = add_playback_call(orchestrator, "call", "en")
    task = asyncio.create_task(
        orchestrator._get_ai_response("call", "check"),
        name="t034c-cancelled-ack",
    )
    try:
        await asyncio.wait_for(stream.started.wait(), timeout=0.5)
        task.cancel()
        await asyncio.wait_for(stream.closing.wait(), timeout=0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.5)
        assert "call" not in orchestrator._active_speech
        assert handler._active_playback is None
        orchestrator._check_booking.assert_not_awaited()
    finally:
        stream.close_release.set()
        await stream.aclose()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await handler.cleanup()
    assert not [
        pending for pending in asyncio.all_tasks()
        if pending.get_name() == "t034c-cancelled-ack"
    ]


@pytest.mark.asyncio
async def test_two_calls_have_independent_owned_acknowledgements():
    orchestrator, _context = make_orchestrator("en", call_sid="A")
    orchestrator._conversations.clear()
    orchestrator._twilio_handlers.clear()
    orchestrator._call_state_locks.clear()
    orchestrator._call_stt_instances.clear()
    orchestrator._barge_in_reset_done.clear()
    context_a, handler_a = add_playback_call(orchestrator, "A", "en")
    context_b, handler_b = add_playback_call(orchestrator, "B", "ar")
    context_a.add_user_message("lookup A")
    context_b.add_user_message("lookup B")
    orchestrator.tts = FakeTTS(
        {ENGLISH_ACK: OneChunkStream(b"A" * 1600), ARABIC_ACK: OneChunkStream(b"B" * 1600)}
    )
    lookups = []

    class RoutingLLM(ToolLLM):
        async def chat_with_tools(self, request):
            marker = request.messages[-1].content
            arguments = dict(
                VALID_ARGUMENTS,
                accountant_name="Rami Kahwaji" if marker.endswith("A") else "Hussam Saadaldin",
            )
            self.tool_requests.append(request)
            return LLMResponse(
                content="",
                tool_calls=[{"name": "check_appointment", "arguments": json.dumps(arguments)}],
            )

    orchestrator.llm = RoutingLLM()

    async def lookup(call_sid, arguments):
        lookups.append((call_sid, dict(arguments)))
        return "SLOT_BUSY: synthetic"

    orchestrator._check_booking = mock.AsyncMock(side_effect=lookup)
    await asyncio.wait_for(
        asyncio.gather(
            orchestrator._get_ai_response("A", "lookup A"),
            orchestrator._get_ai_response("B", "lookup B"),
        ),
        timeout=2,
    )

    assert {call_sid for call_sid, _arguments in lookups} == {"A", "B"}
    assert {text for text, _language in orchestrator.tts.requests} == {
        ENGLISH_ACK, ARABIC_ACK
    }
    assert not orchestrator._active_speech
    assert context_a.state == ConversationState.THINKING
    assert context_b.state == ConversationState.THINKING
    await handler_a.cleanup()
    await handler_b.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("booking_result", "expected_instruction"),
    [
        ("SLOT_AVAILABLE: synthetic", "The time IS AVAILABLE"),
        (
            "SLOT_BUSY: synthetic alternative",
            "time is taken and give ONE alternative",
        ),
        ("SCHEDULE_FULL: synthetic", "time is taken and give ONE alternative"),
        (
            "AVAILABILITY_UNVERIFIED: synthetic error",
            "Relay this outcome to the caller naturally",
        ),
    ],
)
async def test_existing_availability_result_instructions_are_preserved(
    booking_result, expected_instruction
):
    orchestrator, _context = make_orchestrator("en")
    orchestrator._speak_to_caller = mock.AsyncMock(return_value=True)
    orchestrator._check_booking = mock.AsyncMock(return_value=booking_result)

    response = await orchestrator._get_ai_response("call", "check")

    assert response == "Synthetic final response"
    orchestrator._speak_to_caller.assert_awaited_once_with(
        "call", ENGLISH_ACK, "en"
    )
    orchestrator._check_booking.assert_awaited_once_with("call", VALID_ARGUMENTS)
    prompt = orchestrator.llm.stream_requests[0].messages[-1].content
    assert booking_result in prompt
    assert expected_instruction in prompt
    assert ENGLISH_ACK not in prompt


@pytest.mark.asyncio
async def test_missing_tts_allows_current_read_only_lookup_without_ack_history():
    orchestrator, context = make_orchestrator("en")
    orchestrator.tts = None
    handler = orchestrator._twilio_handlers["call"]

    response = await orchestrator._get_ai_response("call", "check")

    assert response == "Synthetic final response"
    orchestrator._check_booking.assert_awaited_once_with("call", VALID_ARGUMENTS)
    assert orchestrator._twilio_handlers["call"] is handler
    assert context.state == ConversationState.THINKING
    assert all(message["content"] != ENGLISH_ACK for message in context.conversation_history)


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [[], "invalid", 7, {}, {"date_time": "bad"}])
async def test_invalid_arguments_never_trigger_acknowledgement(arguments):
    orchestrator, _context = make_orchestrator("en", arguments=arguments)
    orchestrator._speak_to_caller = mock.AsyncMock(return_value=True)
    orchestrator._check_booking = mock.AsyncMock(
        return_value="INVALID_DATE_TIME: repeat the requested date and time"
    )

    response = await orchestrator._get_ai_response("call", "check")

    orchestrator._speak_to_caller.assert_not_awaited()
    orchestrator._check_booking.assert_awaited_once_with("call", arguments)
    assert response == "Synthetic final response"


@pytest.mark.asyncio
async def test_null_arguments_keep_existing_parse_apology_without_acknowledgement():
    orchestrator, _context = make_orchestrator("en", arguments=None)
    orchestrator._speak_to_caller = mock.AsyncMock(return_value=True)

    response = await orchestrator._get_ai_response("call", "check")

    orchestrator._speak_to_caller.assert_not_awaited()
    orchestrator._check_booking.assert_not_awaited()
    assert response == "Sorry, could you repeat that please?"
    assert len(orchestrator.llm.stream_requests) == 0
