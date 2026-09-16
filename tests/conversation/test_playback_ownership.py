import asyncio

import pytest
from loguru import logger

from services.conversation.orchestrator import (
    ConversationContext,
    ConversationOrchestrator,
    ConversationState,
)
from services.telephony.twilio_service import TwilioMediaStreamHandler


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


class PrivateFailureWebSocket(AutoMarkWebSocket):
    def __init__(self, sentinel):
        super().__init__()
        self.sentinel = sentinel
        self.fail = False

    async def send_json(self, message):
        if self.fail:
            try:
                raise ValueError(f"nested-{self.sentinel}")
            except ValueError as cause:
                raise RuntimeError(f"outer-{self.sentinel}") from cause
        await super().send_json(message)


class FakeStream:
    def __init__(self, payload: bytes, *, blocked: bool = False):
        self.payload = payload
        self.blocked = blocked
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.closed = False
        self._sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._sent:
            self._sent = True
            self.started.set()
            return self.payload
        if self.blocked and not self.cancelled:
            await self.release.wait()
        raise StopAsyncIteration

    def cancel(self):
        self.cancelled = True
        self.release.set()

    async def aclose(self):
        self.cancel()
        self.closed = True


class PrivateFailureCloseStream(FakeStream):
    def __init__(self, payload, sentinel):
        super().__init__(payload)
        self.sentinel = sentinel

    async def aclose(self):
        self.cancel()
        self.closed = True
        raise RuntimeError(f"close-{self.sentinel}")


class FakeTTS:
    def __init__(self, streams):
        self.streams = streams
        self.created_requests = []
        self.request_created = asyncio.Event()

    def create_stream(self, request):
        self.created_requests.append(request.text)
        self.request_created.set()
        return self.streams[request.text]


class FakeSTT:
    def __init__(self, language="en"):
        self.language = language
        self.resets = 0
        self.audio = []
        self.disconnected = False

    async def reset_for_listening(self):
        self.resets += 1

    async def stream_audio(self, chunk):
        self.audio.append(chunk.data)

    async def disconnect(self):
        self.disconnected = True


def add_call(orchestrator, call_sid, *, language="en", websocket=None):
    context = ConversationContext(call_sid, "+10000000000", language=language)
    context.state = ConversationState.SPEAKING
    websocket = websocket or AutoMarkWebSocket()
    handler = TwilioMediaStreamHandler(call_sid, f"stream-{call_sid}", websocket)
    websocket.handler = handler
    handler._is_streaming = True
    handler._is_connected = True
    stt = FakeSTT(language)
    orchestrator._conversations[call_sid] = context
    orchestrator._twilio_handlers[call_sid] = handler
    orchestrator._call_stt_instances[call_sid] = stt
    orchestrator._call_state_locks[call_sid] = asyncio.Lock()
    orchestrator._barge_in_reset_done[call_sid] = False
    return context, handler, stt


def make_orchestrator(tts):
    orchestrator = ConversationOrchestrator.__new__(ConversationOrchestrator)
    orchestrator.tts = tts
    orchestrator._conversations = {}
    orchestrator._twilio_handlers = {}
    orchestrator._call_stt_instances = {}
    orchestrator._call_state_locks = {}
    orchestrator._speech_setup_locks = {}
    orchestrator._active_speech = {}
    orchestrator._barge_in_reset_done = {}
    orchestrator._echo_guard_until = {}
    orchestrator._garbled_drop_count = {}
    orchestrator._barge_in_consecutive = {}
    orchestrator._barge_in_speech_start = {}
    return orchestrator


@pytest.mark.asyncio
async def test_barge_in_interrupts_only_call_a_and_outer_coroutine_resumes():
    stream_a = FakeStream(b"A" * 1600, blocked=True)
    stream_b = FakeStream(b"B" * 1600)
    orchestrator = make_orchestrator(FakeTTS({"A": stream_a, "B": stream_b}))
    context_a, _handler_a, stt_a = add_call(orchestrator, "A")
    _context_b, _handler_b, stt_b = add_call(orchestrator, "B")
    resumed = asyncio.Event()

    async def synthetic_tool_coroutine():
        await orchestrator._speak_to_caller("A", "A", "en")
        resumed.set()

    tool_task = asyncio.create_task(synthetic_tool_coroutine())
    task_b = asyncio.create_task(orchestrator._speak_to_caller("B", "B", "en"))
    await asyncio.wait_for(stream_a.started.wait(), timeout=0.5)

    async def detect(_audio, _call_sid):
        return True

    orchestrator._detect_barge_in = detect
    await orchestrator._handle_incoming_audio("A")(b"caller")

    await asyncio.wait_for(resumed.wait(), timeout=0.5)
    await asyncio.wait_for(asyncio.gather(tool_task, task_b), timeout=1)
    assert tool_task.cancelled() is False
    assert stream_a.cancelled is True
    assert stream_b.cancelled is True  # normal final close, not A's interruption
    assert stt_a.resets == 1
    assert stt_b.resets == 1
    assert context_a.state == ConversationState.LISTENING


@pytest.mark.asyncio
async def test_late_old_speech_cannot_mutate_replacement_completion_state():
    old = FakeStream(b"O" * 1600, blocked=True)
    new = FakeStream(b"N" * 1600)
    orchestrator = make_orchestrator(FakeTTS({"old": old, "new": new}))
    context, _handler, stt = add_call(orchestrator, "same", language="en")

    old_task = asyncio.create_task(orchestrator._speak_to_caller("same", "old", "ar"))
    await asyncio.wait_for(old.started.wait(), timeout=0.5)
    new_task = asyncio.create_task(orchestrator._speak_to_caller("same", "new", "en"))
    await asyncio.wait_for(asyncio.gather(old_task, new_task), timeout=1)

    assert stt.resets == 1
    assert stt.language == "en"
    assert context.language == "en"
    assert context.state == ConversationState.LISTENING
    assert orchestrator._echo_guard_until["same"] > 0
    assert not orchestrator._active_speech


@pytest.mark.asyncio
async def test_external_speech_task_cancellation_is_preserved_and_closes_owner():
    blocked = FakeStream(b"C" * 1600, blocked=True)
    orchestrator = make_orchestrator(FakeTTS({"cancel": blocked}))
    add_call(orchestrator, "cancelled")
    speech_task = asyncio.create_task(
        orchestrator._speak_to_caller("cancelled", "cancel", "en")
    )
    await asyncio.wait_for(blocked.started.wait(), timeout=0.5)

    speech_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await speech_task

    assert blocked.cancelled and blocked.closed
    assert "cancelled" not in orchestrator._active_speech


@pytest.mark.asyncio
async def test_end_call_invalidates_old_speech_before_await_and_preserves_reused_sid(monkeypatch):
    blocked = FakeStream(b"O" * 1600, blocked=True)
    orchestrator = make_orchestrator(FakeTTS({"old": blocked}))
    old_context, old_handler, old_stt = add_call(orchestrator, "reused")
    old_context.language = "auto"
    orchestrator.caller_service = object()
    speak_task = asyncio.create_task(orchestrator._speak_to_caller("reused", "old", "en"))
    await asyncio.wait_for(blocked.started.wait(), timeout=0.5)

    database_release = asyncio.Event()

    class Pool:
        async def execute(self, *_args):
            await database_release.wait()

    async def get_pool():
        return Pool()

    import services.database as database
    monkeypatch.setattr(database, "get_db_pool", get_pool)
    end_task = asyncio.create_task(orchestrator.end_call("reused"))
    while old_context.state != ConversationState.ENDED:
        await asyncio.sleep(0)

    new_context, new_handler, new_stt = add_call(orchestrator, "reused")
    new_context.state = ConversationState.LISTENING
    orchestrator._echo_guard_until["reused"] = 123
    database_release.set()
    await asyncio.wait_for(asyncio.gather(end_task, speak_task), timeout=1)

    assert blocked.cancelled and blocked.closed
    assert old_stt.disconnected is True
    assert old_handler._playback_closed is True
    assert orchestrator._conversations["reused"] is new_context
    assert orchestrator._twilio_handlers["reused"] is new_handler
    assert orchestrator._call_stt_instances["reused"] is new_stt
    assert orchestrator._echo_guard_until["reused"] == 123
    assert new_context.state == ConversationState.LISTENING


@pytest.mark.asyncio
async def test_end_call_during_playback_setup_prevents_tts_and_media(monkeypatch):
    stream = FakeStream(b"X" * 1600)
    tts = FakeTTS({"pending": stream})
    orchestrator = make_orchestrator(tts)
    websocket = GatedClearWebSocket()
    context, handler, _stt = add_call(
        orchestrator, "pending", language="auto", websocket=websocket
    )
    orchestrator.caller_service = object()

    database_started = asyncio.Event()
    database_release = asyncio.Event()

    class Pool:
        async def execute(self, *_args):
            database_started.set()
            await database_release.wait()

    async def get_pool():
        return Pool()

    import services.database as database
    monkeypatch.setattr(database, "get_db_pool", get_pool)

    speech_task = asyncio.create_task(
        orchestrator._speak_to_caller("pending", "pending", "en")
    )
    await asyncio.wait_for(websocket.clear_started.wait(), timeout=0.2)
    end_task = asyncio.create_task(orchestrator.end_call("pending"))
    await asyncio.wait_for(database_started.wait(), timeout=0.2)
    assert context.state == ConversationState.ENDED

    websocket.clear_release.set()
    await asyncio.wait_for(speech_task, timeout=0.5)
    database_release.set()
    await asyncio.wait_for(end_task, timeout=0.5)

    assert tts.created_requests == []
    assert not any(message["event"] == "media" for message in websocket.messages)
    assert handler._active_playback is None
    assert not orchestrator._active_speech


@pytest.mark.asyncio
async def test_reused_call_sid_cannot_be_overwritten_by_old_pending_setup(monkeypatch):
    old_stream = FakeStream(b"O" * 1600)
    new_stream = FakeStream(b"N" * 1600)
    tts = FakeTTS({"old-pending": old_stream, "new-live": new_stream})
    orchestrator = make_orchestrator(tts)
    old_websocket = GatedClearWebSocket()
    old_context, old_handler, _old_stt = add_call(
        orchestrator, "same-sid", language="auto", websocket=old_websocket
    )
    orchestrator.caller_service = object()

    database_started = asyncio.Event()
    database_release = asyncio.Event()

    class Pool:
        async def execute(self, *_args):
            database_started.set()
            await database_release.wait()

    async def get_pool():
        return Pool()

    import services.database as database
    monkeypatch.setattr(database, "get_db_pool", get_pool)

    old_speech = asyncio.create_task(
        orchestrator._speak_to_caller("same-sid", "old-pending", "en")
    )
    await asyncio.wait_for(old_websocket.clear_started.wait(), timeout=0.2)
    end_task = asyncio.create_task(orchestrator.end_call("same-sid"))
    await asyncio.wait_for(database_started.wait(), timeout=0.2)
    assert old_context.state == ConversationState.ENDED

    new_context, new_handler, new_stt = add_call(orchestrator, "same-sid")
    new_speech = asyncio.create_task(
        orchestrator._speak_to_caller("same-sid", "new-live", "en")
    )
    old_websocket.clear_release.set()
    await asyncio.wait_for(asyncio.gather(old_speech, new_speech), timeout=1)
    database_release.set()
    await asyncio.wait_for(end_task, timeout=0.5)

    assert tts.created_requests == ["new-live"]
    assert not any(message["event"] == "media" for message in old_websocket.messages)
    assert orchestrator._conversations["same-sid"] is new_context
    assert orchestrator._twilio_handlers["same-sid"] is new_handler
    assert orchestrator._call_stt_instances["same-sid"] is new_stt
    assert new_stt.resets == 1
    assert new_context.state == ConversationState.LISTENING
    assert old_handler._playback_closed is True
    assert not orchestrator._active_speech


@pytest.mark.asyncio
async def test_speech_clear_and_cleanup_logs_never_expose_exception_values():
    sentinel = "PRIVATE-VALUE-7f932"
    messages = []
    sink = logger.add(lambda record: messages.append(str(record)))
    try:
        setup_tts = FakeTTS({"setup": FakeStream(b"S" * 1600)})
        setup_orchestrator = make_orchestrator(setup_tts)
        failing_websocket = PrivateFailureWebSocket(sentinel)
        setup_context, setup_handler, _ = add_call(
            setup_orchestrator, "setup", websocket=failing_websocket
        )
        failing_websocket.fail = True
        await setup_orchestrator._speak_to_caller("setup", "setup", "en")

        failing_websocket.fail = False
        clear_owner = await setup_handler.begin_playback()
        failing_websocket.fail = True
        await setup_handler.clear_audio(clear_owner)

        close_stream = PrivateFailureCloseStream(b"C" * 1600, sentinel)
        close_orchestrator = make_orchestrator(FakeTTS({"close": close_stream}))
        close_context, _close_handler, _ = add_call(close_orchestrator, "close")
        await close_orchestrator._speak_to_caller("close", "close", "en")
    finally:
        logger.remove(sink)

    combined = "".join(messages)
    assert sentinel not in combined
    assert "RuntimeError" in combined
    assert setup_context.state == ConversationState.LISTENING
    assert close_context.state == ConversationState.LISTENING


@pytest.mark.asyncio
async def test_second_cancellation_during_close_still_retires_speech_owner():
    class SlowCloseStream(FakeStream):
        def __init__(self):
            super().__init__(b"X" * 160, blocked=True)
            self.closing = asyncio.Event()
            self.close_release = asyncio.Event()

        async def aclose(self):
            self.cancel()
            self.closing.set()
            await self.close_release.wait()
            self.closed = True

    stream = SlowCloseStream()
    orchestrator = make_orchestrator(FakeTTS({"cancel": stream}))
    _context, handler, _stt = add_call(orchestrator, "cancel-twice")
    task = asyncio.create_task(
        orchestrator._speak_to_caller("cancel-twice", "cancel", "en")
    )
    try:
        await asyncio.wait_for(stream.started.wait(), timeout=0.5)
        task.cancel()
        await asyncio.wait_for(stream.closing.wait(), timeout=0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.5)
        assert "cancel-twice" not in orchestrator._active_speech
        assert handler._active_playback is None
    finally:
        stream.close_release.set()
        await stream.aclose()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await handler.cleanup()
