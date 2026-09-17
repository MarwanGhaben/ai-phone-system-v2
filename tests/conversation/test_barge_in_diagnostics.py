"""Offline contracts for bounded barge-in diagnostics."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time

import pytest

from config.settings import Settings
from services.conversation.barge_in_diagnostics import (
    BargeInDiagnostics,
    DIAGNOSTIC_FAILURE,
    MAX_ACTIVE_CALLS,
    MAX_SPEAKING_WINDOWS,
)
from services.conversation.orchestrator import (
    ConversationContext,
    ConversationOrchestrator,
    ConversationState,
)
from services.telephony.twilio_service import TwilioMediaStreamHandler


PRIVATE = "PRIVATE_CALL_AUDIO_EXCEPTION_TOKEN"


class ControlledClock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeSTT:
    def __init__(self, trace: list[tuple]) -> None:
        self.trace = trace
        self.audio: list[bytes] = []
        self.resets = 0
        self.disconnected = False

    async def stream_audio(self, chunk) -> None:
        self.audio.append(chunk.data)
        self.trace.append(("forward", chunk.data))

    async def reset_for_listening(self) -> None:
        self.resets += 1
        self.trace.append(("reset",))

    async def disconnect(self) -> None:
        self.disconnected = True


class FakeHandler:
    def __init__(self) -> None:
        self.invalidated = 0
        self.cleaned = 0

    def invalidate_playback_session(self) -> None:
        self.invalidated += 1

    async def cleanup(self) -> None:
        self.cleaned += 1


def make_orchestrator(*, enabled: bool, emitted: list[str] | None = None):
    orchestrator = ConversationOrchestrator.__new__(ConversationOrchestrator)
    emitted = emitted if emitted is not None else []
    observer = BargeInDiagnostics(emit=emitted.append) if enabled else None
    orchestrator._barge_in_observer = observer
    orchestrator._conversations = {}
    orchestrator._call_stt_instances = {}
    orchestrator._call_state_locks = {}
    orchestrator._twilio_handlers = {}
    orchestrator._speech_setup_locks = {}
    orchestrator._active_speech = {}
    orchestrator._barge_in_consecutive = {}
    orchestrator._barge_in_speech_start = {}
    orchestrator._barge_in_reset_done = {}
    orchestrator._echo_guard_until = {}
    orchestrator._garbled_drop_count = {}
    return orchestrator, observer, emitted


def add_call(orchestrator, observer, call_sid: str):
    trace: list[tuple] = []
    context = ConversationContext(
        call_sid=call_sid,
        phone_number=f"{PRIVATE}-phone",
        language="auto",
        metadata={"private": PRIVATE},
    )
    context.state = ConversationState.SPEAKING
    if observer is not None:
        context.barge_in_diagnostics = observer.start_call()
    stt = FakeSTT(trace)
    orchestrator._conversations[call_sid] = context
    orchestrator._call_stt_instances[call_sid] = stt
    orchestrator._call_state_locks[call_sid] = asyncio.Lock()
    orchestrator._barge_in_speech_start[call_sid] = 0
    orchestrator._barge_in_reset_done[call_sid] = False

    async def interrupt(_call_sid: str) -> bool:
        trace.append(("interrupt",))
        return True

    orchestrator._interrupt_call_speech = interrupt
    return context, stt, trace


async def run_sequence(
    enabled: bool,
    chunks: list[bytes],
    *,
    grace: bool = False,
    call_sid: str = f"{PRIVATE}-call",
):
    orchestrator, observer, emitted = make_orchestrator(enabled=enabled)
    context, stt, trace = add_call(orchestrator, observer, call_sid)
    if grace:
        orchestrator._barge_in_speech_start[context.call_sid] = time.time()
    callback = orchestrator._handle_incoming_audio(context.call_sid)
    outcomes = []
    for chunk in chunks:
        await callback(chunk)
        outcomes.append(context.state)
    summary = None
    if observer is not None:
        summary = observer.snapshot(context.barge_in_diagnostics)
    return {
        "outcomes": outcomes,
        "audio": stt.audio,
        "resets": stt.resets,
        "trace": trace,
        "consecutive": dict(orchestrator._barge_in_consecutive),
    }, summary, emitted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks,grace",
    [
        ([b""], False),
        ([b"\xff" * 160], False),
        ([b"\x00" * 160] * 3 + [b"\xff" * 160], False),
        ([b"\x00" * 80, b"\x00" * 240, b"\xff" * 37], False),
        ([b"\x00" * 160] * 8, False),
        ([b"\x00" * 160], True),
    ],
    ids=("empty", "silence", "short-burst", "varying", "gate", "grace"),
)
async def test_enabled_and_disabled_preserve_exact_decisions_bytes_and_order(
    chunks, grace
) -> None:
    disabled, _, _ = await run_sequence(False, chunks, grace=grace)
    enabled, summary, _ = await run_sequence(True, chunks, grace=grace)

    assert enabled == disabled
    assert summary["totals"]["received_bytes"] == sum(map(len, chunks))
    if len(chunks) == 8:
        assert enabled["trace"][-3:] == [
            ("interrupt",),
            ("reset",),
            ("forward", chunks[-1]),
        ]
        assert summary["totals"]["trigger_count"] == 1
        assert summary["totals"]["forwarded_bytes"] == len(chunks[-1])
    else:
        assert summary["totals"]["trigger_count"] == 0
        assert summary["totals"]["dropped_bytes"] == sum(map(len, chunks))


@pytest.mark.asyncio
async def test_decode_failure_is_observed_without_changing_drop(monkeypatch) -> None:
    import audioop

    def fail_decode(*_args):
        raise ValueError(PRIVATE)

    monkeypatch.setattr(audioop, "ulaw2lin", fail_decode)
    disabled, _, _ = await run_sequence(False, [b"\x01" * 160])
    enabled, summary, _ = await run_sequence(True, [b"\x01" * 160])

    assert enabled == disabled
    assert summary["windows"][0]["decode_failed_count"] == 1
    assert summary["windows"][0]["analyzed_bytes"] == 160
    assert PRIVATE not in json.dumps(summary)


def test_quantitative_aggregates_and_dispatch_gaps_are_bounded() -> None:
    clock = ControlledClock()
    observer = BargeInDiagnostics(clock=clock)
    state = observer.start_call()
    window = observer.start_speaking(state)
    observer.observe_dispatch(state, 80)
    clock.advance(0.02)
    observer.observe_dispatch(state, 240)
    observer.observe_detector(
        state, "above_threshold", analyzed_bytes=240, rms=901, consecutive=7,
        window=window,
    )
    observer.observe_detector(
        state, "gate_fired", analyzed_bytes=80, rms=850, consecutive=8,
        window=window,
    )
    observer.observe_route(state, "forwarded", 80, window=window)
    observer.observe_route(state, "dropped", 240, window=window)
    observer.observe_action(state, "interrupt", "succeeded", window=window)
    observer.observe_action(state, "reset", "succeeded", window=window)
    clock.advance(0.18)
    observer.end_speaking(state, window, "failed")
    summary = observer.snapshot(state)

    assert summary["totals"] == {
        "received_callbacks": 2,
        "received_bytes": 320,
        "received_min_bytes": 80,
        "received_max_bytes": 240,
        "analyzed_bytes": 320,
        "max_rms": 901,
        "max_consecutive": 8,
        "trigger_count": 1,
        "forwarded_callbacks": 1,
        "forwarded_bytes": 80,
        "dropped_callbacks": 1,
        "dropped_bytes": 240,
        "speaking_seconds": 0.2,
    }
    assert summary["windows"][0]["outcome"] == "interrupted"
    assert summary["windows"][0]["dispatch_gap_count"] == 1
    assert summary["windows"][0]["dispatch_gap_mean_seconds"] == 0.02


def test_window_overflow_and_call_capacity_skip_without_state_growth() -> None:
    observer = BargeInDiagnostics(max_active_calls=MAX_ACTIVE_CALLS)
    states = [observer.start_call() for _ in range(MAX_ACTIVE_CALLS)]
    assert all(state is not None for state in states)
    assert observer.start_call() is None
    assert observer.capacity_skip_count == 1

    selected = states[0]
    for _ in range(MAX_SPEAKING_WINDOWS + 5):
        window = observer.start_speaking(selected)
        observer.end_speaking(selected, window, "completed")
    summary = observer.snapshot(selected)
    assert summary["windows_started"] == MAX_SPEAKING_WINDOWS + 5
    assert summary["windows_retained"] == MAX_SPEAKING_WINDOWS
    assert summary["window_overflow_count"] == 5


@pytest.mark.asyncio
async def test_two_calls_remain_isolated() -> None:
    orchestrator, observer, _ = make_orchestrator(enabled=True)
    context_a, stt_a, _ = add_call(orchestrator, observer, "synthetic-a")
    context_b, stt_b, _ = add_call(orchestrator, observer, "synthetic-b")
    callback_a = orchestrator._handle_incoming_audio("synthetic-a")
    callback_b = orchestrator._handle_incoming_audio("synthetic-b")

    await asyncio.gather(
        callback_a(b"\x00" * 160), callback_b(b"\xff" * 160)
    )
    for _ in range(7):
        await callback_a(b"\x00" * 160)

    summary_a = observer.snapshot(context_a.barge_in_diagnostics)
    summary_b = observer.snapshot(context_b.barge_in_diagnostics)
    assert summary_a["totals"]["trigger_count"] == 1
    assert summary_b["totals"]["trigger_count"] == 0
    assert stt_a.resets == 1 and stt_b.resets == 0
    assert context_b.state == ConversationState.SPEAKING


@pytest.mark.asyncio
async def test_old_callback_and_cleanup_cannot_mutate_replacement_diagnostics() -> None:
    orchestrator, observer, emitted = make_orchestrator(enabled=True)
    old_context, _, _ = add_call(orchestrator, observer, "reused")
    old_callback = orchestrator._handle_incoming_audio("reused")
    old_state = old_context.barge_in_diagnostics
    observer.finish_call(old_state)

    new_context, _, _ = add_call(orchestrator, observer, "reused")
    before = observer.snapshot(new_context.barge_in_diagnostics)
    await old_callback(b"\xff" * 160)
    observer.finish_call(old_state)
    after = observer.snapshot(new_context.barge_in_diagnostics)

    assert after["diagnostic_id"] == before["diagnostic_id"]
    assert after["windows"] == before["windows"] == []
    assert after["totals"] == before["totals"]
    assert observer.active_count == 1
    assert len(emitted) == 1


@pytest.mark.asyncio
async def test_inflight_old_teardown_preserves_replacement_diagnostics(monkeypatch) -> None:
    orchestrator, observer, emitted = make_orchestrator(enabled=True)
    old_context, _, _ = add_call(orchestrator, observer, "reused-teardown")
    old_handler = FakeHandler()
    orchestrator._twilio_handlers[old_context.call_sid] = old_handler
    orchestrator.caller_service = object()
    database_entered = asyncio.Event()
    database_release = asyncio.Event()

    class Pool:
        async def execute(self, *_args) -> None:
            database_entered.set()
            await database_release.wait()

    async def get_pool():
        return Pool()

    import services.database as database

    monkeypatch.setattr(database, "get_db_pool", get_pool)
    teardown = asyncio.create_task(orchestrator.end_call(old_context.call_sid))
    await asyncio.wait_for(database_entered.wait(), timeout=0.5)

    new_context, new_stt, _ = add_call(orchestrator, observer, "reused-teardown")
    new_handler = FakeHandler()
    orchestrator._twilio_handlers[new_context.call_sid] = new_handler
    database_release.set()
    await asyncio.wait_for(teardown, timeout=0.5)

    assert observer.active_count == 1
    assert len(emitted) == 1
    assert orchestrator._conversations[new_context.call_sid] is new_context
    assert orchestrator._call_stt_instances[new_context.call_sid] is new_stt
    assert orchestrator._twilio_handlers[new_context.call_sid] is new_handler
    assert observer.snapshot(new_context.barge_in_diagnostics)["windows"] == []
    assert old_handler.cleaned == 1


@pytest.mark.asyncio
async def test_repeated_actual_teardown_emits_one_redacted_summary(monkeypatch) -> None:
    orchestrator, observer, emitted = make_orchestrator(enabled=True)
    context, stt, _ = add_call(orchestrator, observer, f"{PRIVATE}-sid")
    handler = FakeHandler()
    orchestrator._twilio_handlers[context.call_sid] = handler
    orchestrator.caller_service = object()

    class Pool:
        async def execute(self, *_args) -> None:
            return None

    async def get_pool():
        return Pool()

    import services.database as database

    monkeypatch.setattr(database, "get_db_pool", get_pool)
    await orchestrator.end_call(context.call_sid)
    await orchestrator.end_call(context.call_sid)

    assert len(emitted) == 1
    payload = json.loads(emitted[0])
    assert payload["event"] == "barge_in_diagnostics_summary"
    assert PRIVATE not in emitted[0]
    assert context.call_sid not in emitted[0]
    assert stt.disconnected is True
    assert handler.cleaned == 1
    assert observer.active_count == 0


@pytest.mark.asyncio
async def test_observer_failure_is_fixed_and_never_changes_call_behavior() -> None:
    call_sid = f"{PRIVATE}-failure"
    baseline, _, _ = await run_sequence(
        False, [b"\xff" * 160], call_sid=call_sid
    )
    orchestrator, observer, emitted = make_orchestrator(enabled=True)
    context, stt, trace = add_call(orchestrator, observer, call_sid)

    def fail_observation(*_args, **_kwargs):
        raise RuntimeError(PRIVATE)

    observer.observe_detector = fail_observation
    await orchestrator._handle_incoming_audio(context.call_sid)(b"\xff" * 160)
    enabled = {
        "outcomes": [context.state],
        "audio": stt.audio,
        "resets": stt.resets,
        "trace": trace,
        "consecutive": dict(orchestrator._barge_in_consecutive),
    }
    summary = observer.finish_call(context.barge_in_diagnostics)

    assert enabled == baseline
    assert summary["failure_category"] == DIAGNOSTIC_FAILURE
    assert summary["observer_failure_count"] == 1
    assert PRIVATE not in emitted[0]


def test_emission_failure_is_swallowed_and_not_retried() -> None:
    def fail_emit(_payload: str) -> None:
        raise RuntimeError(PRIVATE)

    observer = BargeInDiagnostics(emit=fail_emit)
    state = observer.start_call()
    summary = observer.finish_call(state)

    assert summary is not None
    assert state.failure_category == DIAGNOSTIC_FAILURE
    assert observer.finish_call(state) is None
    assert observer.active_count == 0


def test_setting_and_compose_passthrough_default_off(monkeypatch) -> None:
    monkeypatch.delenv("BARGE_IN_DIAGNOSTICS_ENABLED", raising=False)
    assert Settings(_env_file=None).barge_in_diagnostics_enabled is False
    monkeypatch.setenv("BARGE_IN_DIAGNOSTICS_ENABLED", "true")
    assert Settings(_env_file=None).barge_in_diagnostics_enabled is True

    compose = (Path(__file__).resolve().parents[2] / "docker-compose.yml").read_text(
        encoding="utf-8"
    )
    assert (
        "BARGE_IN_DIAGNOSTICS_ENABLED: "
        "${BARGE_IN_DIAGNOSTICS_ENABLED:-false}"
    ) in compose


class PlaybackWebSocket:
    def __init__(self, *, gate_interruption_clear: bool = False) -> None:
        self.handler = None
        self.messages: list[dict] = []
        self.clear_count = 0
        self.interruption_clear_started = asyncio.Event()
        self.interruption_clear_release = asyncio.Event()
        self.first_media = asyncio.Event()
        self.gate_interruption_clear = gate_interruption_clear

    async def send_json(self, message: dict) -> None:
        if message["event"] == "clear":
            self.clear_count += 1
            if self.gate_interruption_clear and self.clear_count == 2:
                self.interruption_clear_started.set()
                await self.interruption_clear_release.wait()
        self.messages.append(message)
        if message["event"] == "media":
            self.first_media.set()
        elif message["event"] == "mark":
            self.handler._on_mark(message)


class PlaybackStream:
    def __init__(self, payload: bytes, *, blocked: bool = False) -> None:
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

    def cancel(self) -> None:
        self.cancelled = True
        self.release.set()

    async def aclose(self) -> None:
        self.cancel()
        self.closed = True


class PlaybackTTS:
    def __init__(self, streams: dict[str, PlaybackStream]) -> None:
        self.streams = streams

    def create_stream(self, request):
        return self.streams[request.text]


class GatedResetSTT(FakeSTT):
    def __init__(self, trace: list[tuple]) -> None:
        super().__init__(trace)
        self.reset_started = asyncio.Event()
        self.reset_release = asyncio.Event()
        self.gate_reset = False

    async def reset_for_listening(self) -> None:
        self.resets += 1
        self.trace.append(("reset",))
        if self.gate_reset:
            self.reset_started.set()
            await self.reset_release.wait()


def add_playback_call(
    orchestrator,
    observer,
    call_sid: str,
    websocket: PlaybackWebSocket,
    *,
    gated_reset: bool = False,
):
    trace: list[tuple] = []
    context = ConversationContext(call_sid, "+10000000000", language="en")
    context.state = ConversationState.SPEAKING
    if observer is not None:
        context.barge_in_diagnostics = observer.start_call()
    handler = TwilioMediaStreamHandler(call_sid, f"stream-{call_sid}", websocket)
    websocket.handler = handler
    handler._is_streaming = True
    handler._is_connected = True
    stt = GatedResetSTT(trace) if gated_reset else FakeSTT(trace)
    orchestrator._conversations[call_sid] = context
    orchestrator._twilio_handlers[call_sid] = handler
    orchestrator._call_stt_instances[call_sid] = stt
    orchestrator._call_state_locks[call_sid] = asyncio.Lock()
    orchestrator._barge_in_reset_done[call_sid] = False
    orchestrator._barge_in_speech_start[call_sid] = 0
    return context, handler, stt


async def run_held_interruption(enabled: bool):
    stream = PlaybackStream(b"A" * 1600, blocked=True)
    websocket = PlaybackWebSocket(gate_interruption_clear=True)
    orchestrator, observer, _ = make_orchestrator(enabled=enabled)
    orchestrator.tts = PlaybackTTS({"speech": stream})
    context, handler, stt = add_playback_call(
        orchestrator, observer, "held-clear", websocket
    )
    speech_task = asyncio.create_task(
        orchestrator._speak_to_caller(context.call_sid, "speech", "en")
    )
    callback_task = None
    try:
        await asyncio.wait_for(websocket.first_media.wait(), timeout=0.5)
        callback = orchestrator._handle_incoming_audio(context.call_sid)
        for _ in range(7):
            await callback(b"\x00" * 160)
        callback_task = asyncio.create_task(callback(b"\x00" * 160))
        await asyncio.wait_for(
            websocket.interruption_clear_started.wait(), timeout=0.5
        )
        speech_result = await asyncio.wait_for(speech_task, timeout=0.5)
        websocket.interruption_clear_release.set()
        await asyncio.wait_for(callback_task, timeout=0.5)
        summary = (
            observer.snapshot(context.barge_in_diagnostics)
            if observer is not None
            else None
        )
        behavior = {
            "speech_result": speech_result,
            "state": context.state,
            "resets": stt.resets,
            "audio": list(stt.audio),
            "clear_count": websocket.clear_count,
            "stream_cancelled": stream.cancelled,
            "stream_closed": stream.closed,
        }
        return behavior, summary
    finally:
        websocket.interruption_clear_release.set()
        stream.release.set()
        tasks = [task for task in (speech_task, callback_task) if task is not None]
        await asyncio.gather(*tasks, return_exceptions=True)
        await handler.cleanup()


@pytest.mark.asyncio
async def test_held_interruption_results_stay_in_originating_window() -> None:
    disabled, _ = await run_held_interruption(False)
    enabled, summary = await run_held_interruption(True)

    assert enabled == disabled
    assert summary["windows_started"] == 1
    window = summary["windows"][0]
    assert window["gate_trigger_count"] == 1
    assert window["interrupt_succeeded_count"] == 1
    assert window["reset_succeeded_count"] == 1
    assert window["forwarded_bytes"] == 160
    assert window["outcome"] == "interrupted"


@pytest.mark.asyncio
async def test_post_tts_reset_stays_with_original_when_replacement_starts() -> None:
    first = PlaybackStream(b"A" * 1600)
    second = PlaybackStream(b"B" * 1600, blocked=True)
    websocket = PlaybackWebSocket()
    orchestrator, observer, _ = make_orchestrator(enabled=True)
    orchestrator.tts = PlaybackTTS({"first": first, "second": second})
    context, handler, stt = add_playback_call(
        orchestrator, observer, "replacement", websocket, gated_reset=True
    )
    stt.gate_reset = True
    first_task = asyncio.create_task(
        orchestrator._speak_to_caller(context.call_sid, "first", "en")
    )
    second_task = None
    try:
        await asyncio.wait_for(stt.reset_started.wait(), timeout=0.8)
        second_task = asyncio.create_task(
            orchestrator._speak_to_caller(context.call_sid, "second", "en")
        )
        await asyncio.wait_for(second.started.wait(), timeout=0.5)
        stt.gate_reset = False
        stt.reset_release.set()
        await asyncio.wait_for(first_task, timeout=0.5)
        summary = observer.snapshot(context.barge_in_diagnostics)

        assert summary["windows_started"] == 2
        assert summary["windows"][0]["reset_succeeded_count"] == 1
        assert summary["windows"][1]["reset_succeeded_count"] == 0
    finally:
        stt.reset_release.set()
        second.release.set()
        tasks = [task for task in (first_task, second_task) if task is not None]
        await asyncio.gather(*tasks, return_exceptions=True)
        await handler.cleanup()


def test_late_window_events_do_not_resurrect_or_cross_owners() -> None:
    observer = BargeInDiagnostics()
    state = observer.start_call()
    evicted = observer.start_speaking(state)
    observer.end_speaking(state, evicted, "completed")
    for _ in range(MAX_SPEAKING_WINDOWS):
        retained = observer.start_speaking(state)
        observer.end_speaking(state, retained, "completed")
    before_started = state.windows_started
    observer.observe_action(state, "reset", "succeeded", window=evicted)
    assert evicted.reset_succeeded_count == 0
    assert state.windows_started == before_started

    foreign_state = observer.start_call()
    foreign_window = observer.start_speaking(foreign_state)
    observer.observe_route(state, "forwarded", 160, window=foreign_window)
    assert foreign_window.forwarded_bytes == 0
    assert state.forwarded_bytes == 0

    observer.finish_call(state)
    observer.observe_action(state, "interrupt", "succeeded", window=retained)
    assert retained.interrupt_succeeded_count == 0
    assert observer.ensure_speaking(state) is None


@pytest.mark.parametrize("operation", ["detector", "action", "route", "end"])
def test_missing_window_ownership_does_not_adopt_current_window(operation) -> None:
    observer = BargeInDiagnostics()
    state = observer.start_call()
    window = observer.start_speaking(state)
    if operation == "detector":
        observer.observe_detector(state, "gate_fired", window=None)
    elif operation == "action":
        observer.observe_action(state, "reset", "succeeded", window=None)
    elif operation == "route":
        observer.observe_route(state, "forwarded", 160, window=None)
    else:
        observer.end_speaking(state, None, "completed")
    assert window.gate_trigger_count == 0
    assert window.reset_succeeded_count == 0
    assert window.forwarded_bytes == 0
    assert window.ended_at is None
    assert state.current_window is window


@pytest.mark.parametrize("failure_stage", ["clock", "snapshot", "format"])
def test_teardown_failure_is_terminal_and_releases_capacity(
    monkeypatch, failure_stage
) -> None:
    clock = ControlledClock()
    observer = BargeInDiagnostics(clock=clock, max_active_calls=1)
    state = observer.start_call()
    observer.start_speaking(state)
    orchestrator, _, _ = make_orchestrator(enabled=False)
    orchestrator._barge_in_observer = observer

    if failure_stage == "clock":
        def fail_clock():
            raise RuntimeError(PRIVATE)

        observer._clock = fail_clock
    elif failure_stage == "snapshot":
        monkeypatch.setattr(
            observer, "snapshot", lambda _state: (_ for _ in ()).throw(RuntimeError(PRIVATE))
        )
    else:
        import services.conversation.barge_in_diagnostics as diagnostics

        monkeypatch.setattr(
            diagnostics.json,
            "dumps",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(PRIVATE)),
        )

    assert orchestrator._observe_barge_in(state, "finish_call") is None
    assert state.finished is True
    assert state.failure_category == DIAGNOSTIC_FAILURE
    assert observer.active_count == 0
    assert observer.finish_call(state) is None

    observer._clock = clock
    assert observer.start_call() is not None


def test_emission_failure_is_terminal_and_emitted_once() -> None:
    attempts = 0

    def fail_emit(_payload: str) -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError(PRIVATE)

    observer = BargeInDiagnostics(emit=fail_emit, max_active_calls=1)
    state = observer.start_call()
    assert observer.finish_call(state) is not None
    assert state.finished is True
    assert state.failure_category == DIAGNOSTIC_FAILURE
    assert observer.active_count == 0
    assert observer.finish_call(state) is None
    assert attempts == 1
    assert observer.start_call() is not None


@pytest.mark.asyncio
async def test_actual_speech_completion_opens_and_closes_one_window() -> None:
    stream = PlaybackStream(b"G" * 1600)
    websocket = PlaybackWebSocket()
    orchestrator, observer, _ = make_orchestrator(enabled=True)
    orchestrator.tts = PlaybackTTS({"greeting": stream})
    context, handler, stt = add_playback_call(
        orchestrator, observer, "greeting", websocket
    )
    try:
        assert await orchestrator._speak_to_caller(
            context.call_sid, "greeting", "en"
        ) is True
        summary = observer.snapshot(context.barge_in_diagnostics)
        assert summary["windows_started"] == 1
        assert summary["windows"][0]["outcome"] == "completed"
        assert summary["windows"][0]["reset_succeeded_count"] == 1
        assert stt.resets == 1
    finally:
        await handler.cleanup()
