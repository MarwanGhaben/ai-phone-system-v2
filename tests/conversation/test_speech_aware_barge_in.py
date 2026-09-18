from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from services.conversation.orchestrator import (
    ConversationContext,
    ConversationOrchestrator,
    ConversationState,
    _SpeechOwner,
)
from services.conversation.speech_activity_gate import GateDecision, SpeechActivityGate


class FakeStream:
    def __init__(self) -> None:
        self.cancelled = False
        self.closed = False

    def cancel(self) -> None:
        self.cancelled = True

    async def aclose(self) -> None:
        self.closed = True


class FakeHandler:
    def __init__(self, trace) -> None:
        self.trace = trace
        self.cleaned = 0
        self.invalidated = 0

    async def clear_audio(self, playback) -> None:
        self.trace.append(("clear", playback))

    def invalidate_playback_session(self) -> None:
        self.invalidated += 1

    def invalidate_playback(self, _playback) -> None:
        self.invalidated += 1

    async def cleanup(self) -> None:
        self.cleaned += 1


class FakeSTT:
    def __init__(self, trace, *, reset_release=None) -> None:
        self.trace = trace
        self.audio = []
        self.resets = 0
        self.reset_entered = asyncio.Event()
        self.reset_release = reset_release
        self.disconnected = False

    async def reset_for_listening(self) -> None:
        self.resets += 1
        self.trace.append(("reset",))
        self.reset_entered.set()
        if self.reset_release is not None:
            await self.reset_release.wait()

    async def stream_audio(self, chunk) -> None:
        self.audio.append(chunk.data)
        self.trace.append(("forward", chunk.data))

    async def disconnect(self) -> None:
        self.disconnected = True


class ControlledGate:
    def __init__(self, decisions) -> None:
        self.decisions = iter(decisions)
        self.retired = False
        self.reset_count = 0

    async def analyze(self, _audio):
        result = next(self.decisions)
        if isinstance(result, BaseException):
            raise result
        return result

    def reset_evidence(self) -> None:
        self.reset_count += 1

    def retire(self) -> None:
        self.retired = True


class DelayedGate(ControlledGate):
    def __init__(self, decision) -> None:
        super().__init__([decision])
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def analyze(self, _audio):
        self.entered.set()
        await self.release.wait()
        return await super().analyze(_audio)


def make_orchestrator():
    orchestrator = ConversationOrchestrator.__new__(ConversationOrchestrator)
    orchestrator._speech_aware_barge_in_enabled = True
    orchestrator._barge_in_observer = None
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
    return orchestrator


def add_speaking_call(orchestrator, call_sid, gate, *, reset_release=None):
    trace = []
    context = ConversationContext(call_sid, "+10000000000")
    context.state = ConversationState.SPEAKING
    context.speech_activity_started_at = 0.0
    handler = FakeHandler(trace)
    stt = FakeSTT(trace, reset_release=reset_release)
    stream = FakeStream()
    owner = _SpeechOwner(context, handler, stream, f"playback-{call_sid}")
    context.speech_activity_gate = gate
    context.speech_activity_owner = owner
    orchestrator._conversations[call_sid] = context
    orchestrator._twilio_handlers[call_sid] = handler
    orchestrator._call_stt_instances[call_sid] = stt
    orchestrator._call_state_locks[call_sid] = asyncio.Lock()
    orchestrator._active_speech[call_sid] = owner
    orchestrator._barge_in_reset_done[call_sid] = False
    return context, owner, handler, stt, trace


@pytest.mark.asyncio
async def test_accept_forwards_retained_onset_once_after_clear_and_reset():
    onset = b"onset-and-trigger-tail"
    gate = ControlledGate([GateDecision.accepted(onset)])
    orchestrator = make_orchestrator()
    context, _owner, _handler, stt, trace = add_speaking_call(
        orchestrator, "accepted", gate
    )

    await orchestrator._handle_incoming_audio("accepted")(b"trigger-callback")

    assert trace == [
        ("clear", "playback-accepted"),
        ("reset",),
        ("forward", onset),
    ]
    assert stt.audio == [onset]
    assert context.state == ConversationState.LISTENING
    assert gate.retired


@pytest.mark.asyncio
async def test_rejected_non_speech_never_clears_resets_or_forwards():
    gate = ControlledGate([GateDecision.rejected()])
    orchestrator = make_orchestrator()
    context, owner, _handler, stt, trace = add_speaking_call(
        orchestrator, "noise", gate
    )

    await orchestrator._handle_incoming_audio("noise")(b"tone")

    assert trace == [] and stt.resets == 0 and stt.audio == []
    assert orchestrator._active_speech["noise"] is owner
    assert context.state == ConversationState.SPEAKING


@pytest.mark.asyncio
async def test_classifier_failure_at_callback_seam_retires_without_legacy_fallback():
    class FailingClassifier:
        def create_state(self):
            return object()

        def reset_state(self, _state) -> None:
            return None

        def classify(self, _frame, _state):
            raise RuntimeError("private failure details")

    warnings = []
    gate = SpeechActivityGate(FailingClassifier(), warning=warnings.append)
    orchestrator = make_orchestrator()
    context, owner, _handler, stt, trace = add_speaking_call(
        orchestrator, "failure", gate
    )

    await orchestrator._handle_incoming_audio("failure")(b"\xff" * 256)

    assert warnings == ["inference_failed"]
    assert trace == [] and stt.resets == 0 and stt.audio == []
    assert orchestrator._active_speech["failure"] is owner
    assert context.state == ConversationState.SPEAKING
    assert gate.retired


@pytest.mark.asyncio
async def test_callbacks_during_reset_follow_onset_in_exact_order():
    reset_release = asyncio.Event()
    onset = b"retained-onset"
    gate = ControlledGate([GateDecision.accepted(onset)])
    orchestrator = make_orchestrator()
    context, _owner, _handler, stt, trace = add_speaking_call(
        orchestrator, "ordered", gate, reset_release=reset_release
    )
    callback = orchestrator._handle_incoming_audio("ordered")
    trigger = asyncio.create_task(callback(b"trigger"))
    await stt.reset_entered.wait()

    await callback(b"later-one")
    await callback(b"later-two")
    reset_release.set()
    await trigger

    assert stt.audio == [onset, b"later-onelater-two"]
    assert trace[-2:] == [("forward", onset), ("forward", b"later-onelater-two")]
    assert context.state == ConversationState.LISTENING


@pytest.mark.asyncio
async def test_two_calls_are_isolated():
    orchestrator = make_orchestrator()
    gate_a = ControlledGate([GateDecision.accepted(b"A-onset")])
    gate_b = ControlledGate([GateDecision.rejected()])
    context_a, _, _, stt_a, _ = add_speaking_call(orchestrator, "A", gate_a)
    context_b, owner_b, _, stt_b, _ = add_speaking_call(orchestrator, "B", gate_b)

    await asyncio.gather(
        orchestrator._handle_incoming_audio("A")(b"A"),
        orchestrator._handle_incoming_audio("B")(b"B"),
    )

    assert context_a.state == ConversationState.LISTENING
    assert stt_a.audio == [b"A-onset"]
    assert context_b.state == ConversationState.SPEAKING
    assert stt_b.audio == [] and orchestrator._active_speech["B"] is owner_b


@pytest.mark.asyncio
async def test_same_id_old_callback_and_delayed_inference_cannot_touch_replacement():
    orchestrator = make_orchestrator()
    old_gate = DelayedGate(GateDecision.accepted(b"old"))
    old_context, _, _, old_stt, _ = add_speaking_call(
        orchestrator, "same", old_gate
    )
    callback = orchestrator._handle_incoming_audio("same")
    pending = asyncio.create_task(callback(b"old-input"))
    await old_gate.entered.wait()

    new_gate = ControlledGate([GateDecision.rejected()])
    new_context, new_owner, _, new_stt, _ = add_speaking_call(
        orchestrator, "same", new_gate
    )
    old_context.state = ConversationState.ENDED
    old_gate.retire()
    old_gate.release.set()
    await pending

    assert new_context.state == ConversationState.SPEAKING
    assert orchestrator._active_speech["same"] is new_owner
    assert old_stt.resets == new_stt.resets == 0
    assert old_stt.audio == new_stt.audio == []


@pytest.mark.asyncio
async def test_delayed_reset_cannot_forward_to_replacement_call():
    release = asyncio.Event()
    orchestrator = make_orchestrator()
    gate = ControlledGate([GateDecision.accepted(b"old-onset")])
    old_context, _, _, old_stt, _ = add_speaking_call(
        orchestrator, "reuse", gate, reset_release=release
    )
    pending = asyncio.create_task(
        orchestrator._handle_incoming_audio("reuse")(b"trigger")
    )
    await old_stt.reset_entered.wait()

    new_gate = ControlledGate([GateDecision.rejected()])
    new_context, new_owner, _, new_stt, _ = add_speaking_call(
        orchestrator, "reuse", new_gate
    )
    old_context.state = ConversationState.ENDED
    release.set()
    await pending

    assert old_stt.audio == [] and new_stt.audio == []
    assert orchestrator._conversations["reuse"] is new_context
    assert orchestrator._active_speech["reuse"] is new_owner


@pytest.mark.asyncio
async def test_grace_resets_evidence_and_default_off_retains_legacy_seam(monkeypatch):
    gate = ControlledGate([GateDecision.accepted(b"should-not-run")])
    orchestrator = make_orchestrator()
    context, owner, _, stt, trace = add_speaking_call(orchestrator, "grace", gate)
    context.speech_activity_started_at = __import__("time").monotonic()

    await orchestrator._handle_incoming_audio("grace")(b"quiet")
    assert gate.reset_count == 1 and trace == []
    assert orchestrator._active_speech["grace"] is owner

    orchestrator._speech_aware_barge_in_enabled = False

    async def detect(_audio, _sid):
        return True

    async def interrupt(_sid):
        trace.append(("legacy-interrupt",))
        return True

    monkeypatch.setattr(orchestrator, "_detect_barge_in", detect)
    monkeypatch.setattr(orchestrator, "_interrupt_call_speech", interrupt)
    await orchestrator._handle_incoming_audio("grace")(b"legacy-bytes")

    assert stt.audio == [b"legacy-bytes"]
    assert trace[-3:] == [
        ("legacy-interrupt",),
        ("reset",),
        ("forward", b"legacy-bytes"),
    ]


@pytest.mark.asyncio
async def test_enabled_playback_completion_retires_gate_and_returns_to_listening():
    trace = []

    class CompletingHandler(FakeHandler):
        def __init__(self):
            super().__init__(trace)
            self.playback = object()

        async def begin_playback(self):
            return self.playback

        def is_current_playback(self, playback):
            return playback is self.playback

        async def stream_audio_chunks(self, playback, _stream):
            assert playback is self.playback
            return SimpleNamespace(bytes_sent=1600, completed=True)

        async def wait_for_playback(self, playback, _result, timeout):
            assert playback is self.playback and timeout >= 1
            return True

        def release_playback(self, playback):
            if playback is self.playback:
                self.playback = None

    class Classifier:
        def create_state(self):
            return {}

        def reset_state(self, _state):
            return None

        def classify(self, _frame, _state):
            raise AssertionError("no caller audio was sent")

    class TTS:
        def create_stream(self, _request):
            return FakeStream()

    orchestrator = make_orchestrator()
    orchestrator.tts = TTS()
    orchestrator._local_vad_model = Classifier()
    orchestrator._speech_gate_config = {
        "probability_threshold": 0.5,
        "speech_duration_ms": 160,
        "max_input_gap_ms": 96,
        "max_inference_ms": 20.0,
    }
    context = ConversationContext("complete", "+10000000000")
    context.state = ConversationState.SPEAKING
    handler = CompletingHandler()
    stt = FakeSTT(trace)
    orchestrator._conversations["complete"] = context
    orchestrator._twilio_handlers["complete"] = handler
    orchestrator._call_stt_instances["complete"] = stt
    orchestrator._call_state_locks["complete"] = asyncio.Lock()

    assert await orchestrator._speak_to_caller("complete", "hello", "en")

    assert context.state == ConversationState.LISTENING
    assert context.speech_activity_gate is None
    assert context.speech_activity_owner is None
    assert context.speech_activity_flow is None
    assert stt.resets == 1


@pytest.mark.asyncio
async def test_repeated_teardown_releases_gate_flow_and_owner(monkeypatch):
    orchestrator = make_orchestrator()
    gate = ControlledGate([GateDecision.rejected()])
    context, owner, handler, stt, _ = add_speaking_call(
        orchestrator, "teardown", gate
    )
    context.language = "auto"
    context.speech_activity_flow = SimpleNamespace(trailing=bytearray(b"bounded"))
    orchestrator.caller_service = object()

    class Pool:
        async def execute(self, *_args):
            return None

    async def get_pool():
        return Pool()

    import services.database as database

    monkeypatch.setattr(database, "get_db_pool", get_pool)
    await orchestrator.end_call("teardown")
    await orchestrator.end_call("teardown")

    assert gate.retired and owner.active is False
    assert stt.disconnected and handler.cleaned == 1
    assert "teardown" not in orchestrator._conversations
    assert "teardown" not in orchestrator._active_speech


def test_settings_and_compose_are_bounded_and_default_off(monkeypatch):
    from config.settings import Settings

    required = {
        "SECRET_KEY": "test",
        "DATABASE_URL": "postgresql://test",
        "TWILIO_ACCOUNT_SID": "test",
        "TWILIO_AUTH_TOKEN": "test",
        "TWILIO_PHONE_NUMBER": "test",
        "DEEPGRAM_API_KEY": "test",
        "ELEVENLABS_API_KEY": "test",
        "OPENAI_API_KEY": "test",
    }
    for name in required:
        monkeypatch.setenv(name, required[name])
    monkeypatch.delenv("SPEECH_AWARE_BARGE_IN_ENABLED", raising=False)
    settings = Settings(_env_file=None)
    assert settings.speech_aware_barge_in_enabled is False
    assert settings.local_vad_speech_duration_ms == 160

    monkeypatch.setenv("LOCAL_VAD_PROBABILITY_THRESHOLD", "nan")
    with pytest.raises(ValueError):
        Settings(_env_file=None)
    monkeypatch.setenv("LOCAL_VAD_PROBABILITY_THRESHOLD", "0.5")
    monkeypatch.setenv("LOCAL_VAD_SPEECH_DURATION_MS", "321")
    with pytest.raises(ValueError):
        Settings(_env_file=None)

    compose = (Path(__file__).resolve().parents[2] / "docker-compose.yml").read_text(
        encoding="utf-8"
    )
    assert "SPEECH_AWARE_BARGE_IN_ENABLED: ${SPEECH_AWARE_BARGE_IN_ENABLED:-false}" in compose


def test_disabled_orchestrator_does_not_import_onnx_or_load_model(monkeypatch):
    import sys
    import services.conversation.orchestrator as orchestrator_module
    from config.settings import Settings
    from services.callers import caller_service

    settings = Settings(_env_file=None, SPEECH_AWARE_BARGE_IN_ENABLED=False)
    monkeypatch.setattr(orchestrator_module, "get_settings", lambda: settings)
    monkeypatch.setattr(orchestrator_module, "create_elevenlabs_tts", lambda _config: object())
    monkeypatch.setattr(orchestrator_module, "create_openai_llm", lambda _config: object())
    monkeypatch.setattr(
        orchestrator_module.ConversationOrchestrator,
        "_get_system_prompt",
        lambda _self: "synthetic",
    )
    monkeypatch.setattr(caller_service, "get_caller_service", lambda: object())
    sys.modules.pop("onnxruntime", None)

    orchestrator = orchestrator_module.ConversationOrchestrator()

    assert orchestrator._speech_aware_barge_in_enabled is False
    assert orchestrator._local_vad_model is None
    assert "onnxruntime" not in sys.modules


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["clear", "reset", "forward"])
async def test_failed_interruption_releases_live_call_to_listening(failure_stage):
    orchestrator = make_orchestrator()
    gate = ControlledGate([GateDecision.accepted(b"onset")])
    context, _, handler, stt, _ = add_speaking_call(orchestrator, "failure", gate)

    async def fail(*_args):
        raise RuntimeError("synthetic failure")

    original_forward = stt.stream_audio
    if failure_stage == "clear":
        handler.clear_audio = fail
    elif failure_stage == "reset":
        stt.reset_for_listening = fail
    else:
        stt.stream_audio = fail
    callback = orchestrator._handle_incoming_audio("failure")
    with pytest.raises(RuntimeError):
        await callback(b"trigger")

    assert context.state == ConversationState.LISTENING
    assert context.speech_activity_flow is None
    assert "failure" not in orchestrator._active_speech
    assert gate.retired
    stt.stream_audio = original_forward
    await callback(b"later caller speech")
    assert stt.audio == [b"later caller speech"]


@pytest.mark.asyncio
async def test_cancelled_reset_releases_flow_without_awaiting_cleanup():
    release = asyncio.Event()
    orchestrator = make_orchestrator()
    gate = ControlledGate([GateDecision.accepted(b"onset")])
    context, _, _, stt, _ = add_speaking_call(
        orchestrator, "cancel", gate, reset_release=release
    )
    callback = orchestrator._handle_incoming_audio("cancel")
    task = asyncio.create_task(callback(b"trigger"))
    try:
        await asyncio.wait_for(stt.reset_entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert context.state == ConversationState.LISTENING
        assert context.speech_activity_flow is None
        assert gate.retired
        await callback(b"later caller speech")
        assert stt.audio == [b"later caller speech"]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
