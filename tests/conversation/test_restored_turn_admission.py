"""Actual restored consumer regression, with providers replaced at their boundary."""
import asyncio
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import AsyncMock

import pytest

from services.conversation.events import UtteranceIdentity
from services.conversation.language_policy import explicit_language_request, persistent_language_request
from services.stt.stt_base import STTResult

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runtime(monkeypatch):
    spec = importlib.util.spec_from_file_location("restored_availability_builder", ROOT / "scripts/deploy-restored-availability.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    source = builder.patch_orchestrator(subprocess.check_output([
        "git", "show", builder.RESTORE + ":services/conversation/orchestrator.py"], cwd=ROOT))
    from scripts.build_restored_turn_repair import patch_orchestrator
    from services.conversation import restored_turn_admission as admission
    monkeypatch.setattr(admission, "wall_time", lambda: 101.0)
    module = types.ModuleType("restored_turn_test_runtime")
    sys.modules[module.__name__] = module
    module.repaired_source = patch_orchestrator(source)
    module.baseline_source = source
    exec(compile(module.repaired_source, "restored_runtime", "exec"), module.__dict__)
    return module


def make_agent(runtime, text="Please speak English", *, received=99, language="ar", phase="SPEAKING"):
    agent = runtime.ConversationOrchestrator.__new__(runtime.ConversationOrchestrator)
    context = runtime.ConversationContext("call", "+10000000000", language=language)
    context.state = getattr(runtime.ConversationState, phase)
    context.pending_booking = {"confirmed_available": True}
    result = STTResult(text, "en", None, True,
        utterance_id=UtteranceIdentity("elevenlabs_scribe", "synthetic", 1),
        received_at=datetime.fromtimestamp(received, timezone.utc))
    class STT:
        language = "ar"
        async def get_transcript(self):
            yield result
            context.state = runtime.ConversationState.ENDED
    stt = STT()
    agent._conversations = {"call": context}
    agent._call_stt_instances = {"call": stt}
    agent._call_state_locks = {"call": asyncio.Lock()}
    agent._garbled_drop_count = {}
    agent._echo_guard_until = {}
    agent._barge_in_speech_start = {"call": 100.0}
    agent._get_ai_response = AsyncMock(return_value="response")
    agent._reconnect_stt_with_language = AsyncMock()
    async def speak(*args):
        context.state = runtime.ConversationState.LISTENING
        return True
    agent._speak_to_caller = AsyncMock(side_effect=speak)
    return agent, context, result, stt


@pytest.mark.parametrize("text,expected", [
    ("English please, I want to book with Rami", "en"),
    ("Please speak English and check Rami tomorrow", "en"),
    ("Please speak English I want to book with Rami", "en"),
    ("احكي معي بالعربي بدي موعد مع رامي", "ar"),
    ("احكي معي بالعربي، بدي موعد مع رامي", "ar"),
    ("بالإنجليزي من فضلك، بدي موعد", "en"),
    ("I want to book with Rami, English please", "en"),
    ("English please, Arabic please", None),
    ("He said English please, I want to book", None),
    ('"English please", I want to book', None),
    ("Don't speak English, book Rami", None),
    ("English please, no actually Arabic please", None),
])
def test_combined_language_requests(text, expected):
    assert explicit_language_request(text) == expected


def test_booking_save_word_does_not_persist_language_preference():
    assert explicit_language_request("English please, save my appointment") == "en"
    assert not persistent_language_request("English please, save my appointment")
    assert persistent_language_request("Always speak to me in English")


@pytest.mark.asyncio
@pytest.mark.parametrize("text,initial,expected", [
    ("Please speak English", "ar", "en"),
    ("Arabic please", "en", "ar"),
    ("No, Tuesday at four thirty, not Monday", "en", "en"),
    ("English please, I want to book with Rami", "ar", "en"),
])
async def test_queued_caller_final_survives_playback(runtime, text, initial, expected):
    from services.conversation import restored_turn_admission as admission
    agent, context, _, _ = make_agent(runtime, text, language=initial)
    waiting = asyncio.Event()
    release = asyncio.Event()
    async def pause(_):
        waiting.set()
        await release.wait()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(admission, "pause", pause)
        task = asyncio.create_task(agent._consume_stt_transcripts("call"))
        try:
            await asyncio.wait_for(waiting.wait(), 1)
            agent._get_ai_response.assert_not_awaited()
            context.state = runtime.ConversationState.LISTENING
            release.set()
            await asyncio.wait_for(task, 1)
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    agent._get_ai_response.assert_awaited_once_with("call", text)
    assert context.language == expected


@pytest.mark.asyncio
async def test_uncertain_overlap_cannot_confirm_booking(runtime):
    from services.conversation import restored_turn_admission as admission
    agent, context, result, stt = make_agent(runtime, "yes", received=101)
    async def pause(_):
        context.state = runtime.ConversationState.LISTENING
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(admission, "pause", pause)
        await agent._consume_stt_transcripts("call")
    agent._get_ai_response.assert_not_awaited()
    assert context.pending_booking is None
    assert agent._speak_to_caller.await_count == 1


@pytest.mark.asyncio
async def test_replacement_call_never_gets_delayed_transcript(runtime):
    from services.conversation import restored_turn_admission as admission
    agent, old, result, stt = make_agent(runtime)
    replacement = runtime.ConversationContext("call", "+10000000001", language="en")
    async def pause(_):
        agent._conversations["call"] = replacement
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(admission, "pause", pause)
        assert not await admission.admit_final(agent, "call", old, stt, result)
    agent._get_ai_response.assert_not_awaited()
    agent._speak_to_caller.assert_not_awaited()
    assert replacement.conversation_history == []


@pytest.mark.asyncio
async def test_echo_guard_defers_old_commit_instead_of_discarding_it(runtime):
    from services.conversation import restored_turn_admission as admission
    agent, context, result, stt = make_agent(runtime, phase="LISTENING")
    agent._echo_guard_until["call"] = 101.15
    waited = []
    async def pause(delay):
        waited.append(delay)
        agent._echo_guard_until["call"] = 0
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(admission, "pause", pause)
        await agent._consume_stt_transcripts("call")
    assert waited == [0.02]
    agent._get_ai_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_overlap_committed_during_playback_is_not_authorized_when_dequeued_later(runtime):
    from services.conversation import restored_turn_admission as admission
    agent, context, result, stt = make_agent(runtime, "yes", received=100.5, phase="LISTENING")
    agent._echo_guard_until["call"] = 100.8
    assert not await admission.admit_final(agent, "call", context, stt, result)
    assert context.pending_booking is None
    assert agent._speak_to_caller.await_count == 1


@pytest.mark.asyncio
async def test_repeated_uncertain_finals_prompt_once_then_fresh_speech_recovers(runtime):
    from services.conversation import restored_turn_admission as admission
    agent, context, result, stt = make_agent(runtime, "yes", received=100.5, phase="LISTENING")
    agent._echo_guard_until["call"] = 100.8
    for _ in range(3):
        assert not await admission.admit_final(agent, "call", context, stt, result)
    assert agent._speak_to_caller.await_count == 1
    result.received_at = datetime.fromtimestamp(101, timezone.utc)
    assert await admission.admit_final(agent, "call", context, stt, result)
    assert not context._restored_admission.clarification_latched


@pytest.mark.asyncio
async def test_wait_cancellation_propagates_without_reply(runtime):
    from services.conversation import restored_turn_admission as admission
    agent, context, result, stt = make_agent(runtime)
    async def pause(_):
        raise asyncio.CancelledError
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(admission, "pause", pause)
        with pytest.raises(asyncio.CancelledError):
            await admission.admit_final(agent, "call", context, stt, result)
    agent._speak_to_caller.assert_not_awaited()


@pytest.mark.asyncio
async def test_stuck_playback_is_bounded_and_invalidates_pending_offer(runtime):
    from services.conversation import restored_turn_admission as admission
    agent, context, result, stt = make_agent(runtime)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(admission, "MAX_WAIT_SECONDS", 0)
        assert not await admission.admit_final(agent, "call", context, stt, result)
    assert context.pending_booking is None
    agent._speak_to_caller.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_receipt", [None, datetime(2026, 1, 1), datetime.fromtimestamp(102, timezone.utc)])
async def test_missing_naive_future_receipts_cannot_bypass_overlap(runtime, bad_receipt):
    from services.conversation import restored_turn_admission as admission
    agent, context, result, stt = make_agent(runtime)
    result.received_at = bad_receipt
    async def pause(_):
        context.state = runtime.ConversationState.LISTENING
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(admission, "pause", pause)
        assert not await admission.admit_final(agent, "call", context, stt, result)
    assert context.pending_booking is None


def test_builder_refuses_other_source_and_preserves_booking_methods(runtime):
    import ast
    from scripts.build_restored_turn_repair import patch_orchestrator
    with pytest.raises(ValueError):
        patch_orchestrator(b"unrecognized source")
    before = ast.parse(runtime.baseline_source)
    after = ast.parse(runtime.repaired_source)
    methods = lambda tree: {n.name: ast.dump(n, include_attributes=False)
                            for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)}
    original, repaired = methods(before), methods(after)
    assert original.keys() == repaired.keys()
    assert {name for name in original if original[name] != repaired[name]} == {"_consume_stt_transcripts"}


@pytest.mark.asyncio
async def test_real_playback_mark_and_cleanup_release_queued_language_request(runtime, monkeypatch):
    import time
    from services.conversation import restored_turn_admission as admission
    from services.telephony.twilio_service import TwilioMediaStreamHandler
    from tests.conversation.test_playback_ownership import FakeTTS, FakeStream
    now = time.time()
    monkeypatch.setattr(admission, "wall_time", time.time)
    agent, context, result, stt = make_agent(runtime, received=now - 0.1)
    agent._barge_in_speech_start["call"] = now
    agent._speech_setup_locks = {}
    agent._active_speech = {}
    agent._barge_in_reset_done = {}
    stt.reset_for_listening = AsyncMock()
    mark_seen = asyncio.Event()
    class WebSocket:
        async def send_json(self, message):
            if message["event"] == "mark":
                self.mark = message
                mark_seen.set()
    websocket = WebSocket()
    handler = TwilioMediaStreamHandler("call", "stream", websocket)
    handler._is_connected = handler._is_streaming = True
    agent._twilio_handlers = {"call": handler}
    agent.tts = FakeTTS({"prompt": FakeStream(b"A" * 1600)})
    # Execute the actual restored playback implementation for the first prompt.
    speak = asyncio.create_task(runtime.ConversationOrchestrator._speak_to_caller(
        agent, "call", "prompt", "en"))
    consume = None
    try:
        await asyncio.wait_for(mark_seen.wait(), 2)
        consume = asyncio.create_task(agent._consume_stt_transcripts("call"))
        await asyncio.sleep(0)
        agent._get_ai_response.assert_not_awaited()
        handler._on_mark(websocket.mark)
        assert await asyncio.wait_for(speak, 2)
        await asyncio.wait_for(consume, 2)
        agent._get_ai_response.assert_awaited_once_with("call", "Please speak English")
        assert context.language == "en"
    finally:
        tasks = [t for t in (speak, consume) if t is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await handler.cleanup()
