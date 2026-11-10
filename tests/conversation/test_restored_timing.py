"""Timing instrumentation applied to the exact deployed restored orchestrator."""
import asyncio
import difflib
import hashlib
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import types
from unittest.mock import AsyncMock

from loguru import logger
import pytest

ROOT = Path(__file__).resolve().parents[2]
LINE = re.compile(r"^TurnTiming( [a-z_]+=(-?\d+|en|ar|other))+$")
PINNED_RESULT_SHA = "f8ee6d927ce287425d9ce46a72ec154322e9a6f983fe4381cb5ce23331703f43"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


@pytest.fixture(scope="module")
def timing_builder():
    return _load("restored_timing_builder", "scripts/build_restored_timing.py")


@pytest.fixture(scope="module")
def baseline():
    availability = _load("restored_availability_builder_timing", "scripts/deploy-restored-availability.py")
    turn = _load("restored_turn_builder_timing", "scripts/build_restored_turn_repair.py")
    reply = _load("restored_reply_builder_timing", "scripts/build_restored_reply_repair.py")
    original = subprocess.check_output(
        ["git", "-c", "safe.directory=*", "show",
         availability.RESTORE + ":services/conversation/orchestrator.py"], cwd=ROOT)
    return reply.patch_orchestrator(turn.patch_orchestrator(availability.patch_orchestrator(original)))


@pytest.fixture(scope="module")
def patched(timing_builder, baseline):
    return timing_builder.patch_orchestrator(baseline)


def test_baseline_reconstructs_to_the_pinned_hash(timing_builder, baseline):
    assert _digest(baseline) == timing_builder.BASELINE_SHA
    assert timing_builder.BASELINE_SHA == "5ef96383a930d4fa38e8943bac7abd0afc47fd965b639ac42b6270b6410b7561"


def test_pinned_output_hash(timing_builder, patched):
    assert _digest(patched) == timing_builder.PATCHED_SHA
    assert timing_builder.PATCHED_SHA == PINNED_RESULT_SHA


def test_patch_is_insert_only_and_compiles(baseline, patched):
    old = baseline.decode().split("\n")
    new = patched.decode().split("\n")
    inserted = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        assert tag in ("equal", "insert"), (tag, old[i1:i2][:3])
        if tag == "insert":
            inserted += j2 - j1
    assert inserted == len(new) - len(old) > 0
    # Dropping every inserted line must give the baseline back byte for byte.
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    kept = [new[j] for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag == "equal" for j in range(j1, j2)]
    assert "\n".join(kept).encode() == baseline
    compile(patched, "patched", "exec")


def test_every_insertion_is_guarded_and_awaits_nothing(baseline, patched):
    old = baseline.decode().split("\n")
    new = patched.decode().split("\n")
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    blocks = [new[j1:j2] for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag == "insert"]
    assert len(blocks) == 18
    for block in blocks:
        text = "\n".join(block)
        assert "await " not in text.replace("await event.wait", "")
        if text.startswith("try:\n    from services.conversation import turn_timing"):
            assert "except Exception" in text
            continue
        stripped = [line.strip() for line in block]
        if stripped[0] == "_tt = None":
            assert "except Exception:" in text and "_tt = None" in stripped[-1]
            continue
        assert stripped[0] == "try:" and stripped[-2:] == ["except Exception:", "pass"], block


def test_modified_or_unreviewed_input_is_refused(timing_builder, baseline):
    for mutated in (baseline + b"# unreviewed\n", b"", b"unexpected production source",
                    baseline.replace(b"max_tokens=220,", b"max_tokens=221,", 1)):
        with pytest.raises(RuntimeError):
            timing_builder.patch_orchestrator(mutated)


def test_anchor_drift_with_matching_hash_is_refused(timing_builder, baseline, monkeypatch):
    changed = baseline.replace(b"audio_stream = self.tts.create_stream(request)",
                               b"audio_stream = self.tts.create_stream(request) ")
    monkeypatch.setattr(timing_builder, "BASELINE_SHA", _digest(changed))
    with pytest.raises(RuntimeError, match="anchor differs"):
        timing_builder.patch_orchestrator(changed)


def test_crlf_input_gives_the_same_result(timing_builder, baseline, patched):
    assert timing_builder.patch_orchestrator(baseline.replace(b"\n", b"\r\n")) == patched


# ---------------------------------------------------------------------------
# Behaviour: the real process_transcript, _get_ai_response and _speak_to_caller
# with providers replaced at their boundary.
# ---------------------------------------------------------------------------

def _runtime(source, name):
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, name, "exec"), module.__dict__)
    return module


class FakeStream:
    def __init__(self, delay):
        self.first_chunk = asyncio.Event()
        self.delay = delay
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    async def aclose(self):
        return None


class FakeHandler:
    def __init__(self, events, delay):
        self.events = events
        self.delay = delay

    async def begin_playback(self):
        self.events.append("begin_playback")
        return object()

    def is_current_playback(self, playback):
        return True

    def release_playback(self, playback):
        self.events.append("release_playback")

    async def stream_audio_chunks(self, playback, stream):
        await asyncio.sleep(self.delay)
        stream.first_chunk.set()
        await asyncio.sleep(0.01)
        self.events.append("streamed")
        return types.SimpleNamespace(bytes_sent=8000, completed=True, error=None)

    async def wait_for_playback(self, playback, result, timeout=None):
        self.events.append("playback_confirmed")
        return True


def _agent(runtime, tools=False, language="en"):
    events = []
    agent = runtime.ConversationOrchestrator.__new__(runtime.ConversationOrchestrator)
    context = runtime.ConversationContext("call", "+10000000000", language=language)
    context.state = runtime.ConversationState.LISTENING
    agent._conversations = {"call": context}
    agent._twilio_handlers = {"call": FakeHandler(events, 0.05)}
    agent._call_state_locks = {"call": asyncio.Lock()}
    agent._call_stt_instances = {}
    agent._garbled_drop_count = {}
    agent._echo_guard_until = {}
    agent._barge_in_speech_start = {}
    agent._barge_in_reset_done = {}
    agent._speech_setup_locks = {}
    agent._active_speech = {}
    agent._system_prompt = "synthetic system prompt"
    agent._reconnect_stt_with_language = AsyncMock()
    agent.caller_service = types.SimpleNamespace(register_caller=AsyncMock())

    class TTS:
        def create_stream(self, request):
            events.append("create_stream")
            return FakeStream(0.05)

    agent.tts = TTS()

    class LLM:
        async def chat_with_tools(self, request):
            events.append("llm_first")
            await asyncio.sleep(0.03)
            if tools:
                return types.SimpleNamespace(content="", tool_calls=[
                    {"name": "register_caller_name", "arguments": '{"caller_name": "Canary Name"}'}])
            return types.SimpleNamespace(content="Sure, I can help with that.", tool_calls=[])

        async def chat_stream(self, request):
            events.append("llm_stream")
            await asyncio.sleep(0.02)
            yield types.SimpleNamespace(delta="Welcome ")
            yield types.SimpleNamespace(delta="back.")

    agent.llm = LLM()
    return agent, context, events


async def _turn(runtime, text="Canary Utterance please book me", **kwargs):
    agent, context, events = _agent(runtime, **kwargs)
    await agent.process_transcript("call", text, context.language, True)
    await asyncio.sleep(0.05)
    return agent, context, events


def _capture(coro_factory):
    lines = []
    handle = logger.add(lambda message: lines.append(message.record["message"]), level="DEBUG")
    try:
        result = asyncio.run(coro_factory())
    finally:
        logger.remove(handle)
    return result, lines


def _timing_lines(lines):
    return [line for line in lines if line.startswith("TurnTiming")]


def _observable(runtime, result):
    agent, context, events = result
    return (events, context.state.name, context.ai_response, context.turn_count,
            list(context.conversation_history))


@pytest.mark.parametrize("tools", [False, True])
def test_one_timing_line_per_spoken_reply_with_numbers_only(patched, tools):
    runtime = _runtime(patched, "restored_timing_patched_runtime")
    result, lines = _capture(lambda: _turn(runtime, tools=tools))
    found = _timing_lines(lines)
    assert len(found) == 1 and LINE.match(found[0]), found
    values = dict(part.split("=") for part in found[0].split()[1:])
    assert values["lang"] == "en" and values["tools"] == ("1" if tools else "0")
    assert values["llm_rounds"] == ("2" if tools else "1")
    assert int(values["llm_first_ms"]) >= 25 and int(values["tts_first_audio_ms"]) >= 40
    assert int(values["reply_gap_ms"]) >= int(values["llm_total_ms"]) + int(values["tts_first_audio_ms"])
    assert values["audio_ms"] == "1000"
    for fragment in ("Canary", "call", "+1000", "Welcome", "Sure"):
        assert fragment not in found[0]
    context = result[1]
    assert context.__dict__.get("_turn_timing") is None
    assert context.__dict__.get("_turn_timing_armed") is None


def test_speech_that_is_not_a_timed_reply_logs_nothing(patched):
    runtime = _runtime(patched, "restored_timing_patched_runtime_progress")

    async def scenario():
        agent, context, events = _agent(runtime)
        await agent._speak_to_caller("call", "Greeting or progress speech", "en")
        return events

    events, lines = _capture(scenario)
    assert "streamed" in events and _timing_lines(lines) == []


def test_ignored_and_failed_turns_log_nothing(patched):
    runtime = _runtime(patched, "restored_timing_patched_runtime_ignored")

    async def scenario():
        agent, context, events = _agent(runtime)
        context.state = runtime.ConversationState.SPEAKING
        await agent.process_transcript("call", "ignored while speaking", "en", True)
        context.state = runtime.ConversationState.LISTENING
        agent.tts = types.SimpleNamespace(create_stream=lambda request: (_ for _ in ()).throw(
            RuntimeError("provider down")))
        await agent.process_transcript("call", "provider failure", "en", True)
        await asyncio.sleep(0.05)
        return events

    _events, lines = _capture(scenario)
    assert _timing_lines(lines) == []


def test_broken_timing_does_not_change_anything_the_caller_experiences(baseline, patched, monkeypatch):
    plain = _runtime(baseline, "restored_timing_plain_runtime")
    instrumented = _runtime(patched, "restored_timing_healthy_runtime")
    broken = _runtime(patched, "restored_timing_broken_runtime")
    from services.conversation import turn_timing as module

    expected = {}
    for tools in (False, True):
        outcome, _lines = _capture(lambda: _turn(plain, tools=tools))
        expected[tools] = _observable(plain, outcome)
        outcome, lines = _capture(lambda: _turn(instrumented, tools=tools))
        assert _observable(instrumented, outcome) == expected[tools]
        assert len(_timing_lines(lines)) == 1

    def explode(*args, **kwargs):
        raise RuntimeError("measurement broke")

    for name in ("note_received", "begin", "mark", "llm_start", "llm_first_done",
                 "tool", "arm", "take", "TurnTiming"):
        monkeypatch.setattr(module, name, explode)
    for tools in (False, True):
        outcome, lines = _capture(lambda: _turn(broken, tools=tools))
        assert _observable(broken, outcome) == expected[tools]
        assert _timing_lines(lines) == []

    # A stopwatch whose own methods raise everywhere is equally harmless.
    class Hostile:
        def __getattr__(self, name):
            def fail(*args, **kwargs):
                raise RuntimeError("hostile")
            return fail

    monkeypatch.setattr(module, "take", lambda context: Hostile())
    monkeypatch.setattr(module, "TurnTiming", Hostile)
    outcome, lines = _capture(lambda: _turn(broken, tools=False))
    assert _observable(broken, outcome) == expected[False]
    assert _timing_lines(lines) == []


def test_missing_timing_module_leaves_the_call_flow_untouched(baseline, patched):
    runtime = _runtime(patched, "restored_timing_missing_module_runtime")
    runtime._tt_mod = None
    plain = _runtime(baseline, "restored_timing_plain_runtime_two")
    expected, _ = _capture(lambda: _turn(plain))
    outcome, lines = _capture(lambda: _turn(runtime))
    assert _observable(runtime, outcome) == _observable(plain, expected)
    assert _timing_lines(lines) == []
