"""Offline installed-image probe for the per-reply timing measurement.

Run with --network none. Checks that the installed orchestrator carries the
instrumentation, that the stopwatch's log line holds only numbers and a language
label, and that the instrumented call path behaves exactly as the plain one even when
every timing function raises. Synthetic fakes only: no provider, network or database.
Prints one JSON status line only.
"""
import asyncio
import json
import logging
import os
from pathlib import Path
import re
import sys
import types

from loguru import logger

logger.remove()
logging.disable(logging.CRITICAL)

SYNTHETIC_ENV = {
    "SECRET_KEY": "synthetic", "DATABASE_URL": "postgresql://synthetic:synthetic@localhost/test",
    "TWILIO_ACCOUNT_SID": "ACsynthetic", "TWILIO_AUTH_TOKEN": "synthetic",
    "TWILIO_PHONE_NUMBER": "+14165550100", "DEEPGRAM_API_KEY": "synthetic",
    "ELEVENLABS_API_KEY": "synthetic", "OPENAI_API_KEY": "synthetic",
}
REQUIRED_SNIPPETS = (
    "from services.conversation import turn_timing as _tt_mod",
    "_tt_mod.note_received(context, result)",
    "_tt_mod.begin(context)",
    "_tt_mod.mark(context, 'ai_start')",
    "_tt_mod.mark(context, 'ai_end')",
    "_tt_mod.arm(context)",
    "_tt_mod.llm_start(context)",
    "_tt_mod.llm_first_done(context)",
    "_tt_mod.tool(context)",
    "_tt = _tt_mod.take(self._conversations.get(call_sid))",
    "_tt.request_created(audio_stream)",
    "_tt.set_bytes(stream_result.bytes_sent)",
    "_line = _tt.finish()",
    "logger.info(_line)",
)
LINE = re.compile(r"^TurnTiming( [a-z_]+=(-?\d+|en|ar|other))+$")
CANARY = "Canary Caller +14165550123 CAcanary0123456789 secret@example.com"
HELPERS = ("note_received", "begin", "mark", "llm_start", "llm_first_done", "tool", "arm", "take",
           "TurnTiming")
TOTAL_SECONDS = 30


class Failure(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def check_installed_source(orchestrator):
    text = Path(orchestrator.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    if any(snippet not in text for snippet in REQUIRED_SNIPPETS):
        raise Failure("instrumentation_missing")
    from services.conversation import turn_timing
    if getattr(orchestrator, "_tt_mod", None) is not turn_timing:
        raise Failure("timing_module_not_loaded")


def check_line(turn_timing):
    timing = turn_timing.TurnTiming(0.2, CANARY)
    for name in ("ai_start", "ai_end", "tts_request", "first_audio"):
        timing.mark(name)
    timing.llm_call_started()
    timing.llm_first_done()
    timing.count_tool()
    timing.set_bytes(8000)
    line = timing.line()
    if not LINE.match(line) or any(part in line for part in ("Canary", "+1416", "CAcanary", "secret")):
        raise Failure("line_shape")
    if "audio_ms=1000" not in line or not line.endswith("lang=other"):
        raise Failure("line_values")


class FakeStream:
    def __init__(self):
        self.first_chunk = asyncio.Event()

    def cancel(self):
        pass

    async def aclose(self):
        return None


def make_agent(module, events):
    class Handler:
        async def begin_playback(self):
            events.append("begin_playback")
            return object()

        def is_current_playback(self, playback):
            return True

        def release_playback(self, playback):
            events.append("release_playback")

        async def stream_audio_chunks(self, playback, stream):
            await asyncio.sleep(0.02)
            stream.first_chunk.set()
            await asyncio.sleep(0.01)
            events.append("streamed")
            return types.SimpleNamespace(bytes_sent=8000, completed=True, error=None)

        async def wait_for_playback(self, playback, result, timeout=None):
            events.append("playback_confirmed")
            return True

    class TTS:
        def create_stream(self, request):
            events.append("create_stream")
            return FakeStream()

    class LLM:
        async def chat_with_tools(self, request):
            events.append("llm_first")
            await asyncio.sleep(0.01)
            return types.SimpleNamespace(content="Sure, I can help with that.", tool_calls=[])

    async def reconnect(*args, **kwargs):
        events.append("reconnect")

    agent = module.ConversationOrchestrator.__new__(module.ConversationOrchestrator)
    context = module.ConversationContext("synthetic", "+10000000000", language="en")
    context.state = module.ConversationState.LISTENING
    agent._conversations = {"synthetic": context}
    agent._twilio_handlers = {"synthetic": Handler()}
    agent._call_state_locks = {"synthetic": asyncio.Lock()}
    agent._call_stt_instances = {}
    agent._garbled_drop_count = {}
    agent._echo_guard_until = {}
    agent._barge_in_speech_start = {}
    agent._barge_in_reset_done = {}
    agent._speech_setup_locks = {}
    agent._active_speech = {}
    agent._system_prompt = "synthetic system prompt"
    agent._reconnect_stt_with_language = reconnect
    agent.tts = TTS()
    agent.llm = LLM()
    return agent, context


async def one_turn(module):
    events = []
    agent, context = make_agent(module, events)
    lines = []
    handle = logger.add(lambda message: lines.append(message.record["message"]), level="DEBUG")
    try:
        await agent.process_transcript("synthetic", "I would like to book please", "en", True)
        await asyncio.sleep(0.05)
    finally:
        logger.remove(handle)
    observed = (events, context.state.name, context.ai_response, context.turn_count,
                list(context.conversation_history))
    return observed, [line for line in lines if line.startswith("TurnTiming")], context


async def check_behaviour(module, turn_timing):
    healthy, lines, context = await one_turn(module)
    if len(lines) != 1 or not LINE.match(lines[0]) or "tts_first_audio_ms=" not in lines[0]:
        raise Failure("no_single_timing_line")
    if context.__dict__.get("_turn_timing_armed") is not None or context.__dict__.get("_turn_timing") is not None:
        raise Failure("timing_state_left_behind")
    saved = {name: getattr(turn_timing, name) for name in HELPERS}

    def explode(*args, **kwargs):
        raise RuntimeError("synthetic measurement failure")

    try:
        for name in HELPERS:
            setattr(turn_timing, name, explode)
        broken, broken_lines, _ = await one_turn(module)
    finally:
        for name, value in saved.items():
            setattr(turn_timing, name, value)
    if broken != healthy or broken_lines:
        raise Failure("instrumentation_changed_behaviour")
    module_alias, module._tt_mod = module._tt_mod, None
    try:
        missing, missing_lines, _ = await one_turn(module)
    finally:
        module._tt_mod = module_alias
    if missing != healthy or missing_lines:
        raise Failure("missing_module_changed_behaviour")


async def offline():
    os.environ.update(SYNTHETIC_ENV)
    from services.conversation import orchestrator
    from services.conversation import turn_timing
    check_installed_source(orchestrator)
    check_line(turn_timing)
    await check_behaviour(orchestrator, turn_timing)
    return {"status": "TURN_TIMING_OK", "mode": "offline"}


def main():
    try:
        if sys.argv[1:]:
            raise Failure("arguments")
        result = asyncio.run(asyncio.wait_for(offline(), TOTAL_SECONDS))
    except Failure as error:
        result = {"status": "TURN_TIMING_FAILED", "mode": "offline", "reason": error.reason}
    except BaseException as error:
        result = {"status": "TURN_TIMING_FAILED", "mode": "offline", "reason": type(error).__name__}
    print(json.dumps(result))
    raise SystemExit(0 if result.get("status") == "TURN_TIMING_OK" else 1)


if __name__ == "__main__":
    main()
