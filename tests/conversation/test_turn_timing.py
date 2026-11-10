"""Turn timing helper: fixed-shape numeric line, no free text, never raises."""
import asyncio
from datetime import datetime, timedelta, timezone
import re
import types

import pytest

from services.conversation import turn_timing as tt

LINE = re.compile(r"^TurnTiming( [a-z_]+=(-?\d+|en|ar|other))+$")
CANARY = ("Canary Caller +14165550123 CAcanary0123456789 secret@example.com "
          "مرحبا ")


def _full_timing():
    timing = tt.TurnTiming(0.25, "en")
    timing.mark("ai_start")
    timing.llm_call_started()
    timing.llm_first_done()
    timing.count_tool()
    timing.llm_call_started()
    timing.mark("ai_end")
    timing.mark("tts_request")
    timing.mark("first_audio")
    timing.set_bytes(16000)
    return timing


def test_line_has_fixed_key_order_and_only_numbers_and_a_language_label():
    line = _full_timing().line()
    assert line.startswith("TurnTiming ")
    assert "\n" not in line and LINE.match(line)
    keys = [part.split("=")[0] for part in line.split()[1:]]
    assert keys == [key for key in tt.KEYS if key in keys]
    values = dict(part.split("=") for part in line.split()[1:])
    assert values["llm_rounds"] == "2" and values["tools"] == "1"
    assert values["audio_ms"] == "2000" and values["lang"] == "en"
    assert values["stt_wait_ms"] == "250"
    assert int(values["since_final_ms"]) == int(values["reply_gap_ms"]) + 250


def test_missing_values_are_omitted_not_invented():
    line = tt.TurnTiming().line()
    assert line == "TurnTiming tools=0 lang=other"


@pytest.mark.parametrize("language,label", [("en", "en"), ("ar", "ar"), ("EN", "en"),
                                            (CANARY, "other"), (None, "other"), (7, "other")])
def test_language_label_is_derived_only_from_the_language_code(language, label):
    timing = tt.TurnTiming(language=language)
    assert timing.line().endswith("lang=" + label)
    assert "Canary" not in timing.line()


def test_line_never_contains_free_text_even_if_canaries_reach_every_input():
    timing = tt.TurnTiming(CANARY, CANARY)
    timing.mark(CANARY)
    timing.set_language(CANARY)
    timing.set_bytes(CANARY)
    timing.add_llm_round(CANARY)
    line = timing.line()
    assert LINE.match(line)
    for fragment in ("Canary", "+1416", "CAcanary", "secret", "م"):
        assert fragment not in line


def test_context_helpers_never_put_text_in_the_line():
    context = types.SimpleNamespace(language=CANARY, transcript=CANARY, phone=CANARY)
    tt.note_received(context, types.SimpleNamespace(received_at=datetime.now(timezone.utc), text=CANARY))
    tt.begin(context)
    tt.mark(context, "ai_start")
    tt.llm_start(context)
    tt.tool(context)
    tt.mark(context, "ai_end")
    tt.arm(context)
    timing = tt.take(context)
    assert timing is not None and tt.take(context) is None
    timing.mark("first_audio")
    assert LINE.match(timing.finish())
    assert timing.finish() is None  # one line per reply


def test_stt_wait_uses_provider_receipt_and_ignores_unreasonable_values():
    context = types.SimpleNamespace()
    tt.note_received(context, types.SimpleNamespace(
        received_at=datetime.now(timezone.utc) - timedelta(milliseconds=400)))
    tt.begin(context)
    assert 350 <= int(context._turn_timing.line().split("stt_wait_ms=")[1].split()[0]) < 2000
    for bad in (None, datetime.now(), datetime.now(timezone.utc) + timedelta(seconds=30),
                datetime.now(timezone.utc) - timedelta(hours=2), "yesterday"):
        tt.note_received(context, types.SimpleNamespace(received_at=bad))
        tt.begin(context)
        assert "stt_wait_ms" not in context._turn_timing.line()


def test_everything_swallows_exceptions():
    class Boom:
        def __getattr__(self, name):
            raise RuntimeError("boom")

        def __setattr__(self, name, value):
            raise RuntimeError("boom")

    boom = Boom()
    for call in (lambda: tt.note_received(boom, boom), lambda: tt.begin(boom),
                 lambda: tt.mark(boom, "x"), lambda: tt.llm_start(boom),
                 lambda: tt.llm_first_done(boom), lambda: tt.tool(boom),
                 lambda: tt.arm(boom)):
        assert call() is None
    assert tt.take(boom) is None
    assert tt.take(None) is None
    timing = tt.TurnTiming()
    timing.request_created(boom)
    timing.set_bytes(boom)
    assert timing.finish() is None  # no first audio was ever seen
    timing._marks = None  # corrupt internals: still no exception
    assert timing.line().startswith("TurnTiming")


def test_first_audio_observer_stamps_the_event_and_cleans_up():
    async def scenario():
        stream = types.SimpleNamespace(first_chunk=asyncio.Event())
        timing = tt.TurnTiming(language="ar")
        timing.mark("ai_end")
        timing.request_created(stream)
        task = timing._task
        assert task is not None and not task.done()
        await asyncio.sleep(0.02)
        stream.first_chunk.set()
        await asyncio.sleep(0.02)
        assert "first_audio" in timing._marks
        line = timing.finish()
        assert line and "tts_first_audio_ms=" in line and line.endswith("lang=ar")
        assert timing._task is None and timing._stream is None
        assert task.done()

        # Never produced audio: the watcher is cancelled and nothing is reported.
        silent = types.SimpleNamespace(first_chunk=asyncio.Event())
        quiet = tt.TurnTiming()
        quiet.request_created(silent)
        pending = quiet._task
        assert quiet.finish() is None
        await asyncio.sleep(0)
        assert pending.cancelled() or pending.done()

    asyncio.run(scenario())


def test_stream_without_first_chunk_event_is_tolerated():
    async def scenario():
        timing = tt.TurnTiming()
        timing.request_created(object())
        assert timing._task is None and timing.finish() is None

    asyncio.run(scenario())
