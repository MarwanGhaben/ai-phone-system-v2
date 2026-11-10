"""Per-reply timing measurement for the phone call flow.

Measurement only. Nothing here may change what the agent says, when it says it, or
any booking behaviour, and nothing here may ever raise into the call flow: every
public method and function swallows all ordinary exceptions. The single log line
contains integer millisecond values and a three-way language label, never
transcript text, names, phone numbers, call identifiers or any other free text.

The module uses only the standard library.
"""
import asyncio
from datetime import datetime, timezone
import math
import time

# Fixed output order. Values that were not measured are omitted from the line.
KEYS = (
    "stt_wait_ms",
    "pre_llm_ms",
    "llm_first_ms",
    "llm_total_ms",
    "llm_rounds",
    "tools",
    "speak_setup_ms",
    "tts_first_audio_ms",
    "reply_gap_ms",
    "since_final_ms",
    "audio_ms",
    "lang",
)
_MAX_MS = 3_600_000
_MAX_STT_WAIT_SECONDS = 120.0


def _ms(seconds):
    """Whole milliseconds, or None when the value is missing or not sane."""
    try:
        if seconds is None or isinstance(seconds, bool):
            return None
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds < 0:
            return None
        return min(int(round(seconds * 1000.0)), _MAX_MS)
    except Exception:
        return None


def _language_label(language):
    try:
        if isinstance(language, str):
            value = language.strip().lower()
            if value in ("en", "ar"):
                return value
    except Exception:
        pass
    return "other"


class TurnTiming:
    """Monotonic-clock stopwatch for one spoken reply."""

    def __init__(self, stt_wait_seconds=None, language=None):
        self._t0 = None
        self._marks = {}
        self._stt_wait = None
        self._rounds = 0
        self._tools = 0
        self._llm_first_started = None
        self._llm_first_seconds = None
        self._bytes = None
        self._lang = "other"
        self._task = None
        self._stream = None
        self._emitted = False
        try:
            self._t0 = time.monotonic()
            self._stt_wait = _ms(stt_wait_seconds)
            self._lang = _language_label(language)
        except Exception:
            pass

    def mark(self, name):
        """Record the first time a named point is reached."""
        try:
            self._marks.setdefault(name, time.monotonic())
        except Exception:
            pass

    def llm_call_started(self):
        """Count one language-model call; remember the start of the first one."""
        try:
            self._rounds += 1
            if self._rounds == 1 and self._llm_first_started is None:
                self._llm_first_started = time.monotonic()
        except Exception:
            pass

    def llm_first_done(self):
        try:
            if self._llm_first_started is not None and self._llm_first_seconds is None:
                self._llm_first_seconds = time.monotonic() - self._llm_first_started
        except Exception:
            pass

    def add_llm_round(self, seconds=None):
        """Record one completed language-model call of known duration."""
        try:
            self._rounds += 1
            if self._rounds == 1 and seconds is not None and self._llm_first_seconds is None:
                self._llm_first_seconds = float(seconds)
        except Exception:
            pass

    def count_tool(self):
        try:
            self._tools += 1
        except Exception:
            pass

    def set_language(self, language):
        try:
            self._lang = _language_label(language)
        except Exception:
            pass

    def set_bytes(self, count):
        try:
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                self._bytes = count
        except Exception:
            pass

    def request_created(self, stream):
        """Mark TTS request creation and start observing the first audio bytes.

        The observer is a small background task that only waits on the stream's
        ``first_chunk`` event and stamps the time. The call flow never awaits it;
        ``finish`` cancels it and drops every reference.
        """
        try:
            self.mark("tts_request")
            event = getattr(stream, "first_chunk", None)
            if event is None:
                return
            if event.is_set():
                self.mark("first_audio")
                return
            marks = self

            async def watch():
                try:
                    await event.wait()
                    marks.mark("first_audio")
                except Exception:
                    pass

            self._task = asyncio.get_running_loop().create_task(watch())
            self._stream = stream
        except Exception:
            pass

    def _since(self, later, earlier):
        try:
            if later is None or earlier is None:
                return None
            return _ms(later - earlier)
        except Exception:
            return None

    def values(self):
        try:
            marks = self._marks
            first_audio = marks.get("first_audio")
            ai_start = marks.get("ai_start")
            ai_end = marks.get("ai_end")
            tts_request = marks.get("tts_request")
            reply_gap = self._since(first_audio, self._t0)
            since_final = (None if reply_gap is None or self._stt_wait is None
                           else reply_gap + self._stt_wait)
            return {
                "stt_wait_ms": self._stt_wait,
                "pre_llm_ms": self._since(ai_start, self._t0),
                "llm_first_ms": _ms(self._llm_first_seconds),
                "llm_total_ms": self._since(ai_end, ai_start),
                "llm_rounds": self._rounds if self._rounds > 0 else None,
                "tools": self._tools,
                "speak_setup_ms": self._since(tts_request, ai_end),
                "tts_first_audio_ms": self._since(first_audio, tts_request),
                "reply_gap_ms": reply_gap,
                "since_final_ms": since_final,
                "audio_ms": None if self._bytes is None else min(self._bytes // 8, _MAX_MS),
                "lang": self._lang,
            }
        except Exception:
            return {"lang": "other"}

    def line(self):
        """One log line: fixed key order, integers (and the language label) only."""
        try:
            values = self.values()
            parts = []
            for key in KEYS:
                value = values.get(key)
                if value is None:
                    continue
                if key == "lang":
                    parts.append("lang=" + _language_label(value))
                elif isinstance(value, int) and not isinstance(value, bool):
                    parts.append("%s=%d" % (key, value))
            return "TurnTiming " + " ".join(parts)
        except Exception:
            return "TurnTiming lang=other"

    def finish(self):
        """Stop observing. Return the log line once, and only if audio really started."""
        try:
            task, self._task = self._task, None
            stream = getattr(self, "_stream", None)
            self._stream = None
            if task is not None and not task.done():
                task.cancel()
            if self._emitted:
                return None
            if "first_audio" not in self._marks and stream is not None:
                event = getattr(stream, "first_chunk", None)
                if event is not None and event.is_set():
                    self.mark("first_audio")
            if "first_audio" not in self._marks:
                return None
            self._emitted = True
            return self.line()
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Context helpers used by the instrumented orchestrator. Each one swallows every
# exception and returns nothing the call flow depends on.
# ---------------------------------------------------------------------------

def note_received(context, result):
    """Remember when the provider received the final transcript (consumer side)."""
    try:
        setattr(context, "_turn_timing_received_at", getattr(result, "received_at", None))
    except Exception:
        pass


def begin(context):
    """Start the stopwatch for a final transcript that will be processed."""
    try:
        received = getattr(context, "_turn_timing_received_at", None)
        setattr(context, "_turn_timing_received_at", None)
        wait = None
        if isinstance(received, datetime) and received.tzinfo is not None:
            seconds = (datetime.now(timezone.utc) - received).total_seconds()
            if 0 <= seconds <= _MAX_STT_WAIT_SECONDS:
                wait = seconds
        setattr(context, "_turn_timing", TurnTiming(wait))
    except Exception:
        pass


def _current(context):
    return getattr(context, "_turn_timing", None)


def mark(context, name):
    try:
        timing = _current(context)
        if timing is not None:
            timing.mark(name)
    except Exception:
        pass


def llm_start(context):
    try:
        timing = _current(context)
        if timing is not None:
            timing.llm_call_started()
    except Exception:
        pass


def llm_first_done(context):
    try:
        timing = _current(context)
        if timing is not None:
            timing.llm_first_done()
    except Exception:
        pass


def tool(context):
    try:
        timing = _current(context)
        if timing is not None:
            timing.count_tool()
    except Exception:
        pass


def arm(context):
    """Hand this turn's timing to the very next spoken reply (the final reply)."""
    try:
        timing = _current(context)
        setattr(context, "_turn_timing", None)
        if timing is not None:
            timing.set_language(getattr(context, "language", None))
        setattr(context, "_turn_timing_armed", timing)
    except Exception:
        pass


def take(context):
    """Claim the armed timing at the start of a spoken reply; at most one claimant."""
    try:
        timing = getattr(context, "_turn_timing_armed", None)
        if timing is not None:
            setattr(context, "_turn_timing_armed", None)
        return timing
    except Exception:
        return None
