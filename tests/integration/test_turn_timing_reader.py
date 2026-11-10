"""Owner-run TurnTiming reader: summary numbers only, hostile input tolerated."""
import importlib.util
import io
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("read_turn_timing", ROOT / "scripts/read-turn-timing.py")
reader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reader)

PREFIX = "2026-11-10 17:12:24.515 | INFO     | services.conversation.orchestrator:_speak_to_caller:3373 - "


def line(gap, lang="en", tools=0, **extra):
    values = {"pre_llm_ms": 5, "llm_total_ms": gap // 2, "tools": tools, "reply_gap_ms": gap}
    values.update(extra)
    body = " ".join("%s=%d" % (k, v) for k, v in values.items())
    return PREFIX + "TurnTiming " + body + " lang=" + lang


SAMPLE = "\n".join([
    "unrelated line Canary Caller +14165550123",
    line(1000), line(2000), line(3000), line(4000), line(5000),
    line(900, lang="ar", tools=1),
    line(1500, lang="ar", tools=2),
    line(700, lang="other"),
    "2026-11-10T17:12:24.5Z " + line(1200, stt_wait_ms=30),  # docker -t style prefix
    "\x1b[32m2026-11-10\x1b[0m | \x1b[1mINFO\x1b[0m | x:y:1 - \x1b[1mTurnTiming reply_gap_ms=1100 tools=0 lang=en\x1b[0m",
])


def run(text):
    records, malformed = reader.read(io.StringIO(text))
    return reader.render(records, malformed), records, malformed


def test_summary_counts_and_statistics():
    out, records, malformed = run(SAMPLE)
    assert len(records) == 10 and malformed == 0
    assert out.startswith("TurnTiming summary: 10 replies, 0 malformed lines skipped")
    assert "[lang=en tools=0] replies=7" in out
    assert "[lang=ar tools>0] replies=2" in out
    assert "[lang=other tools=0] replies=1" in out
    assert "[lang=en tools>0]" not in out
    en = reader.summarize([r for r in records if r["lang"] == "en"])
    gaps = sorted([1000, 2000, 3000, 4000, 5000, 1200, 1100])
    assert en["reply_gap_ms"] == (7, gaps[3], gaps[-1], 5000)  # nearest-rank p90 of 7 is the 7th value
    assert en["stt_wait_ms"][0] == 1


def test_percentile_and_median_helpers():
    assert reader.median([1, 2, 3, 4]) == 2
    assert reader.median([5]) == 5
    values = list(range(1, 11))
    assert reader.percentile(values, 90) == 9
    assert reader.percentile([7], 90) == 7


def test_nothing_from_non_matching_lines_is_printed():
    out, _records, _bad = run(SAMPLE)
    for fragment in ("Canary", "+1416", "unrelated", "INFO", "docker"):
        assert fragment not in out


HOSTILE = [
    PREFIX + "Orchestrator: User said: TurnTiming reply_gap_ms=1 tools=0 lang=en",  # transcript text
    PREFIX + "TurnTiming reply_gap_ms=abc lang=en",
    PREFIX + "TurnTiming reply_gap_ms=5 lang=klingon",
    PREFIX + "TurnTiming reply_gap_ms=5 reply_gap_ms=6 lang=en",
    PREFIX + "TurnTiming reply_gap_ms=5 secret=1 lang=en",
    PREFIX + "TurnTiming reply_gap_ms=5",
    PREFIX + "TurnTiming lang=en",
    PREFIX + "TurnTiming reply_gap_ms=-5 lang=en",
    PREFIX + "TurnTiming reply_gap_ms=99999999999 lang=en",
    PREFIX + "TurnTiming reply_gap_ms=5 lang=en trailing words",
    PREFIX + "TurnTiming " + "reply_gap_ms=5 " * 1000 + "lang=en",
    "TurnTiming",
    "TurnTiming ",
    "\u0645\u0631\u062d\u0628\u0627 TurnTiming reply_gap_ms=5 lang=ar",
    "\x00\x01\x02 TurnTiming \xff",
]


def test_malformed_and_hostile_lines_are_skipped_and_counted_only():
    out, records, malformed = run("\n".join(HOSTILE + [line(1000)]))
    assert len(records) == 1 and records[0]["reply_gap_ms"] == 1000
    assert malformed == len([h for h in HOSTILE if "TurnTiming" in h])
    assert "klingon" not in out and "trailing" not in out and "secret" not in out


def test_empty_and_garbage_input():
    assert "0 replies" in run("")[0]
    assert "No usable TurnTiming lines" in run("nothing here\n\n")[0]
    reader.read(io.StringIO("TurnTiming x=" + "9" * 100000))


def test_command_line_use_reads_standard_input_and_exits_cleanly():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/read-turn-timing.py")],
                            input=(SAMPLE + "\n\xff\x00 garbage").encode("utf-8", "replace"),
                            capture_output=True, timeout=60)
    assert result.returncode == 0
    text = result.stdout.decode()
    assert text.startswith("TurnTiming summary: 10 replies")
    assert "Canary" not in text and "+1416" not in text
    assert result.stderr == b""
