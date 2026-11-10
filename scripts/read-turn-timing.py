"""Owner-run reader for the per-reply TurnTiming log lines.

Usage (on the server):
    docker logs --since 60m ai-voice-app 2>&1 | python3 /path/to/read-turn-timing.py

Reads log text from standard input, keeps only well-formed TurnTiming lines (all
other text is ignored and never printed) and prints, for every non-empty group, the
number of replies and the median, 90th percentile and maximum of each timing value
in milliseconds. Groups: all replies, then language by tool use (tools=0 means no
tool call was made for that reply). Standard library only; Python 3.10+.
"""
import re
import sys

KEYS = (
    "stt_wait_ms", "pre_llm_ms", "llm_first_ms", "llm_total_ms", "llm_rounds", "tools",
    "speak_setup_ms", "tts_first_audio_ms", "reply_gap_ms", "since_final_ms", "audio_ms",
)
LANGUAGES = ("en", "ar", "other")
MAX_LINE = 2000
MAX_LINES = 1_000_000
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# The measurement must be the whole logged message: at the start of the line or right
# after the logger's " - " separator, and end the line. Text that merely mentions it
# (for example inside a transcript) does not qualify.
LINE = re.compile(r"(?:^|\s-\s)TurnTiming((?: [a-z_]+=[-a-z0-9]{1,12})+)\s*$")
NUMBER = re.compile(r"^-?[0-9]{1,9}$")


def parse_line(raw):
    """Return a dict of integer values plus 'lang', or None for anything malformed."""
    try:
        text = ANSI.sub("", raw)
        if len(text) > MAX_LINE or "TurnTiming " not in text:
            return None
        found = LINE.search(text.rstrip("\r\n"))
        if found is None:
            return None
        values = {}
        for pair in found.group(1).split():
            key, _, value = pair.partition("=")
            if key in values:
                return None
            if key == "lang":
                if value not in LANGUAGES:
                    return None
                values[key] = value
            elif key in KEYS and NUMBER.match(value):
                values[key] = int(value)
            else:
                return None
        if "lang" not in values or "reply_gap_ms" not in values or values["reply_gap_ms"] < 0:
            return None
        return values
    except Exception:
        return None


def read(stream):
    records = []
    malformed = 0
    for number, raw in enumerate(stream):
        if number >= MAX_LINES:
            break
        if "TurnTiming" not in raw:
            continue
        record = parse_line(raw)
        if record is None:
            malformed += 1
        else:
            records.append(record)
    return records, malformed


def percentile(sorted_values, percent):
    """Nearest-rank percentile (whole percent) of a non-empty sorted list."""
    rank = max(1, -(-percent * len(sorted_values) // 100))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def median(sorted_values):
    count = len(sorted_values)
    middle = count // 2
    if count % 2:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) // 2


def summarize(records):
    out = {}
    for key in KEYS:
        values = sorted(r[key] for r in records if key in r and r[key] >= 0)
        if values:
            out[key] = (len(values), median(values), percentile(values, 90), values[-1])
    return out


def groups(records):
    yield "all replies", records
    for language in LANGUAGES:
        for label, test in (("tools=0", lambda r: r.get("tools", 0) == 0),
                            ("tools>0", lambda r: r.get("tools", 0) > 0)):
            chosen = [r for r in records if r["lang"] == language and test(r)]
            if chosen:
                yield "lang=%s %s" % (language, label), chosen


def render(records, malformed):
    lines = ["TurnTiming summary: %d replies, %d malformed lines skipped" % (len(records), malformed)]
    if not records:
        lines.append("No usable TurnTiming lines found.")
        return "\n".join(lines)
    for title, chosen in groups(records):
        lines.append("")
        lines.append("[%s] replies=%d (milliseconds; tools and llm_rounds are counts)" % (title, len(chosen)))
        for key, (count, mid, p90, high) in summarize(chosen).items():
            lines.append("  %-20s n=%-4d median=%-7d p90=%-7d max=%d" % (key, count, mid, p90, high))
    return "\n".join(lines)


def main():
    try:
        data = sys.stdin
        try:
            data.reconfigure(errors="replace")
        except Exception:
            pass
        records, malformed = read(data)
        print(render(records, malformed))
    except Exception:
        print("TurnTiming summary: input could not be read")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
