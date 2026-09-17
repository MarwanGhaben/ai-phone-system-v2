"""Emit only validated numeric barge-in summaries from bounded recent app logs."""

from __future__ import annotations

import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from typing import BinaryIO, Iterable


CONTAINER = "ai-voice-app"
SUMMARY_EVENT = "barge_in_diagnostics_summary"
READOUT_EVENT = "barge_in_diagnostics_readout"
LOG_SINCE = "30m"
LOG_TAIL = 2000
MAX_CAPTURE_BYTES = 2 * 1024 * 1024
MAX_LINE_BYTES = 32 * 1024
MAX_OUTPUT_SUMMARIES = 5
COMMAND_TIMEOUT_SECONDS = 20.0

TOP_FIELDS = frozenset(
    {
        "schema",
        "event",
        "diagnostic_id",
        "elapsed_seconds",
        "windows_started",
        "windows_retained",
        "window_overflow_count",
        "observer_failure_count",
        "failure_category",
        "totals",
        "windows",
    }
)
TOTAL_FIELDS = (
    "received_callbacks",
    "received_bytes",
    "received_min_bytes",
    "received_max_bytes",
    "analyzed_bytes",
    "max_rms",
    "max_consecutive",
    "trigger_count",
    "forwarded_callbacks",
    "forwarded_bytes",
    "dropped_callbacks",
    "dropped_bytes",
    "speaking_seconds",
)
WINDOW_FIELDS = (
    "sequence",
    "elapsed_seconds",
    "outcome",
    "received_callbacks",
    "received_bytes",
    "received_min_bytes",
    "received_max_bytes",
    "analyzed_callbacks",
    "analyzed_bytes",
    "no_audio_count",
    "grace_suppressed_count",
    "decode_failed_count",
    "below_threshold_count",
    "above_threshold_count",
    "gate_trigger_count",
    "forwarded_callbacks",
    "forwarded_bytes",
    "dropped_callbacks",
    "dropped_bytes",
    "max_rms",
    "max_consecutive",
    "dispatch_gap_count",
    "dispatch_gap_min_seconds",
    "dispatch_gap_mean_seconds",
    "dispatch_gap_max_seconds",
    "interrupt_succeeded_count",
    "interrupt_not_owned_count",
    "interrupt_failed_count",
    "reset_succeeded_count",
    "reset_not_owned_count",
    "reset_failed_count",
)
WINDOW_OUTCOMES = frozenset(
    {"active", "completed", "interrupted", "replaced", "failed", "teardown"}
)
FAILURE_CATEGORIES = frozenset({None, "observer_failure"})


class ReadoutFailure(RuntimeError):
    """Fixed-category readout failure which carries no private command output."""


def _nonnegative_number(value, *, integer: bool = False, nullable: bool = False):
    if value is None and nullable:
        return None
    if isinstance(value, bool):
        raise ValueError
    if integer:
        if not isinstance(value, int) or value < 0:
            raise ValueError
        return value
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError
    return value


def sanitize_summary(value: object) -> dict | None:
    """Return a newly built allowlisted summary, or None for any bad record."""
    try:
        if not isinstance(value, dict) or set(value) != TOP_FIELDS:
            raise ValueError
        if value["schema"] != 1 or value["event"] != SUMMARY_EVENT:
            raise ValueError
        diagnostic_id = value["diagnostic_id"]
        if not isinstance(diagnostic_id, str) or not re.fullmatch(
            r"[0-9a-f]{32}", diagnostic_id
        ):
            raise ValueError
        if value["failure_category"] not in FAILURE_CATEGORIES:
            raise ValueError

        totals = value["totals"]
        if not isinstance(totals, dict) or set(totals) != set(TOTAL_FIELDS):
            raise ValueError
        safe_totals = {}
        for field in TOTAL_FIELDS:
            safe_totals[field] = _nonnegative_number(
                totals[field],
                integer=field != "speaking_seconds",
                nullable=field == "received_min_bytes",
            )

        windows = value["windows"]
        if not isinstance(windows, list) or len(windows) > 16:
            raise ValueError
        safe_windows = []
        for window in windows:
            if not isinstance(window, dict) or set(window) != set(WINDOW_FIELDS):
                raise ValueError
            if window["outcome"] not in WINDOW_OUTCOMES:
                raise ValueError
            safe_window = {}
            for field in WINDOW_FIELDS:
                if field == "outcome":
                    safe_window[field] = window[field]
                elif field in {
                    "elapsed_seconds",
                    "dispatch_gap_min_seconds",
                    "dispatch_gap_mean_seconds",
                    "dispatch_gap_max_seconds",
                }:
                    safe_window[field] = _nonnegative_number(
                        window[field],
                        nullable=field
                        in {
                            "dispatch_gap_min_seconds",
                            "dispatch_gap_mean_seconds",
                        },
                    )
                else:
                    safe_window[field] = _nonnegative_number(
                        window[field],
                        integer=True,
                        nullable=field == "received_min_bytes",
                    )
            safe_windows.append(safe_window)

        windows_started = _nonnegative_number(value["windows_started"], integer=True)
        windows_retained = _nonnegative_number(
            value["windows_retained"], integer=True
        )
        if windows_retained != len(safe_windows) or windows_started < windows_retained:
            raise ValueError
        return {
            "schema": 1,
            "event": SUMMARY_EVENT,
            "diagnostic_id": diagnostic_id,
            "elapsed_seconds": _nonnegative_number(value["elapsed_seconds"]),
            "windows_started": windows_started,
            "windows_retained": windows_retained,
            "window_overflow_count": _nonnegative_number(
                value["window_overflow_count"], integer=True
            ),
            "observer_failure_count": _nonnegative_number(
                value["observer_failure_count"], integer=True
            ),
            "failure_category": value["failure_category"],
            "totals": safe_totals,
            "windows": safe_windows,
        }
    except (KeyError, TypeError, ValueError):
        return None


def _summary_from_line(line: bytes) -> tuple[bool, dict | None]:
    """Find one prefixed JSON record without returning the original line."""
    if len(line) > MAX_LINE_BYTES:
        return True, None
    looks_like_summary = SUMMARY_EVENT.encode() in line
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError:
        return True, None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _end = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("event") == SUMMARY_EVENT:
            return True, sanitize_summary(value)
    return looks_like_summary, None


def build_readout(lines: Iterable[bytes], *, scan_limited: bool = False) -> dict:
    summaries = []
    rejected = 0
    scanned = 0
    for line in lines:
        scanned += 1
        matched, safe = _summary_from_line(line)
        if matched and safe is None:
            rejected += 1
        elif safe is not None:
            summaries.append(safe)
    omitted = max(0, len(summaries) - MAX_OUTPUT_SUMMARIES)
    summaries = summaries[-MAX_OUTPUT_SUMMARIES:]
    if scan_limited:
        status = "scan_limited"
    elif omitted:
        status = "output_limited"
    elif summaries:
        status = "ok"
    else:
        status = "no_valid_summaries"
    return {
        "schema": 1,
        "event": READOUT_EVENT,
        "status": status,
        "scope": "fixed_container_recent_30m_tail_2000",
        "scan_limited": bool(scan_limited),
        "scanned_lines": scanned,
        "rejected_records": rejected,
        "omitted_summaries": omitted,
        "summaries": summaries,
    }


def _reader(
    stream: BinaryIO,
    label: str,
    output: queue.Queue,
    stopped: threading.Event,
) -> None:
    try:
        while not stopped.is_set():
            try:
                block = os.read(stream.fileno(), 8192)
            except (OSError, ValueError):
                break
            if not block:
                break
            while not stopped.is_set():
                try:
                    output.put((label, block), timeout=0.1)
                    break
                except queue.Full:
                    continue
    finally:
        while not stopped.is_set():
            try:
                output.put((label, None), timeout=0.1)
                break
            except queue.Full:
                continue


def _stop_and_join(process, streams, threads, stopped: threading.Event) -> None:
    stopped.set()
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    raise ReadoutFailure("process_cleanup_failed") from None
    except OSError:
        raise ReadoutFailure("process_cleanup_failed") from None
    for stream in streams:
        try:
            stream.close()
        except (OSError, ValueError):
            pass
    for thread in threads:
        thread.join(timeout=2)
    if any(thread.is_alive() for thread in threads):
        raise ReadoutFailure("reader_cleanup_failed")


def _run_bounded(command: tuple[str, ...]) -> tuple[list[bytes], bool]:
    """Run a fixed command with hard byte, line, and time limits."""
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        raise ReadoutFailure("log_command_unavailable") from None
    if process.stdout is None or process.stderr is None:
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            pass
        raise ReadoutFailure("log_stream_unavailable")

    output: queue.Queue = queue.Queue(maxsize=32)
    stopped = threading.Event()
    streams = (process.stdout, process.stderr)
    threads = [
        threading.Thread(
            target=_reader,
            args=(stream, label, output, stopped),
            daemon=True,
        )
        for label, stream in zip(("stdout", "stderr"), streams)
    ]
    for thread in threads:
        thread.start()

    deadline = time.monotonic() + COMMAND_TIMEOUT_SECONDS
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    discarding = {"stdout": False, "stderr": False}
    ended = set()
    lines: list[bytes] = []
    total = 0
    limited = False
    oversize_seen = False
    failure: ReadoutFailure | None = None
    try:
        while len(ended) < 2:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = ReadoutFailure("log_command_timeout")
                break
            try:
                label, block = output.get(timeout=min(0.25, remaining))
            except queue.Empty:
                continue
            if block is None:
                ended.add(label)
                continue
            total += len(block)
            if total > MAX_CAPTURE_BYTES:
                limited = True
                break
            buffer = buffers[label]
            for byte in block:
                if byte == 10:
                    if not discarding[label]:
                        lines.append(bytes(buffer).rstrip(b"\r"))
                    buffer.clear()
                    discarding[label] = False
                    if len(lines) >= LOG_TAIL:
                        limited = True
                        break
                elif not discarding[label]:
                    if len(buffer) >= MAX_LINE_BYTES:
                        buffer.clear()
                        discarding[label] = True
                        oversize_seen = True
                    else:
                        buffer.append(byte)
            if limited:
                break
        if failure is None and not limited:
            for label, buffer in buffers.items():
                if buffer and not discarding[label]:
                    lines.append(bytes(buffer).rstrip(b"\r"))
            try:
                returncode = process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                failure = ReadoutFailure("log_command_timeout")
            else:
                if returncode:
                    failure = ReadoutFailure("log_command_failed")
    finally:
        try:
            _stop_and_join(process, streams, threads, stopped)
        except ReadoutFailure as cleanup_failure:
            if failure is None:
                failure = cleanup_failure
    if failure is not None:
        raise failure
    return lines[:LOG_TAIL], limited or oversize_seen or len(lines) > LOG_TAIL


def read_recent_summaries() -> dict:
    lines, limited = _run_bounded(
        (
            "docker",
            "logs",
            "--since",
            LOG_SINCE,
            "--tail",
            str(LOG_TAIL),
            CONTAINER,
        )
    )
    return build_readout(lines, scan_limited=limited or len(lines) >= LOG_TAIL)


def main() -> int:
    try:
        result = read_recent_summaries()
    except Exception:
        result = {
            "schema": 1,
            "event": READOUT_EVENT,
            "status": "read_failed",
            "scope": "fixed_container_recent_30m_tail_2000",
            "scan_limited": False,
            "scanned_lines": 0,
            "rejected_records": 0,
            "omitted_summaries": 0,
            "summaries": [],
        }
        print(json.dumps(result, separators=(",", ":"), sort_keys=True))
        return 1
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 1:
        os._exit(2)
    raise SystemExit(main())
