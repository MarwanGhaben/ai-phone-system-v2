#!/usr/bin/env python3
"""Offline-first paired Scribe background-audio evaluation pilot."""

from __future__ import annotations

import argparse
import asyncio
import base64
from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Awaitable, Callable, Optional, TextIO
from urllib.parse import urlencode


ENDPOINT = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
REVIEWED_MODEL = "scribe_v2_realtime"
CHUNK_BYTES = 160
CHUNK_SECONDS = 0.020
MIN_CHUNK_GAP_SECONDS = 0.005
SILENCE_BYTES = bytes([0xFF]) * 16000
HANDSHAKE_TIMEOUT_SECONDS = 10.0
OBSERVATION_SECONDS = 5.0
ARM_TIMEOUT_SECONDS = 35.0
PAIR_TIMEOUT_SECONDS = 75.0
CLOSE_TIMEOUT_SECONDS = 2.0
CHILD_CLEANUP_SECONDS = 0.25
SEND_TIMEOUT_SECONDS = 2.0
MAX_MESSAGE_BYTES = 64 * 1024
MAX_PROVIDER_EVENTS = 256
MAX_TRANSCRIPT_CHARS = 16 * 1024
MAX_MANIFEST_BYTES = 256 * 1024
PACING_IMPAIRMENT_SECONDS = 0.040
DEFAULT_SEED = 3304
APPROVED_AUDIO_STATUS = "owner_approved_test_audio"

PROVIDER_ERROR_TYPES = frozenset(
    {
        "auth_error",
        "quota_exceeded",
        "transcriber_error",
        "input_error",
        "invalid_request",
        "error",
        "commit_throttled",
        "unaccepted_terms",
        "rate_limited",
        "queue_overflow",
        "resource_exhausted",
        "session_time_limit_exceeded",
        "chunk_size_exceeded",
        "insufficient_audio_activity",
    }
)

QUERY_PARAMETERS = (
    ("model_id", REVIEWED_MODEL),
    ("audio_format", "ulaw_8000"),
    ("sample_rate", "8000"),
    ("commit_strategy", "vad"),
    ("vad_silence_threshold_secs", "0.3"),
    ("include_language_detection", "true"),
)


class PilotError(RuntimeError):
    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


@dataclass(frozen=True)
class Arm:
    label: str
    filtering: bool


@dataclass(frozen=True)
class ValidatedCase:
    case_id: str
    audio_path: Path
    audio: bytes
    sha256: str
    duration_seconds: float
    language_hint: str
    foreground_expected_wording: str
    expected_entities: tuple[str, ...]
    background_expected_wording: Optional[str]
    foreground_absent: bool
    source_kind: str
    permission_status: str


@dataclass
class TaskOwner:
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)

    def retain(self, task: asyncio.Task[Any]) -> None:
        if task.done():
            _consume_task(task)
            return
        self.tasks.add(task)
        task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Task[Any]) -> None:
        self.tasks.discard(task)
        _consume_task(task)


@dataclass
class ReceiverState:
    accepted: asyncio.Event = field(default_factory=asyncio.Event)
    terminal: asyncio.Event = field(default_factory=asyncio.Event)
    acceptance: bool = False
    filtering_echo: str = "unknown"
    error_category: Optional[str] = None
    finals: list[dict[str, Any]] = field(default_factory=list)
    language_metadata: list[dict[str, Any]] = field(default_factory=list)
    event_count: int = 0
    transcript_chars: int = 0
    first_audio_monotonic: Optional[float] = None


class RealClock:
    @staticmethod
    def monotonic() -> float:
        return time.perf_counter()

    @staticmethod
    async def sleep(seconds: float) -> None:
        if sys.platform == "win32":
            # Python 3.11 uses a high-resolution Windows waitable timer here.
            # Keep the wait off the event loop and bounded to one audio frame;
            # cancellation can leave at most this brief executor wait finishing.
            await asyncio.to_thread(time.sleep, min(max(seconds, 0.0), CHUNK_SECONDS))
        else:
            await asyncio.sleep(seconds)


def _consume_task(task: asyncio.Task[Any]) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        pass


def build_connection_url(language: str, filtering: bool) -> str:
    parameters = list(QUERY_PARAMETERS)
    if language:
        parameters.append(("language_code", language))
    if filtering:
        parameters.append(("filter_background_audio", "true"))
    return f"{ENDPOINT}?{urlencode(parameters)}"


def ordered_arms(seed: int) -> tuple[Arm, Arm]:
    ordered = [Arm("baseline", False), Arm("candidate", True)]
    random.Random(seed).shuffle(ordered)
    return ordered[0], ordered[1]


def _inside_repository(path: Path) -> bool:
    repository = Path(__file__).resolve().parents[1]
    try:
        path.resolve().relative_to(repository)
        return True
    except ValueError:
        return False


def _require_string(value: Any, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise PilotError("manifest_invalid")
    return value


def validate_case(manifest_path: Path, case_id: str) -> ValidatedCase:
    """Validate one external manifest case and load/hash its audio exactly once."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", case_id):
        raise PilotError("case_invalid")
    try:
        with manifest_path.open("rb") as handle:
            raw_manifest = handle.read(MAX_MANIFEST_BYTES + 1)
    except OSError as error:
        raise PilotError("manifest_unavailable") from error
    if len(raw_manifest) > MAX_MANIFEST_BYTES:
        raise PilotError("manifest_invalid")
    try:
        manifest = json.loads(raw_manifest)
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise PilotError("manifest_invalid") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise PilotError("manifest_invalid")
    if manifest.get("test_audio_status") != APPROVED_AUDIO_STATUS:
        raise PilotError("audio_not_approved")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise PilotError("manifest_invalid")
    matches = [item for item in cases if isinstance(item, dict) and item.get("case_id") == case_id]
    if len(matches) != 1:
        raise PilotError("case_invalid")
    item = matches[0]
    if item.get("permission_status") != APPROVED_AUDIO_STATUS:
        raise PilotError("audio_not_approved")
    if item.get("encoding") != "mulaw" or item.get("sample_rate_hz") != 8000:
        raise PilotError("audio_format_invalid")
    if item.get("channels") != 1:
        raise PilotError("audio_format_invalid")
    language = item.get("language_hint")
    if language not in ("auto", "en", "ar"):
        raise PilotError("manifest_invalid")
    audio_relative = _require_string(item.get("audio_path"))
    audio_path = (manifest_path.resolve().parent / audio_relative).resolve()
    if _inside_repository(audio_path):
        raise PilotError("audio_path_invalid")
    try:
        with audio_path.open("rb") as handle:
            audio = handle.read(80001)
    except OSError as error:
        raise PilotError("audio_unavailable") from error
    if not 8000 <= len(audio) <= 80000:
        raise PilotError("audio_duration_invalid")
    duration = len(audio) / 8000.0
    declared_duration = item.get("duration_seconds")
    if (
        isinstance(declared_duration, bool)
        or not isinstance(declared_duration, (int, float))
        or not 1 <= declared_duration <= 10
        or abs(float(declared_duration) - duration) > 0.001
    ):
        raise PilotError("audio_duration_invalid")
    digest = hashlib.sha256(audio).hexdigest()
    declared_hash = item.get("sha256")
    if not isinstance(declared_hash, str) or declared_hash.lower() != digest:
        raise PilotError("audio_hash_mismatch")
    entities = item.get("expected_entities")
    if not isinstance(entities, list) or not all(isinstance(value, str) for value in entities):
        raise PilotError("manifest_invalid")
    background = item.get("background_expected_wording")
    if background is not None and not isinstance(background, str):
        raise PilotError("manifest_invalid")
    foreground_absent = item.get("foreground_absent")
    if not isinstance(foreground_absent, bool):
        raise PilotError("manifest_invalid")
    foreground = _require_string(
        item.get("foreground_expected_wording"), allow_empty=foreground_absent
    )
    if foreground_absent and (foreground or entities):
        raise PilotError("manifest_invalid")
    return ValidatedCase(
        case_id=case_id,
        audio_path=audio_path,
        audio=audio,
        sha256=digest,
        duration_seconds=duration,
        language_hint="" if language == "auto" else language,
        foreground_expected_wording=foreground,
        expected_entities=tuple(entities),
        background_expected_wording=background,
        foreground_absent=foreground_absent,
        source_kind=_require_string(item.get("source_kind")),
        permission_status=item["permission_status"],
    )


@dataclass
class ReportReservation:
    path: Path
    handle: TextIO

    def close(self) -> None:
        self.handle.close()


def reserve_report(path: Path, case: ValidatedCase, seed: int) -> ReportReservation:
    resolved = path.resolve()
    if _inside_repository(resolved) or not resolved.parent.is_dir():
        raise PilotError("report_path_invalid")
    initial = {
        "schema": 1,
        "status": "reserved",
        "case_id": case.case_id,
        "seed": seed,
        "clip": {
            "sha256": case.sha256,
            "byte_count": len(case.audio),
            "duration_seconds": case.duration_seconds,
            "encoding": "mulaw",
            "sample_rate_hz": 8000,
            "channels": 1,
        },
        "arm_order": [],
        "arms": [],
        "error_category": None,
    }
    try:
        descriptor = os.open(str(resolved), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        try:
            json.dump(initial, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            return ReportReservation(resolved, handle)
        except BaseException:
            handle.close()
            raise
    except FileExistsError as error:
        raise PilotError("report_exists") from error
    except OSError as error:
        raise PilotError("report_path_invalid") from error


def write_report(reservation: ReportReservation, report: dict[str, Any]) -> None:
    try:
        handle = reservation.handle
        owned = os.fstat(handle.fileno())
        current = reservation.path.stat()
        if (owned.st_dev, owned.st_ino) != (current.st_dev, current.st_ino):
            raise PilotError("report_write_failed")
        # Write only the exclusive descriptor, never reopen a path that another
        # process could have replaced while provider requests were in flight.
        handle.seek(0)
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.truncate()
        handle.flush()
    except OSError as error:
        raise PilotError("report_write_failed") from error


def _filter_echo(message: dict[str, Any]) -> str:
    config = message.get("config")
    if config is not None and not isinstance(config, dict):
        return "malformed"
    for source in (config, message):
        if isinstance(source, dict) and "filter_background_audio" in source:
            value = source["filter_background_audio"]
            if isinstance(value, bool):
                return "enabled" if value else "disabled"
            return "malformed"
    return "unknown"


def _relative_time(state: ReceiverState, clock: Any) -> Optional[float]:
    if state.first_audio_monotonic is None:
        return None
    return round(max(0.0, clock.monotonic() - state.first_audio_monotonic), 3)


async def receive_events(socket: Any, state: ReceiverState, clock: Any) -> None:
    try:
        while not state.terminal.is_set():
            message = await socket.recv()
            if isinstance(message, bytes):
                if len(message) > MAX_MESSAGE_BYTES:
                    raise PilotError("message_too_large")
                try:
                    message = message.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise PilotError("malformed_event") from error
            if not isinstance(message, str) or len(message.encode("utf-8")) > MAX_MESSAGE_BYTES:
                raise PilotError("message_too_large")
            try:
                event = json.loads(message)
            except (ValueError, TypeError) as error:
                raise PilotError("malformed_event") from error
            if not isinstance(event, dict) or not isinstance(event.get("message_type"), str):
                raise PilotError("malformed_event")
            state.event_count += 1
            if state.event_count > MAX_PROVIDER_EVENTS:
                raise PilotError("event_limit")
            event_type = event["message_type"]
            if event_type == "session_started":
                echo = _filter_echo(event)
                if echo == "malformed":
                    raise PilotError("malformed_event")
                state.acceptance = True
                state.filtering_echo = echo
                state.accepted.set()
            elif event_type in PROVIDER_ERROR_TYPES:
                raise PilotError(event_type)
            elif event_type == "committed_transcript":
                if not state.acceptance:
                    raise PilotError("event_order_invalid")
                text = event.get("text")
                if not isinstance(text, str):
                    raise PilotError("malformed_event")
                state.transcript_chars += len(text)
                if state.transcript_chars > MAX_TRANSCRIPT_CHARS:
                    raise PilotError("transcript_limit")
                state.finals.append(
                    {
                        "sequence": len(state.finals) + 1,
                        "text": text,
                        "received_seconds": _relative_time(state, clock),
                    }
                )
            elif event_type == "committed_transcript_with_timestamps":
                language = event.get("language_code")
                if language is not None and not isinstance(language, str):
                    raise PilotError("malformed_event")
                state.language_metadata.append(
                    {
                        "sequence": len(state.language_metadata) + 1,
                        "detected_language": language,
                        "received_seconds": _relative_time(state, clock),
                    }
                )
            elif event_type == "session_ended":
                raise PilotError("remote_ended")
            elif event_type in ("partial_transcript", "warning"):
                continue
            else:
                raise PilotError("unknown_event")
    except asyncio.CancelledError:
        raise
    except PilotError as error:
        state.error_category = (
            error.category if error.category in PROVIDER_ERROR_TYPES else error.category
        )
        state.terminal.set()
    except Exception:
        state.error_category = "remote_closed"
        state.terminal.set()


async def _dispose_late_connector(task: asyncio.Task[Any], owner: TaskOwner) -> None:
    try:
        socket = await task
    except (asyncio.CancelledError, Exception):
        return
    if socket is not None:
        await close_socket(socket, owner)


async def open_socket(
    connect: Callable[..., Awaitable[Any]],
    url: str,
    api_key: str,
    owner: TaskOwner,
    timeout: Optional[float] = None,
) -> Optional[Any]:
    task = asyncio.ensure_future(
        connect(
            url,
            extra_headers={"xi-api-key": api_key},
            ping_interval=None,
            max_size=MAX_MESSAGE_BYTES,
            max_queue=1,
            open_timeout=HANDSHAKE_TIMEOUT_SECONDS,
            close_timeout=CLOSE_TIMEOUT_SECONDS,
        )
    )
    try:
        done, _ = await asyncio.wait(
            {task}, timeout=HANDSHAKE_TIMEOUT_SECONDS if timeout is None else timeout
        )
    except asyncio.CancelledError:
        task.cancel()
        owner.retain(asyncio.create_task(_dispose_late_connector(task, owner)))
        raise
    if done:
        try:
            return task.result()
        except Exception:
            return None
    task.cancel()
    owner.retain(asyncio.create_task(_dispose_late_connector(task, owner)))
    return None


def _abort_socket(socket: Any) -> None:
    try:
        transport = getattr(socket, "transport", None)
        if transport is not None:
            transport.abort()
    except Exception:
        pass


async def close_socket(socket: Any, owner: TaskOwner) -> bool:
    try:
        task = asyncio.create_task(socket.close())
    except Exception:
        _abort_socket(socket)
        return False
    try:
        done, _ = await asyncio.wait({task}, timeout=CLOSE_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        task.cancel()
        owner.retain(task)
        _abort_socket(socket)
        raise
    if done:
        try:
            task.result()
            return True
        except Exception:
            _consume_task(task)
            _abort_socket(socket)
            return False
    task.cancel()
    owner.retain(task)
    _abort_socket(socket)
    return False


async def _stop_task(task: Optional[asyncio.Task[Any]], owner: TaskOwner) -> None:
    if task is None:
        return
    if task.done():
        _consume_task(task)
        return
    task.cancel()
    try:
        done, _ = await asyncio.wait({task}, timeout=CHILD_CLEANUP_SECONDS)
    except asyncio.CancelledError:
        owner.retain(task)
        raise
    if done:
        _consume_task(task)
    else:
        owner.retain(task)


async def _send_message_or_stop(
    socket: Any,
    message: str,
    terminal: asyncio.Event,
    owner: TaskOwner,
) -> bool:
    if terminal.is_set():
        return False
    send_task = asyncio.create_task(socket.send(message))
    terminal_task = asyncio.create_task(terminal.wait())
    try:
        done, _ = await asyncio.wait(
            {send_task, terminal_task},
            timeout=SEND_TIMEOUT_SECONDS,
            return_when=asyncio.FIRST_COMPLETED,
        )
    except asyncio.CancelledError:
        await _stop_task(send_task, owner)
        await _stop_task(terminal_task, owner)
        raise
    if not done:
        await _stop_task(send_task, owner)
        await _stop_task(terminal_task, owner)
        return False
    if terminal_task in done and terminal.is_set() and not send_task.done():
        send_task.cancel()
        await _stop_task(send_task, owner)
        _consume_task(terminal_task)
        return False
    await _stop_task(terminal_task, owner)
    try:
        send_task.result()
        return not terminal.is_set()
    except (asyncio.CancelledError, Exception):
        _consume_task(send_task)
        return False


async def stream_audio(
    socket: Any,
    case: ValidatedCase,
    state: ReceiverState,
    owner: TaskOwner,
    clock: Any,
) -> dict[str, Any]:
    chunks = [case.audio[index : index + CHUNK_BYTES] for index in range(0, len(case.audio), CHUNK_BYTES)]
    chunks.extend(
        SILENCE_BYTES[index : index + CHUNK_BYTES]
        for index in range(0, len(SILENCE_BYTES), CHUNK_BYTES)
    )
    audio_chunk_count = (len(case.audio) + CHUNK_BYTES - 1) // CHUNK_BYTES
    audio_sent = 0
    silence_sent = 0
    max_deviation = 0.0
    previous_send: Optional[float] = None
    previous_dispatch: Optional[float] = None
    schedule_offset = 0.0

    def metrics(complete: bool, chunks_sent: int) -> dict[str, Any]:
        last_send = None
        if state.first_audio_monotonic is not None and previous_send is not None:
            last_send = round(
                max(0.0, previous_send - state.first_audio_monotonic), 3
            )
        return {
            "complete": complete,
            "audio_bytes_sent": audio_sent,
            "silence_bytes_sent": silence_sent,
            "chunks_sent": chunks_sent,
            "first_audio_send_seconds": (
                0.0 if state.first_audio_monotonic is not None else None
            ),
            "last_chunk_send_seconds": last_send,
            "max_schedule_deviation_seconds": round(max_deviation, 6),
            "latency_scoring_valid": (
                complete and max_deviation <= PACING_IMPAIRMENT_SECONDS
            ),
        }

    for index, chunk in enumerate(chunks):
        if state.terminal.is_set():
            return metrics(False, index)
        now = clock.monotonic()
        if state.first_audio_monotonic is None:
            state.first_audio_monotonic = now
        ideal = state.first_audio_monotonic + index * CHUNK_SECONDS
        # Anchor each chunk to the original audio timeline. Ordinary timer and
        # send overhead must not be added to every subsequent 20 ms interval.
        target = ideal + schedule_offset
        if previous_dispatch is not None:
            target = max(target, previous_dispatch + MIN_CHUNK_GAP_SECONDS)
        # A real stall rebases the schedule rather than flushing overdue chunks.
        if now - target >= CHUNK_SECONDS:
            schedule_offset += now - target
            target = now
        while target > now:
            await clock.sleep(target - now)
            now = clock.monotonic()
        actual = now
        if actual - target >= CHUNK_SECONDS:
            schedule_offset += actual - target
        previous_dispatch = actual
        max_deviation = max(max_deviation, max(0.0, actual - ideal))
        payload = json.dumps(
            {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(chunk).decode("ascii"),
            },
            separators=(",", ":"),
        )
        if not await _send_message_or_stop(socket, payload, state.terminal, owner):
            return metrics(False, index)
        previous_send = clock.monotonic()
        if previous_send - actual >= CHUNK_SECONDS:
            schedule_offset = max(schedule_offset, previous_send - ideal)
        max_deviation = max(max_deviation, max(0.0, previous_send - ideal))
        if index < audio_chunk_count:
            audio_sent += len(chunk)
        else:
            silence_sent += len(chunk)
    return metrics(True, len(chunks))


async def _await_handshake(
    state: ReceiverState, receiver: asyncio.Task[Any], timeout: float,
) -> bool:
    accepted_waiter = asyncio.create_task(state.accepted.wait())
    terminal_waiter = asyncio.create_task(state.terminal.wait())
    try:
        await asyncio.wait(
            {accepted_waiter, terminal_waiter, receiver},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        for task in (accepted_waiter, terminal_waiter):
            task.cancel()
            task.add_done_callback(_consume_task)
        await asyncio.gather(accepted_waiter, terminal_waiter, return_exceptions=True)
    return state.acceptance and not state.terminal.is_set()


def _empty_arm_result(arm: Arm, case: ValidatedCase) -> dict[str, Any]:
    return {
        "label": arm.label,
        "filtering": arm.filtering,
        "status": "incomplete",
        "acceptance": False,
        "filtering_echo": "unknown",
        "clip_sha256": case.sha256,
        "audio_bytes_sent": 0,
        "silence_bytes_sent": 0,
        "chunks_sent": 0,
        "first_audio_send_seconds": None,
        "last_chunk_send_seconds": None,
        "finals": [],
        "language_metadata": [],
        "zero_final_result": False,
        "max_schedule_deviation_seconds": 0.0,
        "latency_scoring_valid": False,
        "error_category": None,
        "human_review": {
            "foreground_preserved": None,
            "entities_correct": None,
            "background_intrusion": None,
            "naturalness": None,
        },
    }


async def run_arm(
    arm: Arm,
    case: ValidatedCase,
    api_key: str,
    connect: Optional[Callable[..., Awaitable[Any]]] = None,
    *,
    clock: Optional[Any] = None,
    budget: Optional[float] = None,
) -> dict[str, Any]:
    owner = TaskOwner()
    clock = clock or RealClock()
    result = _empty_arm_result(arm, case)
    socket = None
    receiver: Optional[asyncio.Task[Any]] = None
    state = ReceiverState()

    async def execute() -> None:
        nonlocal socket, receiver
        actual_connect = connect
        if actual_connect is None:
            import websockets

            actual_connect = websockets.connect
        handshake_deadline = time.monotonic() + HANDSHAKE_TIMEOUT_SECONDS
        socket = await open_socket(
            actual_connect,
            build_connection_url(case.language_hint, arm.filtering),
            api_key,
            owner,
            timeout=max(0.0, handshake_deadline - time.monotonic()),
        )
        if socket is None:
            result["error_category"] = "connection_failed"
            return
        receiver = asyncio.create_task(receive_events(socket, state, clock))
        if not await _await_handshake(
            state, receiver, max(0.0, handshake_deadline - time.monotonic())
        ):
            result["error_category"] = state.error_category or "handshake_timeout"
            return
        result["acceptance"] = True
        result["filtering_echo"] = state.filtering_echo
        sending = await stream_audio(socket, case, state, owner, clock)
        result.update(sending)
        if not sending["complete"]:
            result["error_category"] = state.error_category or "send_failed"
            return
        try:
            await asyncio.wait_for(state.terminal.wait(), timeout=OBSERVATION_SECONDS)
            result["error_category"] = state.error_category or "remote_closed"
            return
        except asyncio.TimeoutError:
            pass
        result["status"] = "complete"
        result["finals"] = list(state.finals)
        result["language_metadata"] = list(state.language_metadata)
        result["zero_final_result"] = not state.finals

    try:
        available = min(ARM_TIMEOUT_SECONDS, budget if budget is not None else ARM_TIMEOUT_SECONDS)
        active_budget = max(0.0, available - 2 * CLOSE_TIMEOUT_SECONDS - 3 * CHILD_CLEANUP_SECONDS)
        await asyncio.wait_for(execute(), timeout=active_budget)
    except asyncio.TimeoutError:
        result["error_category"] = "arm_timeout"
    except Exception:
        result["error_category"] = "arm_failed"
    finally:
        result["acceptance"] = state.acceptance
        result["filtering_echo"] = state.filtering_echo
        result["finals"] = list(state.finals)
        result["language_metadata"] = list(state.language_metadata)
        result["zero_final_result"] = result["status"] == "complete" and not state.finals
        await _stop_task(receiver, owner)
        if socket is not None and not await close_socket(socket, owner):
            result["status"] = "incomplete"
            result["error_category"] = "cleanup_failed"
        if not await drain_tasks(owner):
            result["status"] = "incomplete"
            result["error_category"] = "cleanup_failed"
        result["zero_final_result"] = result["status"] == "complete" and not state.finals
        if result["status"] != "complete":
            result["latency_scoring_valid"] = False
    return result


async def drain_tasks(owner: TaskOwner) -> bool:
    deadline = time.monotonic() + CLOSE_TIMEOUT_SECONDS
    while owner.tasks and time.monotonic() < deadline:
        done, _ = await asyncio.wait(
            tuple(owner.tasks), timeout=max(0.0, deadline - time.monotonic())
        )
        if done:
            await asyncio.gather(*done, return_exceptions=True)
        for task in done:
            owner.tasks.discard(task)
            _consume_task(task)
    return not owner.tasks


def _load_configuration() -> str:
    try:
        from config.settings import Settings

        settings = Settings(_env_file=None)
        api_key = settings.elevenlabs_api_key
        model = settings.elevenlabs_stt_model
    except Exception as error:
        raise PilotError("local_configuration") from error
    if not isinstance(api_key, str) or not api_key.strip() or model != REVIEWED_MODEL:
        raise PilotError("local_configuration")
    return api_key


def _report_base(case: ValidatedCase, seed: int, order: tuple[Arm, Arm]) -> dict[str, Any]:
    return {
        "schema": 1,
        "status": "running",
        "case_id": case.case_id,
        "seed": seed,
        "clip": {
            "sha256": case.sha256,
            "byte_count": len(case.audio),
            "duration_seconds": case.duration_seconds,
            "encoding": "mulaw",
            "sample_rate_hz": 8000,
            "channels": 1,
        },
        "manifest_expectations": {
            "language_hint": case.language_hint or "auto",
            "foreground_expected_wording": case.foreground_expected_wording,
            "expected_entities": list(case.expected_entities),
            "background_expected_wording": case.background_expected_wording,
            "foreground_absent": case.foreground_absent,
            "source_kind": case.source_kind,
            "permission_status": case.permission_status,
        },
        "arm_order": [arm.label for arm in order],
        "arms": [],
        "error_category": None,
        "pacing_version": "absolute-deadlines-v2",
        "timing_note": (
            "Send-to-final timing includes transport, pacing, VAD and provider work; "
            "it is not pure provider inference latency."
        ),
    }


async def run_pair(
    case: ValidatedCase,
    api_key: str,
    seed: int,
    connect: Optional[Callable[..., Awaitable[Any]]] = None,
    *,
    clock_factory: Callable[[], Any] = RealClock,
) -> dict[str, Any]:
    order = ordered_arms(seed)
    report = _report_base(case, seed, order)
    deadline = time.monotonic() + PAIR_TIMEOUT_SECONDS
    for arm in order:
        if time.monotonic() >= deadline:
            report["status"] = "incomplete"
            report["error_category"] = "pair_timeout"
            break
        try:
            result = await run_arm(
                arm, case, api_key, connect, clock=clock_factory(),
                budget=deadline - time.monotonic(),
            )
        except asyncio.TimeoutError:
            result = _empty_arm_result(arm, case)
            result["error_category"] = "arm_timeout"
        report["arms"].append(result)
    if len(report["arms"]) == 2 and all(arm["status"] == "complete" for arm in report["arms"]):
        report["status"] = "complete"
    else:
        report["status"] = "incomplete"
        if report["error_category"] is None:
            report["error_category"] = "arm_incomplete"
    return report


def stdout_summary(report: dict[str, Any], report_written: bool) -> dict[str, Any]:
    return {
        "case_id": report.get("case_id"),
        "status": report.get("status", "failed"),
        "arms": [
            {
                "label": arm.get("label"),
                "status": arm.get("status"),
                "final_count": len(arm.get("finals", [])),
                "latency_scoring_valid": arm.get("latency_scoring_valid", False),
            }
            for arm in report.get("arms", [])
        ],
        "report_written": report_written,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline by default. Validate one approved external raw 8 kHz mu-law "
            "case, or explicitly opt in to two provider audio connections."
        )
    )
    parser.add_argument("--allow-provider-contact", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if not args.allow_provider_contact and not args.validate:
        print(
            "Offline by default; no credential or audio loaded. Use --validate, or "
            "--allow-provider-contact only after lead review and audio approval."
        )
        return 0
    if args.manifest is None or args.case is None:
        print(json.dumps({"status": "input_required", "report_written": False}))
        return 2
    try:
        case = validate_case(args.manifest, args.case)
    except PilotError as error:
        print(
            json.dumps(
                {
                    "case_id": args.case if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.case or "") else None,
                    "status": error.category,
                    "report_written": False,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 1
    if args.validate and not args.allow_provider_contact:
        print(
            json.dumps(
                {
                    "case_id": case.case_id,
                    "status": "valid",
                    "report_written": False,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    if args.report is None:
        print(
            json.dumps(
                {"case_id": case.case_id, "status": "report_required", "report_written": False},
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 2
    try:
        reservation = reserve_report(args.report, case, args.seed)
    except PilotError as error:
        print(
            json.dumps(
                {"case_id": case.case_id, "status": error.category, "report_written": False},
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 1

    report = _report_base(case, args.seed, ordered_arms(args.seed))
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        try:
            api_key = _load_configuration()
            report = asyncio.run(run_pair(case, api_key, args.seed))
        except PilotError as error:
            report["status"] = "incomplete"
            report["error_category"] = error.category
        except Exception:
            report["status"] = "incomplete"
            report["error_category"] = "pilot_failed"
        write_report(reservation, report)
    except PilotError:
        print(
            json.dumps(
                {"case_id": case.case_id, "status": "report_write_failed", "report_written": False},
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 1
    finally:
        logging.disable(previous_disable)
        reservation.close()
    print(json.dumps(stdout_summary(report, True), separators=(",", ":"), sort_keys=True))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
