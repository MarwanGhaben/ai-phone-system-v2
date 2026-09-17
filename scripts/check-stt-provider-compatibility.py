#!/usr/bin/env python3
"""Opt-in, handshake-only Scribe configuration compatibility probe."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import logging
import time
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlencode


ENDPOINT = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
REVIEWED_MODEL = "scribe_v2_realtime"
ARM_TIMEOUT_SECONDS = 10.0
TOTAL_TIMEOUT_SECONDS = 80.0
CLOSE_TIMEOUT_SECONDS = 2.0
MAX_MESSAGE_BYTES = 64 * 1024
MAX_IGNORED_EVENTS = 8

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


class LocalConfigurationError(RuntimeError):
    """A safe local configuration category; never expose its underlying detail."""


@dataclass(frozen=True)
class Arm:
    label: str
    language: str
    filtering: bool


@dataclass
class CleanupOwner:
    """Retain child tasks until they finish, including cancellation-resistant ones."""

    tasks: set[asyncio.Task[Any]]

    def retain(self, task: asyncio.Task[Any]) -> None:
        if task.done():
            _consume_task(task)
            return
        self.tasks.add(task)
        task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Task[Any]) -> None:
        self.tasks.discard(task)
        _consume_task(task)


def _consume_task(task: asyncio.Task[Any]) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        pass


def build_probe_url(language: str, filtering: bool) -> str:
    """Build the reviewed query without adding timestamps or provider options."""
    parameters = list(QUERY_PARAMETERS)
    if language:
        parameters.append(("language_code", language))
    if filtering:
        parameters.append(("filter_background_audio", "true"))
    return f"{ENDPOINT}?{urlencode(parameters)}"


def arms() -> tuple[Arm, ...]:
    return tuple(
        Arm(
            label=f"{language or 'auto'}-{'filtered' if filtering else 'baseline'}",
            language=language,
            filtering=filtering,
        )
        for language in ("", "en", "ar")
        for filtering in (False, True)
    )


def _arm_result(
    arm: Arm,
    started: float,
    outcome: str,
    *,
    provider_error: Optional[str] = None,
    filtering_echo: str = "unknown",
) -> dict[str, Any]:
    return {
        "label": arm.label,
        "language": arm.language or "auto",
        "filtering": arm.filtering,
        "outcome": outcome,
        "provider_error": provider_error,
        "filtering_echo": filtering_echo,
        "elapsed_seconds": round(max(0.0, time.monotonic() - started), 3),
    }


def _filtering_echo(message: dict[str, Any]) -> str:
    """Read only the optional boolean echo; absent remains explicitly unknown."""
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


async def _bounded_await(
    awaitable: Awaitable[Any],
    timeout: float,
    owner: CleanupOwner,
    dispose: Optional[Callable[[Any], Awaitable[Any]]] = None,
) -> tuple[Optional[Any], bool]:
    task = asyncio.ensure_future(awaitable)
    def abandon() -> None:
        task.cancel()

        async def finish() -> None:
            try:
                value = await task
            except (asyncio.CancelledError, Exception):
                return
            if dispose is not None and value is not None:
                await dispose(value)

        owner.retain(asyncio.create_task(finish()))

    try:
        done, pending = await asyncio.wait({task}, timeout=timeout)
    except asyncio.CancelledError:
        abandon()
        raise
    if done:
        return task.result(), False
    if pending:
        abandon()
        return None, True
    return None, True


def _abort_socket(socket: Any) -> None:
    try:
        transport = getattr(socket, "transport", None)
        if transport is not None:
            transport.abort()
    except Exception:
        pass


async def _close_socket(socket: Any, owner: CleanupOwner) -> bool:
    try:
        _, timed_out = await _bounded_await(
            socket.close(), CLOSE_TIMEOUT_SECONDS, owner
        )
        if timed_out:
            _abort_socket(socket)
        return not timed_out
    except asyncio.CancelledError:
        _abort_socket(socket)
        raise
    except Exception:
        _abort_socket(socket)
        return False


async def run_arm(
    arm: Arm,
    api_key: str,
    connect: Optional[Callable[..., Awaitable[Any]]] = None,
    *,
    owner: Optional[CleanupOwner] = None,
    deadline: Optional[float] = None,
) -> dict[str, Any]:
    """Run one handshake-only arm and return only fixed, redacted fields."""
    owner = owner or CleanupOwner(set())
    started = time.monotonic()
    deadline = min(deadline or float("inf"), started + ARM_TIMEOUT_SECONDS)
    socket = None
    try:
        if connect is None:
            import websockets

            connect = websockets.connect
        socket, timed_out = await _bounded_await(
            connect(
                build_probe_url(arm.language, arm.filtering),
                extra_headers={"xi-api-key": api_key},
                ping_interval=None,
                max_size=MAX_MESSAGE_BYTES,
                max_queue=1,
                open_timeout=ARM_TIMEOUT_SECONDS,
                close_timeout=CLOSE_TIMEOUT_SECONDS,
            ),
            max(0.0, deadline - time.monotonic()),
            owner,
            dispose=lambda late_socket: _close_socket(late_socket, owner),
        )
        if timed_out:
            return _arm_result(arm, started, "timeout")
        if socket is None:
            return _arm_result(arm, started, "connection_failed")

        ignored = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return _arm_result(arm, started, "timeout")
            message, timed_out = await _bounded_await(
                socket.recv(), remaining, owner
            )
            if timed_out:
                return _arm_result(arm, started, "timeout")
            if isinstance(message, bytes):
                if len(message) > MAX_MESSAGE_BYTES:
                    return _arm_result(arm, started, "message_too_large")
                try:
                    message = message.decode("utf-8")
                except UnicodeDecodeError:
                    return _arm_result(arm, started, "malformed_event")
            if not isinstance(message, str) or len(message.encode("utf-8")) > MAX_MESSAGE_BYTES:
                return _arm_result(arm, started, "message_too_large")
            try:
                data = json.loads(message)
            except (TypeError, ValueError):
                return _arm_result(arm, started, "malformed_event")
            if not isinstance(data, dict):
                return _arm_result(arm, started, "malformed_event")
            message_type = data.get("message_type")
            if not isinstance(message_type, str):
                return _arm_result(arm, started, "malformed_event")
            if message_type == "session_started":
                echo = _filtering_echo(data)
                if echo == "malformed":
                    return _arm_result(arm, started, "malformed_event")
                return _arm_result(
                    arm, started, "accepted", filtering_echo=echo
                )
            if message_type in PROVIDER_ERROR_TYPES:
                return _arm_result(
                    arm,
                    started,
                    "provider_error",
                    provider_error=message_type,
                )
            ignored += 1
            if ignored > MAX_IGNORED_EVENTS:
                return _arm_result(arm, started, "unknown_event_limit")
    except asyncio.CancelledError:
        raise
    except Exception as error:
        try:
            import websockets

            closed_type = websockets.exceptions.ConnectionClosed
        except Exception:
            closed_type = ()
        if closed_type and isinstance(error, closed_type):
            return _arm_result(arm, started, "remote_closed")
        return _arm_result(arm, started, "connection_failed")
    finally:
        if socket is not None:
            closed = await _close_socket(socket, owner)
            if not closed and not asyncio.current_task().cancelling():
                return _arm_result(arm, started, "cleanup_failed")


def _load_configuration() -> tuple[str, str]:
    """Load only the existing Settings fields required by this diagnostic."""
    try:
        from config.settings import Settings

        settings = Settings(_env_file=None)
        api_key = settings.elevenlabs_api_key
        model = settings.elevenlabs_stt_model
    except Exception as error:
        raise LocalConfigurationError from error
    if not isinstance(api_key, str) or not api_key.strip():
        raise LocalConfigurationError
    if model != REVIEWED_MODEL:
        raise LocalConfigurationError
    return api_key, model


async def _drain_cleanup(owner: CleanupOwner) -> None:
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
        # Pending tasks stay in ``owner.tasks`` with their existing callbacks.


def _local_configuration_output() -> dict[str, Any]:
    return {
        "aggregate_success": False,
        "status": "local_configuration",
        "arms": [
            {
                "label": arm.label,
                "language": arm.language or "auto",
                "filtering": arm.filtering,
                "outcome": "unattempted",
                "provider_error": None,
                "filtering_echo": "unknown",
                "elapsed_seconds": 0.0,
            }
            for arm in arms()
        ],
    }


async def run_probe() -> dict[str, Any]:
    api_key, _ = _load_configuration()
    owner = CleanupOwner(set())
    results = _local_configuration_output()["arms"]
    # Leave time for the last socket close and late-child cleanup inside the
    # overall budget. Each arm still has its own combined open/receive deadline.
    deadline = time.monotonic() + TOTAL_TIMEOUT_SECONDS - 2 * CLOSE_TIMEOUT_SECONDS
    status = "complete"
    try:
        for index, arm in enumerate(arms()):
            if time.monotonic() >= deadline:
                status = "overall_timeout"
                break
            results[index] = await run_arm(arm, api_key, owner=owner, deadline=deadline)
            if time.monotonic() >= deadline:
                status = "overall_timeout"
                break
    finally:
        await _drain_cleanup(owner)
    return {
        "aggregate_success": status == "complete" and all(
            result["outcome"] == "accepted" for result in results
        ),
        "status": status,
        "arms": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline by default. Use --allow-provider-contact only for six "
            "reviewed, handshake-only Scribe configuration checks."
        )
    )
    parser.add_argument(
        "--allow-provider-contact",
        action="store_true",
        help="explicitly opt in to six short real provider connections",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if not args.allow_provider_contact:
        print(
            "Offline by default; no provider contacted. Re-run with "
            "--allow-provider-contact only after lead review."
        )
        return 0

    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        try:
            output = asyncio.run(run_probe())
        except LocalConfigurationError:
            output = _local_configuration_output()
        except Exception:
            output = {"aggregate_success": False, "status": "probe_failed", "arms": []}
    finally:
        logging.disable(previous_disable)
    print(json.dumps(output, separators=(",", ":"), sort_keys=True))
    return 0 if output["aggregate_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
