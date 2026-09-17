"""T033-B contracts for truthful provider failure and transport ownership."""
from __future__ import annotations

import asyncio
from collections import deque
import gc
import json
from dataclasses import dataclass

import pytest
from loguru import logger

import services.stt.elevenlabs_stt_service as stt_module
from services.stt.elevenlabs_stt_service import (
    ElevenLabsSTT,
    STTTransportOutcome,
)
from services.stt.stt_base import STTStatus


DOCUMENTED_PROVIDER_ERRORS = {
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


class SyntheticRemoteClose(Exception):
    pass


@pytest.mark.asyncio
async def test_cancel_resistant_old_receive_error_is_owned_and_consumed(monkeypatch):
    class LateErrorSocket(FakeWebSocket):
        def __init__(self):
            super().__init__()
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()

        async def recv(self):
            self.recv_waiting.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                await self.release.wait()
            raise SyntheticRemoteClose("private late provider failure")

    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    old, replacement = LateErrorSocket(), FakeWebSocket()
    monkeypatch.setattr(stt_module, "WebSocketConnectionClosed", SyntheticRemoteClose)
    monkeypatch.setattr(stt_module.websockets, "connect", ScriptedConnector(
        ConnectStep(socket=old), ConnectStep(socket=replacement),
    ))
    stt = ElevenLabsSTT(api_key="synthetic", filter_background_audio=True)
    stt.RECEIVER_CANCEL_TIMEOUT_SECONDS = 0.01
    try:
        assert await stt.connect()
        await asyncio.wait_for(old.recv_waiting.wait(), 0.5)
        old_task = stt._receive_task
        await asyncio.wait_for(stt.reset_for_listening(), 0.5)
        await asyncio.wait_for(old.cancelled.wait(), 0.5)
        old.release.set()
        await asyncio.wait_for(old_task, 0.5)
        await finish_cleanup(stt)
        gc.collect()
        assert not unhandled, "late receive errors must be consumed, never printed"
        assert stt._websocket is replacement
        assert stt.status is STTStatus.CONNECTED
        assert stt.transport_outcome is STTTransportOutcome.TRANSPORT_OPEN
        assert not replacement.closed
    finally:
        old.release.set()
        await stt.disconnect()
        await finish_cleanup(stt)
        loop.set_exception_handler(previous_handler)
    assert not stt._cleanup_tasks


class FakeWebSocket:
    def __init__(self) -> None:
        self.events: asyncio.Queue[dict | BaseException] = asyncio.Queue()
        self.recv_waiting = asyncio.Event()
        self.sent: list[str] = []
        self.closed = False

    def feed(self, event: dict | BaseException) -> None:
        self.events.put_nowait(event)

    async def recv(self) -> str:
        self.recv_waiting.set()
        event = await self.events.get()
        if isinstance(event, BaseException):
            raise event
        return json.dumps(event)

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def close(self) -> None:
        self.closed = True

    async def ping(self) -> None:
        return None


class GatedDeliveryWebSocket(FakeWebSocket):
    def __init__(self) -> None:
        super().__init__()
        self.delivery_started = asyncio.Event()
        self.delivery_release = asyncio.Event()

    async def recv(self) -> str:
        self.recv_waiting.set()
        event = await self.events.get()
        self.delivery_started.set()
        await self.delivery_release.wait()
        if isinstance(event, BaseException):
            raise event
        return json.dumps(event)


class DelayedCleanupWebSocket(FakeWebSocket):
    def __init__(self) -> None:
        super().__init__()
        self.io_release = asyncio.Event()
        self.send_cancelled = asyncio.Event()
        self.close_cancelled = asyncio.Event()
        self.transport = _AbortableTransport()

    async def send(self, message: str) -> None:
        try:
            await self.io_release.wait()
        except asyncio.CancelledError:
            self.send_cancelled.set()
            await self.io_release.wait()
        self.sent.append(message)

    async def close(self) -> None:
        try:
            await self.io_release.wait()
        except asyncio.CancelledError:
            self.close_cancelled.set()
            await self.io_release.wait()
        self.closed = True


class _AbortableTransport:
    def __init__(self) -> None:
        self.aborted = False

    def abort(self) -> None:
        self.aborted = True


@dataclass
class ConnectStep:
    socket: FakeWebSocket | None = None
    error: BaseException | None = None


class ScriptedConnector:
    def __init__(self, *steps: ConnectStep) -> None:
        self.steps = deque(steps)
        self.calls: list[str] = []

    async def __call__(self, url: str, **_kwargs):
        self.calls.append(url)
        step = self.steps.popleft()
        if step.error is not None:
            raise step.error
        return step.socket


async def connect_stt(monkeypatch, socket: FakeWebSocket, **kwargs) -> ElevenLabsSTT:
    connector = ScriptedConnector(ConnectStep(socket=socket))
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    stt = ElevenLabsSTT(api_key="synthetic-key", **kwargs)
    assert await stt.connect() is True
    await asyncio.wait_for(socket.recv_waiting.wait(), timeout=0.5)
    return stt


async def wait_for_receiver(stt: ElevenLabsSTT) -> None:
    task = stt._receive_task
    assert task is not None
    await asyncio.wait_for(task, timeout=0.5)


async def finish_cleanup(stt: ElevenLabsSTT) -> None:
    tasks = tuple(stt._cleanup_tasks)
    if tasks:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=0.5,
        )


@pytest.mark.asyncio
async def test_invalid_request_is_truthful_redacted_and_not_retried(monkeypatch) -> None:
    raw_error = "synthetic-private-error-body-4831"
    socket = FakeWebSocket()
    connector = ScriptedConnector(ConnectStep(socket=socket))
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    captured: list[str] = []
    sink = logger.add(lambda message: captured.append(str(message)), level="DEBUG")
    stt = ElevenLabsSTT(
        api_key="synthetic-key",
        language="ar",
        filter_background_audio=True,
    )
    try:
        assert await stt.connect() is True
        await asyncio.wait_for(socket.recv_waiting.wait(), timeout=0.5)
        socket.feed({
            "message_type": "invalid_request",
            "error": raw_error,
            "url": "wss://private.invalid/secret",
        })
        socket.feed(SyntheticRemoteClose())
        await wait_for_receiver(stt)

        assert stt.status is STTStatus.ERROR
        assert stt.transport_outcome is STTTransportOutcome.PROVIDER_ERROR
        assert stt.provider_error_category == "invalid_request"
        assert stt._is_listening is False
        assert stt._transcript_queue.empty()
        assert socket.events.qsize() == 1
        assert [result async for result in stt.get_transcript()] == []
        assert len(connector.calls) == 1
        assert "filter_background_audio=true" in connector.calls[0]
        assert "language_code=ar" in connector.calls[0]
        assert raw_error not in "".join(captured)
        assert "private.invalid" not in "".join(captured)
        assert raw_error not in repr(stt.transport_outcome)
        await finish_cleanup(stt)
        assert socket.closed is True
    finally:
        logger.remove(sink)
        await stt.disconnect()


def test_provider_error_allowlist_matches_documented_contract() -> None:
    assert ElevenLabsSTT.PROVIDER_ERROR_TYPES == DOCUMENTED_PROVIDER_ERRORS


@pytest.mark.asyncio
@pytest.mark.parametrize("category", sorted(DOCUMENTED_PROVIDER_ERRORS))
async def test_documented_provider_errors_have_allowlisted_category(
    monkeypatch, category: str
) -> None:
    socket = FakeWebSocket()
    stt = await connect_stt(monkeypatch, socket)
    try:
        socket.feed({"message_type": category, "error": "untrusted detail"})
        await wait_for_receiver(stt)
        assert stt.status is STTStatus.ERROR
        assert stt.transport_outcome is STTTransportOutcome.PROVIDER_ERROR
        assert stt.provider_error_category == category
        assert stt._transcript_queue.empty()
    finally:
        await stt.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ({"message_type": "session_ended"}, STTTransportOutcome.REMOTE_ENDED),
        (SyntheticRemoteClose(), STTTransportOutcome.REMOTE_CLOSED),
    ],
)
async def test_ordinary_remote_end_and_close_are_not_provider_errors(
    monkeypatch, event: dict | BaseException, expected: STTTransportOutcome
) -> None:
    monkeypatch.setattr(stt_module, "WebSocketConnectionClosed", SyntheticRemoteClose)
    socket = FakeWebSocket()
    stt = await connect_stt(monkeypatch, socket)
    try:
        socket.feed(event)
        await wait_for_receiver(stt)
        assert stt.status is STTStatus.DISCONNECTED
        assert stt.transport_outcome is expected
        assert stt.provider_error_category is None
        assert stt._is_listening is False
    finally:
        await stt.disconnect()


@pytest.mark.asyncio
async def test_commit_before_failure_drains_once_with_original_identity(monkeypatch) -> None:
    socket = FakeWebSocket()
    stt = await connect_stt(monkeypatch, socket, language="en")
    try:
        socket.feed({
            "message_type": "committed_transcript",
            "text": "accepted before failure",
            "language_code": "en",
        })
        socket.feed({"message_type": "quota_exceeded", "error": "private"})
        await wait_for_receiver(stt)

        iterator = stt.get_transcript()
        result = await anext(iterator)
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(iterator), timeout=0.2)
        assert result.text == "accepted before failure"
        assert result.utterance_id.provider == stt.PROVIDER_ID
        assert result.utterance_id.commit_sequence == 1
        assert stt.status is STTStatus.ERROR
    finally:
        await stt.disconnect()


@pytest.mark.asyncio
async def test_messages_after_provider_failure_are_not_processed(monkeypatch) -> None:
    socket = FakeWebSocket()
    stt = await connect_stt(monkeypatch, socket)
    try:
        socket.feed({"message_type": "input_error", "error": "private"})
        socket.feed({"message_type": "committed_transcript", "text": "too late"})
        await wait_for_receiver(stt)
        assert stt._transcript_queue.empty()
        assert socket.events.qsize() == 1
    finally:
        await stt.disconnect()


@pytest.mark.asyncio
async def test_late_old_failure_and_close_cannot_change_replacement(monkeypatch) -> None:
    monkeypatch.setattr(stt_module, "WebSocketConnectionClosed", SyntheticRemoteClose)
    old_socket = FakeWebSocket()
    new_socket = FakeWebSocket()
    connector = ScriptedConnector(
        ConnectStep(socket=old_socket),
        ConnectStep(socket=new_socket),
    )
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    stt = ElevenLabsSTT(
        api_key="synthetic-key",
        language="en",
        filter_background_audio=True,
    )
    assert await stt.connect() is True
    await asyncio.wait_for(old_socket.recv_waiting.wait(), timeout=0.5)
    old_receiver = stt._current_receiver
    await stt.reset_for_listening()
    await asyncio.wait_for(new_socket.recv_waiting.wait(), timeout=0.5)
    replacement = stt._current_receiver

    old_socket.recv_waiting.clear()
    late_loop = asyncio.create_task(stt._receive_loop(old_receiver))
    await asyncio.wait_for(old_socket.recv_waiting.wait(), timeout=0.5)
    old_socket.feed({"message_type": "auth_error", "error": "late private"})
    old_socket.feed(SyntheticRemoteClose())
    await asyncio.wait_for(late_loop, timeout=0.5)

    assert stt._current_receiver is replacement
    assert stt._websocket is new_socket
    assert stt.status is STTStatus.CONNECTED
    assert stt.transport_outcome is STTTransportOutcome.TRANSPORT_OPEN
    assert stt.provider_error_category is None
    assert new_socket.closed is False
    await stt.disconnect()


@pytest.mark.asyncio
async def test_terminal_disconnect_wins_over_concurrent_failure(monkeypatch) -> None:
    socket = GatedDeliveryWebSocket()
    stt = await connect_stt(monkeypatch, socket)
    socket.feed({"message_type": "invalid_request", "error": "private"})
    await asyncio.wait_for(socket.delivery_started.wait(), timeout=0.5)

    disconnect_task = asyncio.create_task(stt.disconnect())
    socket.delivery_release.set()
    await asyncio.wait_for(disconnect_task, timeout=0.5)

    assert stt.status is STTStatus.DISCONNECTED
    assert stt.transport_outcome is STTTransportOutcome.LOCAL_DISCONNECT
    assert stt.provider_error_category is None
    assert stt._session_active is False
    assert stt._transcript_queue is None


@pytest.mark.asyncio
async def test_explicit_recovery_retains_options_and_clears_only_after_adoption(
    monkeypatch,
) -> None:
    rejected_socket = FakeWebSocket()
    recovered_socket = FakeWebSocket()
    connector = ScriptedConnector(
        ConnectStep(socket=rejected_socket),
        ConnectStep(error=RuntimeError("synthetic connection refusal")),
        ConnectStep(socket=recovered_socket),
    )
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    stt = ElevenLabsSTT(
        api_key="synthetic-key",
        language="en",
        filter_background_audio=True,
    )
    try:
        assert await stt.connect() is True
        await asyncio.wait_for(rejected_socket.recv_waiting.wait(), timeout=0.5)
        rejected_socket.feed({"message_type": "invalid_request", "error": "private"})
        await wait_for_receiver(stt)
        await finish_cleanup(stt)

        assert await stt.reconnect_with_language("ar") is False
        assert stt.transport_outcome is STTTransportOutcome.PROVIDER_ERROR
        assert stt.provider_error_category == "invalid_request"

        assert await stt.reconnect_with_language("ar") is True
        await asyncio.wait_for(recovered_socket.recv_waiting.wait(), timeout=0.5)
        assert stt.transport_outcome is STTTransportOutcome.TRANSPORT_OPEN
        assert stt.provider_error_category is None
        assert stt.language == "ar"
        assert stt.filter_background_audio is True
        assert "language_code=ar" in connector.calls[-1]
        assert "filter_background_audio=true" in connector.calls[-1]
        assert len(connector.calls) == 3
    finally:
        await stt.disconnect()


@pytest.mark.asyncio
async def test_session_started_is_distinct_from_transport_open(monkeypatch) -> None:
    socket = FakeWebSocket()
    stt = await connect_stt(monkeypatch, socket)
    try:
        assert stt.transport_outcome is STTTransportOutcome.TRANSPORT_OPEN
        socket.recv_waiting.clear()
        socket.feed({"message_type": "session_started", "session_id": "private"})
        await asyncio.wait_for(socket.recv_waiting.wait(), timeout=0.5)
        assert stt.status is STTStatus.CONNECTED
        assert stt.transport_outcome is STTTransportOutcome.SESSION_STARTED
    finally:
        await stt.disconnect()


@pytest.mark.asyncio
async def test_instances_keep_failure_state_isolated(monkeypatch) -> None:
    first_socket = FakeWebSocket()
    second_socket = FakeWebSocket()
    connector = ScriptedConnector(
        ConnectStep(socket=first_socket),
        ConnectStep(socket=second_socket),
    )
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    first = ElevenLabsSTT(api_key="first", filter_background_audio=True)
    second = ElevenLabsSTT(api_key="second", filter_background_audio=False)
    try:
        assert await first.connect() is True
        assert await second.connect() is True
        await asyncio.wait_for(first_socket.recv_waiting.wait(), timeout=0.5)
        await asyncio.wait_for(second_socket.recv_waiting.wait(), timeout=0.5)
        first_socket.feed({"message_type": "quota_exceeded", "error": "private"})
        await wait_for_receiver(first)

        assert first.status is STTStatus.ERROR
        assert first.provider_error_category == "quota_exceeded"
        assert second.status is STTStatus.CONNECTED
        assert second.transport_outcome is STTTransportOutcome.TRANSPORT_OPEN
        assert second.provider_error_category is None
        assert second._is_listening is True
    finally:
        await first.disconnect()
        await second.disconnect()


@pytest.mark.asyncio
async def test_failed_socket_cleanup_is_bounded_owned_and_eventually_finishes(
    monkeypatch,
) -> None:
    socket = DelayedCleanupWebSocket()
    monkeypatch.setattr(stt_module.ElevenLabsSTT, "CLOSE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(stt_module.ElevenLabsSTT, "CLEANUP_WAIT_SECONDS", 0.01)
    stt = await connect_stt(monkeypatch, socket)
    socket.feed({"message_type": "resource_exhausted", "error": "private"})
    await wait_for_receiver(stt)
    try:
        await asyncio.wait_for(socket.send_cancelled.wait(), timeout=0.2)
        await asyncio.wait_for(socket.close_cancelled.wait(), timeout=0.2)
        assert socket.transport.aborted is True
        pending = tuple(stt._cleanup_tasks)
        assert pending
        assert stt.status is STTStatus.ERROR

        socket.io_release.set()
        await asyncio.wait_for(
            asyncio.gather(*pending, return_exceptions=True),
            timeout=0.5,
        )
        await finish_cleanup(stt)
        assert socket.closed is True
        assert not stt._cleanup_tasks
    finally:
        socket.io_release.set()
        await stt.disconnect()
