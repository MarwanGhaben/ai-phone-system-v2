"""T030-B behavioral contracts for queue ownership and STT replacement."""

from __future__ import annotations

import asyncio
from collections import deque
import json
from dataclasses import dataclass, field

import pytest
from loguru import logger

import services.stt.elevenlabs_stt_service as stt_module
from services.conversation.events import MetadataDisposition
from services.stt.elevenlabs_stt_service import ElevenLabsSTT
from services.stt.stt_base import STTStatus


class FakeWebSocket:
    def __init__(
        self,
        *,
        send_release: asyncio.Event | None = None,
        close_release: asyncio.Event | None = None,
    ) -> None:
        self._events: asyncio.Queue[str] = asyncio.Queue()
        self.recv_waiting = asyncio.Event()
        self.sent_messages: list[str] = []
        self.closed = False
        self.closed_event = asyncio.Event()
        self.send_release = send_release
        self.close_release = close_release

    def feed(self, event: dict) -> None:
        self._events.put_nowait(json.dumps(event))

    async def recv(self) -> str:
        self.recv_waiting.set()
        return await self._events.get()

    async def send(self, message: str) -> None:
        if self.send_release is not None:
            await self.send_release.wait()
        self.sent_messages.append(message)

    async def close(self) -> None:
        if self.close_release is not None:
            await self.close_release.wait()
        self.closed = True
        self.closed_event.set()

    async def ping(self) -> None:
        return None


@dataclass
class ConnectStep:
    socket: FakeWebSocket | None = None
    error: BaseException | None = None
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event | None = None


class ScriptedConnector:
    def __init__(self, *steps: ConnectStep) -> None:
        self.steps = deque(steps)
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        step = self.steps.popleft()
        step.started.set()
        if step.release is not None:
            await step.release.wait()
        if step.error is not None:
            raise step.error
        return step.socket


class DequeueGateQueue(asyncio.Queue):
    """Pause after removing an item so terminal invalidation can win the race."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.removed = asyncio.Event()
        self.release = asyncio.Event()

    async def get(self):
        self.started.set()
        item = await super().get()
        self.removed.set()
        await self.release.wait()
        return item


class DelayedCancellationWebSocket(FakeWebSocket):
    """Transport whose receiver and close calls delay cancellation cleanup."""

    def __init__(self) -> None:
        super().__init__()
        self.receiver_cancelled = asyncio.Event()
        self.send_cancelled = asyncio.Event()
        self.close_cancelled = asyncio.Event()
        self.receiver_release = asyncio.Event()
        self.io_release = asyncio.Event()

    async def recv(self) -> str:
        self.recv_waiting.set()
        try:
            return await self._events.get()
        except asyncio.CancelledError:
            self.receiver_cancelled.set()
            await self.receiver_release.wait()
            raise

    async def send(self, message: str) -> None:
        try:
            await self.io_release.wait()
        except asyncio.CancelledError:
            self.send_cancelled.set()
            await self.io_release.wait()
        self.sent_messages.append(message)

    async def close(self) -> None:
        try:
            await self.io_release.wait()
        except asyncio.CancelledError:
            self.close_cancelled.set()
            await self.io_release.wait()
        self.closed = True
        self.closed_event.set()


def committed(text: str, **metadata) -> dict:
    return {"message_type": "committed_transcript", "text": text, **metadata}


async def wait_until_receiving(socket: FakeWebSocket) -> None:
    await asyncio.wait_for(socket.recv_waiting.wait(), timeout=0.5)


async def feed_and_wait(socket: FakeWebSocket, event: dict) -> None:
    socket.recv_waiting.clear()
    socket.feed(event)
    await wait_until_receiving(socket)


async def collect_transcripts(stt: ElevenLabsSTT):
    return [result async for result in stt.get_transcript()]


@pytest.mark.asyncio
async def test_queued_finals_survive_reset_in_order_and_unchanged(monkeypatch) -> None:
    old_socket = FakeWebSocket()
    new_socket = FakeWebSocket()
    connector = ScriptedConnector(
        ConnectStep(socket=old_socket),
        ConnectStep(socket=new_socket),
    )
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    stt = ElevenLabsSTT(api_key="synthetic-key", language="en")

    assert await stt.connect() is True
    await wait_until_receiving(old_socket)
    original_queue = stt._transcript_queue
    await feed_and_wait(
        old_socket,
        committed("yes", language_code="en", confidence=0.7),
    )
    await feed_and_wait(
        old_socket,
        committed("لا، الثلاثاء", language_code="ar", confidence=0.8),
    )
    queued = tuple(original_queue._queue)

    await stt.reset_for_listening()
    await wait_until_receiving(new_socket)
    await feed_and_wait(
        new_socket,
        committed("after reset", language_code="en", confidence=0.9),
    )
    iterator = stt.get_transcript()
    results = [await anext(iterator) for _ in range(3)]

    assert stt._transcript_queue is original_queue
    assert results[0] is queued[0]
    assert results[1] is queued[1]
    assert [result.text for result in results] == [
        "yes",
        "لا، الثلاثاء",
        "after reset",
    ]
    assert [result.language for result in results] == ["en", "ar", "en"]
    assert [result.confidence for result in results] == [0.7, 0.8, 0.9]
    assert results[0].utterance_id.connection_epoch == results[1].utterance_id.connection_epoch
    assert results[2].utterance_id.connection_epoch != results[0].utterance_id.connection_epoch
    assert len({result.utterance_id for result in results}) == 3
    await iterator.aclose()
    await stt.disconnect()


@pytest.mark.asyncio
async def test_waiting_iterator_continues_through_reset(monkeypatch) -> None:
    old_socket = FakeWebSocket()
    new_socket = FakeWebSocket()
    connector = ScriptedConnector(
        ConnectStep(socket=old_socket),
        ConnectStep(socket=new_socket),
    )
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    stt = ElevenLabsSTT(api_key="synthetic-key")
    assert await stt.connect() is True
    await wait_until_receiving(old_socket)

    iterator = stt.get_transcript()
    waiting_result = asyncio.create_task(anext(iterator))
    await stt.reset_for_listening()
    await wait_until_receiving(new_socket)
    await feed_and_wait(new_socket, committed("new epoch answer"))

    result = await asyncio.wait_for(waiting_result, timeout=0.5)
    assert result.text == "new epoch answer"
    await iterator.aclose()
    await stt.disconnect()


@pytest.mark.asyncio
async def test_consumed_result_is_not_replayed_after_reset(monkeypatch) -> None:
    old_socket = FakeWebSocket()
    new_socket = FakeWebSocket()
    connector = ScriptedConnector(
        ConnectStep(socket=old_socket),
        ConnectStep(socket=new_socket),
    )
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    stt = ElevenLabsSTT(api_key="synthetic-key")
    assert await stt.connect() is True
    await wait_until_receiving(old_socket)
    iterator = stt.get_transcript()

    await feed_and_wait(old_socket, committed("already consumed"))
    first = await anext(iterator)
    await stt.reset_for_listening()
    await wait_until_receiving(new_socket)
    await feed_and_wait(new_socket, committed("only new result"))
    second = await asyncio.wait_for(anext(iterator), timeout=0.5)

    assert [first.text, second.text] == ["already consumed", "only new result"]
    assert first.utterance_id != second.utterance_id
    assert stt._transcript_queue.empty()
    await iterator.aclose()
    await stt.disconnect()


@pytest.mark.asyncio
async def test_language_reconnect_preserves_queue_and_original_result(monkeypatch) -> None:
    english_socket = FakeWebSocket()
    arabic_socket = FakeWebSocket()
    connector = ScriptedConnector(
        ConnectStep(socket=english_socket),
        ConnectStep(socket=arabic_socket),
    )
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    stt = ElevenLabsSTT(api_key="synthetic-key", language="en")
    assert await stt.connect() is True
    await wait_until_receiving(english_socket)
    queue = stt._transcript_queue
    await feed_and_wait(
        english_socket,
        committed("preserved", language_code="en", confidence=0.6),
    )
    preserved = queue._queue[0]

    assert await stt.reconnect_with_language("ar") is True
    assert stt._transcript_queue is queue
    assert "language_code=ar" in connector.calls[1][0]
    result = await anext(stt.get_transcript())

    assert result is preserved
    assert result.language == "en"
    assert result.confidence == 0.6
    assert result.utterance_id.connection_epoch != stt._current_receiver.connection_epoch
    await stt.disconnect()


@pytest.mark.asyncio
async def test_failed_reconnect_preserves_finals_and_iterator_finishes(monkeypatch) -> None:
    old_socket = FakeWebSocket()
    connector = ScriptedConnector(
        ConnectStep(socket=old_socket),
        ConnectStep(error=RuntimeError("private reconnect failure")),
    )
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    stt = ElevenLabsSTT(api_key="synthetic-key")
    assert await stt.connect() is True
    await wait_until_receiving(old_socket)
    queue = stt._transcript_queue
    await feed_and_wait(old_socket, committed("must survive"))

    assert await stt.reconnect_with_language("ar") is False
    results = await asyncio.wait_for(collect_transcripts(stt), timeout=0.5)

    assert stt._transcript_queue is queue
    assert [result.text for result in results] == ["must survive"]
    assert stt.status is STTStatus.ERROR
    assert stt._is_listening is False
    assert stt._websocket is None
    await stt.disconnect()


@pytest.mark.asyncio
async def test_failed_reset_raises_safely_and_preserves_finals(monkeypatch) -> None:
    old_socket = FakeWebSocket()
    connector = ScriptedConnector(
        ConnectStep(socket=old_socket),
        ConnectStep(error=RuntimeError("private reset failure")),
    )
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    stt = ElevenLabsSTT(api_key="synthetic-key")
    assert await stt.connect() is True
    await wait_until_receiving(old_socket)
    await feed_and_wait(old_socket, committed("queued correction"))

    with pytest.raises(RuntimeError, match="STT reset failed") as caught:
        await stt.reset_for_listening()
    results = await asyncio.wait_for(collect_transcripts(stt), timeout=0.5)

    assert "private" not in str(caught.value)
    assert [result.text for result in results] == ["queued correction"]
    assert stt.status is STTStatus.ERROR
    assert stt._is_listening is False
    await stt.disconnect()


@pytest.mark.asyncio
async def test_concurrent_replacements_serialize_and_old_epoch_stays_revoked(monkeypatch) -> None:
    old_socket = FakeWebSocket()
    middle_socket = FakeWebSocket()
    final_socket = FakeWebSocket()
    middle_release = asyncio.Event()
    final_release = asyncio.Event()
    middle_step = ConnectStep(socket=middle_socket, release=middle_release)
    final_step = ConnectStep(socket=final_socket, release=final_release)
    connector = ScriptedConnector(
        ConnectStep(socket=old_socket),
        middle_step,
        final_step,
    )
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    stt = ElevenLabsSTT(api_key="synthetic-key")
    assert await stt.connect() is True
    await wait_until_receiving(old_socket)
    old_receiver = stt._current_receiver

    reset_task = asyncio.create_task(stt.reset_for_listening())
    await middle_step.started.wait()
    reconnect_task = asyncio.create_task(stt.reconnect_with_language("ar"))
    assert not final_step.started.is_set()
    middle_release.set()
    await final_step.started.wait()
    final_release.set()
    await reset_task
    assert await reconnect_task is True
    await wait_until_receiving(final_socket)

    late_loop = asyncio.create_task(stt._receive_loop(old_receiver))
    await wait_until_receiving(old_socket)
    await feed_and_wait(old_socket, committed("late old speech"))
    old_socket.feed({"message_type": "session_ended"})
    await late_loop
    await feed_and_wait(final_socket, committed("current speech"))
    result = await anext(stt.get_transcript())

    assert result.text == "current speech"
    assert any(
        event.disposition is MetadataDisposition.OLD_EPOCH
        for event in stt.metadata_events
    )
    assert old_socket.closed is True
    assert middle_socket.closed is True
    assert final_socket.closed is False
    assert stt._websocket is final_socket
    assert stt._receive_task is not None and not stt._receive_task.done()
    await stt.disconnect()


@pytest.mark.asyncio
async def test_disconnect_wins_over_blocked_connect_and_closes_candidate(monkeypatch) -> None:
    old_socket = FakeWebSocket()
    candidate = FakeWebSocket()
    release = asyncio.Event()
    step = ConnectStep(socket=candidate, release=release)
    connector = ScriptedConnector(ConnectStep(socket=old_socket), step)
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    stt = ElevenLabsSTT(api_key="synthetic-key")
    assert await stt.connect() is True
    await wait_until_receiving(old_socket)

    connect_task = asyncio.create_task(stt.reconnect_with_language("ar"))
    await step.started.wait()
    terminal_started = asyncio.Event()
    original_disconnect = stt.disconnect

    async def signaled_disconnect() -> None:
        terminal_started.set()
        await original_disconnect()

    disconnect_task = asyncio.create_task(signaled_disconnect())
    await terminal_started.wait()
    release.set()

    assert await connect_task is False
    await disconnect_task
    assert old_socket.closed is True
    assert candidate.closed is True
    assert stt.status is STTStatus.DISCONNECTED
    assert stt._is_listening is False
    assert stt._websocket is None
    assert stt._current_receiver is None
    assert stt._receive_task is None


@pytest.mark.asyncio
async def test_replacement_cancellation_cleans_up_and_propagates(monkeypatch) -> None:
    old_socket = FakeWebSocket()
    candidate = FakeWebSocket()
    replacement_started = asyncio.Event()
    never_release = asyncio.Event()
    calls = 0

    async def cancellation_connector(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return old_socket
        replacement_started.set()
        try:
            await never_release.wait()
        except asyncio.CancelledError:
            # Simulate a provider connector that completes socket creation at the
            # same boundary where its caller is cancelled.
            return candidate

    monkeypatch.setattr(
        stt_module.websockets,
        "connect",
        cancellation_connector,
    )
    stt = ElevenLabsSTT(api_key="synthetic-key")
    assert await stt.connect() is True
    await wait_until_receiving(old_socket)

    reset_task = asyncio.create_task(stt.reset_for_listening())
    await replacement_started.wait()
    reset_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reset_task

    assert old_socket.closed is True
    assert candidate.closed is True
    assert stt._websocket is None
    assert stt._receive_task is None
    assert stt.status is STTStatus.ERROR
    await asyncio.wait_for(stt.disconnect(), timeout=0.5)


@pytest.mark.asyncio
async def test_cancellation_while_waiting_for_lifecycle_lock_has_no_side_effect(
    monkeypatch,
) -> None:
    old_socket = FakeWebSocket()
    replacement = FakeWebSocket()
    release = asyncio.Event()
    step = ConnectStep(socket=replacement, release=release)
    connector = ScriptedConnector(ConnectStep(socket=old_socket), step)
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    stt = ElevenLabsSTT(api_key="synthetic-key", language="en")
    assert await stt.connect() is True
    await wait_until_receiving(old_socket)

    reset_task = asyncio.create_task(stt.reset_for_listening())
    await step.started.wait()
    reconnect_started = asyncio.Event()
    original_reconnect = stt.reconnect_with_language

    async def signaled_reconnect(language: str) -> bool:
        reconnect_started.set()
        return await original_reconnect(language)

    waiting_reconnect = asyncio.create_task(signaled_reconnect("ar"))
    await reconnect_started.wait()
    waiting_reconnect.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting_reconnect
    release.set()
    await reset_task

    assert stt.language == "en"
    assert stt._websocket is replacement
    assert replacement.closed is False
    await stt.disconnect()


@pytest.mark.asyncio
async def test_instances_keep_independent_queues_and_receivers(monkeypatch) -> None:
    a_old = FakeWebSocket()
    b_socket = FakeWebSocket()
    a_new = FakeWebSocket()
    connector = ScriptedConnector(
        ConnectStep(socket=a_old),
        ConnectStep(socket=b_socket),
        ConnectStep(socket=a_new),
    )
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    first = ElevenLabsSTT(api_key="first-key")
    second = ElevenLabsSTT(api_key="second-key")
    assert await first.connect() is True
    assert await second.connect() is True
    await wait_until_receiving(a_old)
    await wait_until_receiving(b_socket)
    second_queue = second._transcript_queue
    second_receiver = second._current_receiver

    await first.reset_for_listening()
    await first.disconnect()
    await feed_and_wait(b_socket, committed("second call intact"))
    result = await anext(second.get_transcript())

    assert result.text == "second call intact"
    assert second._transcript_queue is second_queue
    assert second._current_receiver is second_receiver
    assert second._websocket is b_socket
    assert b_socket.closed is False
    await second.disconnect()


@pytest.mark.asyncio
async def test_terminal_disconnect_ends_iterator_and_new_session_is_empty(monkeypatch) -> None:
    old_socket = FakeWebSocket()
    new_socket = FakeWebSocket()
    connector = ScriptedConnector(
        ConnectStep(socket=old_socket),
        ConnectStep(socket=new_socket),
    )
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    stt = ElevenLabsSTT(api_key="synthetic-key")
    assert await stt.connect() is True
    await wait_until_receiving(old_socket)
    await feed_and_wait(old_socket, committed("discard at teardown"))
    old_queue = stt._transcript_queue

    await stt.disconnect()
    await stt.disconnect()
    assert stt._transcript_queue is None
    assert old_queue.empty()
    assert await stt.connect() is True
    await wait_until_receiving(new_socket)
    assert stt._transcript_queue is not old_queue
    iterator = stt.get_transcript()
    waiting = asyncio.create_task(anext(iterator))
    await feed_and_wait(new_socket, committed("new session only"))
    assert (await waiting).text == "new session only"

    terminal_wait = asyncio.create_task(anext(iterator))
    await stt.disconnect()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(terminal_wait, timeout=0.5)


@pytest.mark.asyncio
async def test_old_iterator_stays_terminated_after_immediate_new_connect(
    monkeypatch,
) -> None:
    old_socket = FakeWebSocket()
    new_socket = FakeWebSocket()
    connector = ScriptedConnector(
        ConnectStep(socket=old_socket),
        ConnectStep(socket=new_socket),
    )
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    stt = ElevenLabsSTT(api_key="synthetic-key")
    assert await stt.connect() is True
    await wait_until_receiving(old_socket)
    old_queue = DequeueGateQueue()
    stt._transcript_queue = old_queue
    old_iterator = stt.get_transcript()
    old_waiter = asyncio.create_task(anext(old_iterator))
    await old_queue.started.wait()

    await stt.disconnect()
    assert await stt.connect() is True
    await wait_until_receiving(new_socket)
    await feed_and_wait(new_socket, committed("new session speech"))
    new_result = await anext(stt.get_transcript())

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(old_waiter, timeout=0.5)
    assert new_result.text == "new session speech"
    await stt.disconnect()


@pytest.mark.asyncio
async def test_dequeued_final_cannot_yield_after_terminal_disconnect(
    monkeypatch,
) -> None:
    old_socket = FakeWebSocket()
    new_socket = FakeWebSocket()
    connector = ScriptedConnector(
        ConnectStep(socket=old_socket),
        ConnectStep(socket=new_socket),
    )
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    stt = ElevenLabsSTT(api_key="synthetic-key")
    assert await stt.connect() is True
    await wait_until_receiving(old_socket)
    gated_queue = DequeueGateQueue()
    stt._transcript_queue = gated_queue
    old_waiter = asyncio.create_task(anext(stt.get_transcript()))
    await feed_and_wait(old_socket, committed("terminal race speech"))
    await gated_queue.removed.wait()

    await stt.disconnect()
    gated_queue.release.set()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(old_waiter, timeout=0.5)

    assert await stt.connect() is True
    await wait_until_receiving(new_socket)
    await feed_and_wait(new_socket, committed("new session speech"))
    assert (await anext(stt.get_transcript())).text == "new session speech"
    await stt.disconnect()


@pytest.mark.asyncio
async def test_internal_replacement_cannot_reopen_terminal_session(monkeypatch) -> None:
    old_socket = FakeWebSocket()
    reopened_socket = FakeWebSocket()
    reset_socket = FakeWebSocket()
    connector = ScriptedConnector(
        ConnectStep(socket=old_socket),
        ConnectStep(socket=reopened_socket),
        ConnectStep(socket=reset_socket),
    )
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    stt = ElevenLabsSTT(api_key="synthetic-key")
    assert await stt.connect() is True
    await wait_until_receiving(old_socket)
    await stt.disconnect()

    with pytest.raises(RuntimeError, match="STT reset failed"):
        await stt.reset_for_listening()
    assert await stt.reconnect_with_language("ar") is False
    assert len(connector.calls) == 1
    assert stt._transcript_queue is None
    assert stt._websocket is None
    assert stt.status is STTStatus.DISCONNECTED

    assert await stt.connect() is True
    await wait_until_receiving(reopened_socket)
    await stt.reset_for_listening()
    await wait_until_receiving(reset_socket)
    assert len(connector.calls) == 3
    assert stt._websocket is reset_socket
    await stt.disconnect()


@pytest.mark.asyncio
async def test_connection_timeout_returns_before_late_socket_cleanup(monkeypatch) -> None:
    cancellation_started = asyncio.Event()
    connector_release = asyncio.Event()
    candidate = FakeWebSocket()

    async def delayed_cancellation_connector(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_started.set()
            await connector_release.wait()
            return candidate

    monkeypatch.setattr(
        stt_module.websockets,
        "connect",
        delayed_cancellation_connector,
    )
    monkeypatch.setattr(stt_module.ElevenLabsSTT, "CONNECT_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(stt_module.ElevenLabsSTT, "CLEANUP_WAIT_SECONDS", 0.01)
    stt = ElevenLabsSTT(api_key="synthetic-key")
    connect_task = asyncio.create_task(stt.connect())
    try:
        await cancellation_started.wait()
        assert await asyncio.wait_for(connect_task, timeout=0.2) is False
        cleanup_handles = tuple(stt._cleanup_tasks)
        assert cleanup_handles
        assert candidate.closed is False
        assert stt._websocket is None

        connector_release.set()
        await asyncio.wait_for(
            asyncio.gather(*cleanup_handles, return_exceptions=True),
            timeout=0.5,
        )
        assert candidate.closed is True
        assert stt._websocket is None
        assert not stt._cleanup_tasks
    finally:
        connector_release.set()
        if not connect_task.done():
            connect_task.cancel()
        await asyncio.gather(connect_task, return_exceptions=True)
        await stt.disconnect()


@pytest.mark.asyncio
async def test_repeated_caller_cancellation_keeps_late_cleanup_owned(monkeypatch) -> None:
    connector_started = asyncio.Event()
    cancellation_started = asyncio.Event()
    connector_release = asyncio.Event()
    candidate = FakeWebSocket()

    async def delayed_cancellation_connector(*args, **kwargs):
        connector_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_started.set()
            await connector_release.wait()
            return candidate

    monkeypatch.setattr(
        stt_module.websockets,
        "connect",
        delayed_cancellation_connector,
    )
    stt = ElevenLabsSTT(api_key="synthetic-key")
    connect_task = asyncio.create_task(stt.connect())
    try:
        await connector_started.wait()
        connect_task.cancel()
        await cancellation_started.wait()
        assert stt._cleanup_tasks
        connect_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(connect_task, timeout=0.2)
        cleanup_handles = tuple(stt._cleanup_tasks)
        assert cleanup_handles

        connector_release.set()
        await asyncio.wait_for(
            asyncio.gather(*cleanup_handles, return_exceptions=True),
            timeout=0.5,
        )
        assert candidate.closed is True
        assert not stt._cleanup_tasks
    finally:
        connector_release.set()
        if not connect_task.done():
            connect_task.cancel()
        await asyncio.gather(connect_task, return_exceptions=True)
        await stt.disconnect()


@pytest.mark.asyncio
async def test_receiver_and_close_delays_remain_tracked_after_disconnect(
    monkeypatch,
) -> None:
    socket = DelayedCancellationWebSocket()
    connector = ScriptedConnector(ConnectStep(socket=socket))
    monkeypatch.setattr(stt_module.websockets, "connect", connector)
    monkeypatch.setattr(
        stt_module.ElevenLabsSTT,
        "RECEIVER_CANCEL_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(stt_module.ElevenLabsSTT, "CLOSE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(stt_module.ElevenLabsSTT, "CLEANUP_WAIT_SECONDS", 0.01)
    stt = ElevenLabsSTT(api_key="synthetic-key")
    assert await stt.connect() is True
    await wait_until_receiving(socket)
    disconnect_task = asyncio.create_task(stt.disconnect())
    try:
        await socket.receiver_cancelled.wait()
        await asyncio.wait_for(disconnect_task, timeout=0.2)
        cleanup_handles = tuple(stt._cleanup_tasks)
        assert cleanup_handles
        assert stt.status is STTStatus.DISCONNECTED
        assert stt._websocket is None

        socket.receiver_release.set()
        socket.io_release.set()
        await asyncio.wait_for(
            asyncio.gather(*cleanup_handles, return_exceptions=True),
            timeout=0.5,
        )
        assert socket.closed is True
        assert not stt._cleanup_tasks
    finally:
        socket.receiver_release.set()
        socket.io_release.set()
        if not disconnect_task.done():
            disconnect_task.cancel()
        await asyncio.gather(disconnect_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_connection_and_close_timeouts_are_bounded_and_logs_are_safe(
    monkeypatch,
) -> None:
    transcript_sentinel = "synthetic-transcript-secret-8301"
    credential_sentinel = "synthetic-api-secret-4219"
    captured: list[str] = []
    sink = logger.add(lambda message: captured.append(str(message)), level="DEBUG")
    try:
        send_release = asyncio.Event()
        close_release = asyncio.Event()
        socket = FakeWebSocket(
            send_release=send_release,
            close_release=close_release,
        )
        never_release = asyncio.Event()
        blocked_step = ConnectStep(socket=FakeWebSocket(), release=never_release)
        connector = ScriptedConnector(ConnectStep(socket=socket), blocked_step)
        monkeypatch.setattr(stt_module.websockets, "connect", connector)
        monkeypatch.setattr(stt_module.ElevenLabsSTT, "CONNECT_TIMEOUT_SECONDS", 0.01)
        monkeypatch.setattr(stt_module.ElevenLabsSTT, "CLOSE_TIMEOUT_SECONDS", 0.01)
        stt = ElevenLabsSTT(api_key=credential_sentinel)
        assert await stt.connect() is True
        await wait_until_receiving(socket)
        await feed_and_wait(socket, committed(transcript_sentinel))

        await asyncio.wait_for(stt.disconnect(), timeout=0.2)
        assert socket.closed is False
        assert stt._transcript_queue is None

        timed_out = ElevenLabsSTT(api_key=credential_sentinel)
        assert await asyncio.wait_for(timed_out.connect(), timeout=0.2) is False
        assert timed_out.status is STTStatus.ERROR
        logs = "".join(captured)
        assert transcript_sentinel not in logs
        assert credential_sentinel not in logs
    finally:
        logger.remove(sink)
