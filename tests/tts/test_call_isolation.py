import asyncio

import pytest

from services.tts.elevenlabs_service import ElevenLabsTTS
from services.tts.tts_base import TTSRequest


class FakeResponse:
    def __init__(
        self,
        chunks,
        gate: asyncio.Event | None = None,
        error=None,
        close_gate: asyncio.Event | None = None,
    ):
        self._chunks = list(chunks)
        self._gate = gate
        self._error = error
        self._close_gate = close_gate
        self.closed = False
        self.close_started = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.close_started.set()
        if self._close_gate is not None:
            await self._close_gate.wait()
        self.closed = True

    def raise_for_status(self):
        return None

    async def aiter_bytes(self, _size):
        if self._chunks:
            yield self._chunks.pop(0)
        if self._gate is not None:
            await self._gate.wait()
        if self._error is not None:
            raise self._error
        for chunk in self._chunks:
            yield chunk


class DelayedReaderCancellationResponse(FakeResponse):
    def __init__(self, release: asyncio.Event):
        super().__init__([])
        self._reader_release = release
        self.reader_started = asyncio.Event()
        self.reader_cancelled = asyncio.Event()
        self.reader_finished = asyncio.Event()

    async def aiter_bytes(self, _size):
        yield b"first"
        self.reader_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.reader_cancelled.set()
            await self._reader_release.wait()
            self.reader_finished.set()
            yield b"late"


class DelayedEntryResponse(FakeResponse):
    def __init__(self, entry_release: asyncio.Event, exit_release: asyncio.Event):
        super().__init__([])
        self._entry_release = entry_release
        self._exit_release = exit_release
        self.entry_started = asyncio.Event()
        self.exit_started = asyncio.Event()
        self.iter_calls = 0

    async def __aenter__(self):
        self.entry_started.set()
        await self._entry_release.wait()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exit_started.set()
        await self._exit_release.wait()
        self.closed = True

    async def aiter_bytes(self, _size):
        self.iter_calls += 1
        yield b"forbidden"


class FakeClient:
    is_closed = False

    def __init__(self, responses):
        self.responses = responses
        self.stream_calls = 0

    def stream(self, _method, _url, **kwargs):
        self.stream_calls += 1
        return self.responses[kwargs["json"]["text"]]


def make_tts(responses):
    tts = ElevenLabsTTS.__new__(ElevenLabsTTS)
    tts.api_key = "test-key"
    tts.default_voice_id = "voice"
    tts.model = "model"
    tts.stability = 0.5
    tts.similarity_boost = 0.75
    tts.output_format = "ulaw_8000"
    tts._http_client = FakeClient(responses)
    tts._active_streams = set()
    tts._legacy_streams = set()
    return tts


async def collect(stream, consumed=None):
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
        if consumed is not None:
            consumed.set()
    return chunks


async def cleanup_stream_test(tts, tasks, gates):
    for gate in gates:
        gate.set()
    streams = tuple(tts._active_streams)
    for stream in streams:
        stream.cancel()
    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=1)
    for stream in streams:
        await stream.wait_closed(timeout=1)


@pytest.mark.asyncio
async def test_stopping_one_shared_tts_request_does_not_stop_or_resurrect_others():
    release_a = asyncio.Event()
    responses = {
        "A": FakeResponse([b"a1", b"a-late"], release_a),
        "B": FakeResponse([b"b1", b"b2"]),
        "C": FakeResponse([b"c1"]),
    }
    tts = make_tts(responses)
    stream_a = tts.create_stream(TTSRequest("A"))
    stream_b = tts.create_stream(TTSRequest("B"))

    consumed = asyncio.Event()
    task_a = asyncio.create_task(collect(stream_a, consumed))
    task_b = asyncio.create_task(collect(stream_b))
    try:
        await asyncio.wait_for(consumed.wait(), timeout=0.2)
        await asyncio.wait_for(stream_b.first_chunk.wait(), timeout=0.2)

        stream_a.cancel()
        assert await asyncio.wait_for(task_a, timeout=0.2) == [b"a1"]
        assert await asyncio.wait_for(task_b, timeout=0.2) == [b"b1", b"b2"]

        stream_c = tts.create_stream(TTSRequest("C"))
        assert await asyncio.wait_for(collect(stream_c), timeout=0.2) == [b"c1"]
        release_a.set()
        await stream_a.wait_closed(timeout=0.2)
        await stream_b.wait_closed(timeout=0.2)
        await stream_c.wait_closed(timeout=0.2)
        assert responses["A"].closed is True
        assert stream_a.cleanup_pending is False
        assert not tts._active_streams
    finally:
        await cleanup_stream_test(tts, [task_a, task_b], [release_a])


@pytest.mark.asyncio
async def test_blocked_request_stop_is_bounded_closes_response_and_ignores_late_data():
    release = asyncio.Event()
    response = FakeResponse([b"first", b"late"], release)
    tts = make_tts({"blocked": response, "after": FakeResponse([b"usable"])})
    stream = tts.create_stream(TTSRequest("blocked"))
    consumed = asyncio.Event()
    task = asyncio.create_task(collect(stream, consumed))
    try:
        await asyncio.wait_for(consumed.wait(), timeout=0.2)

        stream.cancel()
        assert await asyncio.wait_for(task, timeout=0.2) == [b"first"]
        await stream.wait_closed(timeout=0.2)
        assert response.closed is True

        release.set()
        await stream.aclose()
        await stream.aclose()
        assert await collect(tts.create_stream(TTSRequest("after"))) == [b"usable"]
    finally:
        await cleanup_stream_test(tts, [task], [release])


@pytest.mark.asyncio
async def test_failure_is_request_scoped_and_scoped_stream_never_uses_blocking_fallback(monkeypatch):
    tts = make_tts({
        "bad": FakeResponse([b"partial"], error=RuntimeError("broken stream")),
        "good": FakeResponse([b"complete"]),
    })
    bad = tts.create_stream(TTSRequest("bad"))
    good = tts.create_stream(TTSRequest("good"))

    with pytest.raises(RuntimeError, match="broken stream"):
        await collect(bad)
    assert await collect(good) == [b"complete"]

    import services.tts.elevenlabs_service as module
    monkeypatch.setattr(module, "HTTPX_AVAILABLE", False)
    fallback = tts.create_stream(TTSRequest("good"))
    with pytest.raises(RuntimeError, match="async streaming is unavailable"):
        await collect(fallback)


@pytest.mark.asyncio
async def test_delayed_transport_cleanup_remains_owned_and_observable_until_release():
    chunk_release = asyncio.Event()
    close_release = asyncio.Event()
    response = FakeResponse(
        [b"first"], gate=chunk_release, close_gate=close_release
    )
    tts = make_tts({"slow-close": response})
    stream = tts.create_stream(TTSRequest("slow-close"))
    consumed = asyncio.Event()
    consumer = asyncio.create_task(collect(stream, consumed))
    try:
        await asyncio.wait_for(consumed.wait(), timeout=0.2)

        stream.cancel()
        assert await asyncio.wait_for(consumer, timeout=0.2) == [b"first"]
        await asyncio.wait_for(response.close_started.wait(), timeout=0.2)
        await asyncio.wait_for(stream.aclose(), timeout=0.5)
        assert stream.cleanup_pending is True
        assert stream in tts._active_streams

        close_release.set()
        await stream.wait_closed(timeout=0.2)
        assert response.closed is True
        assert stream.cleanup_pending is False
        assert stream not in tts._active_streams
    finally:
        await cleanup_stream_test(tts, [consumer], [chunk_release, close_release])


@pytest.mark.asyncio
async def test_preclosed_stream_never_opens_http_and_cannot_be_restarted():
    tts = make_tts({"closed": FakeResponse([b"forbidden"])})
    stream = tts.create_stream(TTSRequest("closed"))

    await stream.aclose()
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    with pytest.raises(StopAsyncIteration):
        await anext(stream)

    assert tts._http_client.stream_calls == 0
    assert not stream.cleanup_pending
    assert stream not in tts._active_streams


@pytest.mark.asyncio
async def test_natural_eof_is_terminal_for_repeated_anext_calls():
    tts = make_tts({"once": FakeResponse([b"only"])})
    stream = tts.create_stream(TTSRequest("once"))

    assert await anext(stream) == b"only"
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    calls_after_eof = tts._http_client.stream_calls
    with pytest.raises(StopAsyncIteration):
        await anext(stream)

    assert calls_after_eof == 1
    assert tts._http_client.stream_calls == 1


@pytest.mark.asyncio
async def test_delayed_reader_cancellation_stays_owned_until_reader_finishes():
    reader_release = asyncio.Event()
    response = DelayedReaderCancellationResponse(reader_release)
    tts = make_tts({"reader": response})
    stream = tts.create_stream(TTSRequest("reader"))
    consumer = asyncio.create_task(collect(stream))
    await asyncio.wait_for(response.reader_started.wait(), timeout=0.2)

    stream.cancel()
    assert await asyncio.wait_for(consumer, timeout=0.2) == [b"first"]
    await asyncio.wait_for(response.reader_cancelled.wait(), timeout=0.2)
    assert response.closed is True
    with pytest.raises(asyncio.TimeoutError):
        await stream.wait_closed(timeout=0.01)
    assert stream.cleanup_pending is True
    assert stream in tts._active_streams

    reader_release.set()
    await asyncio.wait_for(response.reader_finished.wait(), timeout=0.2)
    await stream.wait_closed(timeout=0.2)
    assert stream.cleanup_pending is False
    assert not stream._child_tasks
    assert stream not in tts._active_streams


@pytest.mark.asyncio
async def test_cancellation_during_response_entry_closes_without_reading_audio():
    entry_release = asyncio.Event()
    exit_release = asyncio.Event()
    response = DelayedEntryResponse(entry_release, exit_release)
    tts = make_tts({"opening": response})
    stream = tts.create_stream(TTSRequest("opening"))
    consumer = asyncio.create_task(collect(stream))
    await asyncio.wait_for(response.entry_started.wait(), timeout=0.2)

    stream.cancel()
    assert await asyncio.wait_for(consumer, timeout=0.2) == []
    with pytest.raises(asyncio.TimeoutError):
        await stream.wait_closed(timeout=0.01)
    entry_release.set()
    await asyncio.wait_for(response.exit_started.wait(), timeout=0.2)
    assert response.iter_calls == 0
    assert stream.cleanup_pending is True

    exit_release.set()
    await stream.wait_closed(timeout=0.2)
    assert response.closed is True
    assert not stream._child_tasks


@pytest.mark.asyncio
async def test_repeated_consumer_cancellation_keeps_no_anonymous_wait_tasks():
    response_release = asyncio.Event()
    tts = make_tts({"consumer": FakeResponse([], gate=response_release)})
    stream = tts.create_stream(TTSRequest("consumer"))

    for _ in range(2):
        next_chunk = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        next_chunk.cancel()
        with pytest.raises(asyncio.CancelledError):
            await next_chunk

    stream.cancel()
    response_release.set()
    await stream.wait_closed(timeout=0.2)
    assert not stream._child_tasks
    assert stream.cleanup_pending is False
    assert stream not in tts._active_streams
