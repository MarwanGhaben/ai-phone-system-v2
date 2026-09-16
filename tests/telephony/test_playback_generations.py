import asyncio
import base64
import json

import pytest

from services.telephony.twilio_service import TwilioMediaStreamHandler


class RecordingWebSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)


class FailingMediaWebSocket(RecordingWebSocket):
    async def send_json(self, message):
        if message["event"] == "media":
            raise RuntimeError("websocket send failed")
        await super().send_json(message)


class GatedMarkWebSocket(RecordingWebSocket):
    def __init__(self):
        super().__init__()
        self.mark_started = asyncio.Event()
        self.mark_release = asyncio.Event()

    async def send_json(self, message):
        if message["event"] == "mark":
            self.mark_started.set()
            await self.mark_release.wait()
        await super().send_json(message)


class DelayedCancellationMarkWebSocket(GatedMarkWebSocket):
    def __init__(self):
        super().__init__()
        self.cancellation_seen = asyncio.Event()

    async def send_json(self, message):
        if message["event"] == "mark":
            self.mark_started.set()
            try:
                await self.mark_release.wait()
            except asyncio.CancelledError:
                self.cancellation_seen.set()
                await self.mark_release.wait()
        self.messages.append(message)


async def chunks(*values):
    for value in values:
        yield value


async def wait_for_message(websocket, event, *, after=0):
    async def find_message():
        while True:
            for message in reversed(websocket.messages[after:]):
                if message["event"] == event:
                    return message
            await asyncio.sleep(0)

    return await asyncio.wait_for(find_message(), timeout=0.2)


@pytest.fixture
def handler():
    result = TwilioMediaStreamHandler("call", "stream", RecordingWebSocket())
    result._is_streaming = True
    result._is_connected = True
    return result


@pytest.mark.asyncio
async def test_replaced_stream_cannot_send_late_or_residual_media(handler):
    old_release = asyncio.Event()

    async def old_chunks():
        yield b"A" * 1600
        await old_release.wait()
        yield b"X" * 160

    old = await handler.begin_playback()
    old_task = asyncio.create_task(handler.stream_audio_chunks(old, old_chunks()))
    while not any(m["event"] == "media" for m in handler.websocket.messages):
        await asyncio.sleep(0)

    new = await handler.begin_playback()
    replacement_clear = len(handler.websocket.messages) - 1
    new_result = await handler.stream_audio_chunks(new, chunks(b"B" * 1600, b"R" * 17))
    old_release.set()
    old_result = await asyncio.wait_for(old_task, timeout=0.2)

    after_clear = handler.websocket.messages[replacement_clear + 1:]
    decoded = [base64.b64decode(m["media"]["payload"]) for m in after_clear if m["event"] == "media"]
    assert decoded
    assert all(set(payload) <= {ord("B"), ord("R")} for payload in decoded)
    assert old_result.completed is False
    assert new_result.completed is True


@pytest.mark.asyncio
async def test_owner_is_rechecked_inside_send_lock_before_old_media_send(handler):
    old = await handler.begin_playback()
    await handler._send_lock.acquire()
    old_task = asyncio.create_task(handler.stream_audio_chunks(old, chunks(b"A" * 1600)))
    await asyncio.sleep(0)
    replacement_task = asyncio.create_task(handler.begin_playback())
    await asyncio.sleep(0)
    handler._send_lock.release()
    new = await asyncio.wait_for(replacement_task, timeout=0.2)
    old_result = await asyncio.wait_for(old_task, timeout=0.2)

    assert old_result.completed is False
    replacement_clear = max(i for i, m in enumerate(handler.websocket.messages) if m["event"] == "clear")
    assert not any(m["event"] == "media" for m in handler.websocket.messages[replacement_clear + 1:])
    assert handler.is_current_playback(new)


@pytest.mark.asyncio
async def test_marks_are_bound_to_exact_live_successful_generation(handler):
    first = await handler.begin_playback()
    completed = await handler.stream_audio_chunks(first, chunks(b"A" * 1600))
    mark_start = len(handler.websocket.messages)
    wait = asyncio.create_task(handler.wait_for_playback(first, completed, timeout=1))
    mark = await wait_for_message(handler.websocket, "mark", after=mark_start)
    await handler._process_message(json.dumps({"event": "mark", "mark": mark["mark"]}))
    await handler._process_message(json.dumps({"event": "mark", "mark": mark["mark"]}))
    assert await wait is True

    retired = await handler.begin_playback()
    retired_result = await handler.stream_audio_chunks(retired, chunks(b"B" * 1600))
    retired_mark_start = len(handler.websocket.messages)
    retired_wait = asyncio.create_task(handler.wait_for_playback(retired, retired_result, timeout=1))
    retired_mark = await wait_for_message(
        handler.websocket, "mark", after=retired_mark_start
    )
    await handler.clear_audio(retired)
    await handler._process_message(json.dumps({"event": "mark", "mark": retired_mark["mark"]}))
    assert await retired_wait is False

    missing = await handler.begin_playback()
    missing_result = await handler.stream_audio_chunks(missing, chunks(b"C" * 1600))
    assert await handler.wait_for_playback(missing, missing_result, timeout=0.01) is False
    await handler.cleanup()
    await handler._process_message(json.dumps({"event": "mark", "mark": retired_mark["mark"]}))
    assert not handler._pending_marks


@pytest.mark.asyncio
async def test_partial_transport_failure_cannot_be_playback_success(handler):
    async def failing_chunks():
        yield b"P" * 1600
        raise RuntimeError("transport failed")

    owner = await handler.begin_playback()
    result = await handler.stream_audio_chunks(owner, failing_chunks())
    assert result.bytes_sent > 0
    assert result.completed is False
    assert await handler.wait_for_playback(owner, result, timeout=0.01) is False
    assert not any(message["event"] == "mark" for message in handler.websocket.messages)


@pytest.mark.asyncio
async def test_websocket_failure_has_no_success_mark_or_live_owner_after_cleanup():
    handler = TwilioMediaStreamHandler("call", "stream", FailingMediaWebSocket())
    handler._is_streaming = True
    handler._is_connected = True
    owner = await handler.begin_playback()

    result = await handler.stream_audio_chunks(owner, chunks(b"W" * 1600))
    assert result.completed is False
    assert result.error == "transport_error"
    assert await handler.wait_for_playback(owner, result, timeout=0.01) is False
    await handler.cleanup()
    await handler.cleanup()
    assert owner.retired is True
    assert handler._active_playback is None
    assert not handler._pending_marks


@pytest.mark.asyncio
async def test_mark_deadline_includes_waiting_for_send_lock():
    handler = TwilioMediaStreamHandler("call", "stream", RecordingWebSocket())
    handler._is_streaming = True
    owner = await handler.begin_playback()
    result = await handler.stream_audio_chunks(owner, chunks(b"L" * 1600))
    await handler._send_lock.acquire()
    wait = asyncio.create_task(handler.wait_for_playback(owner, result, timeout=0.01))
    while not handler._pending_marks:
        await asyncio.sleep(0)

    done, _ = await asyncio.wait({wait}, timeout=0.1)
    bounded = wait in done
    handler._send_lock.release()
    outcome = await asyncio.wait_for(wait, timeout=0.2)

    assert bounded is True
    assert outcome is False
    assert not handler._pending_marks
    assert not any(message["event"] == "mark" for message in handler.websocket.messages)


@pytest.mark.asyncio
async def test_mark_deadline_includes_stalled_websocket_submission():
    websocket = GatedMarkWebSocket()
    handler = TwilioMediaStreamHandler("call", "stream", websocket)
    handler._is_streaming = True
    owner = await handler.begin_playback()
    result = await handler.stream_audio_chunks(owner, chunks(b"S" * 1600))
    wait = asyncio.create_task(handler.wait_for_playback(owner, result, timeout=0.01))
    await asyncio.wait_for(websocket.mark_started.wait(), timeout=0.2)

    done, _ = await asyncio.wait({wait}, timeout=0.1)
    bounded = wait in done
    websocket.mark_release.set()
    outcome = await asyncio.wait_for(wait, timeout=0.2)

    assert bounded is True
    assert outcome is False
    assert not handler._pending_marks


@pytest.mark.asyncio
async def test_delayed_mark_send_cancellation_remains_owned_and_cannot_revive():
    websocket = DelayedCancellationMarkWebSocket()
    handler = TwilioMediaStreamHandler("call", "stream", websocket)
    handler._is_streaming = True
    owner = await handler.begin_playback()
    result = await handler.stream_audio_chunks(owner, chunks(b"D" * 1600))
    wait = asyncio.create_task(handler.wait_for_playback(owner, result, timeout=0.01))
    await asyncio.wait_for(websocket.mark_started.wait(), timeout=0.2)

    done, _ = await asyncio.wait({wait}, timeout=0.1)
    bounded = wait in done
    if bounded:
        await asyncio.wait_for(websocket.cancellation_seen.wait(), timeout=0.2)
        assert handler.cleanup_pending is True

    other_handler = TwilioMediaStreamHandler(
        "other-call", "other-stream", RecordingWebSocket()
    )
    other_handler._is_streaming = True
    other_owner = await other_handler.begin_playback()
    other_result = await other_handler.stream_audio_chunks(
        other_owner, chunks(b"O" * 1600)
    )
    assert other_result.completed is True
    assert other_handler.is_current_playback(other_owner)

    replacement_task = asyncio.create_task(handler.begin_playback())
    websocket.mark_release.set()
    replacement = await asyncio.wait_for(replacement_task, timeout=0.2)
    outcome = await asyncio.wait_for(wait, timeout=0.2)
    await handler.wait_for_cleanup(timeout=0.2)

    late_mark = next(
        message for message in websocket.messages if message["event"] == "mark"
    )
    handler._on_mark(late_mark)
    assert bounded is True
    assert outcome is False
    assert not handler._pending_marks
    assert handler.cleanup_pending is False
    assert handler.is_current_playback(replacement)
