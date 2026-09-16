import asyncio
import json

import pytest

from services.telephony.twilio_service import TwilioMediaStreamHandler


class DetectingWebSocket:
    def __init__(self) -> None:
        self.active_sends = 0
        self.overlap_detected = False
        self.first_send_started = asyncio.Event()
        self.release_first_send = asyncio.Event()
        self.messages: list[dict] = []

    async def send_json(self, message: dict) -> None:
        self.active_sends += 1
        self.overlap_detected |= self.active_sends > 1
        self.messages.append(message)
        try:
            if len(self.messages) == 1:
                self.first_send_started.set()
                await self.release_first_send.wait()
        finally:
            self.active_sends -= 1


class RecordingWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.message_sent = asyncio.Event()

    async def send_json(self, message: dict) -> None:
        self.messages.append(message)
        self.message_sent.set()


@pytest.mark.asyncio
async def test_control_messages_are_serialized() -> None:
    websocket = DetectingWebSocket()
    handler = TwilioMediaStreamHandler("call-1", "stream-1", websocket)

    first_send = asyncio.create_task(handler.send_event("clear"))
    await websocket.first_send_started.wait()
    second_send = asyncio.create_task(handler.send_event("mark", mark={"name": "m1"}))
    await asyncio.sleep(0)

    try:
        assert websocket.overlap_detected is False
    finally:
        websocket.release_first_send.set()
        await asyncio.gather(first_send, second_send)


@pytest.mark.asyncio
async def test_matching_mark_confirms_playback() -> None:
    websocket = RecordingWebSocket()
    handler = TwilioMediaStreamHandler("call-1", "stream-1", websocket)
    handler._is_streaming = True
    owner = await handler.begin_playback()
    stream_result = await handler.stream_audio_chunks(owner, _audio_chunks(b"x" * 1600))
    websocket.message_sent.clear()

    playback_wait = asyncio.create_task(handler.wait_for_playback(owner, stream_result, timeout=1))
    await websocket.message_sent.wait()
    mark_name = next(
        message["mark"]["name"]
        for message in reversed(websocket.messages)
        if message["event"] == "mark"
    )
    await handler._process_message(
        json.dumps({"event": "mark", "mark": {"name": mark_name}})
    )

    assert await playback_wait is True


@pytest.mark.asyncio
async def test_missing_mark_has_bounded_timeout() -> None:
    handler = TwilioMediaStreamHandler("call-1", "stream-1", RecordingWebSocket())
    handler._is_streaming = True
    owner = await handler.begin_playback()
    stream_result = await handler.stream_audio_chunks(owner, _audio_chunks(b"x" * 1600))

    assert await handler.wait_for_playback(owner, stream_result, timeout=0.01) is False


async def _audio_chunks(audio: bytes):
    yield audio
