import asyncio
from types import SimpleNamespace

import pytest

import services.stt.elevenlabs_stt_service as stt_module
from services.stt.elevenlabs_stt_service import ElevenLabsSTT
from services.stt.stt_base import AudioChunk, STTStatus


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent_messages: list[str] = []

    async def send(self, message: str) -> None:
        self.sent_messages.append(message)

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_audio_waits_for_replacement_socket_during_reset(monkeypatch) -> None:
    old_socket = FakeWebSocket()
    new_socket = FakeWebSocket()
    reconnect_started = asyncio.Event()
    allow_reconnect = asyncio.Event()

    async def delayed_connect(*args, **kwargs):
        reconnect_started.set()
        await allow_reconnect.wait()
        return new_socket

    monkeypatch.setattr(
        stt_module,
        "websockets",
        SimpleNamespace(connect=delayed_connect),
    )

    stt = ElevenLabsSTT(api_key="test-key")
    stt._status = STTStatus.CONNECTED
    stt._websocket = old_socket
    stt._receive_task = None
    stt._transcript_queue = asyncio.Queue()
    stt._is_listening = True

    reset_task = asyncio.create_task(stt.reset_for_listening())
    await reconnect_started.wait()
    audio_task = asyncio.create_task(stt.stream_audio(AudioChunk(data=b"caller audio")))
    await asyncio.sleep(0)
    audio_waited_for_reset = not audio_task.done()

    stt._is_listening = False
    allow_reconnect.set()
    await reset_task
    await audio_task

    assert audio_waited_for_reset
    assert len(new_socket.sent_messages) == 1
