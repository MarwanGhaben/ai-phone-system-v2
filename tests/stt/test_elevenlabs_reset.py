import asyncio

import pytest

import services.stt.elevenlabs_stt_service as stt_module
from services.stt.elevenlabs_stt_service import ElevenLabsSTT
from services.stt.stt_base import AudioChunk


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent_messages: list[str] = []

    async def send(self, message: str) -> None:
        self.sent_messages.append(message)

    async def close(self) -> None:
        return None

    async def recv(self) -> str:
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_audio_waits_for_replacement_socket_during_reset(monkeypatch) -> None:
    old_socket = FakeWebSocket()
    new_socket = FakeWebSocket()
    reconnect_started = asyncio.Event()
    allow_reconnect = asyncio.Event()

    connections = 0

    async def delayed_connect(*args, **kwargs):
        nonlocal connections
        connections += 1
        if connections == 1:
            return old_socket
        reconnect_started.set()
        await allow_reconnect.wait()
        return new_socket

    monkeypatch.setattr(
        stt_module.websockets,
        "connect",
        delayed_connect,
    )

    stt = ElevenLabsSTT(api_key="test-key")
    assert await stt.connect() is True

    reset_task = asyncio.create_task(stt.reset_for_listening())
    await asyncio.wait_for(reconnect_started.wait(), timeout=1)
    audio_task = asyncio.create_task(stt.stream_audio(AudioChunk(data=b"caller audio")))
    await asyncio.sleep(0)
    audio_waited_for_reset = not audio_task.done()

    allow_reconnect.set()
    try:
        await asyncio.wait_for(reset_task, timeout=1)
        await asyncio.wait_for(audio_task, timeout=1)
        assert audio_waited_for_reset
        assert len(new_socket.sent_messages) == 1
    finally:
        await stt.disconnect()
