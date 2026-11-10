"""The speech request must match what each ElevenLabs model accepts.

Owner server check 2026-11-10: eleven_v4_turbo answers HTTP 400 unsupported_model
when the deprecated optimize_streaming_latency option is sent, and HTTP 200
without it. eleven_multilingual_v2 accepts the option and must keep it.
"""
import pytest

from services.tts.elevenlabs_service import ElevenLabsTTS
from services.tts.tts_base import TTSRequest


class Response:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def raise_for_status(self):
        return None

    async def aiter_bytes(self, _size):
        yield b"audio"


class Client:
    is_closed = False

    def __init__(self):
        self.calls = []

    def stream(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return Response()


def make_tts(model, voice="voice123"):
    tts = ElevenLabsTTS.__new__(ElevenLabsTTS)
    tts.api_key = "test-key"
    tts.default_voice_id = voice
    tts.model = model
    tts.stability = 0.5
    tts.similarity_boost = 0.75
    tts.output_format = "ulaw_8000"
    tts._http_client = Client()
    tts._active_streams = set()
    tts._legacy_streams = set()
    return tts


async def request_for(model):
    tts = make_tts(model)
    stream = tts.create_stream(TTSRequest(text="hello"))
    chunks = [chunk async for chunk in stream]
    await stream.wait_closed(timeout=1)
    assert chunks == [b"audio"]
    (method, url, kwargs), = tts._http_client.calls
    return method, url, kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["eleven_v4_turbo", "eleven_v4"])
async def test_v4_models_do_not_receive_the_refused_latency_option(model):
    method, url, kwargs = await request_for(model)
    assert method == "POST"
    assert url == "https://api.elevenlabs.io/v1/text-to-speech/voice123/stream"
    assert kwargs["params"] == {"output_format": "ulaw_8000"}
    assert kwargs["json"]["model_id"] == model


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["eleven_multilingual_v2", "eleven_flash_v2_5"])
async def test_earlier_models_keep_the_existing_request_unchanged(model):
    _, _, kwargs = await request_for(model)
    assert kwargs["params"] == {
        "output_format": "ulaw_8000",
        "optimize_streaming_latency": "2",
    }
    assert kwargs["json"] == {
        "text": "hello",
        "model_id": model,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
