"""Installed-image voice probe.

Offline (no argument): the real speech service builds the request each model
accepts; no network. Live (--live): one short spoken sentence through the real
service with the container's own settings. Prints status only: no key, no audio.
"""
import asyncio
import json
import logging
import sys

from loguru import logger

logger.remove()
logging.disable(logging.CRITICAL)

MODEL = "eleven_v4_turbo"
# "Hello, your appointment is tomorrow at two thirty." Escaped so no console or
# transport can alter the Arabic characters.
TEXT = ("مرحباً، موعدك "
        "غداً الساعة "
        "الثانية والنصف.")


class _Response:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def raise_for_status(self):
        return None

    async def aiter_bytes(self, _size):
        yield b"audio"


class _Client:
    is_closed = False

    def __init__(self):
        self.calls = []

    def stream(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return _Response()


async def offline():
    from services.tts.elevenlabs_service import create_elevenlabs_tts
    from services.tts.tts_base import TTSRequest
    expected = {
        MODEL: {"output_format": "ulaw_8000"},
        "eleven_multilingual_v2": {"output_format": "ulaw_8000",
                                   "optimize_streaming_latency": "2"},
    }
    for model, params in expected.items():
        tts = create_elevenlabs_tts({"elevenlabs_api_key": "synthetic",
                                     "elevenlabs_voice_id": "syntheticvoice01",
                                     "elevenlabs_model": model})
        if tts is None or tts.model != model:
            raise RuntimeError("speech service was not created")
        client = _Client()
        tts._http_client = client
        stream = tts.create_stream(TTSRequest(text="hello"))
        chunks = [chunk async for chunk in stream]
        await stream.wait_closed(timeout=2)
        (method, url, kwargs), = client.calls
        if (chunks != [b"audio"] or method != "POST"
                or url != "https://api.elevenlabs.io/v1/text-to-speech/syntheticvoice01/stream"
                or kwargs["params"] != params or kwargs["json"]["model_id"] != model):
            raise RuntimeError("speech request differs for " + model)
    return {"status": "VOICE_MODEL_OK", "mode": "offline"}


async def live():
    from config.settings import settings
    from services.tts.elevenlabs_service import create_elevenlabs_tts
    from services.tts.tts_base import TTSRequest
    if settings.elevenlabs_model != MODEL:
        return {"status": "VOICE_MODEL_FAILED", "mode": "live", "reason": "model_setting"}
    tts = create_elevenlabs_tts(settings.model_dump())
    if tts is None or tts.model != MODEL or tts.default_voice_id != settings.elevenlabs_voice_id:
        return {"status": "VOICE_MODEL_FAILED", "mode": "live", "reason": "service"}
    total = 0
    stream = tts.create_stream(TTSRequest(text=TEXT))
    try:
        async for chunk in stream:
            total += len(chunk)
    except Exception as error:
        response = getattr(error, "response", None)
        code = getattr(response, "status_code", None)
        return {"status": "VOICE_MODEL_FAILED", "mode": "live", "reason": "request",
                "http_status": code if type(code) is int else None}
    finally:
        await stream.aclose()
        client = tts._http_client
        if client is not None and not client.is_closed:
            await client.aclose()
    if total < 8000:  # less than one second of 8 kHz phone audio
        return {"status": "VOICE_MODEL_FAILED", "mode": "live", "reason": "short_audio"}
    return {"status": "VOICE_MODEL_OK", "mode": "live", "audio_seconds": round(total / 8000, 1)}


def main():
    mode = live if sys.argv[1:] == ["--live"] else offline
    try:
        result = asyncio.run(asyncio.wait_for(mode(), 60))
    except BaseException as error:
        result = {"status": "VOICE_MODEL_FAILED", "reason": type(error).__name__}
    print(json.dumps(result))
    raise SystemExit(0 if result.get("status") == "VOICE_MODEL_OK" else 1)


if __name__ == "__main__":
    main()
