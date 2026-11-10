"""Installed-image probe for the caller-pause and reply-length repair.

Offline (no argument, run with --network none): checks the installed speech
recognition connection parameters and the installed reply length caps. Live
(--live): opens one real speech recognition session with the container's own
settings and the new pause threshold, sends no audio, then disconnects.
Prints one JSON status line only: no key, token, audio or transcript.
"""
import asyncio
import json
import logging
import os
from pathlib import Path
import sys

from loguru import logger

logger.remove()
logging.disable(logging.CRITICAL)

ENV_NAME = "ELEVENLABS_STT_VAD_SILENCE_SECS"
SYNTHETIC_ENV = {
    "SECRET_KEY": "synthetic", "DATABASE_URL": "postgresql://synthetic:synthetic@localhost/test",
    "TWILIO_ACCOUNT_SID": "ACsynthetic", "TWILIO_AUTH_TOKEN": "synthetic",
    "TWILIO_PHONE_NUMBER": "+14165550100", "DEEPGRAM_API_KEY": "synthetic",
    "ELEVENLABS_API_KEY": "synthetic", "OPENAI_API_KEY": "synthetic",
}
CAPS = {"max_tokens=220,": 3, "max_tokens=160,": 2, "max_tokens=120,": 1}
OLD_CAPS = ("max_tokens=80,", "max_tokens=60,", "max_tokens=50,")
FUNCTION_START = "    async def _get_ai_response("
FUNCTION_END = "    async def _speak_to_caller("
CONNECT_WAIT_SECONDS = 10.0
TOTAL_SECONDS = 30
POLL_SECONDS = 0.1


class Failure(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _threshold_for(value):
    """The connection parameter the installed service builds for one env value."""
    from services.stt.elevenlabs_stt_service import create_elevenlabs_stt
    if value is None:
        os.environ.pop(ENV_NAME, None)
    else:
        os.environ[ENV_NAME] = value
    service = create_elevenlabs_stt({"elevenlabs_api_key": "synthetic"})
    for part in service._connection_url().split("?", 1)[1].split("&"):
        key, _, found = part.partition("=")
        if key == "vad_silence_threshold_secs":
            return found
    raise Failure("threshold_missing")


def check_threshold():
    saved = os.environ.get(ENV_NAME)
    try:
        cases = [(None, "0.7"), ("", "0.7"), ("0.3", "0.3"), ("0.45", "0.45"),
                 ("1", "1.0"), ("1.5", "1.5"), ("0.29", "0.7"), ("1.51", "0.7"),
                 ("abc", "0.7"), ("nan", "0.7"), ("inf", "0.7"), ("-1", "0.7")]
        for value, expected in cases:
            if _threshold_for(value) != expected:
                raise Failure("threshold_value")
    finally:
        if saved is None:
            os.environ.pop(ENV_NAME, None)
        else:
            os.environ[ENV_NAME] = saved


def check_reply_caps():
    from services.conversation import orchestrator
    text = Path(orchestrator.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    if text.count(FUNCTION_START) != 1 or text.count(FUNCTION_END) != 1:
        raise Failure("reply_generator")
    start = text.index(FUNCTION_START)
    end = text.index(FUNCTION_END)
    if end <= start:
        raise Failure("reply_generator")
    body = text[start:end]
    if any(old in body for old in OLD_CAPS):
        raise Failure("old_reply_cap")
    if any(body.count(cap) != count for cap, count in CAPS.items()):
        raise Failure("reply_cap")


async def offline():
    os.environ.update(SYNTHETIC_ENV)
    os.environ.pop(ENV_NAME, None)
    check_threshold()
    check_reply_caps()
    return {"status": "REPLY_PAUSE_OK", "mode": "offline"}


async def live():
    """Open the same per-call speech recognition service the orchestrator builds."""
    from config.settings import settings
    from services.stt import create_stt_service
    from services.stt.elevenlabs_stt_service import STTTransportOutcome
    os.environ.pop(ENV_NAME, None)
    if settings.stt_provider != "elevenlabs":
        raise Failure("provider_setting")
    if _threshold_for(None) != "0.7":
        raise Failure("threshold_value")
    config = settings.model_dump()
    config["elevenlabs_stt_language"] = ""  # automatic language, as for a new caller
    stt = create_stt_service(settings.stt_provider, config)
    try:
        if await stt.connect() is not True:
            raise Failure("connect")
        waited = 0.0
        while waited < CONNECT_WAIT_SECONDS:
            outcome = stt.transport_outcome
            if outcome is STTTransportOutcome.SESSION_STARTED:
                return {"status": "REPLY_PAUSE_OK", "mode": "live"}
            if outcome not in (STTTransportOutcome.TRANSPORT_OPEN,):
                category = stt.provider_error_category
                raise Failure(category if outcome is STTTransportOutcome.PROVIDER_ERROR
                              and isinstance(category, str) else "rejected")
            await asyncio.sleep(POLL_SECONDS)
            waited += POLL_SECONDS
        raise Failure("no_session_started")
    finally:
        try:
            await asyncio.wait_for(stt.disconnect(), 5)
        except BaseException:
            pass


def main():
    live_mode = sys.argv[1:] == ["--live"]
    mode = "live" if live_mode else "offline"
    try:
        result = asyncio.run(asyncio.wait_for((live if live_mode else offline)(), TOTAL_SECONDS))
    except Failure as error:
        result = {"status": "REPLY_PAUSE_FAILED", "mode": mode, "reason": error.reason}
    except BaseException as error:
        result = {"status": "REPLY_PAUSE_FAILED", "mode": mode, "reason": type(error).__name__}
    print(json.dumps(result))
    raise SystemExit(0 if result.get("status") == "REPLY_PAUSE_OK" else 1)


if __name__ == "__main__":
    main()
