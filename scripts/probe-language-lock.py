"""Installed-image probe for the owner-reversible speech-recognition language lock.

Offline (no argument, run with --network none): checks, with synthetic values only,
how the installed speech recognition service builds its connection URL and answers
a language change while locked and unlocked. Live (--live): reads the lock mode from
the container's own environment, builds the per-call service for a known English
caller, checks the URL it would send, opens one real session, sends no audio, then
disconnects. Prints one JSON status line only: no key, URL, audio or transcript.
"""
import asyncio
import json
import logging
import os
import sys
from urllib.parse import parse_qs, urlparse

from loguru import logger

logger.remove()
logging.disable(logging.CRITICAL)

ENV_NAME = "ELEVENLABS_STT_LANGUAGE_LOCK"
SYNTHETIC_ENV = {
    "SECRET_KEY": "synthetic", "DATABASE_URL": "postgresql://synthetic:synthetic@localhost/test",
    "TWILIO_ACCOUNT_SID": "ACsynthetic", "TWILIO_AUTH_TOKEN": "synthetic",
    "TWILIO_PHONE_NUMBER": "+14165550100", "DEEPGRAM_API_KEY": "synthetic",
    "ELEVENLABS_API_KEY": "synthetic", "OPENAI_API_KEY": "synthetic",
}
# (environment value or None for unset, expected to be locked)
ENV_CASES = [(None, True), ("", True), ("on", True), ("junk", True), ("locked", True),
             ("off", False), ("OFF", False), (" false ", False), ("0", False),
             ("no", False), ("unlocked", False)]
LANGUAGES = ("en", "ar", "")
CONNECT_WAIT_SECONDS = 10.0
TOTAL_SECONDS = 30
POLL_SECONDS = 0.1


class Failure(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _set_env(value):
    if value is None:
        os.environ.pop(ENV_NAME, None)
    else:
        os.environ[ENV_NAME] = value


def _language_in(url):
    found = parse_qs(urlparse(url).query).get("language_code")
    return None if found is None else found[0]


def check_urls():
    from services.stt.elevenlabs_stt_service import ElevenLabsSTT, language_lock_enabled
    for value, locked in ENV_CASES:
        _set_env(value)
        if language_lock_enabled() is not locked:
            raise Failure("lock_parse")
        for language in LANGUAGES:
            service = ElevenLabsSTT(api_key="synthetic", language=language)
            expected = language if (locked and language) else None
            if _language_in(service._connection_url()) != expected:
                raise Failure("url_language")
            if service.language != language:
                raise Failure("language_state")


async def check_reconnect():
    from services.stt.elevenlabs_stt_service import ElevenLabsSTT
    for value, locked in (("on", True), (None, True), ("off", False)):
        _set_env(value)
        service = ElevenLabsSTT(api_key="synthetic", language="en")
        replaced = []

        async def replace(revision, _calls=replaced):
            _calls.append(revision)
            return True

        service._replace_transport_locked = replace
        service._session_active = True
        # A healthy transport: only then may the unlocked mode leave it untouched.
        from services.stt.stt_base import STTStatus
        service._websocket = object()
        service._status = STTStatus.CONNECTED
        if await service.reconnect_with_language("ar") is not True:
            raise Failure("reconnect_result")
        if service.language != "ar":
            raise Failure("language_state")
        if bool(replaced) is not locked:
            raise Failure("transport_replacement")
        idle = ElevenLabsSTT(api_key="synthetic", language="en")
        idle._replace_transport_locked = replace
        if await idle.reconnect_with_language("ar") is not False or idle.language != "en":
            raise Failure("inactive_session")


async def offline():
    os.environ.update(SYNTHETIC_ENV)
    saved = os.environ.get(ENV_NAME)
    try:
        check_urls()
        await check_reconnect()
    finally:
        _set_env(saved)
    return {"status": "LANGUAGE_LOCK_OK", "mode": "offline"}


async def live():
    """Open the per-call service as for a known English caller, with the real environment."""
    from config.settings import settings
    from services.stt import create_stt_service
    from services.stt.elevenlabs_stt_service import STTTransportOutcome, language_lock_enabled
    if settings.stt_provider != "elevenlabs":
        raise Failure("provider_setting")
    lock = "locked" if language_lock_enabled() else "unlocked"
    config = settings.model_dump()
    config["elevenlabs_stt_language"] = "en"  # known English caller
    stt = create_stt_service(settings.stt_provider, config)
    try:
        if getattr(stt, "language", None) != "en":
            raise Failure("language_state")
        sent = _language_in(stt._connection_url())
        if sent != ("en" if lock == "locked" else None):
            raise Failure("url_language")
        if await stt.connect() is not True:
            raise Failure("connect")
        waited = 0.0
        while waited < CONNECT_WAIT_SECONDS:
            outcome = stt.transport_outcome
            if outcome is STTTransportOutcome.SESSION_STARTED:
                return {"status": "LANGUAGE_LOCK_OK", "mode": "live", "lock": lock}
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
        result = {"status": "LANGUAGE_LOCK_FAILED", "mode": mode, "reason": error.reason}
    except BaseException as error:
        result = {"status": "LANGUAGE_LOCK_FAILED", "mode": mode, "reason": type(error).__name__}
    print(json.dumps(result))
    raise SystemExit(0 if result.get("status") == "LANGUAGE_LOCK_OK" else 1)


if __name__ == "__main__":
    main()
