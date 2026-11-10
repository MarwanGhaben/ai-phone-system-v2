"""Owner-reversible language lock for the realtime speech-to-text connection."""
import asyncio
import os
import types
from urllib.parse import parse_qs, urlparse

import pytest

from services.stt.stt_base import STTStatus

import services.stt.elevenlabs_stt_service as stt_module
from services.stt.elevenlabs_stt_service import (
    LANGUAGE_LOCK_ENV,
    ElevenLabsSTT,
    language_lock_enabled,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(LANGUAGE_LOCK_ENV, raising=False)


def query(url):
    return parse_qs(urlparse(url).query)


def test_env_name_is_fixed():
    assert LANGUAGE_LOCK_ENV == "ELEVENLABS_STT_LANGUAGE_LOCK"


@pytest.mark.parametrize("raw", ["off", "false", "0", "no", "unlocked",
                                 "OFF", "False", "UNLOCKED", " off ", "\tNo\n", " Unlocked "])
def test_unlocking_values(monkeypatch, raw):
    monkeypatch.setenv(LANGUAGE_LOCK_ENV, raw)
    assert language_lock_enabled() is False


@pytest.mark.parametrize("raw", ["on", "true", "1", "yes", "locked", "", "   ", "junk",
                                 "of", "offf", "off off", "disabled", "none", "-1", "nan"])
def test_everything_else_stays_locked(monkeypatch, raw):
    monkeypatch.setenv(LANGUAGE_LOCK_ENV, raw)
    assert language_lock_enabled() is True


def test_unset_is_locked():
    assert LANGUAGE_LOCK_ENV not in os.environ
    assert language_lock_enabled() is True


def test_value_is_read_on_every_call(monkeypatch):
    monkeypatch.setenv(LANGUAGE_LOCK_ENV, "off")
    assert language_lock_enabled() is False
    monkeypatch.setenv(LANGUAGE_LOCK_ENV, "on")
    assert language_lock_enabled() is True
    monkeypatch.delenv(LANGUAGE_LOCK_ENV)
    assert language_lock_enabled() is True


def test_broken_environment_never_raises(monkeypatch):
    class Broken:
        def get(self, *_args):
            raise OSError("synthetic")

    class NotText:
        def get(self, *_args):
            return b"off"

    real_os = stt_module.os
    for fake in (Broken(), NotText()):
        monkeypatch.setattr(stt_module, "os", types.SimpleNamespace(environ=fake))
        assert language_lock_enabled() is True
    monkeypatch.setattr(stt_module, "os", real_os)


@pytest.mark.parametrize("language", ["en", "ar"])
def test_locked_url_includes_language(language):
    service = ElevenLabsSTT(api_key="synthetic-key", language=language)
    assert query(service._connection_url())["language_code"] == [language]


@pytest.mark.parametrize("language", ["en", "ar"])
def test_unlocked_url_omits_language_but_keeps_the_rest(monkeypatch, language):
    locked = ElevenLabsSTT(api_key="synthetic-key", language=language)
    locked_query = query(locked._connection_url())
    monkeypatch.setenv(LANGUAGE_LOCK_ENV, "off")
    service = ElevenLabsSTT(api_key="synthetic-key", language=language)
    unlocked_query = query(service._connection_url())
    assert "language_code" not in unlocked_query
    assert service.language == language
    del locked_query["language_code"]
    assert unlocked_query == locked_query


@pytest.mark.parametrize("raw", [None, "off", "on"])
def test_empty_language_never_has_language_code(monkeypatch, raw):
    if raw is not None:
        monkeypatch.setenv(LANGUAGE_LOCK_ENV, raw)
    service = ElevenLabsSTT(api_key="synthetic-key", language="")
    assert "language_code" not in query(service._connection_url())


def test_url_follows_the_environment_each_time(monkeypatch):
    service = ElevenLabsSTT(api_key="synthetic-key", language="en")
    assert "language_code=en" in service._connection_url()
    monkeypatch.setenv(LANGUAGE_LOCK_ENV, "unlocked")
    assert "language_code" not in service._connection_url()
    monkeypatch.setenv(LANGUAGE_LOCK_ENV, "on")
    assert "language_code=en" in service._connection_url()


def service_with_spy(monkeypatch, *, active=True, language="en", healthy=True):
    service = ElevenLabsSTT(api_key="synthetic-key", language=language)
    service._session_active = active
    if healthy:
        service._websocket = object()
        service._status = STTStatus.CONNECTED
    calls = []

    async def replace(revision):
        calls.append(revision)
        return True

    monkeypatch.setattr(service, "_replace_transport_locked", replace)
    return service, calls


@pytest.mark.asyncio
async def test_unlocked_reconnect_records_language_without_touching_transport(monkeypatch):
    monkeypatch.setenv(LANGUAGE_LOCK_ENV, "off")
    service, calls = service_with_spy(monkeypatch)
    queue_before = service._transcript_queue
    assert await service.reconnect_with_language("ar") is True
    assert calls == []
    assert service.language == "ar"
    assert service._transcript_queue is queue_before
    assert service._session_active is True
    assert await service.reconnect_with_language("en") is True
    assert service.language == "en"
    assert calls == []


@pytest.mark.asyncio
async def test_locked_reconnect_replaces_transport(monkeypatch):
    service, calls = service_with_spy(monkeypatch)
    assert await service.reconnect_with_language("ar") is True
    assert calls == [service._terminal_revision]
    assert service.language == "ar"


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [None, "off"])
async def test_inactive_session_refuses_in_both_modes(monkeypatch, raw):
    if raw is not None:
        monkeypatch.setenv(LANGUAGE_LOCK_ENV, raw)
    service, calls = service_with_spy(monkeypatch, active=False)
    assert await service.reconnect_with_language("ar") is False
    assert calls == []
    assert service.language == "en"


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [None, "off"])
async def test_terminal_revision_change_while_waiting_for_lock_refuses(monkeypatch, raw):
    if raw is not None:
        monkeypatch.setenv(LANGUAGE_LOCK_ENV, raw)
    service, calls = service_with_spy(monkeypatch)
    await service._connection_lock.acquire()
    task = asyncio.create_task(service.reconnect_with_language("ar"))
    await asyncio.sleep(0)
    service._terminal_revision += 1
    service._session_active = False
    service._connection_lock.release()
    assert await task is False
    assert calls == []
    assert service.language == "en"


class FakeWebSocket:
    def __init__(self):
        self.closed = False
        self._events = asyncio.Queue()

    async def recv(self):
        return await self._events.get()

    async def send(self, _message):
        return None

    async def close(self):
        self.closed = True

    async def ping(self):
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,connections,language_in_urls", [
    ("off", 1, [False]),
    (None, 2, [True, True]),
])
async def test_real_session_connection_count_and_urls(monkeypatch, raw, connections, language_in_urls):
    if raw is not None:
        monkeypatch.setenv(LANGUAGE_LOCK_ENV, raw)
    urls = []

    async def connect(url, **_kwargs):
        urls.append(url)
        return FakeWebSocket()

    monkeypatch.setattr(stt_module.websockets, "connect", connect)
    service = ElevenLabsSTT(api_key="synthetic-key", language="en")
    try:
        assert await service.connect() is True
        assert await service.reconnect_with_language("ar") is True
        assert len(urls) == connections
        assert ["language_code" in query(url) for url in urls] == language_in_urls
        assert service.language == "ar"
        if raw is None:
            assert query(urls[-1])["language_code"] == ["ar"]
    finally:
        await service.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("broken", ["no_socket", "error_status"])
async def test_unlocked_reconnect_still_replaces_a_dead_transport(monkeypatch, broken):
    monkeypatch.setenv(LANGUAGE_LOCK_ENV, "off")
    service, calls = service_with_spy(monkeypatch)
    if broken == "no_socket":
        service._websocket = None
    else:
        service._status = STTStatus.ERROR
    assert await service.reconnect_with_language("ar") is True
    assert calls == [service._terminal_revision]
    assert service.language == "ar"
