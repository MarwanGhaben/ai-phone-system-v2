"""Contract tests for the opt-in ElevenLabs background-audio filter."""
from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

import config.settings as settings_module
import services.stt.elevenlabs_stt_service as stt_module
from services.stt.elevenlabs_stt_service import (
    ElevenLabsSTT,
    create_elevenlabs_stt,
)
from services.stt.stt_base import STTStatus
from services.conversation.events import MetadataDisposition


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = {
    "elevenlabs_api_key": "synthetic-key",
    "elevenlabs_stt_language": "",
    "elevenlabs_stt_model": "scribe_v2_realtime",
}
BASE_QUERY = {
    "model_id": ["scribe_v2_realtime"],
    "audio_format": ["ulaw_8000"],
    "sample_rate": ["8000"],
    "commit_strategy": ["vad"],
    "vad_silence_threshold_secs": ["0.3"],
    "include_language_detection": ["true"],
}
BASE_URL = (
    "wss://api.elevenlabs.io/v1/speech-to-text/realtime?"
    "model_id=scribe_v2_realtime&audio_format=ulaw_8000&sample_rate=8000&"
    "commit_strategy=vad&vad_silence_threshold_secs=0.3&"
    "include_language_detection=true"
)


class FakeWebSocket:
    def __init__(self) -> None:
        self.closed = asyncio.Event()
        self.sent: list[str] = []

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        await self.closed.wait()
        raise ConnectionError("synthetic socket closed")

    async def close(self) -> None:
        self.closed.set()


def query(url: str) -> dict[str, list[str]]:
    return parse_qs(urlsplit(url).query, keep_blank_values=True)


def config_with_filter(value: object) -> dict:
    config = dict(BASE_CONFIG)
    config["elevenlabs_stt_filter_background_audio"] = value
    return config


def test_default_constructor_and_factory_preserve_exact_prior_query() -> None:
    direct = ElevenLabsSTT(api_key="synthetic-key")
    factory_missing = create_elevenlabs_stt(dict(BASE_CONFIG))
    factory_false = create_elevenlabs_stt(config_with_filter(False))

    assert direct.filter_background_audio is False
    assert factory_missing.filter_background_audio is False
    assert factory_false.filter_background_audio is False
    assert direct._connection_url() == BASE_URL
    assert factory_missing._connection_url() == BASE_URL
    assert factory_false._connection_url() == BASE_URL
    assert query(direct._connection_url()) == BASE_QUERY
    assert query(factory_missing._connection_url()) == BASE_QUERY
    assert query(factory_false._connection_url()) == BASE_QUERY


def test_true_adds_only_filter_and_never_timestamps() -> None:
    stt = create_elevenlabs_stt(config_with_filter(True))
    filtered = query(stt._connection_url())

    assert filtered == BASE_QUERY | {"filter_background_audio": ["true"]}
    assert "include_timestamps" not in filtered


@pytest.mark.parametrize(
    ("language", "expected"),
    [("en", "en"), ("ar", "ar"), ("", None)],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_language_hints_remain_unchanged(language: str, expected: str | None, enabled: bool) -> None:
    stt = ElevenLabsSTT(
        api_key="synthetic-key",
        language=language,
        filter_background_audio=enabled,
    )
    parsed = query(stt._connection_url())

    if expected is None:
        assert "language_code" not in parsed
    else:
        assert parsed["language_code"] == [expected]
    assert parsed["model_id"] == ["scribe_v2_realtime"]
    assert parsed["audio_format"] == ["ulaw_8000"]
    assert parsed["sample_rate"] == ["8000"]
    assert parsed["commit_strategy"] == ["vad"]
    assert parsed["include_language_detection"] == ["true"]
    assert ("filter_background_audio" in parsed) is enabled


@pytest.mark.parametrize("raw", ["false", "true"])
def test_settings_parses_boolean_environment_values(monkeypatch, raw: str) -> None:
    monkeypatch.setenv("ELEVENLABS_STT_FILTER_BACKGROUND_AUDIO", raw)
    configured = settings_module.Settings(_env_file=None)
    assert configured.elevenlabs_stt_filter_background_audio is (raw == "true")
    instance = create_elevenlabs_stt(configured.model_dump())
    assert ("filter_background_audio" in query(instance._connection_url())) is (raw == "true")


@pytest.mark.parametrize("invalid", ["false", "true", None, 1, 0, [], {}, object()])
def test_constructor_rejects_non_boolean_without_echoing_value(invalid: object) -> None:
    with pytest.raises(TypeError) as error:
        ElevenLabsSTT(
            api_key="synthetic-key",
            filter_background_audio=invalid,
        )
    assert str(error.value) == "filter_background_audio must be a bool"


@pytest.mark.parametrize("invalid", ["false", "true", None, 1, [], {}])
def test_factory_rejects_non_boolean_without_silent_disable(invalid: object) -> None:
    with pytest.raises(TypeError) as error:
        create_elevenlabs_stt(config_with_filter(invalid))
    assert str(error.value) == "filter_background_audio must be a bool"


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_actual_factory_connection_uses_candidate_query(monkeypatch, enabled: bool) -> None:
    sockets: list[FakeWebSocket] = []
    urls: list[str] = []

    async def connect(url: str, **_kwargs):
        urls.append(url)
        socket = FakeWebSocket()
        sockets.append(socket)
        return socket

    monkeypatch.setattr(stt_module.websockets, "connect", connect)
    stt = create_elevenlabs_stt(config_with_filter(enabled))
    try:
        assert await stt.connect() is True
        assert len(urls) == 1
        parsed = query(urls[0])
        assert ("filter_background_audio" in parsed) is enabled
        assert "include_timestamps" not in parsed
        assert parsed["include_language_detection"] == ["true"]
    finally:
        await stt.disconnect()
    assert stt.status is STTStatus.DISCONNECTED
    assert all(socket.closed.is_set() for socket in sockets)


@pytest.mark.asyncio
async def test_reset_and_language_reconnect_retain_instance_option(monkeypatch) -> None:
    urls: list[str] = []

    async def connect(url: str, **_kwargs):
        urls.append(url)
        return FakeWebSocket()

    monkeypatch.setattr(stt_module.websockets, "connect", connect)
    stt = ElevenLabsSTT(
        api_key="synthetic-key",
        language="en",
        filter_background_audio=True,
    )
    try:
        assert await stt.connect() is True
        await stt.reset_for_listening()
        assert await stt.reconnect_with_language("ar") is True
        assert stt.filter_background_audio is True
        assert [query(url).get("filter_background_audio") for url in urls] == [
            ["true"],
            ["true"],
            ["true"],
        ]
        assert query(urls[-1])["language_code"] == ["ar"]
    finally:
        await stt.disconnect()


def test_separate_instances_do_not_share_filter_state() -> None:
    enabled = ElevenLabsSTT(api_key="first", filter_background_audio=True)
    disabled = ElevenLabsSTT(api_key="second", filter_background_audio=False)

    assert enabled.filter_background_audio is True
    assert disabled.filter_background_audio is False
    assert "filter_background_audio=true" in enabled._connection_url()
    assert "filter_background_audio" not in disabled._connection_url()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_filtered_mode_preserves_identity_and_unknown_metadata(enabled: bool) -> None:
    stt = ElevenLabsSTT(api_key="synthetic-key", filter_background_audio=enabled)
    stt._transcript_queue = asyncio.Queue()
    stt._session_active = True
    receiver = stt._activate_receiver(FakeWebSocket())

    await stt._handle_commit(receiver, {"text": "yes"})
    await stt._handle_commit(receiver, {"text": "yes"})
    first = await stt._transcript_queue.get()
    second = await stt._transcript_queue.get()
    assert first.utterance_id != second.utterance_id
    assert first.confidence is None
    assert second.confidence is None

    stt._handle_enrichment(
        receiver,
        {"text": "yes", "language_code": "en"},
        "committed_transcript_with_timestamps",
    )
    event = stt.metadata_events[-1]
    assert event.disposition is MetadataDisposition.AMBIGUOUS
    assert event.utterance_id is None
    assert event.confidence is None
    assert event.language_confidence is None
    assert event.word_timings == ()


@pytest.mark.asyncio
async def test_rejected_filtered_request_is_not_retried_unfiltered(monkeypatch) -> None:
    urls: list[str] = []

    async def reject(url: str, **_kwargs):
        urls.append(url)
        raise RuntimeError("synthetic provider rejection")

    monkeypatch.setattr(stt_module.websockets, "connect", reject)
    stt = ElevenLabsSTT(api_key="synthetic-key", filter_background_audio=True)

    assert await stt.connect() is False
    assert len(urls) == 1
    assert query(urls[0])["filter_background_audio"] == ["true"]
    assert stt.status is STTStatus.ERROR


def test_compose_declares_default_false_passthrough() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    environment = compose["services"]["app"]["environment"]
    expression = environment["ELEVENLABS_STT_FILTER_BACKGROUND_AUDIO"]
    expected = "${ELEVENLABS_STT_FILTER_BACKGROUND_AUDIO:-false}"
    assert expression == expected

def test_invalid_filter_request_does_not_leak_into_test_output() -> None:
    # Keep this contract explicit: errors identify the field/type only.
    with pytest.raises(TypeError, match="^filter_background_audio must be a bool$"):
        ElevenLabsSTT(api_key="synthetic-key", filter_background_audio={"secret": "value"})
