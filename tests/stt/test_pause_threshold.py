"""Caller-pause threshold for the ElevenLabs realtime connection query."""
from urllib.parse import parse_qs, urlparse

import pytest

from services.stt.elevenlabs_stt_service import (
    VAD_SILENCE_ENV,
    create_elevenlabs_stt,
    vad_silence_threshold_param,
)


def _threshold_in_url():
    service = create_elevenlabs_stt({"elevenlabs_api_key": "synthetic-key"})
    query = parse_qs(urlparse(service._connection_url()).query)
    return query["vad_silence_threshold_secs"]


def test_default_when_unset(monkeypatch):
    monkeypatch.delenv(VAD_SILENCE_ENV, raising=False)
    assert vad_silence_threshold_param() == "0.7"
    assert _threshold_in_url() == ["0.7"]


@pytest.mark.parametrize("raw", ["", "   "])
def test_default_when_empty(monkeypatch, raw):
    monkeypatch.setenv(VAD_SILENCE_ENV, raw)
    assert vad_silence_threshold_param() == "0.7"


@pytest.mark.parametrize("raw,expected", [
    ("0.3", "0.3"), ("0.45", "0.45"), ("0.7", "0.7"), ("1", "1.0"),
    ("1.0", "1.0"), ("1.2", "1.2"), ("1.5", "1.5"), (" 0.9 ", "0.9"),
    ("0.30", "0.3"), ("1.50", "1.5"),
])
def test_valid_values_including_boundaries(monkeypatch, raw, expected):
    monkeypatch.setenv(VAD_SILENCE_ENV, raw)
    assert vad_silence_threshold_param() == expected
    assert _threshold_in_url() == [expected]


@pytest.mark.parametrize("raw", [
    "0.29", "0.299999", "1.51", "1.5000001", "0", "-1", "-0.5", "100",   # out of range
    "abc", "0.7s", "1,0", "0.7 0.8", "true", "None",                       # text
    "nan", "NaN", "inf", "-inf", "Infinity", "1e999",                      # not finite
])
def test_invalid_values_fall_back_to_default(monkeypatch, raw):
    monkeypatch.setenv(VAD_SILENCE_ENV, raw)
    assert vad_silence_threshold_param() == "0.7"
    assert _threshold_in_url() == ["0.7"]


def test_value_is_read_each_time_connection_parameters_are_built(monkeypatch):
    service = create_elevenlabs_stt({"elevenlabs_api_key": "synthetic-key"})
    monkeypatch.setenv(VAD_SILENCE_ENV, "1.1")
    assert "vad_silence_threshold_secs=1.1&" in service._connection_url()
    monkeypatch.setenv(VAD_SILENCE_ENV, "0.5")
    assert "vad_silence_threshold_secs=0.5&" in service._connection_url()


def test_other_connection_parameters_are_unchanged(monkeypatch):
    monkeypatch.delenv(VAD_SILENCE_ENV, raising=False)
    service = create_elevenlabs_stt({"elevenlabs_api_key": "synthetic-key"})
    assert service._connection_url() == (
        "wss://api.elevenlabs.io/v1/speech-to-text/realtime?"
        "model_id=scribe_v2_realtime&audio_format=ulaw_8000&sample_rate=8000&"
        "commit_strategy=vad&vad_silence_threshold_secs=0.7&"
        "include_language_detection=true")
