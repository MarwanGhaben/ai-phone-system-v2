import pytest

from services.stt.whisper_service import WhisperSTT


def scalar_pcm_reference(mulaw_data: bytes) -> bytes:
    decode_table = []
    for encoded_sample in range(256):
        mu = 255 - encoded_sample
        magnitude = (mu & 0x0F) << 3
        exponent = (mu & 0x70) >> 4
        if exponent > 0:
            magnitude = (magnitude + 0x84) << (exponent - 1)
        if mu & 0x80:
            magnitude = -magnitude
        decode_table.append(magnitude)

    pcm_bytes = bytearray()
    for encoded_sample in mulaw_data:
        sample = max(-32768, min(32767, decode_table[encoded_sample]))
        pcm_bytes.extend((sample & 0xFF, (sample >> 8) & 0xFF))
    return bytes(pcm_bytes)


def test_mulaw_conversion_preserves_pcm_output() -> None:
    stt = WhisperSTT(api_key="test-key")
    all_mulaw_values = bytes(range(256))

    assert stt._mulaw_to_pcm(all_mulaw_values) == scalar_pcm_reference(all_mulaw_values)


@pytest.mark.parametrize(
    ("audio_data", "threshold", "expected"),
    [(b"", 0.02, False), (bytes([0xFF]) * 160, 0.02, False), (bytes([0x00]) * 160, 0.02, True)],
)
def test_energy_detection_preserves_threshold_decisions(
    audio_data: bytes,
    threshold: float,
    expected: bool,
) -> None:
    stt = WhisperSTT(api_key="test-key", silence_threshold=threshold)

    assert stt._has_speech_energy(audio_data, threshold) is expected
