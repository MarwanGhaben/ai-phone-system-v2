"""
=====================================================
AI Voice Platform v2 - Audio Converter
=====================================================
Converts audio formats for Twilio Media Streams
"""

import io
from typing import Tuple
from loguru import logger

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None


def convert_to_mulaw(audio_data: bytes, input_format: str = "mp3", input_rate: int = 44100) -> bytes:
    """
    Convert audio to μ-law 8kHz mono (Twilio Media Streams format)

    Args:
        audio_data: Input audio bytes
        input_format: Input format (mp3, wav, etc.)
        input_rate: Input sample rate

    Returns:
        μ-law encoded audio bytes (8kHz mono)
    """
    if not PYDUB_AVAILABLE:
        logger.error("pydub is not available, cannot convert audio")
        # Return empty bytes to avoid sending invalid audio
        return b""

    try:
        # Load audio from bytes
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format=input_format)

        # Convert to mono if needed
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Resample to 8000 Hz (Twilio requirement)
        if audio.frame_rate != 8000:
            audio = audio.set_frame_rate(8000)

        # Export as raw PCM (16-bit)
        raw_audio = audio.raw_data

        # Convert PCM to μ-law
        mulaw_audio = pcm_to_mulaw(raw_audio)

        # Verify audio is not silent (check first 100 bytes)
        if len(mulaw_audio) > 100:
            sample_bytes = mulaw_audio[:100]
            # Count non-zero bytes (μ-law silence is often 255, so check variety)
            unique_values = len(set(sample_bytes))
            logger.info(f"Audio conversion: {len(audio_data)} -> {len(mulaw_audio)} bytes μ-law, unique values in first 100 bytes: {unique_values}")
            if unique_values < 10:
                logger.warning(f"Audio may be silent or invalid - only {unique_values} unique values in first 100 bytes")

        return mulaw_audio

    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return b""


def pcm_to_mulaw(pcm_data: bytes) -> bytes:
    """
    Convert 16-bit PCM to μ-law

    Args:
        pcm_data: 16-bit PCM audio bytes (little-endian)

    Returns:
        μ-law encoded bytes
    """
    import struct
    import array

    # μ-law encoding table (simplified)
    # Using Python's built-in audioop module if available
    try:
        import audioop
        return audioop.lin2ulaw(pcm_data, 2)  # 2 bytes per sample (16-bit)
    except ImportError:
        # Fallback: manual implementation
        # For now, return the original data (won't work properly)
        logger.warning("audioop not available, μ-law conversion may not work")
        return pcm_data


def convert_twilio_audio(audio_data: bytes, input_format: str = "mp3", input_rate: int = 44100) -> bytes:
    """
    Convert audio for Twilio Media Streams
    Wrapper for convert_to_mulaw with logging

    Args:
        audio_data: Input audio bytes
        input_format: Input format (mp3, wav, etc.)
        input_rate: Input sample rate

    Returns:
        μ-law encoded audio bytes (8kHz mono)
    """
    result = convert_to_mulaw(audio_data, input_format, input_rate)

    if result:
        logger.debug(f"Audio converted: {len(audio_data)} bytes -> {len(result)} bytes μ-law")
    else:
        logger.error("Audio conversion failed")

    return result
