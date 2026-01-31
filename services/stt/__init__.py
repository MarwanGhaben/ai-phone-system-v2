"""
=====================================================
AI Voice Platform v2 - STT (Speech-to-Text) Services
=====================================================
"""

from .stt_base import STTServiceBase, STTResult, STTStatus, AudioChunk
from .deepgram_service import DeepgramSTT, create_deepgram_stt
from .whisper_service import WhisperSTT, create_whisper_stt
from .elevenlabs_stt_service import ElevenLabsSTT, create_elevenlabs_stt

__all__ = [
    'STTServiceBase',
    'STTResult',
    'STTStatus',
    'AudioChunk',
    'DeepgramSTT',
    'create_deepgram_stt',
    'WhisperSTT',
    'create_whisper_stt',
    'ElevenLabsSTT',
    'create_elevenlabs_stt',
]


def create_stt_service(provider: str, config: dict) -> STTServiceBase:
    """
    Factory function to create STT service by provider name

    Args:
        provider: 'deepgram', 'whisper', or 'elevenlabs'
        config: Configuration dictionary

    Returns:
        Configured STT service instance

    Raises:
        ValueError: If provider is not supported
    """
    providers = {
        'deepgram': create_deepgram_stt,
        'whisper': create_whisper_stt,
        'elevenlabs': create_elevenlabs_stt,
    }

    if provider not in providers:
        raise ValueError(f"Unknown STT provider: {provider}. Available: {list(providers.keys())}")

    return providers[provider](config)
