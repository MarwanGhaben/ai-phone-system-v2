"""
=====================================================
AI Voice Platform v2 - ElevenLabs TTS Service
=====================================================
Real-time streaming Text-to-Speech using ElevenLabs
"""

import asyncio
import io
from typing import AsyncIterator, Optional, List
from loguru import logger

try:
    from elevenlabs import generate, stream, Voice, VoiceSettings, set_api_key
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    generate = stream = Voice = VoiceSettings = set_api_key = None

from .tts_base import TTSServiceBase, TTSRequest, TTSResponse, TTSChunk, TTSStatus


class ElevenLabsTTS(TTSServiceBase):
    """
    ElevenLabs Streaming TTS Service

    Features:
    - Human-level voice quality
    - Streaming synthesis with low latency
    - Emotional inflection and prosody
    - Multi-language support
    - Voice customization (stability, similarity boost)
    """

    # Popular voices for different languages/accents
    POPULAR_VOICES = {
        # English voices
        "Rachel": {"name": "Rachel", "gender": "F", "accent": "American"},
        "Drew": {"name": "Drew", "gender": "M", "accent": "American"},
        "Clyde": {"name": "Clyde", "gender": "M", "accent": "American"},
        "Sarah": {"name": "Sarah", "gender": "F", "accent": "British"},
        "Adam": {"name": "Adam", "gender": "M", "accent": "American"},
        "Emily": {"name": "Emily", "gender": "F", "accent": "American"},
        "Josh": {"name": "Josh", "gender": "M", "accent": "Canadian"},

        # Multilingual voices (good for Arabic)
        "Antoni": {"name": "Antoni", "gender": "M", "accent": "American", "languages": ["en", "es"]},
        "Fin": {"name": "Fin", "gender": "M", "accent": "Irish", "languages": ["en", "es"]},
    }

    def __init__(
        self,
        api_key: str,
        default_voice_id: str = "Rachel",
        model: str = "eleven_multilingual_v2",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        output_format: str = "mp3_44100_128"
    ):
        """
        Initialize ElevenLabs TTS service

        Args:
            api_key: ElevenLabs API key
            default_voice_id: Default voice ID
            model: Model to use (eleven_multilingual_v2 recommended)
            stability: Voice stability (0-1, lower = more expressive)
            similarity_boost: Voice similarity (0-1, higher = more similar to original)
            output_format: Audio output format
        """
        if not ELEVENLABS_AVAILABLE:
            raise ImportError("elevenlabs is not installed. Install with: pip install elevenlabs")

        super().__init__(api_key, default_voice_id)

        self.model = model
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.output_format = output_format

        # Set API key globally for elevenlabs 1.5.0
        set_api_key(api_key)

        self._voices_cache: Optional[List[dict]] = None
        self._stop_event = asyncio.Event()

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize speech from text (blocking)

        Args:
            request: TTS request

        Returns:
            TTS response with audio data
        """
        self._status = TTSStatus.SPEAKING
        self._current_request = request
        self._stop_event.clear()

        try:
            voice_id = request.voice_id or self.default_voice_id

            logger.info(f"ElevenLabs: Synthesizing '{request.text[:50]}...' with voice {voice_id}")

            # Configure voice settings
            voice_settings = VoiceSettings(
                stability=self.stability,
                similarity_boost=self.similarity_boost
            )

            # Generate audio (elevenlabs 1.5.0 uses sync API)
            audio_generator = generate(
                text=request.text,
                voice=voice_id,
                model=self.model,
                voice_settings=voice_settings,
                api_key=self.api_key
            )

            # Collect all audio data
            audio_buffer = bytearray()
            duration_chunks = []

            for chunk in audio_generator:
                if self._stop_event.is_set():
                    self._status = TTSStatus.INTERRUPTED
                    logger.info("ElevenLabs: Synthesis interrupted")
                    break

                audio_buffer.extend(chunk)
                duration_chunks.append(len(chunk))

            # Calculate approximate duration
            total_bytes = len(audio_buffer)
            # MP3 at 128 kbps = 16 KB/sec
            duration_ms = int((total_bytes / 16000) * 1000)

            if not self._stop_event.is_set():
                self._status = TTSStatus.IDLE

            return TTSResponse(
                audio_data=bytes(audio_buffer),
                sample_rate=44100,
                format="mp3",
                duration_ms=duration_ms,
                text=request.text,
                is_final=True,
                metadata={"voice_id": voice_id}
            )

        except Exception as e:
            self._status = TTSStatus.ERROR
            logger.error(f"ElevenLabs: Synthesis error: {e}")
            raise

    async def synthesize_stream(self, request: TTSRequest) -> AsyncIterator[TTSChunk]:
        """
        Synthesize speech with streaming output

        Args:
            request: TTS request

        Yields:
            TTSChunk objects as audio is generated
        """
        self._status = TTSStatus.SPEAKING
        self._current_request = request
        self._stop_event.clear()

        try:
            voice_id = request.voice_id or self.default_voice_id

            logger.info(f"ElevenLabs: Streaming '{request.text[:50]}...' with voice {voice_id}")

            voice_settings = VoiceSettings(
                stability=self.stability,
                similarity_boost=self.similarity_boost
            )

            # Stream audio generation (elevenlabs 1.5.0)
            audio_stream = stream(
                text=request.text,
                voice=voice_id,
                model=self.model,
                voice_settings=voice_settings,
                api_key=self.api_key
            )

            chunk_index = 0
            for chunk in audio_stream:
                if self._stop_event.is_set():
                    self._status = TTSStatus.INTERRUPTED
                    logger.info("ElevenLabs: Stream interrupted")
                    break

                yield TTSChunk(
                    audio_data=chunk,
                    is_final=False,
                    text_offset=chunk_index
                )
                chunk_index += len(chunk)

            # Send final chunk
            if not self._stop_event.is_set():
                yield TTSChunk(
                    audio_data=b"",
                    is_final=True,
                    text_offset=chunk_index
                )
                self._status = TTSStatus.IDLE

        except Exception as e:
            self._status = TTSStatus.ERROR
            logger.error(f"ElevenLabs: Streaming error: {e}")
            raise

    async def stop(self) -> None:
        """Stop current synthesis"""
        self._stop_event.set()
        self._status = TTSStatus.IDLE

    async def get_available_voices(self, language: str = "all") -> List[dict]:
        """
        Get list of available voices

        Args:
            language: Filter by language ('all' for no filter)

        Returns:
            List of voice metadata
        """
        # TODO: Implement for elevenlabs 1.5.0 (voices API changed)
        # For now, return the popular voices defined in the class
        return [
            {"voice_id": k, **v} for k, v in self.POPULAR_VOICES.items()
        ]


# Factory function
def create_elevenlabs_tts(config: dict) -> ElevenLabsTTS | None:
    """
    Factory function to create ElevenLabs TTS service from config

    Args:
        config: Configuration dictionary (from Settings)

    Returns:
        Configured ElevenLabsTTS instance or None if not available
    """
    if not ELEVENLABS_AVAILABLE:
        logger.warning("ElevenLabs is not available, TTS will be disabled")
        return None

    try:
        return ElevenLabsTTS(
            api_key=config.get('elevenlabs_api_key'),
            default_voice_id=config.get('elevenlabs_voice_id', 'Rachel'),
            model=config.get('elevenlabs_model', 'eleven_multilingual_v2'),
            stability=config.get('elevenlabs_stability', 0.5),
            similarity_boost=config.get('elevenlabs_similarity_boost', 0.75),
            output_format=config.get('elevenlabs_output_format', 'mp3_44100_128')
        )
    except Exception as e:
        logger.error(f"Failed to initialize ElevenLabs TTS: {e}")
        return None
