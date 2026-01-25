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
    from elevenlabs import Voice, VoiceSettings, generate, stream, Voices
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    Voice = VoiceSettings = ElevenLabs = None

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

        self._client: Optional[ElevenLabs] = None
        self._voices_cache: Optional[List[dict]] = None
        self._stop_event = asyncio.Event()

    async def _get_client(self) -> ElevenLabs:
        """Get or create ElevenLabs client"""
        if self._client is None:
            self._client = ElevenLabs(api_key=self.api_key)
        return self._client

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
            client = await self._get_client()
            voice_id = request.voice_id or self.default_voice_id

            logger.info(f"ElevenLabs: Synthesizing '{request.text[:50]}...' with voice {voice_id}")

            # Configure voice settings
            voice_settings = VoiceSettings(
                stability=self.stability,
                similarity_boost=self.similarity_boost,
                model=self.model,
            )

            # Generate audio
            audio_generator = generate(
                text=request.text,
                voice=voice_id,
                model=self.model,
                voice_settings=voice_settings
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
            client = await self._get_client()
            voice_id = request.voice_id or self.default_voice_id

            logger.info(f"ElevenLabs: Streaming '{request.text[:50]}...' with voice {voice_id}")

            voice_settings = VoiceSettings(
                stability=self.stability,
                similarity_boost=self.similarity_boost,
                model=self.model,
            )

            # Stream audio generation
            audio_stream = stream(
                text=request.text,
                voice=voice_id,
                model=self.model,
                voice_settings=voice_settings
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
        try:
            if self._voices_cache is None:
                client = await self._get_client()
                voices: Voices = client.voices.get_all()

                # Convert to simpler format
                self._voices_cache = [
                    {
                        "voice_id": voice.voice_id,
                        "name": voice.name,
                        "category": voice.category,
                        "labels": voice.labels or {},
                    }
                    for voice in voices
                ]

            # Filter by language if requested
            if language != "all":
                return [
                    v for v in self._voices_cache
                    if language in str(v.get("labels", {})).lower()
                ]

            return self._voices_cache

        except Exception as e:
            logger.error(f"ElevenLabs: Error fetching voices: {e}")
            return []


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
