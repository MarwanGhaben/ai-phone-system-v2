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
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    ElevenLabs = None

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
        # English voices (use voice_ids)
        "Rachel": {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "gender": "F", "accent": "American"},
        "Drew": {"voice_id": "29vD33N1CtxCmqQRPOHJ", "name": "Drew", "gender": "M", "accent": "American"},
        "Clyde": {"voice_id": "2EiwWnXFnvU5JabPnv8n", "name": "Clyde", "gender": "M", "accent": "American"},
        "Sarah": {"voice_id": "EXHAITRWHUWQO296QKJI", "name": "Sarah", "gender": "F", "accent": "British"},
        "Adam": {"voice_id": "ADq4zsqJPsd4acy0B6B1", "name": "Adam", "gender": "M", "accent": "American"},
        "Emily": {"voice_id": "AZnzlk1XvdvUeBnXmlld", "name": "Emily", "gender": "F", "accent": "American"},
        "Josh": {"voice_id": "TxGEqnHWrfWFTfGW9XjX", "name": "Josh", "gender": "M", "accent": "Canadian"},

        # Multilingual voices (good for Arabic)
        "Antoni": {"voice_id": "ErXwobaRi7UmFJ9fQaF1", "name": "Antoni", "gender": "M", "accent": "American", "languages": ["en", "es"]},
        "Fin": {"voice_id": "YOZ27uZTVtijvd1HfGBq", "name": "Fin", "gender": "M", "accent": "Irish", "languages": ["en", "es"]},
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
            default_voice_id: Default voice name (will be mapped to voice_id)
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

        # Create ElevenLabs client
        self._client = Optional[ElevenLabs]
        self._voices_cache: Optional[List[dict]] = None
        self._stop_event = asyncio.Event()

    def _get_client(self) -> ElevenLabs:
        """Get or create ElevenLabs client"""
        return ElevenLabs(api_key=self.api_key)

    def _get_voice_id(self, voice_name: str) -> str:
        """Convert voice name to voice_id"""
        voice_info = self.POPULAR_VOICES.get(voice_name)
        if voice_info:
            return voice_info.get("voice_id", voice_name)
        return voice_name  # Assume it's already a voice_id

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
            client = self._get_client()
            voice_id = self._get_voice_id(request.voice_id or self.default_voice_id)

            logger.info(f"ElevenLabs: Synthesizing '{request.text[:50]}...' with voice {voice_id}")

            # Generate audio using new API
            audio = client.text_to_speech.convert(
                text=request.text,
                voice_id=voice_id,
                model_id=self.model,
                output_format=self.output_format,
            )

            # audio is bytes directly in new API
            if isinstance(audio, bytes):
                audio_data = audio
            else:
                # If it's a generator, collect all chunks
                audio_buffer = bytearray()
                for chunk in audio:
                    if self._stop_event.is_set():
                        self._status = TTSStatus.INTERRUPTED
                        logger.info("ElevenLabs: Synthesis interrupted")
                        break
                    audio_buffer.extend(chunk)
                audio_data = bytes(audio_buffer)

            if not self._stop_event.is_set():
                self._status = TTSStatus.IDLE

            # Calculate approximate duration
            total_bytes = len(audio_data)
            # MP3 at 128 kbps = 16 KB/sec
            duration_ms = int((total_bytes / 16000) * 1000)

            return TTSResponse(
                audio_data=audio_data,
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
            client = self._get_client()
            voice_id = self._get_voice_id(request.voice_id or self.default_voice_id)

            logger.info(f"ElevenLabs: Streaming '{request.text[:50]}...' with voice {voice_id}")

            # Stream audio generation using new API
            audio_stream = client.text_to_speech.convert_as_stream(
                text=request.text,
                voice_id=voice_id,
                model_id=self.model,
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
            client = self._get_client()
            response = client.voices.get_all()

            voices_list = []
            if hasattr(response, 'voices'):
                for voice in response.voices:
                    voices_list.append({
                        "voice_id": voice.voice_id,
                        "name": voice.name,
                        "category": voice.category if hasattr(voice, 'category') else None,
                        "labels": voice.labels if hasattr(voice, 'labels') else {},
                    })

            # Filter by language if requested
            if language != "all":
                return [
                    v for v in voices_list
                    if language in str(v.get("labels", {})).lower()
                ]

            return voices_list

        except Exception as e:
            logger.error(f"ElevenLabs: Error fetching voices: {e}")
            # Return popular voices as fallback
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
