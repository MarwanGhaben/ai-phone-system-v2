"""
=====================================================
AI Voice Platform v2 - ElevenLabs TTS Service
=====================================================
Real-time streaming Text-to-Speech using ElevenLabs

KEY DESIGN: Uses async HTTP streaming to send audio to Twilio
as it's generated, reducing time-to-first-audio from seconds
to ~200ms. The caller hears audio almost immediately.
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

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

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
        output_format: str = "ulaw_8000"
    ):
        """
        Initialize ElevenLabs TTS service

        Args:
            api_key: ElevenLabs API key
            default_voice_id: Default voice name (will be mapped to voice_id)
            model: Model to use (eleven_multilingual_v2 recommended)
            stability: Voice stability (0-1, lower = more expressive)
            similarity_boost: Voice similarity (0-1, higher = more similar to original)
            output_format: Audio output format (ulaw_8000 for direct Twilio compatibility)
        """
        if not ELEVENLABS_AVAILABLE:
            raise ImportError("elevenlabs is not installed. Install with: pip install elevenlabs")

        super().__init__(api_key, default_voice_id)

        self.model = model
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.output_format = output_format

        # Reuse ElevenLabs client (don't recreate per call)
        self._client: Optional[ElevenLabs] = ElevenLabs(api_key=self.api_key) if ELEVENLABS_AVAILABLE else None
        # Reuse httpx client for streaming TTS
        self._http_client: Optional[httpx.AsyncClient] = None
        self._voices_cache: Optional[List[dict]] = None
        self._stop_event = asyncio.Event()

    def _get_client(self) -> ElevenLabs:
        """Get ElevenLabs SDK client (reused)"""
        return self._client

    async def _get_http_client(self) -> "httpx.AsyncClient":
        """Get or create async HTTP client for streaming TTS"""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

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

            # Calculate approximate duration based on output format
            total_bytes = len(audio_data)
            is_mulaw = "ulaw" in self.output_format or "mulaw" in self.output_format
            if is_mulaw:
                # μ-law 8kHz = 8000 bytes/sec (1 byte per sample)
                duration_ms = int((total_bytes / 8000) * 1000)
                sample_rate = 8000
                fmt = "mulaw"
            else:
                # MP3 at 128 kbps = 16 KB/sec
                duration_ms = int((total_bytes / 16000) * 1000)
                sample_rate = 44100
                fmt = "mp3"

            return TTSResponse(
                audio_data=audio_data,
                sample_rate=sample_rate,
                format=fmt,
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

    async def synthesize_stream_async(self, request: TTSRequest) -> AsyncIterator[bytes]:
        """
        Stream TTS audio using async HTTP — yields raw audio chunks as they're generated.

        This uses the ElevenLabs /stream endpoint with httpx async streaming,
        so audio bytes flow to the caller as they're generated (~200ms to first chunk)
        instead of waiting for the entire audio to be synthesized (~2-3s).

        The output is raw ulaw_8000 bytes ready for Twilio — no conversion needed.

        Args:
            request: TTS request with text and voice settings

        Yields:
            Raw audio bytes (ulaw_8000 format) as they arrive from ElevenLabs
        """
        if not HTTPX_AVAILABLE:
            logger.warning("ElevenLabs: httpx not available, falling back to blocking synthesize")
            response = await self.synthesize(request)
            yield response.audio_data
            return

        self._status = TTSStatus.SPEAKING
        self._current_request = request
        self._stop_event.clear()

        voice_id = self._get_voice_id(request.voice_id or self.default_voice_id)
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        params = {
            "output_format": self.output_format,
            "optimize_streaming_latency": "3",  # Max latency optimization for faster TTFB
        }
        body = {
            "text": request.text,
            "model_id": self.model,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
            }
        }

        logger.info(f"ElevenLabs: Streaming async '{request.text[:50]}...' voice={voice_id} model={self.model}")

        try:
            client = await self._get_http_client()
            async with client.stream(
                "POST", url,
                headers=headers,
                params=params,
                json=body,
                timeout=30.0,
            ) as response:
                response.raise_for_status()
                chunk_count = 0
                async for chunk in response.aiter_bytes(1024):
                    if self._stop_event.is_set():
                        self._status = TTSStatus.INTERRUPTED
                        logger.info("ElevenLabs: Async stream interrupted by stop event")
                        break
                    chunk_count += 1
                    if chunk_count == 1:
                        logger.info(f"ElevenLabs: First audio chunk received ({len(chunk)} bytes)")
                    yield chunk

            if not self._stop_event.is_set():
                self._status = TTSStatus.IDLE
                logger.info(f"ElevenLabs: Async stream complete ({chunk_count} chunks)")

        except httpx.HTTPStatusError as e:
            self._status = TTSStatus.ERROR
            logger.error(f"ElevenLabs: HTTP streaming error {e.response.status_code}: {e.response.text[:200]}")
            raise
        except Exception as e:
            self._status = TTSStatus.ERROR
            logger.error(f"ElevenLabs: Async streaming error: {e}")
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
            output_format=config.get('elevenlabs_output_format', 'ulaw_8000')
        )
    except Exception as e:
        logger.error(f"Failed to initialize ElevenLabs TTS: {e}")
        return None
