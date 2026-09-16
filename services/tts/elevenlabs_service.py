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


_STREAM_DONE = object()


class ElevenLabsStream:
    """One independently cancellable use of the shared ElevenLabs HTTP pool.

    Cancellation wakes a blocked consumer immediately.  HTTP response shutdown is
    owned by the producer task and remains observable through ``cleanup_pending``
    and ``wait_closed`` if an uncooperative transport outlives the 250 ms close
    budget used by ``aclose``.
    """

    CLOSE_TIMEOUT_SECONDS = 0.25

    def __init__(self, service: "ElevenLabsTTS", request: TTSRequest, *, scoped: bool = True):
        self._service = service
        self._request = request
        self._scoped = scoped
        self._cancelled = asyncio.Event()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._producer_task: Optional[asyncio.Task] = None
        self._child_tasks: set[asyncio.Task] = set()
        self._consumer_done = False
        self._cleanup_complete = asyncio.Event()
        self.first_chunk = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if self._consumer_done or self._cancelled.is_set():
            raise StopAsyncIteration
        self._ensure_started()

        queued = self._create_child(self._queue.get())
        stopped = self._create_child(self._cancelled.wait())
        try:
            done, _ = await asyncio.wait(
                {queued, stopped}, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            queued.cancel()
            stopped.cancel()
            raise
        if stopped in done and self._cancelled.is_set():
            self._consumer_done = True
            queued.cancel()
            raise StopAsyncIteration

        stopped.cancel()
        item = queued.result()
        if item is _STREAM_DONE:
            self._consumer_done = True
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            self._consumer_done = True
            raise item
        return item

    @property
    def cleanup_pending(self) -> bool:
        return (
            (self._producer_task is not None and not self._producer_task.done())
            or any(not task.done() for task in self._child_tasks)
        )

    def cancel(self) -> None:
        """Permanently retire this request without affecting any other request."""
        self._consumer_done = True
        self._cancelled.set()
        if self._producer_task is None:
            self._finish_cleanup()

    async def aclose(self) -> None:
        self.cancel()
        try:
            await self.wait_closed(timeout=self.CLOSE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning("ElevenLabs: scoped stream cleanup remains pending")

    async def wait_closed(self, timeout: float | None = None) -> None:
        if self._producer_task is None:
            self._finish_cleanup()
            return
        if timeout is None:
            await self._cleanup_complete.wait()
        else:
            async with asyncio.timeout(timeout):
                await self._cleanup_complete.wait()

    def _ensure_started(self) -> None:
        if self._producer_task is None and not self._consumer_done:
            self._service._active_streams.add(self)
            if not self._scoped:
                self._service._legacy_streams.add(self)
            self._producer_task = asyncio.create_task(self._produce())
            self._producer_task.add_done_callback(self._producer_done)

    def _create_child(self, awaitable) -> asyncio.Task:
        task = asyncio.create_task(awaitable)
        self._child_tasks.add(task)
        task.add_done_callback(self._child_done)
        return task

    def _child_done(self, task: asyncio.Task) -> None:
        self._child_tasks.discard(task)
        self._consume_task_result(task)
        self._maybe_finish_cleanup()

    def _producer_done(self, task: asyncio.Task) -> None:
        self._consume_task_result(task)
        self._maybe_finish_cleanup()

    def _maybe_finish_cleanup(self) -> None:
        if (
            self._producer_task is not None
            and self._producer_task.done()
            and not self._child_tasks
        ):
            self._finish_cleanup()

    def _finish_cleanup(self) -> None:
        self._service._active_streams.discard(self)
        self._service._legacy_streams.discard(self)
        self._cleanup_complete.set()

    async def _produce(self) -> None:
        try:
            if not HTTPX_AVAILABLE:
                raise RuntimeError("ElevenLabs async streaming is unavailable")

            client = await self._service._get_http_client()
            if self._cancelled.is_set():
                return
            voice_id = self._service._get_voice_id(
                self._request.voice_id or self._service.default_voice_id
            )
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
            headers = {
                "xi-api-key": self._service.api_key,
                "Content-Type": "application/json",
            }
            params = {
                "output_format": self._service.output_format,
                "optimize_streaming_latency": "2",
            }
            body = {
                "text": self._request.text,
                "model_id": self._service.model,
                "voice_settings": {
                    "stability": self._service.stability,
                    "similarity_boost": self._service.similarity_boost,
                },
            }

            logger.info("ElevenLabs: Starting request-scoped async stream")
            async with client.stream(
                "POST", url, headers=headers, params=params, json=body, timeout=30.0
            ) as response:
                if self._cancelled.is_set():
                    return
                response.raise_for_status()
                if self._cancelled.is_set():
                    return
                iterator = response.aiter_bytes(1024).__aiter__()
                chunk_count = 0
                while not self._cancelled.is_set():
                    next_chunk = self._create_child(iterator.__anext__())
                    stopped = self._create_child(self._cancelled.wait())
                    done, _ = await asyncio.wait(
                        {next_chunk, stopped}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if stopped in done and self._cancelled.is_set():
                        next_chunk.cancel()
                        break
                    stopped.cancel()
                    try:
                        chunk = next_chunk.result()
                    except StopAsyncIteration:
                        break
                    if self._cancelled.is_set():
                        break
                    chunk_count += 1
                    self.first_chunk.set()
                    await self._queue.put(chunk)
                logger.info(
                    f"ElevenLabs: Request-scoped stream ended ({chunk_count} chunks)"
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._cancelled.is_set():
                await self._queue.put(exc)
        finally:
            await self._queue.put(_STREAM_DONE)

    @staticmethod
    def _consume_task_result(task: asyncio.Task) -> None:
        if not task.cancelled():
            task.exception()


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
        self._active_streams: set[ElevenLabsStream] = set()
        self._legacy_streams: set[ElevenLabsStream] = set()

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
        stream = ElevenLabsStream(self, request, scoped=False)
        try:
            async for chunk in stream:
                yield chunk
        finally:
            await stream.aclose()

    def create_stream(self, request: TTSRequest) -> ElevenLabsStream:
        """Create a stream whose cancellation belongs only to this invocation."""
        return ElevenLabsStream(self, request, scoped=True)

    async def stop(self) -> None:
        """Stop legacy synthesis without touching scoped live-call streams."""
        self._stop_event.set()
        for stream in tuple(self._legacy_streams):
            stream.cancel()
        self._status = TTSStatus.IDLE

    async def prewarm(self) -> bool:
        """
        Pre-warm the HTTP connection pool to ElevenLabs API.

        Eliminates the "underwater" / cold-start audio quality issue on the
        first call after app restart by establishing TCP/TLS connections
        and DNS cache before any real call happens.

        Returns:
            True if prewarm succeeded, False otherwise
        """
        if not HTTPX_AVAILABLE:
            return False
        try:
            logger.info("ElevenLabs: Pre-warming HTTP connection pool...")
            client = await self._get_http_client()
            # Lightweight call to /user endpoint - establishes connection without using TTS credits
            response = await client.get(
                "https://api.elevenlabs.io/v1/user",
                headers={"xi-api-key": self.api_key},
                timeout=10.0,
            )
            if response.status_code == 200:
                logger.info("ElevenLabs: Connection pool warmed up successfully")
                return True
            logger.warning(f"ElevenLabs: Prewarm got status {response.status_code}")
            return False
        except Exception as e:
            logger.warning(f"ElevenLabs: Prewarm failed (non-critical): {e}")
            return False

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
