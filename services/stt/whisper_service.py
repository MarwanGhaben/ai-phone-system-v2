"""
=====================================================
AI Voice Platform v2 - OpenAI Whisper STT Service
=====================================================
Speech-to-Text using OpenAI Whisper API
Supports 50+ languages including Arabic
"""

import asyncio
import io
import tempfile
import wave
from typing import AsyncIterator, Optional, List
from loguru import logger

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .stt_base import STTServiceBase, STTResult, STTStatus, AudioChunk


class WhisperSTT(STTServiceBase):
    """
    OpenAI Whisper Streaming STT Service

    Features:
    - Supports 50+ languages including Arabic (ar), English (en)
    - High accuracy with noisy audio
    - Auto language detection
    - Uses buffered approach (not native streaming)

    Note: Whisper processes complete audio files, not streams.
    This service buffers audio and transcribes on speech pauses.
    """

    # Supported languages ( Whisper supports 50+ )
    SUPPORTED_LANGUAGES = ["en", "ar", "es", "fr", "de", "zh", "ja", "ko", "pt", "ru", "it", "nl", "tr", "pl", "sv", "fi"]

    def __init__(
        self,
        api_key: str,
        language: str = "en",  # Whisper uses ISO 639-1 codes (en, ar, not en-US)
        model: str = "whisper-1",
        silence_threshold: float = 0.3,  # Energy threshold for silence detection
        silence_duration: float = 1.0,  # Seconds of silence to trigger transcription
        min_audio_length: float = 0.5,  # Minimum audio length before transcribing
    ):
        """
        Initialize Whisper STT service

        Args:
            api_key: OpenAI API key
            language: Default language code (en, ar, etc.) or None for auto-detect
            model: Whisper model (whisper-1)
            silence_threshold: Energy level (0-1) below which is considered silence
            silence_duration: Seconds of silence to trigger transcription
            min_audio_length: Minimum seconds of audio before transcribing
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai is not installed. Install with: pip install openai")

        super().__init__(api_key, language)

        self.model = model
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_audio_length = min_audio_length

        self._client: Optional[AsyncOpenAI] = None
        self._transcript_queue: asyncio.Queue = None
        self._is_listening = False

        # Audio buffer
        self._audio_buffer: List[bytes] = []
        self._buffer_sample_rate = 8000  # Twilio default
        self._buffer_start_time: Optional[float] = None
        self._last_speech_time: Optional[float] = None
        self._total_samples = 0

        # Silence detection task
        self._silence_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """
        Initialize OpenAI client

        Returns:
            True if connected successfully
        """
        try:
            self._status = STTStatus.CONNECTING
            logger.info("Whisper: Initializing OpenAI client...")

            self._client = AsyncOpenAI(api_key=self.api_key)
            self._transcript_queue = asyncio.Queue()
            self._is_listening = True

            # Clear buffer
            self._audio_buffer = []
            self._buffer_start_time = None
            self._last_speech_time = None
            self._total_samples = 0

            # Start silence detection task
            self._silence_task = asyncio.create_task(self._silence_detector())

            self._status = STTStatus.CONNECTED
            logger.info(f"Whisper: Connected (language={self.language or 'auto-detect'})")
            return True

        except Exception as e:
            self._status = STTStatus.ERROR
            import traceback
            logger.error(f"Whisper: Connection error: {e}")
            logger.error(f"Whisper: Traceback:\n{traceback.format_exc()}")
            return False

    async def disconnect(self) -> None:
        """Close connection and transcribe any remaining audio"""
        self._is_listening = False

        # Cancel silence detection task
        if self._silence_task:
            self._silence_task.cancel()
            try:
                await self._silence_task
            except asyncio.CancelledError:
                pass

        # Transcribe any remaining audio
        if self._audio_buffer and self._total_samples > 0:
            logger.info("Whisper: Transcribing final audio buffer...")
            await self._transcribe_buffer(is_final=True)

        self._status = STTStatus.DISCONNECTED
        logger.info("Whisper: Disconnected")

    async def stream_audio(self, audio_chunk: AudioChunk) -> None:
        """
        Stream audio chunk to buffer

        Args:
            audio_chunk: Audio data (typically 8kHz μ-law from Twilio)
        """
        if self._status != STTStatus.CONNECTED or not self._is_listening:
            return

        try:
            # Store chunk for later transcription
            self._audio_buffer.append(audio_chunk.data)
            self._total_samples += len(audio_chunk.data)  # μ-law = 8-bit = 1 byte per sample

            # Track timing
            import time
            now = time.monotonic()
            if self._buffer_start_time is None:
                self._buffer_start_time = now
                self._last_speech_time = now

            # Simple energy-based speech detection
            # Convert μ-law to PCM for energy calculation
            if self._has_speech_energy(audio_chunk.data):
                self._last_speech_time = now

            # Debug logging
            if not hasattr(self, '_chunk_count'):
                self._chunk_count = 0
            self._chunk_count += 1
            if self._chunk_count == 1 or self._chunk_count % 100 == 0:
                duration = self._total_samples / 8000  # 8kHz sample rate
                logger.info(f"Whisper: Buffered {self._chunk_count} chunks, {duration:.2f}s audio")

        except Exception as e:
            logger.error(f"Whisper: Error buffering audio: {e}")
            import traceback
            logger.error(f"Whisper: Traceback:\n{traceback.format_exc()}")

    async def get_transcript(self) -> AsyncIterator[STTResult]:
        """
        Get transcription results as they arrive

        Yields:
            STTResult objects
        """
        if self._transcript_queue is None:
            return

        while self._is_listening:
            try:
                result = await asyncio.wait_for(
                    self._transcript_queue.get(),
                    timeout=1.0
                )
                yield result

                # Emit to callbacks for final results
                if result.is_final:
                    await self._emit_transcript(result)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Whisper: Error getting transcript: {e}")
                break

    async def detect_language(self, audio_data: bytes) -> str:
        """
        Detect language from audio sample

        Args:
            audio_data: Audio sample

        Returns:
            Detected language code
        """
        # Whisper auto-detects language during transcription
        logger.info("Whisper: Language detection is automatic during transcription")
        return self.language or "en"

    def _has_speech_energy(self, audio_data: bytes, threshold: float = None) -> bool:
        """
        Speech energy detection using proper μ-law decoding

        Args:
            audio_data: μ-law encoded audio
            threshold: Energy threshold (0-1, relative to max amplitude)

        Returns:
            True if speech energy detected
        """
        if threshold is None:
            threshold = self.silence_threshold

        try:
            # Decode μ-law to get actual sample values
            # Use a simpler decode table for performance
            energy = 0
            sample_count = len(audio_data)

            for byte in audio_data:
                # Proper μ-law decode (simplified)
                mu = 255 - byte
                magnitude = ((mu & 0x0F) << 3) + 0x84
                exponent = (mu & 0x70) >> 4
                if exponent > 0:
                    magnitude = magnitude << (exponent - 1)
                if mu & 0x80:
                    magnitude = -magnitude
                # Normalize to 0-1 range (max is ~32000)
                energy += abs(magnitude) / 32000.0

            avg_energy = energy / sample_count if sample_count > 0 else 0

            # Log periodically for debugging
            if not hasattr(self, '_energy_log_count'):
                self._energy_log_count = 0
            self._energy_log_count += 1
            if self._energy_log_count == 1 or self._energy_log_count % 200 == 0:
                logger.info(f"Whisper: Energy={avg_energy:.4f}, threshold={threshold}, has_speech={avg_energy > threshold}")

            return avg_energy > threshold

        except Exception:
            # If energy detection fails, assume speech
            return True

    async def _silence_detector(self):
        """
        Background task that detects silence and triggers transcription

        Runs continuously while listening, checking for speech pauses.
        """
        import time

        while self._is_listening:
            try:
                await asyncio.sleep(0.1)  # Check every 100ms

                now = time.monotonic()

                # Check if we should transcribe
                if self._last_speech_time is None:
                    continue

                silence_duration = now - self._last_speech_time

                # Calculate buffer duration
                buffer_duration = self._total_samples / 8000  # 8kHz

                # Trigger transcription if:
                # 1. Enough silence detected AND
                # 2. Minimum audio length reached
                if (silence_duration >= self.silence_duration and
                    buffer_duration >= self.min_audio_length):

                    logger.info(f"Whisper: Silence detected ({silence_duration:.1f}s), buffer={buffer_duration:.1f}s, transcribing...")
                    await self._transcribe_buffer()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Whisper: Error in silence detector: {e}")

    async def _transcribe_buffer(self, is_final: bool = False):
        """
        Transcribe accumulated audio buffer

        Args:
            is_final: True if this is the final transcription (call ending)
        """
        if not self._audio_buffer or self._total_samples == 0:
            return

        # Get audio data
        audio_data = b''.join(self._audio_buffer)
        duration = self._total_samples / 8000

        logger.info(f"Whisper: Transcribing {duration:.2f}s audio ({len(audio_data)} bytes)")

        # Clear buffer
        self._audio_buffer = []
        self._total_samples = 0
        self._last_speech_time = None
        self._buffer_start_time = None

        try:
            # Convert μ-law to WAV format
            wav_data = self._mulaw_to_wav(audio_data, sample_rate=8000)

            # Transcribe with Whisper
            import time
            start_time = time.time()

            transcription = await self._client.audio.transcriptions.create(
                model=self.model,
                file=("audio.wav", io.BytesIO(wav_data), "audio/wav"),
                language=None if is_final else (self.language or None),  # Auto-detect if not set
                response_format="verbose_json"
            )

            elapsed = time.time() - start_time

            # Extract results
            text = transcription.text.strip()
            detected_language = getattr(transcription, 'language', self.language or 'en')
            confidence = self._calculate_confidence(transcription)

            if text:
                logger.info(f"Whisper: Transcription [{detected_language}] ({elapsed:.2f}s): {text}")

                result = STTResult(
                    text=text,
                    language=detected_language,
                    confidence=confidence,
                    is_final=True,  # Whisper always returns final results
                    alternatives=[],
                    metadata={
                        'duration': duration,
                        'processing_time': elapsed,
                        'model': self.model,
                        'words': getattr(transcription, 'words', [])
                    }
                )

                # Put in queue
                if self._transcript_queue:
                    self._transcript_queue.put_nowait(result)
            else:
                logger.debug("Whisper: Empty transcription result")

        except Exception as e:
            logger.error(f"Whisper: Transcription error: {e}")
            import traceback
            logger.error(f"Whisper: Traceback:\n{traceback.format_exc()}")

    def _mulaw_to_wav(self, mulaw_data: bytes, sample_rate: int = 8000) -> bytes:
        """
        Convert μ-law audio to WAV format

        Args:
            mulaw_data: μ-law encoded audio bytes
            sample_rate: Sample rate in Hz

        Returns:
            WAV format audio bytes
        """
        # Create WAV in memory
        output = io.BytesIO()

        with wave.open(output, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)

            # Convert μ-law to 16-bit PCM
            pcm_data = self._mulaw_to_pcm(mulaw_data)
            wav_file.writeframes(pcm_data)

        return output.getvalue()

    def _mulaw_to_pcm(self, mulaw_data: bytes) -> bytes:
        """
        Convert μ-law encoded bytes to 16-bit PCM

        Args:
            mulaw_data: μ-law encoded bytes

        Returns:
            16-bit PCM bytes
        """
        # μ-law decoding table (simplified)
        BIAS = 0x84
        CLIP = 32635

        # Build decode table
        mu_law_decode_table = []
        for i in range(256):
            mu = 255 - i
            magnitude = (mu & 0x0F) << 3
            exponent = (mu & 0x70) >> 4
            sign = (mu & 0x80) >> 7

            if exponent > 0:
                magnitude += 0x84
                magnitude <<= (exponent - 1)

            if sign == 1:
                magnitude = -magnitude

            mu_law_decode_table.append(magnitude)

        # Convert each byte
        pcm_samples = []
        for byte in mulaw_data:
            sample = mu_law_decode_table[byte]
            # Clamp to 16-bit range
            sample = max(-32768, min(32767, sample))
            # Convert to little-endian bytes
            pcm_samples.append(sample & 0xFF)
            pcm_samples.append((sample >> 8) & 0xFF)

        return bytes(pcm_samples)

    def _calculate_confidence(self, transcription) -> float:
        """
        Calculate confidence from transcription

        Args:
            transcription: OpenAI transcription response

        Returns:
            Confidence score (0-1)
        """
        # Whisper doesn't provide per-word confidence in standard API
        # Use avg(logprob) if available
        if hasattr(transcription, 'avg_logprob'):
            # Convert logprob to confidence (rough approximation)
            return max(0.0, min(1.0, transcription.avg_logprob + 1.0))

        return 0.85  # Default confidence


# Factory function for easy instantiation
def create_whisper_stt(config: dict) -> WhisperSTT:
    """
    Factory function to create Whisper STT service from config

    Args:
        config: Configuration dictionary (from Settings)

    Returns:
        Configured WhisperSTT instance
    """
    return WhisperSTT(
        api_key=config.get('openai_api_key'),
        language=config.get('whisper_language', None),  # None = auto-detect
        model=config.get('whisper_model', 'whisper-1'),
        silence_threshold=config.get('whisper_silence_threshold', 0.3),
        silence_duration=config.get('whisper_silence_duration', 1.0),
        min_audio_length=config.get('whisper_min_audio_length', 0.5)
    )
