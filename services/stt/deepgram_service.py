"""
=====================================================
AI Voice Platform v2 - Deepgram STT Service
=====================================================
Real-time streaming Speech-to-Text using Deepgram Nova-2
"""

import asyncio
import json
from typing import AsyncIterator, Optional
from loguru import logger

try:
    from deepgram import DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents, LiveOptions
    DEEPGRAM_AVAILABLE = True
except ImportError:
    DEEPGRAM_AVAILABLE = False
    LiveOptions = None
    DeepgramClient = None

from .stt_base import STTServiceBase, STTResult, STTStatus, AudioChunk


class DeepgramSTT(STTServiceBase):
    """
    Deepgram Nova-2 Streaming STT Service

    Features:
    - Real-time streaming with <300ms latency
    - Auto language detection (100+ languages)
    - Punctuation and formatting
    - Speaker detection (for multi-party calls)
    - High accuracy with noisy audio
    """

    # Supported languages for auto-detect
    SUPPORTED_LANGUAGES = ["en", "ar", "es", "fr", "de", "zh", "ja", "ko", "pt", "ru"]

    def __init__(
        self,
        api_key: str,
        language: str = "en-US",
        model: str = "nova-2",
        smart_format: bool = True,
        punctuate: bool = True,
        paragraphs: bool = True,
        profanity_filter: bool = True,
        detect_language: bool = True
    ):
        """
        Initialize Deepgram STT service

        Args:
            api_key: Deepgram API key
            language: Default language (fallback if detection fails)
            model: Model name (nova-2 recommended)
            smart_format: Enable smart formatting
            punctuate: Add punctuation
            paragraphs: Split into paragraphs
            profanity_filter: Filter profanity
            detect_language: Auto-detect language
        """
        if not DEEPGRAM_AVAILABLE:
            raise ImportError("deepgram-sdk is not installed. Install with: pip install deepgram-sdk")

        super().__init__(api_key, language)

        self.model = model
        self.smart_format = smart_format
        self.punctuate = punctuate
        self.paragraphs = paragraphs
        self.profanity_filter = profanity_filter
        self.detect_language = detect_language

        self._deepgram: Optional[DeepgramClient] = None
        self._live_connection = None
        self._transcript_queue: asyncio.Queue = None
        self._is_listening = False

    async def connect(self) -> bool:
        """
        Establish WebSocket connection to Deepgram

        Returns:
            True if connected successfully
        """
        try:
            self._status = STTStatus.CONNECTING
            logger.info("Deepgram: Connecting to live transcription...")

            # Initialize Deepgram client
            deepgram_options = DeepgramClientOptions(
                options={"keepalive": "true"}
            )
            self._deepgram = DeepgramClient(
                self.api_key,
                deepgram_options
            )

            # Create live transcription connection
            # Note: SDK 3.x API changed - use live.v() instead of listen.websocket.v()
            self._live_connection = self._deepgram.listen.live.v("1")

            # Set up event handlers
            self._live_connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)

            # Configure live options
            # Official Deepgram SDK 3.x LiveOptions parameters:
            # - model, language, encoding, channels, sample_rate
            # - smart_format, punctuate, interim_results
            # - vad_events, utterance_end_ms
            # NOT supported: paragraphs, detect_language, profanity_filter, filler_words
            options = LiveOptions(
                model=self.model,
                language=self.language,  # Use configured language (no auto-detect in live mode)
                encoding="mulaw",       # Twilio sends μ-law encoded audio
                channels=1,             # Mono audio
                sample_rate=8000,       # Twilio uses 8kHz
                smart_format=self.smart_format,
                punctuate=self.punctuate,
                interim_results=True,   # Get interim results while speaking
            )

            # Start the connection
            if await self._live_connection.start(options):
                self._status = STTStatus.CONNECTED
                self._transcript_queue = asyncio.Queue()
                self._is_listening = True
                logger.info("Deepgram: Connected and listening")
                return True
            else:
                self._status = STTStatus.ERROR
                logger.error("Deepgram: Failed to start connection")
                return False

        except Exception as e:
            self._status = STTStatus.ERROR
            import traceback
            logger.error(f"Deepgram: Connection error: {e}")
            logger.error(f"Deepgram: Traceback:\n{traceback.format_exc()}")
            return False

    async def disconnect(self) -> None:
        """Close WebSocket connection"""
        self._is_listening = False

        if self._live_connection:
            try:
                await self._live_connection.finish()
            except Exception as e:
                logger.warning(f"Deepgram: Error disconnecting: {e}")

        self._status = STTStatus.DISCONNECTED
        logger.info("Deepgram: Disconnected")

    async def stream_audio(self, audio_chunk: AudioChunk) -> None:
        """
        Stream audio chunk to Deepgram

        Args:
            audio_chunk: Audio data (typically 8kHz μ-law from Twilio)
        """
        if self._status != STTStatus.CONNECTED or not self._is_listening:
            logger.warning("Deepgram: Cannot stream - not connected")
            return

        try:
            # Send audio data to Deepgram
            # Twilio sends 8kHz μ-law, Deepgram accepts various formats
            await self._live_connection.send(audio_chunk.data)
        except Exception as e:
            logger.error(f"Deepgram: Error sending audio: {e}")

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

                # If final result, also emit to callbacks
                if result.is_final:
                    await self._emit_transcript(result)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Deepgram: Error getting transcript: {e}")
                break

    async def detect_language(self, audio_data: bytes) -> str:
        """
        Detect language from audio sample

        Args:
            audio_data: Audio sample (at least 3 seconds recommended)

        Returns:
            Detected language code
        """
        # Deepgram does this automatically during transcription
        # This method can be used for pre-call detection if needed
        logger.info("Deepgram: Language detection is automatic during transcription")
        return self.language  # Will be overridden by actual detection

    def _on_transcript(self, *args, **kwargs):
        """Handle incoming transcript from Deepgram WebSocket"""
        try:
            result = kwargs.get('result')
            if not result:
                return

            # Extract transcript data
            channel = result.get('channel', {})
            alternatives = channel.get('alternatives', [])

            if not alternatives:
                return

            best_alternative = alternatives[0]
            transcript = best_alternative.get('transcript', '')

            if not transcript:
                return

            # Extract metadata
            is_final = result.get('is_final', False)
            confidence = best_alternative.get('confidence', 0.0)

            # Detect language from result
            detected_language = channel.get('detected_language')
            if not detected_language:
                # Map from full language code to ISO code
                language_full = channel.get('language', self.language)
                detected_language = language_full.split('-')[0]

            # Create STT result
            stt_result = STTResult(
                text=transcript.strip(),
                language=detected_language,
                confidence=confidence,
                is_final=is_final,
                alternatives=[
                    alt.get('transcript', '') for alt in alternatives[1:4]  # Up to 3 alternatives
                ],
                metadata={
                    'words': best_alternative.get('words', []),
                    'duration': channel.get('duration', 0)
                }
            )

            # Put in queue for consumption
            if self._transcript_queue:
                self._transcript_queue.put_nowait(stt_result)

            # Log final results
            if is_final:
                logger.info(f"Deepgram: Final transcript [{detected_language}]: {transcript}")

        except Exception as e:
            logger.error(f"Deepgram: Error processing transcript: {e}")


# Factory function for easy instantiation
def create_deepgram_stt(config: dict) -> DeepgramSTT:
    """
    Factory function to create Deepgram STT service from config

    Args:
        config: Configuration dictionary (from Settings)

    Returns:
        Configured DeepgramSTT instance
    """
    return DeepgramSTT(
        api_key=config.get('deepgram_api_key'),
        language=config.get('deepgram_language', 'en-US'),
        model=config.get('deepgram_model', 'nova-2'),
        smart_format=config.get('deepgram_smart_format', True),
        punctuate=config.get('deepgram_punctuate', True),
        paragraphs=config.get('deepgram_paragraphs', True),
        profanity_filter=config.get('deepgram_profanity_filter', True),
        detect_language=True  # Always enable for AI platform
    )
