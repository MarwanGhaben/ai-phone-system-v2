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

            # Set up event handlers - DEBUG: log all possible events
            logger.info(f"Deepgram: Setting up event handlers...")
            logger.info(f"Deepgram: LiveTranscriptionEvents available: {[e for e in dir(LiveTranscriptionEvents) if not e.startswith('_')]}")

            # Register transcript handler
            self._live_connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
            logger.info(f"Deepgram: Registered Transcript event handler")

            # DEBUG: Register handlers for other events to see what Deepgram sends
            self._live_connection.on(LiveTranscriptionEvents.Open, self._on_open)
            self._live_connection.on(LiveTranscriptionEvents.Close, self._on_close)
            self._live_connection.on(LiveTranscriptionEvents.Error, self._on_error)
            self._live_connection.on(LiveTranscriptionEvents.Warning, self._on_warning)
            self._live_connection.on(LiveTranscriptionEvents.Metadata, self._on_metadata)
            logger.info(f"Deepgram: Registered all event handlers")

            # Configure live options
            # Official Deepgram SDK 3.x LiveOptions parameters:
            # - model, language, encoding, channels, sample_rate
            # - smart_format, punctuate, interim_results
            # - vad_events, utterance_end_ms
            # NOT supported: paragraphs, detect_language, profanity_filter, filler_words

            # Use multilingual mode to support English AND Arabic
            # "mul" enables automatic language detection for 100+ languages
            stt_language = "mul" if self.detect_language else self.language

            options = LiveOptions(
                model=self.model,
                language=stt_language,  # "mul" for multilingual (en, ar, es, etc.)
                encoding="mulaw",       # Twilio sends μ-law encoded audio
                channels=1,             # Mono audio
                sample_rate=8000,       # Twilio uses 8kHz
                smart_format=self.smart_format,
                punctuate=self.punctuate,
                interim_results=True,   # Get interim results while speaking
            )

            # Start the connection (start() returns bool directly, not awaitable)
            logger.info(f"Deepgram: Starting with language={stt_language}, model={self.model}")
            if self._live_connection.start(options):
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
                self._live_connection.finish()  # Synchronous method
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
            # Send audio data to Deepgram (synchronous method)
            # Twilio sends 8kHz μ-law, Deepgram accepts various formats

            # DEBUG: Log audio being sent
            data_size = len(audio_chunk.data) if audio_chunk.data else 0
            if data_size > 0:
                # Only log first chunk and every 50 chunks to avoid spam
                if not hasattr(self, '_chunk_count'):
                    self._chunk_count = 0
                self._chunk_count += 1
                if self._chunk_count == 1 or self._chunk_count % 50 == 0:
                    logger.info(f"Deepgram: Sending audio chunk #{self._chunk_count}, size={data_size} bytes")

            self._live_connection.send(audio_chunk.data)
        except Exception as e:
            logger.error(f"Deepgram: Error sending audio: {e}")
            import traceback
            logger.error(f"Deepgram: Traceback:\n{traceback.format_exc()}")

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
            # Deepgram SDK 3.x passes result via kwargs, not positional args
            # The first arg is the LiveClient itself, not the result
            result = kwargs.get('result')

            if not result:
                logger.warning("Deepgram: _on_transcript called but no result in kwargs")
                logger.debug(f"Deepgram: args={[type(a).__name__ for a in args]}, kwargs_keys={list(kwargs.keys())}")
                return

            # result should be a LiveResultResponse object
            logger.debug(f"Deepgram: Raw result type={type(result).__name__}")

            # Try to get transcript from result object
            # Deepgram SDK 3.x structure: result.channel.alternatives[0].transcript
            if not hasattr(result, 'channel'):
                logger.warning(f"Deepgram: Result type {type(result).__name__} has no 'channel' attribute")
                # Try to see if result has a different structure
                logger.debug(f"Deepgram: Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')][:20]}")
                return

            channel = result.channel
            if not hasattr(channel, 'alternatives') or not channel.alternatives:
                logger.warning("Deepgram: Channel has no alternatives")
                return

            alternatives = channel.alternatives
            best_alternative = alternatives[0]

            if not hasattr(best_alternative, 'transcript'):
                logger.warning("Deepgram: Best alternative has no 'transcript' attribute")
                return

            transcript = best_alternative.transcript
            if not transcript:
                logger.debug(f"Deepgram: Empty transcript received (is_final={getattr(result, 'is_final', False)})")
                return

            # Extract metadata from object attributes
            is_final = getattr(result, 'is_final', False)
            confidence = getattr(best_alternative, 'confidence', 0.0)

            # Detect language from result
            detected_language = getattr(channel, 'detected_language', None)
            if not detected_language:
                # Map from full language code to ISO code
                language_full = getattr(channel, 'language', self.language)
                detected_language = language_full.split('-')[0] if language_full else self.language

            # Log final transcripts
            if is_final:
                logger.info(f"Deepgram: Final transcript [{detected_language}]: {transcript}")
            else:
                logger.debug(f"Deepgram: Interim transcript: {transcript}")

            # Create STT result
            stt_result = STTResult(
                text=transcript.strip(),
                language=detected_language,
                confidence=confidence,
                is_final=is_final,
                alternatives=[
                    alt.transcript for alt in alternatives[1:4] if hasattr(alt, 'transcript')
                ],
                metadata={
                    'words': getattr(best_alternative, 'words', []),
                    'duration': getattr(channel, 'duration', 0)
                }
            )

            # Put in queue for consumption
            if self._transcript_queue:
                self._transcript_queue.put_nowait(stt_result)
            else:
                logger.warning("Deepgram: Transcript queue is None!")

        except Exception as e:
            logger.error(f"Deepgram: Error processing transcript: {e}")
            import traceback
            logger.error(f"Deepgram: Traceback:\n{traceback.format_exc()}")

    def _on_open(self, *args, **kwargs):
        """Handle Deepgram connection opened event"""
        logger.info(f"Deepgram: WebSocket OPENED - args={len(args)}, kwargs={list(kwargs.keys())}")

    def _on_close(self, *args, **kwargs):
        """Handle Deepgram connection closed event"""
        logger.info(f"Deepgram: WebSocket CLOSED - args={len(args)}, kwargs={list(kwargs.keys())}")

    def _on_error(self, *args, **kwargs):
        """Handle Deepgram error event"""
        logger.error(f"Deepgram: WebSocket ERROR - args={len(args)}, kwargs={list(kwargs.keys())}")

    def _on_warning(self, *args, **kwargs):
        """Handle Deepgram warning event"""
        logger.warning(f"Deepgram: WebSocket WARNING - args={len(args)}, kwargs={list(kwargs.keys())}")

    def _on_metadata(self, *args, **kwargs):
        """Handle Deepgram metadata event"""
        logger.info(f"Deepgram: METADATA - args={len(args)}, kwargs={list(kwargs.keys())}")


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
