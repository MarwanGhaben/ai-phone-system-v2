"""
=====================================================
AI Voice Platform v2 - ElevenLabs Scribe STT Service
=====================================================
Real-time Speech-to-Text using ElevenLabs Scribe v2 Realtime API
Supports 90+ languages including Arabic with 150ms latency

AUDIO FORMAT: Accepts raw μ-law 8kHz audio directly from Twilio
Media Streams — no conversion needed. ElevenLabs natively supports
ulaw_8000 format, avoiding spectral artifacts from upsampling.
"""

import asyncio
import base64
import json
from typing import AsyncIterator, Optional, List
from loguru import logger

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None

from .stt_base import STTServiceBase, STTResult, STTStatus, AudioChunk


class ElevenLabsSTT(STTServiceBase):
    """
    ElevenLabs Scribe v2 Realtime STT Service

    Features:
    - Real-time streaming with ~150ms latency
    - 90+ languages including Arabic (ar), English (en)
    - Auto language detection with include_language_detection
    - WebSocket-based for true real-time transcription
    - No hallucination on silence (unlike Whisper)
    - Native μ-law 8kHz support for telephony audio
    """

    # WebSocket endpoint for Scribe v2 Realtime
    WEBSOCKET_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

    def __init__(
        self,
        api_key: str,
        language: str = "",  # Empty = auto-detect
        model: str = "scribe_v2_realtime",
        sample_rate: int = 8000,  # Twilio native rate
    ):
        """
        Initialize ElevenLabs STT service

        Args:
            api_key: ElevenLabs API key
            language: Language code (en, ar, etc.) or empty for auto-detect
            model: Model to use (scribe_v2_realtime)
            sample_rate: Sample rate for audio (8000 for Twilio μ-law)
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets is not installed. Install with: pip install websockets")

        super().__init__(api_key, language)

        self.model = model
        self.sample_rate = sample_rate

        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._transcript_queue: asyncio.Queue = None
        self._is_listening = False
        self._receive_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """
        Establish WebSocket connection to ElevenLabs STT

        Returns:
            True if connected successfully
        """
        try:
            self._status = STTStatus.CONNECTING
            logger.info("ElevenLabs STT: Connecting to Scribe v2 Realtime...")

            # Build WebSocket URL with config as query parameters
            # Use ulaw_8000 to accept raw Twilio audio — no conversion needed
            params = {
                "model_id": self.model,
                "audio_format": "ulaw_8000",
                "sample_rate": "8000",
                "commit_strategy": "vad",
                "vad_silence_threshold_secs": "0.3",
                "include_language_detection": "true",
            }

            # Add language if specified
            if self.language:
                params["language_code"] = self.language

            query_string = "&".join(f"{k}={v}" for k, v in params.items())
            ws_url = f"{self.WEBSOCKET_URL}?{query_string}"

            logger.info(f"ElevenLabs STT: Connecting to {self.WEBSOCKET_URL} with model={self.model}")

            # Connect with API key in header for authentication
            # websockets 12.x uses 'extra_headers', not 'additional_headers'
            self._websocket = await websockets.connect(
                ws_url,
                extra_headers={"xi-api-key": self.api_key},
                ping_interval=30,
                ping_timeout=10,
            )

            # Initialize transcript queue
            self._transcript_queue = asyncio.Queue()
            self._is_listening = True

            # Start receive task (will process session_started message)
            self._receive_task = asyncio.create_task(
                self._receive_loop(),
                name="elevenlabs_stt_receive"
            )

            self._status = STTStatus.CONNECTED
            lang_str = self.language if self.language else "auto-detect"
            logger.info(f"ElevenLabs STT: Connected (language={lang_str}, format=ulaw_8000)")

            return True

        except Exception as e:
            self._status = STTStatus.ERROR
            logger.error(f"ElevenLabs STT: Connection error: {e}")
            return False

    async def reconnect_with_language(self, language_code: str) -> bool:
        """
        Disconnect and reconnect with a specific language code.
        This dramatically improves accuracy vs auto-detect for phone audio.

        Args:
            language_code: Language code (e.g., 'ar', 'en')

        Returns:
            True if reconnected successfully
        """
        logger.info(f"ElevenLabs STT: Reconnecting with language={language_code}")

        # Save queue contents to avoid losing pending transcripts
        pending_results = []
        if self._transcript_queue:
            while not self._transcript_queue.empty():
                try:
                    pending_results.append(self._transcript_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

        # Disconnect current session
        await self.disconnect()

        # Update language
        self.language = language_code

        # Reconnect
        success = await self.connect()

        # Restore pending results
        if success and pending_results:
            for result in pending_results:
                await self._transcript_queue.put(result)

        return success

    async def disconnect(self) -> None:
        """Close WebSocket connection"""
        self._is_listening = False

        # Cancel receive task
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket
        if self._websocket:
            try:
                # Send end of stream message
                await self._websocket.send(json.dumps({"message_type": "end_of_stream"}))
                await self._websocket.close()
            except Exception as e:
                logger.debug(f"ElevenLabs STT: Error closing WebSocket: {e}")
            self._websocket = None

        self._status = STTStatus.DISCONNECTED
        logger.info("ElevenLabs STT: Disconnected")

    async def reset_for_listening(self) -> None:
        """
        Reset STT for a fresh listening session by swapping the WebSocket.

        Called when the AI finishes speaking. During SPEAKING, the AI's voice
        comes back through the phone mic as echo, polluting the STT's VAD buffer.
        Without a reset, the first 1-3 user utterances after AI speech are missed
        or delayed by 5-15 seconds.

        IMPORTANT: This does NOT call disconnect()/connect() because that would
        toggle _is_listening to False, causing get_transcript() to exit and the
        transcript consumer to spin-loop. Instead, we swap the WebSocket and
        receive task while keeping _is_listening=True and the same queue.
        """
        try:
            logger.info(f"ElevenLabs STT: Resetting for listening (language={self.language})")

            # Drain any echo transcripts from queue
            drained = 0
            if self._transcript_queue:
                while not self._transcript_queue.empty():
                    try:
                        self._transcript_queue.get_nowait()
                        drained += 1
                    except asyncio.QueueEmpty:
                        break
            if drained:
                logger.info(f"ElevenLabs STT: Drained {drained} echo transcripts from queue")

            # Cancel old receive task (but do NOT set _is_listening = False)
            old_receive_task = self._receive_task
            self._receive_task = None
            if old_receive_task and not old_receive_task.done():
                old_receive_task.cancel()
                try:
                    await old_receive_task
                except asyncio.CancelledError:
                    pass

            # Close old WebSocket
            old_ws = self._websocket
            self._websocket = None
            if old_ws:
                try:
                    await old_ws.send(json.dumps({"message_type": "end_of_stream"}))
                    await old_ws.close()
                except Exception:
                    pass

            # Build new WebSocket connection (reuse same params)
            params = {
                "model_id": self.model,
                "audio_format": "ulaw_8000",
                "sample_rate": "8000",
                "commit_strategy": "vad",
                "vad_silence_threshold_secs": "0.3",
                "include_language_detection": "true",
            }
            if self.language:
                params["language_code"] = self.language

            query_string = "&".join(f"{k}={v}" for k, v in params.items())
            ws_url = f"{self.WEBSOCKET_URL}?{query_string}"

            self._websocket = await websockets.connect(
                ws_url,
                extra_headers={"xi-api-key": self.api_key},
                ping_interval=30,
                ping_timeout=10,
            )

            # Start new receive task (pushes to the SAME queue)
            # _is_listening was never set to False, so get_transcript() keeps blocking
            self._receive_task = asyncio.create_task(
                self._receive_loop(),
                name="elevenlabs_stt_receive"
            )

            self._status = STTStatus.CONNECTED
            logger.info(f"ElevenLabs STT: Reset complete (language={self.language})")

        except Exception as e:
            logger.warning(f"ElevenLabs STT: Error during reset: {e}")
            # Try full reconnect as fallback
            try:
                await self.disconnect()
                await self.connect()
            except Exception:
                logger.error("ElevenLabs STT: Failed to recover after reset error")

    async def stream_audio(self, audio_chunk: AudioChunk) -> None:
        """
        Stream audio chunk to ElevenLabs STT

        Sends raw μ-law 8kHz audio directly from Twilio — no conversion.
        ElevenLabs natively supports ulaw_8000 format.

        Args:
            audio_chunk: Audio data to transcribe (μ-law 8kHz from Twilio)
        """
        if self._status != STTStatus.CONNECTED or not self._websocket:
            return

        try:
            # Send raw μ-law audio directly — no conversion needed
            message = {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(audio_chunk.data).decode("utf-8"),
            }

            await self._websocket.send(json.dumps(message))

        except websockets.exceptions.ConnectionClosed:
            logger.warning("ElevenLabs STT: WebSocket closed while sending audio")
            self._status = STTStatus.DISCONNECTED
        except Exception as e:
            logger.error(f"ElevenLabs STT: Error streaming audio: {e}")

    async def _receive_loop(self) -> None:
        """Receive and process messages from WebSocket"""
        # Deduplicate: ElevenLabs sends both committed_transcript AND
        # final_transcript for the same utterance. Track last text to skip dupes.
        last_final_text = None

        try:
            while self._is_listening and self._websocket:
                try:
                    message = await asyncio.wait_for(
                        self._websocket.recv(),
                        timeout=60.0  # 1 minute timeout
                    )

                    data = json.loads(message)
                    msg_type = data.get("message_type", data.get("type", ""))

                    if msg_type in ("partial_transcript", "transcript"):
                        # Partial/interim result — log only, don't queue.
                        # Partials are discarded by process_transcript (is_final=False),
                        # so queuing them just adds latency and overhead.
                        text = data.get("text", "")
                        if text:
                            logger.debug(f"ElevenLabs STT: Partial: '{text}'")

                    elif msg_type in ("committed_transcript", "committed_transcript_with_timestamps", "final_transcript"):
                        # Final result — deduplicate (ElevenLabs sends 2 events per utterance)
                        text = data.get("text", "")
                        if text:
                            if text == last_final_text:
                                logger.debug(f"ElevenLabs STT: Skipping duplicate final: '{text}'")
                                continue
                            last_final_text = text

                            detected_lang = self._extract_language(data)
                            result = STTResult(
                                text=text,
                                language=detected_lang,
                                confidence=data.get("confidence", 0.95),
                                is_final=True,
                            )
                            await self._transcript_queue.put(result)
                            logger.info(f"ElevenLabs STT: Final: '{text}' (lang={detected_lang})")

                    elif msg_type == "session_started":
                        session_id = data.get("session_id", "unknown")
                        logger.info(f"ElevenLabs STT: Session started (id={session_id}, config={json.dumps({k: v for k, v in data.items() if k != 'session_id'}, default=str)})")

                    elif msg_type == "error":
                        error_msg = data.get("message", data.get("error", "Unknown error"))
                        logger.error(f"ElevenLabs STT: API error: {error_msg}")

                    elif msg_type == "session_ended":
                        logger.info("ElevenLabs STT: Session ended")
                        break

                    else:
                        logger.debug(f"ElevenLabs STT: Unknown message type '{msg_type}': {json.dumps(data, default=str)[:200]}")

                except asyncio.TimeoutError:
                    # Send keepalive
                    if self._websocket:
                        try:
                            await self._websocket.ping()
                        except Exception:
                            break

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"ElevenLabs STT: WebSocket closed: {e}")
        except asyncio.CancelledError:
            logger.debug("ElevenLabs STT: Receive loop cancelled")
            return  # Don't touch _is_listening — reset_for_listening manages it
        except Exception as e:
            logger.error(f"ElevenLabs STT: Receive loop error: {e}")

    def _extract_language(self, data: dict) -> str:
        """
        Extract detected language from STT response.

        With include_language_detection=true, the API returns language info
        in the response. Falls back to configured language or 'auto'.

        Args:
            data: Response message from ElevenLabs

        Returns:
            Language code (e.g., 'ar', 'en')
        """
        # Try language_code field directly
        lang = data.get("language_code", "")
        if lang:
            return lang

        # Try nested language_detection object
        lang_detection = data.get("language_detection", {})
        if isinstance(lang_detection, dict):
            lang = lang_detection.get("language_code", "")
            if lang:
                return lang

        # Fall back to configured language
        return self.language or "auto"

    async def get_transcript(self) -> AsyncIterator[STTResult]:
        """
        Get transcription results as they arrive

        Yields:
            STTResult objects with transcribed text
        """
        if not self._transcript_queue:
            return

        while self._is_listening or not self._transcript_queue.empty():
            try:
                result = await asyncio.wait_for(
                    self._transcript_queue.get(),
                    timeout=0.1
                )
                yield result
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def detect_language(self, audio_data: bytes) -> str:
        """
        Detect language from audio

        Note: ElevenLabs Scribe auto-detects language, so this returns
        the detected language from the most recent transcription or 'auto'.

        Args:
            audio_data: Audio sample for detection

        Returns:
            Detected language code
        """
        # ElevenLabs auto-detects, return configured or auto
        return self.language if self.language else "auto"


def create_elevenlabs_stt(config: dict) -> ElevenLabsSTT:
    """
    Factory function to create ElevenLabs STT service from config

    Args:
        config: Configuration dictionary (from Settings)

    Returns:
        Configured ElevenLabsSTT instance
    """
    if not WEBSOCKETS_AVAILABLE:
        raise ImportError("websockets is not installed. Install with: pip install websockets")

    # Get language setting
    language = config.get('elevenlabs_stt_language', '')
    if language == 'auto':
        language = ''  # Empty = auto-detect

    return ElevenLabsSTT(
        api_key=config.get('elevenlabs_api_key'),
        language=language,
        model=config.get('elevenlabs_stt_model', 'scribe_v2_realtime'),
        sample_rate=8000,  # Native Twilio μ-law rate
    )
