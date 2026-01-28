"""
=====================================================
AI Voice Platform v2 - ElevenLabs Scribe STT Service
=====================================================
Real-time Speech-to-Text using ElevenLabs Scribe v2 Realtime API
Supports 90+ languages including Arabic with 150ms latency
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
    - Auto language detection
    - WebSocket-based for true real-time transcription
    - No hallucination on silence (unlike Whisper)
    """

    # WebSocket endpoint for Scribe v2 Realtime
    WEBSOCKET_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

    def __init__(
        self,
        api_key: str,
        language: str = "",  # Empty = auto-detect
        model: str = "scribe_v2_realtime",
        sample_rate: int = 16000,  # Target sample rate for the API
    ):
        """
        Initialize ElevenLabs STT service

        Args:
            api_key: ElevenLabs API key
            language: Language code (en, ar, etc.) or empty for auto-detect
            model: Model to use (scribe_v2_realtime)
            sample_rate: Sample rate for audio (16000 recommended)
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

        # Audio buffer for resampling (Twilio sends 8kHz, API wants 16kHz)
        self._audio_buffer: bytearray = bytearray()
        self._input_sample_rate = 8000  # Twilio default

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
            # ElevenLabs Scribe Realtime uses query params for session config
            params = {
                "model_id": self.model,
                "audio_format": "pcm_16000",
                "sample_rate": str(self.sample_rate),
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
            logger.info(f"ElevenLabs STT: Connected (language={lang_str})")

            return True

        except Exception as e:
            self._status = STTStatus.ERROR
            logger.error(f"ElevenLabs STT: Connection error: {e}")
            return False

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

    async def stream_audio(self, audio_chunk: AudioChunk) -> None:
        """
        Stream audio chunk to ElevenLabs STT

        Args:
            audio_chunk: Audio data to transcribe (μ-law 8kHz from Twilio)
        """
        if self._status != STTStatus.CONNECTED or not self._websocket:
            return

        try:
            # Convert μ-law 8kHz to PCM 16kHz
            pcm_audio = self._convert_mulaw_to_pcm16(audio_chunk.data)

            # Send audio chunk with message_type field per ElevenLabs API
            message = {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(pcm_audio).decode("utf-8"),
                "sample_rate": self.sample_rate,
            }

            await self._websocket.send(json.dumps(message))

        except websockets.exceptions.ConnectionClosed:
            logger.warning("ElevenLabs STT: WebSocket closed while sending audio")
            self._status = STTStatus.DISCONNECTED
        except Exception as e:
            logger.error(f"ElevenLabs STT: Error streaming audio: {e}")

    def _convert_mulaw_to_pcm16(self, mulaw_data: bytes) -> bytes:
        """
        Convert μ-law 8kHz audio to PCM 16-bit 16kHz

        Args:
            mulaw_data: μ-law encoded audio at 8kHz

        Returns:
            PCM 16-bit audio at 16kHz
        """
        # μ-law to linear decode table
        MULAW_DECODE = [
            -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
            -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
            -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
            -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
            -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
            -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
            -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
            -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
            -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
            -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
            -876, -844, -812, -780, -748, -716, -684, -652,
            -620, -588, -556, -524, -492, -460, -428, -396,
            -372, -356, -340, -324, -308, -292, -276, -260,
            -244, -228, -212, -196, -180, -164, -148, -132,
            -120, -112, -104, -96, -88, -80, -72, -64,
            -56, -48, -40, -32, -24, -16, -8, 0,
            32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
            23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
            15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
            11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
            7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
            5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
            3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
            2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
            1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
            1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
            876, 844, 812, 780, 748, 716, 684, 652,
            620, 588, 556, 524, 492, 460, 428, 396,
            372, 356, 340, 324, 308, 292, 276, 260,
            244, 228, 212, 196, 180, 164, 148, 132,
            120, 112, 104, 96, 88, 80, 72, 64,
            56, 48, 40, 32, 24, 16, 8, 0,
        ]

        # Decode μ-law to 16-bit PCM
        pcm_samples = []
        for byte in mulaw_data:
            pcm_samples.append(MULAW_DECODE[byte])

        # Simple 2x upsampling (8kHz -> 16kHz) using linear interpolation
        upsampled = []
        for i in range(len(pcm_samples)):
            upsampled.append(pcm_samples[i])
            if i < len(pcm_samples) - 1:
                # Interpolate between samples
                upsampled.append((pcm_samples[i] + pcm_samples[i + 1]) // 2)
            else:
                upsampled.append(pcm_samples[i])

        # Convert to bytes (16-bit little-endian)
        pcm_bytes = bytearray()
        for sample in upsampled:
            # Clamp to 16-bit range
            sample = max(-32768, min(32767, sample))
            pcm_bytes.extend(sample.to_bytes(2, byteorder='little', signed=True))

        return bytes(pcm_bytes)

    async def _receive_loop(self) -> None:
        """Receive and process messages from WebSocket"""
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
                        # Partial/interim result
                        text = data.get("text", "")
                        if text:
                            result = STTResult(
                                text=text,
                                language=data.get("language_code", self.language or "en"),
                                confidence=data.get("confidence", 0.9),
                                is_final=False,
                            )
                            await self._transcript_queue.put(result)
                            logger.debug(f"ElevenLabs STT: Partial: '{text}'")

                    elif msg_type in ("committed_transcript", "committed_transcript_with_timestamps", "final_transcript"):
                        # Final result
                        text = data.get("text", "")
                        if text:
                            result = STTResult(
                                text=text,
                                language=data.get("language_code", self.language or "en"),
                                confidence=data.get("confidence", 0.95),
                                is_final=True,
                            )
                            await self._transcript_queue.put(result)
                            logger.info(f"ElevenLabs STT: Final: '{text}'")

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
        except Exception as e:
            logger.error(f"ElevenLabs STT: Receive loop error: {e}")
        finally:
            self._is_listening = False

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
                    timeout=0.5
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
        sample_rate=config.get('elevenlabs_stt_sample_rate', 16000),
    )
