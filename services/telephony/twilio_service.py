"""
=====================================================
AI Voice Platform v2 - Twilio Media Streams Service
=====================================================
Handles Twilio Media Streams for real-time bidirectional audio
"""

import asyncio
import json
import base64
from typing import Callable, Optional, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
import httpx


class TwilioMediaStreamHandler:
    """
    Handles Twilio Media Streams WebSocket connection

    This service manages the WebSocket connection from Twilio,
    converting between Twilio's audio format and our internal format.
    """

    # Twilio Media Streams uses μ-law 8kHz mono
    SAMPLE_RATE = 8000
    CHANNELS = 1

    # Twilio media event types
    EVENT_CONNECTED = "connected"
    EVENT_START = "start"
    EVENT_MEDIA = "media"
    EVENT_STOP = "stop"
    EVENT_DISCONNECTED = "disconnected"
    EVENT_ERROR = "error"

    def __init__(self, call_sid: str | None, stream_sid: str, websocket: WebSocket):
        """
        Initialize media stream handler

        Args:
            call_sid: Twilio call SID
            stream_sid: Twilio Media Stream SID (from start event)
            websocket: WebSocket connection from Twilio
        """
        self.call_sid = call_sid or "unknown"
        self.stream_sid = stream_sid
        self.websocket = websocket

        # Event handlers
        self._on_media_received: Optional[Callable] = None
        self._on_call_ended: Optional[Callable] = None

        # State
        self._is_connected = False
        self._is_streaming = False
        self._interrupted = False  # Set to True to abort current audio playback

        # Queue for audio to send (consumed by event loop)
        # Initialize immediately so audio can be queued before handle_connection() starts
        self._audio_queue: asyncio.Queue = asyncio.Queue()

    async def handle_connection(self) -> None:
        """
        Handle the WebSocket connection lifecycle

        This method processes incoming events from Twilio and sends queued audio.
        Runs both tasks concurrently: receiving messages and sending audio.
        """
        self._is_connected = True
        logger.info(f"Twilio: Media stream connected for call {self.call_sid}, starting event loop...")

        # Create tasks for receiving and sending
        receive_task = asyncio.create_task(self._receive_messages())
        send_task = asyncio.create_task(self._send_queued_audio())

        try:
            # Wait for either task to complete (usually connection ends)
            done, pending = await asyncio.wait(
                [receive_task, send_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except WebSocketDisconnect:
            logger.info(f"Twilio: WebSocket disconnected for call {self.call_sid}")
        except Exception as e:
            logger.error(f"Twilio: Error in connection: {e}")
            import traceback
            logger.error(f"Twilio: Traceback:\n{traceback.format_exc()}")
        finally:
            logger.info(f"Twilio: handle_connection() finally block for call {self.call_sid}")
            await self.cleanup()

    async def _receive_messages(self) -> None:
        """Receive and process incoming messages from Twilio"""
        try:
            loop_count = 0
            while self._is_connected:
                loop_count += 1
                if loop_count % 50 == 0:  # Log every 5 seconds
                    logger.debug(f"Twilio: Receive loop running ({loop_count} iterations)")

                # Wait for message
                message = await self.websocket.receive_text()

                # Log non-media events (media events are too verbose)
                data = json.loads(message)
                if data.get("event") != self.EVENT_MEDIA:
                    logger.info(f"Twilio: Received message: {message[:150] if len(message) > 150 else message}")

                await self._process_message(message)

        except WebSocketDisconnect:
            logger.info(f"Twilio: Receive loop: WebSocket disconnected")
            self._is_connected = False
        except Exception as e:
            logger.error(f"Twilio: Receive loop error: {e}")
            self._is_connected = False

    async def _send_queued_audio(self) -> None:
        """Send queued audio to Twilio"""
        try:
            while self._is_connected:
                # Wait for audio to send (with timeout to allow checking _is_connected)
                try:
                    audio_data = await asyncio.wait_for(
                        self._audio_queue.get(),
                        timeout=0.5
                    )
                    await self._send_audio_chunked(audio_data)
                except asyncio.TimeoutError:
                    # No audio to send, continue loop
                    continue

        except Exception as e:
            logger.error(f"Twilio: Send loop error: {e}")
            import traceback
            logger.error(f"Twilio: Traceback:\n{traceback.format_exc()}")

    async def _process_message(self, message: str) -> None:
        """
        Process incoming message from Twilio

        Args:
            message: JSON message from Twilio
        """
        try:
            data = json.loads(message)
            event = data.get("event")

            # Log ALL events for debugging
            if event == self.EVENT_MEDIA:
                # For media events, just log the payload size
                payload_size = len(data.get("media", {}).get("payload", ""))
                logger.debug(f"Twilio: Received media event - payload: {payload_size} bytes base64")
            else:
                logger.info(f"Twilio: Received event: {event} - {data}")

            if event == self.EVENT_CONNECTED:
                await self._on_connected(data)
            elif event == self.EVENT_START:
                self._is_streaming = True
                await self._on_start(data)
            elif event == self.EVENT_MEDIA:
                await self._on_media(data)
            elif event == self.EVENT_STOP:
                self._is_streaming = False
                await self._on_stop(data)
            elif event == self.EVENT_DISCONNECTED:
                self._is_connected = False
                await self._on_disconnected(data)
            else:
                logger.debug(f"Twilio: Unknown event: {event}")

        except json.JSONDecodeError:
            logger.warning(f"Twilio: Invalid JSON: {message[:100]}")

    async def _on_connected(self, data: dict) -> None:
        """Handle connected event"""
        logger.info(f"Twilio: Connected event: {data}")

    async def _on_start(self, data: dict) -> None:
        """Handle start event - call started streaming"""
        # Extract CallSid from start event if not already set
        # CallSid is nested in data['start']['callSid']
        if not self.call_sid or self.call_sid == "unknown":
            start_data = data.get("start", {})
            self.call_sid = start_data.get("callSid", "unknown")
            # Also store streamSid for reference
            self.stream_sid = start_data.get("streamSid", "")

        logger.info(f"Twilio: Start event for call {self.call_sid}")
        logger.info(f"Twilio: Full start event data: {data}")

        # Send clear message
        await self.send_event("clear")

    async def _on_media(self, data: dict) -> None:
        """
        Handle media event - incoming audio

        Args:
            data: Media data with base64 encoded audio
        """
        if not self._is_streaming:
            logger.debug(f"Twilio: _on_media skipping - not streaming")
            return

        # Extract base64 encoded μ-law audio
        media_payload = data.get("media", {})
        raw_audio = media_payload.get("payload")

        if raw_audio:
            # Decode base64
            audio_data = base64.b64decode(raw_audio)

            # Pass to registered handler
            if self._on_media_received:
                await self._on_media_received(audio_data)
            else:
                logger.warning(f"Twilio: _on_media - no handler registered!")

    async def _on_stop(self, data: dict) -> None:
        """Handle stop event - call stopped streaming"""
        logger.info(f"Twilio: Stop event for call {self.call_sid}")

        # Send clear message
        await self.send_event("clear")

    async def _on_disconnected(self, data: dict) -> None:
        """Handle disconnected event"""
        logger.info(f"Twilio: Disconnected event for call {self.call_sid}")

        # Notify registered handler
        if self._on_call_ended:
            await self._on_call_ended(self.call_sid)

    async def send_audio(self, audio_data: bytes) -> None:
        """
        Send audio to Twilio (play to caller)

        Audio is queued and sent from the event loop's send task.
        NOTE: This clears any pending audio before sending new audio (interruption).

        Args:
            audio_data: Audio data (μ-law 8kHz for Twilio)
        """
        if not self._is_streaming:
            logger.warning(f"Twilio: Cannot send audio - not streaming")
            return

        if not self.stream_sid:
            logger.error(f"Twilio: Cannot send audio - stream_sid is empty!")
            return

        if self._audio_queue is None:
            logger.error(f"Twilio: Cannot send audio - queue not initialized!")
            return

        # CRITICAL: Clear any pending audio before sending new audio
        # This prevents old audio from playing after new audio (repeating bug)
        cleared = 0
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break

        if cleared > 0:
            logger.info(f"Twilio: Cleared {cleared} pending audio(s) from queue")

        # Send clear event to Twilio to stop any currently playing audio
        await self.send_event("clear")

        # Put new audio in queue - will be sent by send task
        await self._audio_queue.put(audio_data)
        logger.info(f"Twilio: Queued {len(audio_data)} bytes audio for sending")

    async def clear_audio(self) -> None:
        """
        Clear all pending and currently playing audio (for barge-in).
        Sends a 'clear' event to Twilio to stop playback immediately.
        """
        self._interrupted = True

        # Drain the queue
        cleared = 0
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break

        # Tell Twilio to stop playing any buffered audio
        try:
            await self.send_event("clear")
        except Exception as e:
            logger.warning(f"Twilio: Failed to send clear event: {e}")

        if cleared > 0:
            logger.info(f"Twilio: Barge-in cleared {cleared} queued audio(s)")
        logger.info(f"Twilio: Audio playback interrupted")

    async def stream_audio_chunks(self, audio_stream) -> int:
        """
        Stream audio chunks directly to Twilio WebSocket in real-time.

        Instead of waiting for full audio, this sends 160-byte (20ms) chunks
        to Twilio as they arrive from TTS. The caller starts hearing audio
        within ~200ms of the TTS request, not after the full synthesis.

        Args:
            audio_stream: Async iterator yielding raw ulaw_8000 audio bytes

        Returns:
            Total bytes sent to Twilio
        """
        if not self._is_streaming or not self.stream_sid:
            logger.warning("Twilio: Cannot stream audio - not streaming or no stream_sid")
            return 0

        self._interrupted = False

        # Clear any pending audio and stop current playback
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        await self.send_event("clear")

        CHUNK_SIZE = 160  # 20ms at 8kHz ulaw
        CHUNK_INTERVAL = 0.015  # Send slightly faster than real-time
        buffer = bytearray()
        total_bytes = 0
        chunks_sent = 0

        try:
            async for audio_data in audio_stream:
                if self._interrupted:
                    logger.info(f"Twilio: Stream interrupted by barge-in at {total_bytes} bytes")
                    await self.send_event("clear")
                    break

                buffer.extend(audio_data)

                # Send complete 160-byte chunks as they accumulate
                while len(buffer) >= CHUNK_SIZE:
                    if self._interrupted:
                        break

                    chunk = bytes(buffer[:CHUNK_SIZE])
                    buffer = bytearray(buffer[CHUNK_SIZE:])

                    payload = base64.b64encode(chunk).decode("utf-8")
                    media_event = {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {
                            "payload": payload
                        }
                    }
                    await self.websocket.send_json(media_event)
                    total_bytes += CHUNK_SIZE
                    chunks_sent += 1

                    if chunks_sent == 1:
                        logger.info(f"Twilio: First audio chunk streamed to caller")
                    elif chunks_sent % 50 == 0:
                        logger.info(f"Twilio: Streamed {chunks_sent} chunks ({total_bytes} bytes)")

                    await asyncio.sleep(CHUNK_INTERVAL)

            # Send any remaining bytes in the buffer
            if buffer and not self._interrupted:
                payload = base64.b64encode(bytes(buffer)).decode("utf-8")
                media_event = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {
                        "payload": payload
                    }
                }
                await self.websocket.send_json(media_event)
                total_bytes += len(buffer)

            logger.info(f"Twilio: Stream complete - {total_bytes} bytes in {chunks_sent} chunks")
            return total_bytes

        except Exception as e:
            logger.error(f"Twilio: Stream error: {e}")
            import traceback
            logger.error(f"Twilio: Traceback:\n{traceback.format_exc()}")
            return total_bytes

    async def _send_audio_chunked(self, audio_data: bytes) -> None:
        """
        Send audio to Twilio with proper chunking

        Twilio Media Streams requires audio in 20ms chunks (160 bytes at 8kHz μ-law).
        Sending larger payloads can cause playback issues.
        Checks _interrupted flag each chunk to allow barge-in to abort playback.

        Args:
            audio_data: Audio data (μ-law 8kHz for Twilio)
        """
        total_bytes = len(audio_data)
        self._interrupted = False  # Reset at start of new playback

        # Twilio Media Streams: 20ms = 160 bytes at 8kHz μ-law
        CHUNK_SIZE = 160  # 20ms at 8kHz μ-law
        CHUNK_INTERVAL = 0.015  # Send every 15ms (slightly faster than real-time)

        logger.info(f"Twilio: Sending {total_bytes} bytes audio in {CHUNK_SIZE}-byte chunks (streamSid={self.stream_sid})")

        try:
            # Note: clear event is already sent in send_audio() before queuing
            # No need to send it again here

            # Calculate number of chunks
            num_chunks = (len(audio_data) + CHUNK_SIZE - 1) // CHUNK_SIZE
            logger.info(f"Twilio: Will send {num_chunks} chunks of {CHUNK_SIZE} bytes each")

            # Send each chunk
            for i in range(num_chunks):
                # Check if interrupted by barge-in
                if self._interrupted:
                    logger.info(f"Twilio: Audio playback interrupted at chunk {i+1}/{num_chunks} (barge-in)")
                    await self.send_event("clear")
                    return

                start = i * CHUNK_SIZE
                end = min(start + CHUNK_SIZE, len(audio_data))
                chunk = audio_data[start:end]

                # Encode chunk to base64
                payload = base64.b64encode(chunk).decode("utf-8")

                # Create media event for this chunk
                # Per Twilio docs: media events sent TO Twilio should NOT include 'track' field
                # The 'track' field is only present in media events FROM Twilio
                media_event = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {
                        "payload": payload
                    }
                }

                # Send the chunk
                await self.websocket.send_json(media_event)

                # Log progress every 50 chunks
                if (i + 1) % 50 == 0 or i == 0:
                    logger.info(f"Twilio: Sent chunk {i+1}/{num_chunks} ({len(chunk)} bytes)")

                # Small delay between chunks (except last one)
                if i < num_chunks - 1:
                    await asyncio.sleep(CHUNK_INTERVAL)

            logger.info(f"Twilio: Audio sent successfully - {total_bytes} bytes in {num_chunks} chunks")

        except Exception as e:
            logger.error(f"Twilio: Failed to send audio: {e}")
            import traceback
            logger.error(f"Twilio: Traceback:\n{traceback.format_exc()}")
            raise  # Re-raise so caller knows send failed

    async def send_event(self, event: str, **kwargs) -> None:
        """
        Send control event to Twilio

        Args:
            event: Event type
            **kwargs: Additional event data
        """
        message = {
            "event": event,
            "streamSid": self.stream_sid,
            **kwargs
        }

        await self.websocket.send_json(message)

    def set_media_handler(self, handler: Callable[[bytes], Any]) -> None:
        """Set handler for incoming audio"""
        self._on_media_received = handler

    def set_call_ended_handler(self, handler: Callable[[str], Any]) -> None:
        """Set handler for call ended"""
        self._on_call_ended = handler

    async def cleanup(self) -> None:
        """Cleanup resources"""
        self._is_connected = False
        self._is_streaming = False
        logger.info(f"Twilio: Cleaned up handler for call {self.call_sid}")


class TwilioService:
    """
    Twilio service for managing calls and Media Streams

    Handles Twilio REST API calls for call control.
    """

    def __init__(self, account_sid: str, auth_token: str, phone_number: str):
        """
        Initialize Twilio service

        Args:
            account_sid: Twilio Account SID
            auth_token: Twilio Auth Token
            phone_number: Twilio phone number
        """
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.phone_number = phone_number
        self._base_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}"
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                auth=(self.account_sid, self.auth_token),
                timeout=30.0
            )
        return self._client

    async def generate_twiml(self, websocket_url: str, caller_number: str = "") -> str:
        """
        Generate TwiML for connecting to Media Stream

        Using <Connect><Stream> for BIDIRECTIONAL audio.
        <Start><Stream> only sends audio TO the server (unidirectional).

        Args:
            websocket_url: WebSocket URL for Media Stream
            caller_number: Caller's phone number to pass as custom parameter

        Returns:
            TwiML as string
        """
        from xml.sax.saxutils import escape as xml_escape

        # Pass caller number as a custom parameter so the WebSocket handler can access it
        # SECURITY: Escape all values to prevent XML injection attacks
        param_tag = ""
        if caller_number:
            # Escape special XML characters: &, <, >, ", '
            safe_caller = xml_escape(caller_number, {'"': '&quot;', "'": '&apos;'})
            param_tag = f'\n            <Parameter name="callerNumber" value="{safe_caller}" />'

        # Also escape the websocket URL in case it contains special characters
        safe_ws_url = xml_escape(websocket_url, {'"': '&quot;', "'": '&apos;'})

        twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{safe_ws_url}">{param_tag}
        </Stream>
    </Connect>
</Response>'''
        return twiml

    async def end_call(self, call_sid: str) -> bool:
        """
        End an active call

        Args:
            call_sid: Call SID to end

        Returns:
            True if successful
        """
        client = await self._get_client()

        try:
            url = f"{self._base_url}/Calls/{call_sid}.json"
            response = await client.post(
                url,
                data={"Status": "completed"}
            )

            if response.status_code == 200:
                logger.info(f"Twilio: Ended call {call_sid}")
                return True
            else:
                logger.error(f"Twilio: Failed to end call: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Twilio: Error ending call: {e}")
            return False

    async def redirect_call(self, call_sid: str, to_number: str) -> bool:
        """
        Redirect call to another number (transfer)

        Args:
            call_sid: Call SID to redirect
            to_number: Phone number to transfer to

        Returns:
            True if successful
        """
        client = await self._get_client()

        try:
            url = f"{self._base_url}/Calls/{call_sid}.json"
            response = await client.post(
                url,
                data={"Url": f"http://demo.twilio.com/docs/voice.xml?To={to_number}"}
            )

            if response.status_code == 200:
                logger.info(f"Twilio: Redirected call {call_sid} to {to_number}")
                return True
            else:
                logger.error(f"Twilio: Failed to redirect: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Twilio: Error redirecting call: {e}")
            return False


# Factory function
def create_twilio_service(config: dict) -> TwilioService:
    """
    Factory function to create Twilio service from config

    Args:
        config: Configuration dictionary

    Returns:
        Configured TwilioService instance
    """
    return TwilioService(
        account_sid=config.get('twilio_account_sid'),
        auth_token=config.get('twilio_auth_token'),
        phone_number=config.get('twilio_phone_number')
    )
