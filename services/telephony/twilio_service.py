"""
=====================================================
AI Voice Platform v2 - Twilio Media Streams Service
=====================================================
Handles Twilio Media Streams for real-time bidirectional audio
"""

import asyncio
import json
import base64
from dataclasses import dataclass
from uuid import uuid4
from typing import Callable, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
import httpx


@dataclass(eq=False)
class PlaybackOwner:
    """Opaque, permanently retireable playback identity for one handler session."""

    session_id: str
    generation: int
    retired: bool = False


@dataclass(frozen=True)
class PlaybackStreamResult:
    """Transport outcome kept separate from the useful diagnostic byte count."""

    bytes_sent: int
    completed: bool
    error: str | None = None


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
    EVENT_MARK = "mark"
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
        self._send_lock = asyncio.Lock()
        self._handler_session_id = uuid4().hex
        self._playback_generation = 0
        self._active_playback: PlaybackOwner | None = None
        self._playback_closed = False
        self._pending_marks: dict[str, tuple[PlaybackOwner, asyncio.Future]] = {}
        self._send_tasks: set[asyncio.Task] = set()
        self._send_cleanup_complete = asyncio.Event()
        self._send_cleanup_complete.set()

    async def handle_connection(self) -> None:
        """
        Handle the WebSocket connection lifecycle

        This method processes incoming events from Twilio.
        """
        self._is_connected = True
        logger.info(f"Twilio: Media stream connected for call {self.call_sid}, starting event loop...")

        try:
            await self._receive_messages()

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
            elif event == self.EVENT_MARK:
                self._on_mark(data)
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
        self._retire_all_playback()

        # Notify registered handler
        if self._on_call_ended:
            await self._on_call_ended(self.call_sid)

    def _on_mark(self, data: dict) -> None:
        mark_name = data.get("mark", {}).get("name", "")
        pending = self._pending_marks.get(mark_name)
        if not pending:
            return
        owner, playback_wait = pending
        if self.is_current_playback(owner) and not playback_wait.done():
            playback_wait.set_result(True)

    async def begin_playback(self) -> PlaybackOwner:
        """Retire the prior generation and order its clear before new media."""
        if self._playback_closed:
            raise RuntimeError("Twilio playback handler is closed")

        previous = self._active_playback
        if previous is not None:
            previous.retired = True
            self._resolve_pending_marks(False, previous)

        self._playback_generation += 1
        owner = PlaybackOwner(self._handler_session_id, self._playback_generation)
        self._active_playback = owner

        try:
            async with self._send_lock:
                if self.is_current_playback(owner):
                    await self.websocket.send_json({
                        "event": "clear",
                        "streamSid": self.stream_sid,
                    })
        except Exception:
            self.release_playback(owner)
            raise
        return owner

    def is_current_playback(self, owner: PlaybackOwner) -> bool:
        return (
            not self._playback_closed
            and owner.session_id == self._handler_session_id
            and self._active_playback is owner
            and not owner.retired
        )

    def release_playback(self, owner: PlaybackOwner) -> None:
        """Retire an exact completed owner without clearing a replacement."""
        if self._active_playback is owner:
            owner.retired = True
            self._active_playback = None
        self._resolve_pending_marks(False, owner)

    def invalidate_playback(self, owner: PlaybackOwner) -> None:
        """Synchronously invalidate an owner for terminal call teardown."""
        self.release_playback(owner)

    def invalidate_playback_session(self) -> None:
        """Permanently invalidate installed and provisional handler ownership."""
        self._retire_all_playback()

    async def clear_audio(self, owner: PlaybackOwner | None = None) -> bool:
        """
        Clear all pending and currently playing audio (for barge-in).
        Sends a 'clear' event to Twilio to stop playback immediately.
        """
        target = owner or self._active_playback
        if target is None or not self.is_current_playback(target):
            return False

        # Permanent retirement happens before the first await.  A sender already
        # queued on the lock will therefore fail its ownership check when released.
        target.retired = True
        self._active_playback = None
        self._resolve_pending_marks(False, target)

        # Tell Twilio to stop playing any buffered audio
        try:
            async with self._send_lock:
                if not self._playback_closed:
                    await self.websocket.send_json({
                        "event": "clear",
                        "streamSid": self.stream_sid,
                    })
        except Exception as exc:
            logger.warning(
                f"Twilio: Clear send failed ({type(exc).__name__})"
            )

        logger.info(f"Twilio: Audio playback interrupted")
        return True

    async def stream_audio_chunks(
        self, owner: PlaybackOwner, audio_stream
    ) -> PlaybackStreamResult:
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
            return PlaybackStreamResult(0, False, "not_streaming")
        if not self.is_current_playback(owner):
            return PlaybackStreamResult(0, False, "retired")

        CHUNK_SIZE = 160  # 20ms at 8kHz ulaw
        CHUNK_INTERVAL = 0.015  # Send slightly faster than real-time
        # Pre-buffer 200ms (10 chunks) before starting to send to Twilio.
        # This prevents buffer underrun at the start (the "underwater" sound)
        # when ElevenLabs chunks arrive irregularly during cold start.
        PREBUFFER_BYTES = CHUNK_SIZE * 10  # 200ms at 8kHz ulaw
        buffer = bytearray()
        total_bytes = 0
        chunks_sent = 0
        prebuffered = False

        try:
            async for audio_data in audio_stream:
                if not self.is_current_playback(owner):
                    logger.info(f"Twilio: Stream interrupted by barge-in at {total_bytes} bytes")
                    break

                buffer.extend(audio_data)

                # Wait until we have enough pre-buffer before sending the first chunk.
                # After that, stream normally as chunks arrive.
                if not prebuffered:
                    if len(buffer) < PREBUFFER_BYTES:
                        continue
                    prebuffered = True
                    logger.info(f"Twilio: Pre-buffer filled ({len(buffer)} bytes), starting playback")

                # Send complete 160-byte chunks as they accumulate
                while len(buffer) >= CHUNK_SIZE:
                    if not self.is_current_playback(owner):
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
                    if not await self._send_owned(owner, media_event):
                        break
                    total_bytes += CHUNK_SIZE
                    chunks_sent += 1

                    if chunks_sent == 1:
                        logger.info(f"Twilio: First audio chunk streamed to caller")
                    elif chunks_sent % 50 == 0:
                        logger.info(f"Twilio: Streamed {chunks_sent} chunks ({total_bytes} bytes)")

                    await asyncio.sleep(CHUNK_INTERVAL)

            # Send any remaining bytes in the buffer
            if buffer and self.is_current_playback(owner):
                payload = base64.b64encode(bytes(buffer)).decode("utf-8")
                media_event = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {
                        "payload": payload
                    }
                }
                if await self._send_owned(owner, media_event):
                    total_bytes += len(buffer)
                else:
                    return PlaybackStreamResult(total_bytes, False, "retired")

            logger.info(f"Twilio: Stream complete - {total_bytes} bytes in {chunks_sent} chunks")
            return PlaybackStreamResult(
                total_bytes,
                self.is_current_playback(owner),
                None if self.is_current_playback(owner) else "retired",
            )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Twilio: Stream transport error ({type(e).__name__})")
            return PlaybackStreamResult(total_bytes, False, "transport_error")

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

        await self._send_message(message)

    async def _send_message(self, message: dict) -> None:
        async with self._send_lock:
            await self.websocket.send_json(message)

    async def _send_owned(self, owner: PlaybackOwner, message: dict) -> bool:
        async with self._send_lock:
            if not self.is_current_playback(owner):
                return False
            await self.websocket.send_json(message)
            return True

    async def wait_for_playback(
        self,
        owner: PlaybackOwner,
        stream_result: PlaybackStreamResult,
        timeout: float,
    ) -> bool:
        if not stream_result.completed or not self.is_current_playback(owner):
            return False
        mark_name = f"playback-{uuid4().hex}"
        playback_wait = asyncio.get_running_loop().create_future()
        self._pending_marks[mark_name] = (owner, playback_wait)
        send_task: asyncio.Task | None = None
        deadline = asyncio.get_running_loop().time() + max(0.0, timeout)

        try:
            send_task = self._track_send_task(self._send_owned(owner, {
                "event": "mark",
                "streamSid": self.stream_sid,
                "mark": {"name": mark_name},
            }))
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            submitted_done, _ = await asyncio.wait({send_task}, timeout=remaining)
            if send_task not in submitted_done:
                self._expire_mark_wait(owner, playback_wait, send_task)
                logger.warning(f"Twilio: Playback mark timed out for call {self.call_sid}")
                return False
            try:
                submitted = send_task.result()
            except Exception as exc:
                self.release_playback(owner)
                logger.warning(
                    f"Twilio: Playback mark submission failed ({type(exc).__name__})"
                )
                return False
            if not submitted:
                return False
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            acknowledged, _ = await asyncio.wait({playback_wait}, timeout=remaining)
            if playback_wait not in acknowledged:
                self._expire_mark_wait(owner, playback_wait, send_task)
                logger.warning(f"Twilio: Playback mark timed out for call {self.call_sid}")
                return False
            return playback_wait.result()
        except asyncio.CancelledError:
            self._expire_mark_wait(owner, playback_wait, send_task)
            raise
        finally:
            self._pending_marks.pop(mark_name, None)

    @property
    def cleanup_pending(self) -> bool:
        return any(not task.done() for task in self._send_tasks)

    async def wait_for_cleanup(self, timeout: float | None = None) -> None:
        if not self.cleanup_pending:
            return
        if timeout is None:
            await self._send_cleanup_complete.wait()
        else:
            async with asyncio.timeout(timeout):
                await self._send_cleanup_complete.wait()

    def _track_send_task(self, awaitable) -> asyncio.Task:
        task = asyncio.create_task(awaitable)
        self._send_tasks.add(task)
        self._send_cleanup_complete.clear()
        task.add_done_callback(self._send_task_done)
        return task

    def _send_task_done(self, task: asyncio.Task) -> None:
        self._send_tasks.discard(task)
        if not task.cancelled():
            task.exception()
        if not self._send_tasks:
            self._send_cleanup_complete.set()

    def _expire_mark_wait(
        self,
        owner: PlaybackOwner,
        playback_wait: asyncio.Future,
        send_task: asyncio.Task | None,
    ) -> None:
        self.release_playback(owner)
        if not playback_wait.done():
            playback_wait.set_result(False)
        if send_task is not None and not send_task.done():
            send_task.cancel()

    def _resolve_pending_marks(
        self, played: bool, owner: PlaybackOwner | None = None
    ) -> None:
        for mark_owner, playback_wait in self._pending_marks.values():
            if owner is not None and mark_owner is not owner:
                continue
            if not playback_wait.done():
                playback_wait.set_result(played)

    def _retire_all_playback(self) -> None:
        self._playback_closed = True
        if self._active_playback is not None:
            self._active_playback.retired = True
            self._active_playback = None
        self._resolve_pending_marks(False)

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
        self.invalidate_playback_session()
        for task in tuple(self._send_tasks):
            task.cancel()
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
