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

        # Pending audio to send (simplified - just one at a time)
        self._pending_audio: Optional[bytes] = None

    async def handle_connection(self) -> None:
        """
        Handle the WebSocket connection lifecycle

        This method processes incoming events from Twilio:
        - connected: Initial connection
        - start: Call started
        - media: Audio data (μ-law 8kHz)
        - stop: Call ended
        - disconnected: Cleanup
        """
        self._is_connected = True
        logger.info(f"Twilio: Media stream connected for call {self.call_sid}, starting event loop...")

        try:
            loop_count = 0
            while self._is_connected:
                loop_count += 1
                if loop_count % 10 == 0:  # Log every 1 second
                    logger.info(f"Twilio: Event loop iteration {loop_count}, pending_audio={bool(self._pending_audio)}")

                # Wait for message with short timeout to check for pending audio
                try:
                    message = await asyncio.wait_for(
                        self.websocket.receive_text(),
                        timeout=0.1  # 100ms timeout
                    )
                    logger.info(f"Twilio: Received message: {message[:100] if len(message) > 100 else message}")
                    await self._process_message(message)
                except asyncio.TimeoutError:
                    # Timeout is OK - check for pending audio
                    pass

                # Check if we have pending audio to send
                if self._pending_audio:
                    logger.info(f"Twilio: Sending pending audio ({len(self._pending_audio)} bytes)...")
                    await self._send_audio_now(self._pending_audio)
                    self._pending_audio = None

        except WebSocketDisconnect:
            logger.info(f"Twilio: WebSocket disconnected for call {self.call_sid}")
        except Exception as e:
            logger.error(f"Twilio: Error in connection: {e}")
            import traceback
            logger.error(f"Twilio: Traceback:\n{traceback.format_exc()}")
        finally:
            logger.info(f"Twilio: handle_connection() finally block for call {self.call_sid}")
            await self.cleanup()

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

        Audio is marked as pending and will be sent from within the event loop
        after the next incoming message is processed.

        Args:
            audio_data: Audio data (μ-law 8kHz for Twilio)
        """
        if not self._is_streaming:
            logger.warning(f"Twilio: Cannot send audio - not streaming")
            return

        if not self.stream_sid:
            logger.error(f"Twilio: Cannot send audio - stream_sid is empty!")
            return

        # Mark audio as pending - will be sent from event loop
        self._pending_audio = audio_data
        logger.info(f"Twilio: Marked {len(audio_data)} bytes audio as pending")

    async def _send_audio_now(self, audio_data: bytes) -> None:
        """
        Send audio to Twilio immediately (called from within event loop)

        Args:
            audio_data: Audio data (μ-law 8kHz for Twilio)
        """
        total_bytes = len(audio_data)

        logger.info(f"Twilio: Sending {total_bytes} bytes audio (streamSid={self.stream_sid})")

        try:
            # Encode entire audio to base64 (no chunking for now)
            payload = base64.b64encode(audio_data).decode("utf-8")

            # Create media event
            media_event = {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {
                    "payload": payload
                }
            }

            logger.info(f"Twilio: Media event - event={media_event['event']}, streamSid={media_event['streamSid'][:20]}..., payload_len={len(payload)}")
            logger.info(f"Twilio: First 200 chars of payload: {payload[:200]}")

            await self.websocket.send_json(media_event)
            logger.info(f"Twilio: Audio sent successfully")

        except Exception as e:
            logger.error(f"Twilio: Failed to send audio: {e}")
            import traceback
            logger.error(f"Twilio: Traceback:\n{traceback.format_exc()}")

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

    async def generate_twiml(self, websocket_url: str) -> str:
        """
        Generate TwiML for connecting to Media Stream

        Args:
            websocket_url: WebSocket URL for Media Stream

        Returns:
            TwiML as string
        """
        twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="{websocket_url}" />
    </Start>
    <Pause length="60" />
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
