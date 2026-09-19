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
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, AsyncIterator, Deque, Optional
from uuid import uuid4
from loguru import logger

try:
    import websockets
    from websockets.exceptions import ConnectionClosed as WebSocketConnectionClosed
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None
    WebSocketConnectionClosed = ConnectionError

from .stt_base import STTServiceBase, STTResult, STTStatus, AudioChunk
from services.conversation.events import (
    MetadataDisposition,
    UtteranceIdentity,
    UtteranceMetadataEvent,
    WordTiming,
)


@dataclass
class _PendingCommit:
    text: str
    utterance_id: UtteranceIdentity
    correlation_eligible: bool = True


@dataclass
class _ReceiverState:
    websocket: Any
    connection_epoch: str
    commit_sequence: int = 0
    pending_commits: OrderedDict = field(default_factory=OrderedDict)
    seen_metadata_digests: OrderedDict = field(default_factory=OrderedDict)
    correlation_text_history: OrderedDict = field(default_factory=OrderedDict)
    correlation_disabled: bool = False


class STTTransportOutcome(str, Enum):
    """Bounded public outcome for the most recently owned provider transport."""

    INACTIVE = "inactive"
    TRANSPORT_OPEN = "transport_open"
    SESSION_STARTED = "session_started"
    PROVIDER_ERROR = "provider_error"
    REMOTE_ENDED = "remote_ended"
    REMOTE_CLOSED = "remote_closed"
    RECEIVER_ERROR = "receiver_error"
    LOCAL_DISCONNECT = "local_disconnect"


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
    PROVIDER_ID = "elevenlabs_scribe"
    METADATA_PROVENANCE = "elevenlabs_realtime_metadata"
    MAX_PENDING_COMMITS = 64
    MAX_METADATA_EVENTS = 128
    MAX_SEEN_METADATA = 128
    MAX_CORRELATION_TEXT_HISTORY = 128
    CONNECT_TIMEOUT_SECONDS = 10.0
    CLOSE_TIMEOUT_SECONDS = 2.0
    RECEIVER_CANCEL_TIMEOUT_SECONDS = 2.0
    CLEANUP_WAIT_SECONDS = 0.05
    _HISTORY_COMMIT = 1
    _HISTORY_ENRICHMENT = 2
    PROVIDER_ERROR_TYPES = frozenset({
        "auth_error",
        "quota_exceeded",
        "transcriber_error",
        "input_error",
        "invalid_request",
        "error",
        "commit_throttled",
        "unaccepted_terms",
        "rate_limited",
        "queue_overflow",
        "resource_exhausted",
        "session_time_limit_exceeded",
        "chunk_size_exceeded",
        "insufficient_audio_activity",
    })

    def __init__(
        self,
        api_key: str,
        language: str = "",  # Empty = auto-detect
        model: str = "scribe_v2_realtime",
        sample_rate: int = 8000,  # Twilio native rate
        filter_background_audio: bool = False,
    ):
        """
        Initialize ElevenLabs STT service

        Args:
            api_key: ElevenLabs API key
            language: Language code (en, ar, etc.) or empty for auto-detect
            model: Model to use (scribe_v2_realtime)
            sample_rate: Sample rate for audio (8000 for Twilio μ-law)
            filter_background_audio: Whether ElevenLabs should suppress background speech/noise
        """
        if not isinstance(filter_background_audio, bool):
            raise TypeError("filter_background_audio must be a bool")
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets is not installed. Install with: pip install websockets")

        super().__init__(api_key, language)

        self.model = model
        self.sample_rate = sample_rate
        self.filter_background_audio = filter_background_audio

        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._transcript_queue: asyncio.Queue = None
        self._is_listening = False
        self._receive_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._connection_lock = asyncio.Lock()
        self._instance_token = uuid4().hex
        self._connection_number = 0
        self._terminal_revision = 0
        self._session_active = False
        self._cleanup_tasks: set[asyncio.Task] = set()
        self._current_receiver: Optional[_ReceiverState] = None
        self._transport_outcome = STTTransportOutcome.INACTIVE
        self._provider_error_category: Optional[str] = None
        self._metadata_events: Deque[UtteranceMetadataEvent] = deque(
            maxlen=self.MAX_METADATA_EVENTS
        )

    @property
    def metadata_events(self) -> tuple[UtteranceMetadataEvent, ...]:
        """Return the bounded, non-actionable metadata/diagnostic snapshot."""
        return tuple(self._metadata_events)

    @property
    def pending_commit_count(self) -> int:
        """Number of commits retained only for possible metadata correlation."""
        receiver = self._current_receiver
        return len(receiver.pending_commits) if receiver else 0

    @property
    def transport_outcome(self) -> STTTransportOutcome:
        """Return the bounded outcome for the latest owned transport."""
        return self._transport_outcome

    @property
    def provider_error_category(self) -> Optional[str]:
        """Return only a documented, allowlisted provider error category."""
        return self._provider_error_category

    def _activate_receiver(self, websocket) -> _ReceiverState:
        """Create a connection epoch and bind it to one captured WebSocket."""
        self._connection_number += 1
        receiver = _ReceiverState(
            websocket=websocket,
            connection_epoch=f"{self._instance_token}.{self._connection_number}",
        )
        self._current_receiver = receiver
        return receiver

    def _receiver_for_loop(self) -> _ReceiverState:
        receiver = self._current_receiver
        if receiver is None or receiver.websocket is not self._websocket:
            receiver = self._activate_receiver(self._websocket)
        return receiver

    async def connect(self) -> bool:
        """Open a provider transport for the current logical listening session."""
        revision = self._terminal_revision
        async with self._connection_lock:
            if revision != self._terminal_revision:
                return False
            if self._websocket is not None and self._status is STTStatus.CONNECTED:
                return True
            if not self._session_active:
                self._transcript_queue = asyncio.Queue()
                self._metadata_events.clear()
                self._session_active = True
            if self._websocket is not None or self._current_receiver is not None:
                await self._stop_transport_locked()
            return await self._open_transport_locked(revision)

    async def reconnect_with_language(self, language_code: str) -> bool:
        """Replace the provider transport while retaining accepted finals."""
        revision = self._terminal_revision
        session_active = self._session_active
        async with self._connection_lock:
            if (
                not session_active
                or not self._session_active
                or revision != self._terminal_revision
            ):
                return False
            logger.info("ElevenLabs STT: Replacing transport for language change")
            self.language = language_code
            return await self._replace_transport_locked(revision)

    async def disconnect(self) -> None:
        """End the logical session and discard undelivered session finals."""
        # Change the revision before waiting for the lifecycle lock. A connection
        # already in progress must then close its candidate instead of adopting it.
        self._terminal_revision += 1
        self._session_active = False
        async with self._connection_lock:
            self._is_listening = False
            await self._stop_transport_locked()
            queue = self._transcript_queue
            if queue is not None:
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            self._transcript_queue = None
            self._metadata_events.clear()
            self._status = STTStatus.DISCONNECTED
            self._transport_outcome = STTTransportOutcome.LOCAL_DISCONNECT
            self._provider_error_category = None
            await self._wait_for_cleanup_tasks()
            logger.info("ElevenLabs STT: Disconnected")

    async def reset_for_listening(self) -> None:
        """Replace the provider connection without replacing the session queue."""
        revision = self._terminal_revision
        session_active = self._session_active
        async with self._connection_lock:
            logger.info("ElevenLabs STT: Replacing transport for listening reset")
            if (
                not session_active
                or not self._session_active
                or revision != self._terminal_revision
                or not await self._replace_transport_locked(revision)
            ):
                raise RuntimeError("ElevenLabs STT reset failed")

    def _connection_url(self) -> str:
        """Build the unchanged Scribe v2 transport query."""
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
        if self.filter_background_audio:
            params["filter_background_audio"] = "true"
        query_string = "&".join(f"{key}={value}" for key, value in params.items())
        return f"{self.WEBSOCKET_URL}?{query_string}"

    async def _replace_transport_locked(self, revision: int) -> bool:
        """Replace transport state while retaining the logical-session queue."""
        if (
            not self._session_active
            or self._transcript_queue is None
            or revision != self._terminal_revision
        ):
            return False
        # Keep logical listening active while the socket is swapped so an iterator
        # already blocked on the stable queue remains valid.
        self._is_listening = True
        try:
            await self._stop_transport_locked()
            if revision != self._terminal_revision:
                self._is_listening = False
                self._status = STTStatus.DISCONNECTED
                return False
            return await self._open_transport_locked(revision)
        except asyncio.CancelledError:
            self._is_listening = False
            self._status = STTStatus.ERROR
            raise

    async def _open_transport_locked(self, revision: int) -> bool:
        """Open and adopt one socket while the lifecycle lock is held."""
        self._status = STTStatus.CONNECTING
        logger.info("ElevenLabs STT: Connecting to Scribe v2 Realtime")
        try:
            connection_awaitable = websockets.connect(
                self._connection_url(),
                extra_headers={"xi-api-key": self.api_key},
                ping_interval=30,
                ping_timeout=10,
            )
            connection_task = asyncio.ensure_future(connection_awaitable)
        except Exception as error:
            self._is_listening = False
            self._status = STTStatus.ERROR
            logger.error(
                "ElevenLabs STT: Connection setup failed category={}",
                type(error).__name__,
            )
            return False
        try:
            websocket = await asyncio.wait_for(
                asyncio.shield(connection_task),
                timeout=self.CONNECT_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            self._track_abandoned_connection(connection_task)
            self._is_listening = False
            self._status = STTStatus.ERROR
            logger.warning("ElevenLabs STT: Connection attempt cancelled")
            await self._wait_for_cleanup_tasks()
            raise
        except asyncio.TimeoutError:
            self._track_abandoned_connection(connection_task)
            self._is_listening = False
            self._status = STTStatus.ERROR
            logger.error("ElevenLabs STT: Connection attempt timed out")
            return False
        except Exception as error:
            self._is_listening = False
            self._status = STTStatus.ERROR
            logger.error(
                "ElevenLabs STT: Connection failed category={}",
                type(error).__name__,
            )
            return False

        if websocket is None:
            self._is_listening = False
            self._status = STTStatus.ERROR
            logger.error("ElevenLabs STT: Connection returned no socket")
            return False

        if (
            not self._session_active
            or revision != self._terminal_revision
        ):
            await self._close_socket(websocket)
            self._is_listening = False
            self._status = STTStatus.DISCONNECTED
            return False

        self._websocket = websocket
        receiver = self._activate_receiver(websocket)
        self._is_listening = True
        self._receive_task = asyncio.create_task(
            self._receive_loop(receiver),
            name="elevenlabs_stt_receive",
        )
        self._status = STTStatus.CONNECTED
        self._transport_outcome = STTTransportOutcome.TRANSPORT_OPEN
        self._provider_error_category = None
        logger.info("ElevenLabs STT: Connected")
        return True

    def _track_abandoned_connection(self, connection_task: asyncio.Task) -> None:
        """Retain ownership until a cancelled connector ends and its socket closes."""
        connection_task.cancel()
        cleanup_task = asyncio.create_task(
            self._finish_abandoned_connection(connection_task),
            name="elevenlabs_stt_connection_cleanup",
        )
        self._retain_cleanup_task(cleanup_task)

    async def _finish_abandoned_connection(
        self,
        connection_task: asyncio.Task,
    ) -> None:
        try:
            websocket = await connection_task
        except asyncio.CancelledError:
            return
        except Exception as error:
            logger.debug(
                "ElevenLabs STT: Abandoned connection ended category={}",
                type(error).__name__,
            )
            return
        if websocket is not None:
            await self._close_socket(websocket)

    def _retain_cleanup_task(self, task: asyncio.Task) -> None:
        """Track late cleanup without allowing it to adopt application state."""
        if task.done():
            self._consume_cleanup_result(task)
            return
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_finished)

    def _cleanup_finished(self, task: asyncio.Task) -> None:
        self._cleanup_tasks.discard(task)
        self._consume_cleanup_result(task)

    @staticmethod
    def _consume_cleanup_result(task: asyncio.Task) -> None:
        try:
            task.exception()
        except asyncio.CancelledError:
            pass

    async def _wait_for_cleanup_tasks(self) -> None:
        """Give tracked cleanup a bounded grace period without abandoning it."""
        tasks = {task for task in self._cleanup_tasks if not task.done()}
        if tasks:
            await asyncio.wait(tasks, timeout=self.CLEANUP_WAIT_SECONDS)

    async def _stop_transport_locked(self) -> None:
        """Revoke ownership, then run bounded task and socket cleanup."""
        receiver = self._current_receiver
        receive_task = self._receive_task
        websocket = self._websocket
        self._current_receiver = None
        self._receive_task = None
        self._websocket = None

        cleanup_task = asyncio.create_task(
            self._finish_transport_cleanup(receiver, receive_task, websocket),
            name="elevenlabs_stt_transport_cleanup",
        )
        self._retain_cleanup_task(cleanup_task)
        timeout = (
            self.RECEIVER_CANCEL_TIMEOUT_SECONDS
            + (2 * self.CLOSE_TIMEOUT_SECONDS)
            + self.CLEANUP_WAIT_SECONDS
        )
        await asyncio.wait({cleanup_task}, timeout=timeout)

    async def _finish_transport_cleanup(
        self,
        receiver: Optional[_ReceiverState],
        receive_task: Optional[asyncio.Task],
        websocket: Any,
    ) -> None:
        if receive_task is not None and not receive_task.done():
            receive_task.cancel()
            _, pending = await asyncio.wait(
                {receive_task},
                timeout=self.RECEIVER_CANCEL_TIMEOUT_SECONDS,
            )
            if pending:
                self._retain_cleanup_task(receive_task)
                logger.debug("ElevenLabs STT: Receiver cleanup remains pending")
            else:
                self._consume_cleanup_result(receive_task)
        if receiver is not None:
            receiver.pending_commits.clear()
            receiver.seen_metadata_digests.clear()
            receiver.correlation_text_history.clear()
        if websocket is not None:
            await self._close_socket(websocket)

    async def _close_socket(self, websocket: Any) -> None:
        await self._run_bounded_socket_operation(
            websocket.send(json.dumps({"message_type": "end_of_stream"})),
            "end_of_stream",
        )
        close_completed = await self._run_bounded_socket_operation(
            websocket.close(),
            "socket_close",
        )
        if not close_completed:
            self._abort_socket_transport(websocket)

    async def _run_bounded_socket_operation(
        self,
        operation: Any,
        category: str,
    ) -> bool:
        task = asyncio.ensure_future(operation)
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self.CLOSE_TIMEOUT_SECONDS,
            )
            return True
        except asyncio.TimeoutError:
            task.cancel()
            self._retain_cleanup_task(task)
            logger.debug(
                "ElevenLabs STT: Socket operation timed out category={}",
                category,
            )
            return False
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                task.cancel()
                self._retain_cleanup_task(task)
                raise
            return False
        except Exception as error:
            logger.debug(
                "ElevenLabs STT: Socket operation failed category={} error={}",
                category,
                type(error).__name__,
            )
            return False

    @staticmethod
    def _abort_socket_transport(websocket: Any) -> None:
        transport = getattr(websocket, "transport", None)
        abort = getattr(transport, "abort", None)
        if callable(abort):
            try:
                abort()
            except Exception as error:
                logger.debug(
                    "ElevenLabs STT: Transport abort failed category={}",
                    type(error).__name__,
                )

    async def stream_audio(self, audio_chunk: AudioChunk) -> None:
        """
        Stream audio chunk to ElevenLabs STT

        Sends raw μ-law 8kHz audio directly from Twilio — no conversion.
        ElevenLabs natively supports ulaw_8000 format.

        Args:
            audio_chunk: Audio data to transcribe (μ-law 8kHz from Twilio)
        """
        try:
            async with self._connection_lock:
                if self._status != STTStatus.CONNECTED or not self._websocket:
                    return

                message = {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": base64.b64encode(audio_chunk.data).decode("utf-8"),
                }
                websocket = self._websocket
                await websocket.send(json.dumps(message))

        except WebSocketConnectionClosed:
            logger.warning("ElevenLabs STT: WebSocket closed while sending audio")
            self._status = STTStatus.DISCONNECTED
        except Exception as error:
            logger.error(
                "ElevenLabs STT: Audio send failed category={}",
                type(error).__name__,
            )

    def _receiver_is_current(self, receiver: _ReceiverState) -> bool:
        return (
            self._current_receiver is receiver
            and self._websocket is receiver.websocket
        )

    def _finish_receiver_outcome(
        self,
        receiver: _ReceiverState,
        outcome: Optional[STTTransportOutcome],
        provider_error_category: Optional[str],
    ) -> None:
        """Apply one terminal outcome only to the exactly owned receiver."""
        if outcome is None or not self._receiver_is_current(receiver):
            return

        self._is_listening = False
        self._transport_outcome = outcome
        self._provider_error_category = provider_error_category
        if outcome in (
            STTTransportOutcome.PROVIDER_ERROR,
            STTTransportOutcome.RECEIVER_ERROR,
        ):
            self._status = STTStatus.ERROR
        else:
            self._status = STTStatus.DISCONNECTED

        receive_task = asyncio.current_task()
        if self._receive_task is receive_task:
            cleanup_task = asyncio.create_task(
                self._retire_completed_receiver(receiver, receive_task),
                name="elevenlabs_stt_receiver_cleanup",
            )
            self._retain_cleanup_task(cleanup_task)

    async def _retire_completed_receiver(
        self,
        receiver: _ReceiverState,
        receive_task: asyncio.Task,
    ) -> None:
        """Revoke a completed current receiver, then reuse bounded socket cleanup."""
        async with self._connection_lock:
            if (
                not self._receiver_is_current(receiver)
                or self._receive_task is not receive_task
            ):
                return
            websocket = self._websocket
            self._current_receiver = None
            self._receive_task = None
            self._websocket = None
        await self._finish_transport_cleanup(receiver, None, websocket)

    async def _receive_loop(self, receiver: Optional[_ReceiverState] = None) -> None:
        """Normalize messages from one captured WebSocket/identity epoch."""
        receiver = receiver or self._receiver_for_loop()
        terminal_outcome: Optional[STTTransportOutcome] = None
        provider_error_category: Optional[str] = None

        try:
            while self._is_listening and receiver.websocket:
                try:
                    receive_operation = asyncio.ensure_future(receiver.websocket.recv())
                    try:
                        message = await asyncio.wait_for(
                            receive_operation,
                            timeout=60.0  # 1 minute timeout
                        )
                    finally:
                        # wait_for cancellation can leave a cancellation-resistant
                        # recv finishing with an exception. Retain/consume that
                        # child too, so its raw exception never reaches asyncio's
                        # unhandled-task logger after transport replacement.
                        self._retain_cleanup_task(receive_operation)

                    data = json.loads(message)
                    if not isinstance(data, dict):
                        self._record_metadata(
                            receiver,
                            MetadataDisposition.MALFORMED,
                            "invalid_message_shape",
                        )
                        continue
                    msg_type = data.get("message_type", data.get("type", ""))
                    if not isinstance(msg_type, str) or not msg_type:
                        self._record_metadata(
                            receiver,
                            MetadataDisposition.MALFORMED,
                            "missing_message_type",
                        )
                        continue

                    # A receiver awaiting data while a reset completes still owns
                    # its old epoch. Its late events are diagnostic only.
                    if receiver is not self._current_receiver:
                        if msg_type == "session_ended":
                            break
                        self._record_metadata(
                            receiver,
                            MetadataDisposition.OLD_EPOCH,
                            "old_epoch_event",
                        )
                        continue

                    if msg_type in ("partial_transcript", "transcript"):
                        # Partial/interim result — log only, don't queue.
                        # Partials are discarded by process_transcript (is_final=False),
                        # so queuing them just adds latency and overhead.
                        logger.debug("ElevenLabs STT: Partial event ignored")

                    elif msg_type == "committed_transcript":
                        await self._handle_commit(receiver, data)

                    elif msg_type == "committed_transcript_with_timestamps":
                        self._handle_enrichment(receiver, data, msg_type)

                    elif msg_type == "final_transcript":
                        # Not documented as a second canonical Scribe v2 commit.
                        self._record_metadata(
                            receiver,
                            MetadataDisposition.UNSUPPORTED,
                            "legacy_final_transcript",
                        )

                    elif msg_type == "session_started":
                        if self._receiver_is_current(receiver):
                            self._transport_outcome = STTTransportOutcome.SESSION_STARTED
                            self._provider_error_category = None
                            logger.info(
                                "ElevenLabs STT: Session started (epoch={})",
                                receiver.connection_epoch,
                            )

                    elif msg_type in self.PROVIDER_ERROR_TYPES:
                        terminal_outcome = STTTransportOutcome.PROVIDER_ERROR
                        provider_error_category = msg_type
                        logger.error(
                            "ElevenLabs STT: Provider rejected transport category={}",
                            msg_type,
                        )
                        break

                    elif msg_type == "session_ended":
                        terminal_outcome = STTTransportOutcome.REMOTE_ENDED
                        logger.info("ElevenLabs STT: Session ended")
                        break

                    else:
                        self._record_metadata(
                            receiver,
                            MetadataDisposition.UNSUPPORTED,
                            "unknown_event_type",
                        )

                except asyncio.TimeoutError:
                    # Send keepalive
                    if receiver.websocket and receiver is self._current_receiver:
                        try:
                            await receiver.websocket.ping()
                        except Exception:
                            terminal_outcome = STTTransportOutcome.REMOTE_CLOSED
                            break

        except WebSocketConnectionClosed:
            terminal_outcome = STTTransportOutcome.REMOTE_CLOSED
            logger.info("ElevenLabs STT: WebSocket closed")
        except asyncio.CancelledError:
            if self._receiver_is_current(receiver):
                terminal_outcome = STTTransportOutcome.RECEIVER_ERROR
            logger.debug("ElevenLabs STT: Receive loop cancelled")
            return  # Revoked lifecycle receivers cannot change replacement state.
        except Exception as e:
            terminal_outcome = STTTransportOutcome.RECEIVER_ERROR
            logger.error(
                "ElevenLabs STT: Receive loop error category={}",
                type(e).__name__,
            )
        finally:
            self._finish_receiver_outcome(
                receiver,
                terminal_outcome,
                provider_error_category,
            )
            receiver.pending_commits.clear()
            receiver.seen_metadata_digests.clear()
            receiver.correlation_text_history.clear()

    async def _handle_commit(self, receiver: _ReceiverState, data: dict) -> None:
        # Local receipt time belongs to the canonical commit, before queueing.
        # Provider event timestamps are separate metadata, never approval clocks.
        from datetime import datetime, timezone
        received_at = datetime.now(timezone.utc)
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            self._record_metadata(
                receiver,
                MetadataDisposition.MALFORMED,
                "committed_transcript",
                invalid_fields=("text",),
            )
            return

        receiver.commit_sequence += 1
        utterance_id = UtteranceIdentity(
            provider=self.PROVIDER_ID,
            connection_epoch=receiver.connection_epoch,
            commit_sequence=receiver.commit_sequence,
        )
        history = receiver.correlation_text_history.get(text, 0)
        if history & self._HISTORY_COMMIT:
            for pending in receiver.pending_commits.values():
                if pending.text == text:
                    pending.correlation_eligible = False
        self._remember_correlation_text(receiver, text, self._HISTORY_COMMIT)
        receiver.pending_commits[utterance_id] = _PendingCommit(
            text,
            utterance_id,
            correlation_eligible=(
                not receiver.correlation_disabled and history == 0
            ),
        )
        if len(receiver.pending_commits) > self.MAX_PENDING_COMMITS:
            evicted_id, _ = receiver.pending_commits.popitem(last=False)
            self._record_metadata(
                receiver,
                MetadataDisposition.EVICTED,
                "pending_commit",
                utterance_id=evicted_id,
            )

        confidence = self._validated_confidence(data.get("confidence"))
        detected_lang, language_provenance = self._extract_language_with_provenance(data)
        metadata = {
            "provider": self.PROVIDER_ID,
            "provider_event_type": "committed_transcript",
            "identity_provenance": "local_stt_connection_commit_sequence",
            "language_provenance": language_provenance,
        }
        if confidence is not None:
            metadata["confidence_provenance"] = "provider_committed_transcript"

        await self._transcript_queue.put(
            STTResult(
                text=text,
                language=detected_lang,
                confidence=confidence,
                is_final=True,
                metadata=metadata,
                utterance_id=utterance_id,
                received_at=received_at,
            )
        )
        logger.info(
            "ElevenLabs STT: Committed transcript queued (epoch={}, sequence={})",
            receiver.connection_epoch,
            receiver.commit_sequence,
        )

    def _handle_enrichment(
        self,
        receiver: _ReceiverState,
        data: dict,
        msg_type: str,
    ) -> None:
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            self._record_metadata(
                receiver,
                MetadataDisposition.MALFORMED,
                msg_type,
                invalid_fields=("text",),
            )
            return

        pending = tuple(receiver.pending_commits.values())
        eligible_pending = tuple(
            commit for commit in pending if commit.correlation_eligible
        )
        matching_pending = tuple(commit for commit in pending if commit.text == text)
        history = receiver.correlation_text_history.get(text, 0)
        digest = hashlib.blake2s(text.encode("utf-8"), digest_size=12).hexdigest()
        utterance_id = None
        # The provider supplies no documented utterance ID for correlation.
        # Text equality is only a consistency check. Matching is allowed once
        # per text and epoch, when exactly one still-eligible commit exists and
        # no earlier enrichment for that text makes its source ambiguous.
        if (
            not receiver.correlation_disabled
            and not history & self._HISTORY_ENRICHMENT
            and len(eligible_pending) == 1
            and eligible_pending[0].text == text
        ):
            disposition = MetadataDisposition.MATCHED
            utterance_id = eligible_pending[0].utterance_id
            receiver.pending_commits.pop(utterance_id, None)
        elif (
            matching_pending
            or len(eligible_pending) > 1
            or receiver.correlation_disabled
        ):
            disposition = MetadataDisposition.AMBIGUOUS
        elif (
            history & self._HISTORY_ENRICHMENT
            or digest in receiver.seen_metadata_digests
        ):
            disposition = MetadataDisposition.REPEATED
        else:
            disposition = MetadataDisposition.UNMATCHED

        self._remember_correlation_text(receiver, text, self._HISTORY_ENRICHMENT)

        receiver.seen_metadata_digests[digest] = None
        receiver.seen_metadata_digests.move_to_end(digest)
        while len(receiver.seen_metadata_digests) > self.MAX_SEEN_METADATA:
            receiver.seen_metadata_digests.popitem(last=False)

        language, invalid_language = self._validated_language(data.get("language_code"))
        language_confidence = self._validated_confidence(
            data.get("language_probability")
        )
        confidence = self._validated_confidence(data.get("confidence"))
        word_timings, timing_invalid = self._validated_word_timings(data.get("words"))
        invalid_fields = []
        if invalid_language:
            invalid_fields.append("language_code")
        if "language_probability" in data and language_confidence is None:
            invalid_fields.append("language_probability")
        if "confidence" in data and confidence is None:
            invalid_fields.append("confidence")
        if timing_invalid:
            invalid_fields.append("words")

        self._record_metadata(
            receiver,
            disposition,
            msg_type,
            utterance_id=utterance_id,
            language=language,
            language_confidence=language_confidence,
            confidence=confidence,
            word_timings=word_timings,
            invalid_fields=tuple(invalid_fields),
        )

    def _remember_correlation_text(
        self,
        receiver: _ReceiverState,
        text: str,
        event_flag: int,
    ) -> None:
        """Retain bounded proof state, failing closed if any would be forgotten."""
        history = receiver.correlation_text_history
        if text in history:
            history[text] |= event_flag
            return
        if len(history) >= self.MAX_CORRELATION_TEXT_HISTORY:
            receiver.correlation_disabled = True
            for pending in receiver.pending_commits.values():
                pending.correlation_eligible = False
            return
        history[text] = event_flag

    def _record_metadata(
        self,
        receiver: _ReceiverState,
        disposition: MetadataDisposition,
        provider_event_type: str,
        *,
        utterance_id: Optional[UtteranceIdentity] = None,
        language: Optional[str] = None,
        language_confidence: Optional[float] = None,
        confidence: Optional[float] = None,
        word_timings: tuple[WordTiming, ...] = (),
        invalid_fields: tuple[str, ...] = (),
    ) -> None:
        self._metadata_events.append(
            UtteranceMetadataEvent(
                connection_epoch=receiver.connection_epoch,
                disposition=disposition,
                provider_event_type=provider_event_type,
                provenance=self.METADATA_PROVENANCE,
                utterance_id=utterance_id,
                language=language,
                language_confidence=language_confidence,
                confidence=confidence,
                word_timings=word_timings,
                invalid_fields=invalid_fields,
            )
        )
        logger.debug(
            "ElevenLabs STT: Non-actionable event classified (category={}, epoch={})",
            disposition.value,
            receiver.connection_epoch,
        )

    @staticmethod
    def _validated_confidence(value: Any) -> Optional[float]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            numeric = float(value)
        except (OverflowError, ValueError):
            return None
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            return None
        return numeric

    @staticmethod
    def _validated_language(value: Any) -> tuple[Optional[str], bool]:
        if value is None:
            return None, False
        if not isinstance(value, str):
            return None, True
        language = value.strip()
        if not language or len(language) > 32:
            return None, True
        return language, False

    @staticmethod
    def _validated_word_timings(
        value: Any,
    ) -> tuple[tuple[WordTiming, ...], bool]:
        if value is None:
            return (), False
        if not isinstance(value, list) or len(value) > 512:
            return (), True
        timings = []
        for word in value:
            if not isinstance(word, dict):
                return (), True
            start = word.get("start")
            end = word.get("end")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
            ):
                return (), True
            try:
                start_number = float(start)
                end_number = float(end)
            except (OverflowError, ValueError):
                return (), True
            if (
                not math.isfinite(start_number)
                or not math.isfinite(end_number)
                or start_number < 0
                or end_number < start_number
            ):
                return (), True
            timings.append(WordTiming(start_number, end_number))
        return tuple(timings), False

    def _extract_language_with_provenance(self, data: dict) -> tuple[str, str]:
        language, invalid = self._validated_language(data.get("language_code"))
        if language:
            return language, "provider_committed_transcript"

        lang_detection = data.get("language_detection")
        if isinstance(lang_detection, dict):
            language, _ = self._validated_language(lang_detection.get("language_code"))
            if language:
                return language, "provider_committed_transcript"

        if invalid:
            return self.language or "auto", "configured_fallback_after_invalid_provider_value"
        return self.language or "auto", "configured_or_auto_fallback"

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
        language, _ = self._extract_language_with_provenance(data)
        return language

    async def get_transcript(self) -> AsyncIterator[STTResult]:
        """
        Get transcription results as they arrive

        Yields:
            STTResult objects with transcribed text
        """
        queue = self._transcript_queue
        session_revision = self._terminal_revision
        if queue is None or not self._session_active:
            return

        while (
            self._session_active
            and session_revision == self._terminal_revision
            and (self._is_listening or not queue.empty())
        ):
            try:
                result = await asyncio.wait_for(
                    queue.get(),
                    timeout=0.1
                )
                if (
                    not self._session_active
                    or session_revision != self._terminal_revision
                    or queue is not self._transcript_queue
                ):
                    break
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
    # Get language setting
    language = config.get('elevenlabs_stt_language', '')
    if language == 'auto':
        language = ''  # Empty = auto-detect
    filter_background_audio = config.get(
        'elevenlabs_stt_filter_background_audio', False
    )
    if not isinstance(filter_background_audio, bool):
        raise TypeError("filter_background_audio must be a bool")
    if not WEBSOCKETS_AVAILABLE:
        raise ImportError("websockets is not installed. Install with: pip install websockets")

    return ElevenLabsSTT(
        api_key=config.get('elevenlabs_api_key'),
        language=language,
        model=config.get('elevenlabs_stt_model', 'scribe_v2_realtime'),
        sample_rate=8000,  # Native Twilio μ-law rate
        filter_background_audio=filter_background_audio,
    )
