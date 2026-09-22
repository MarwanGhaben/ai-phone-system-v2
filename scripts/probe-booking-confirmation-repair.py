"""Offline real Twilio mark, proposal completion and consent probe. No database, audio or provider traffic."""
import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from types import SimpleNamespace
from unittest import mock
from zoneinfo import ZoneInfo

os.environ.update({"SECRET_KEY": "synthetic", "DATABASE_URL": "postgresql://synthetic:synthetic@localhost/test",
    "TWILIO_ACCOUNT_SID": "ACsynthetic", "TWILIO_AUTH_TOKEN": "synthetic", "TWILIO_PHONE_NUMBER": "+14165550100",
    "DEEPGRAM_API_KEY": "synthetic", "ELEVENLABS_API_KEY": "synthetic", "OPENAI_API_KEY": "synthetic"})
from loguru import logger
logger.remove()
logging.disable(logging.CRITICAL)

from services.conversation.orchestrator import ConversationOrchestrator, ConversationContext, ConversationState
from services.conversation.booking_session import BookingSession
from services.conversation.events import UtteranceIdentity
from services.scheduling.models import AvailabilityQuery, CalendarScope, TimeInterval
from services.scheduling.policy import AppointmentCandidate
from services.scheduling.proposals import CustomerSnapshot
from services.scheduling.booking_service import BookingResult, BookingOutcome
from services.telephony.twilio_service import TwilioMediaStreamHandler

class AutoMarkWebSocket:

    def __init__(self):
        self.handler = None
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)
        if message['event'] == 'mark':
            self.handler._on_mark(message)

class FakeStream:

    def __init__(self, payload: bytes, *, blocked: bool=False):
        self.payload = payload
        self.blocked = blocked
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.closed = False
        self._sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._sent:
            self._sent = True
            self.started.set()
            return self.payload
        if self.blocked and (not self.cancelled):
            await self.release.wait()
        raise StopAsyncIteration

    def cancel(self):
        self.cancelled = True
        self.release.set()

    async def aclose(self):
        self.cancel()
        self.closed = True

class FakeSTT:

    def __init__(self, language='en'):
        self.language = language
        self.resets = 0
        self.audio = []
        self.disconnected = False

    async def reset_for_listening(self):
        self.resets += 1

    async def stream_audio(self, chunk):
        self.audio.append(chunk.data)

    async def disconnect(self):
        self.disconnected = True

def add_call(orchestrator, call_sid, *, language='en', websocket=None):
    context = ConversationContext(call_sid, '+10000000000', language=language)
    context.state = ConversationState.SPEAKING
    websocket = websocket or AutoMarkWebSocket()
    handler = TwilioMediaStreamHandler(call_sid, f'stream-{call_sid}', websocket)
    websocket.handler = handler
    handler._is_streaming = True
    handler._is_connected = True
    stt = FakeSTT(language)
    orchestrator._conversations[call_sid] = context
    orchestrator._twilio_handlers[call_sid] = handler
    orchestrator._call_stt_instances[call_sid] = stt
    orchestrator._call_state_locks[call_sid] = asyncio.Lock()
    orchestrator._barge_in_reset_done[call_sid] = False
    return (context, handler, stt)

def make_orchestrator(tts):
    orchestrator = ConversationOrchestrator.__new__(ConversationOrchestrator)
    orchestrator.tts = tts
    orchestrator._conversations = {}
    orchestrator._twilio_handlers = {}
    orchestrator._call_stt_instances = {}
    orchestrator._call_state_locks = {}
    orchestrator._speech_setup_locks = {}
    orchestrator._active_speech = {}
    orchestrator._barge_in_reset_done = {}
    orchestrator._echo_guard_until = {}
    orchestrator._garbled_drop_count = {}
    orchestrator._barge_in_consecutive = {}
    orchestrator._barge_in_speech_start = {}
    return orchestrator

def final(sequence, text, *, when=None, epoch='epoch'):
    return SimpleNamespace(text=text, language='en', is_final=True, utterance_id=UtteranceIdentity('stt', epoch, sequence), received_at=when or datetime.now(timezone.utc))

def take_pending(session):
    """Mirror the worker's receipt/text binding for direct session tests."""
    batch = session.turn.take_pending()
    if batch.inputs:
        latest = batch.inputs[-1]
        session.current_receipt = session.receipts.get(latest.identity)
        session.current_text = latest.text
    return batch

def candidate():
    now = datetime.now(timezone.utc)
    start = now + timedelta(hours=3)
    scope = CalendarScope('tenant', 'business', 'service')
    query = AvailabilityQuery(scope, TimeInterval(now, start + timedelta(days=1)), ('staff',), 'request')
    return AppointmentCandidate(scope, 'staff', TimeInterval(start, start + timedelta(minutes=30)), 'policy', query, now)

async def test_confirmation_follows_exact_mark_not_cleanup(arrival, answer):
    entered, released = (asyncio.Event(), asyncio.Event())

    class Stream(FakeStream):

        async def aclose(self):
            if arrival == 'close':
                entered.set()
                await released.wait()
            await super().aclose()

    class TTS:

        def create_stream(self, request):
            return Stream(b'A' * 1600)
    orchestrator = make_orchestrator(TTS())
    context, handler, stt = add_call(orchestrator, 'call')
    selected = candidate()
    clock = [datetime.now(timezone.utc)]
    session = BookingSession(context, handler, object(), object(), SimpleNamespace(tenant_id='tenant', business_id='business'), clock=lambda: clock[0], on_new_input=lambda: orchestrator._revoke_verified_speech('call', context, handler))
    context.booking_session = session

    async def reset():
        if arrival in ('reset', 'after'):
            entered.set()
            await released.wait()
    stt.reset_for_listening = reset
    wait_for_playback = handler.wait_for_playback

    async def wait(*args, **kwargs):
        if arrival == 'before_mark':
            entered.set()
            await released.wait()
        return await wait_for_playback(*args, **kwargs)
    handler.wait_for_playback = wait
    assert session.admit(final(1, 'check a time', when=clock[0]))
    token = take_pending(session).token
    session.offer(selected, CustomerSnapshot('Synthetic Customer', '+14165550100', ''), 'Rami', 'appointment', '1901 Banff Ave, Ottawa', 'America/Toronto', timedelta(), timedelta())
    original_send = handler.websocket.send_json

    async def send(message):
        await original_send(message)
        if arrival == 'at_mark' and message['event'] == 'mark':
            clock[0] += timedelta(seconds=1)
            assert session.admit(final(2, answer, when=clock[0]))
            entered.set()
    handler.websocket.send_json = send
    task = asyncio.create_task(session.present(orchestrator, 'call', 'en', token))
    create = mock.AsyncMock(return_value=BookingResult(BookingOutcome.VERIFIED))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert any((message['event'] == 'mark' for message in handler.websocket.messages)) == (arrival != 'before_mark')
        if arrival not in ('after', 'at_mark'):
            clock[0] += timedelta(seconds=1)
            assert session.admit(final(2, answer, when=clock[0]))
        released.set()
        await asyncio.wait_for(task, 2)
        if arrival == 'after':
            clock[0] += timedelta(seconds=1)
            assert session.admit(final(2, answer, when=clock[0]))
        batch = take_pending(session)
        with mock.patch('services.conversation.booking_session.BookingService.create', create):
            outcome = await session.confirm(answer, 'en', batch.token)
        if arrival != 'before_mark' and answer == 'yes':
            assert outcome is BookingOutcome.VERIFIED
            create.assert_awaited_once()
        else:
            assert outcome is None
            create.assert_not_awaited()
    finally:
        released.set()
        await asyncio.gather(task, return_exceptions=True)
        session.close()
        await asyncio.gather(*getattr(orchestrator, '_verified_clear_tasks', ()), return_exceptions=True)

async def main():
    for arrival in ("before_mark", "at_mark", "reset", "close", "after"):
        for answer in ("yes", "yes, but change the time"):
            await test_confirmation_follows_exact_mark_not_cleanup(arrival, answer)
    print("BOOKING_CONFIRMATION_OFFLINE_MARK_AND_CONSENT_OK")

if __name__ == "__main__":
    asyncio.run(main())
