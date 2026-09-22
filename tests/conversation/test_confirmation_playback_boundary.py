import asyncio
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest import mock
import pytest
from tests.conversation.test_playback_ownership import make_orchestrator, add_call, FakeStream
from tests.conversation.test_safe_booking_flow import candidate, final, take_pending
from services.conversation.booking_session import BookingSession
from services.scheduling.proposals import CustomerSnapshot
from services.scheduling.booking_service import BookingResult, BookingOutcome
from services.telephony.twilio_service import TwilioMediaStreamHandler, PlaybackStreamResult
from tests.telephony.test_playback_generations import RecordingWebSocket, wait_for_message

@pytest.mark.asyncio
@pytest.mark.parametrize("arrival", ["before_mark", "at_mark", "reset", "close", "after"])
@pytest.mark.parametrize("answer", ["yes", "yes, but change the time"])
async def test_confirmation_follows_exact_mark_not_cleanup(arrival, answer):
    entered, released = asyncio.Event(), asyncio.Event()
    class Stream(FakeStream):
        async def aclose(self):
            if arrival == "close":
                entered.set()
                await released.wait()
            await super().aclose()
    class TTS:
        def create_stream(self, request):
            return Stream(b"A" * 1600)
    orchestrator = make_orchestrator(TTS())
    context, handler, stt = add_call(orchestrator, "call")
    selected = candidate()
    clock = [datetime.now(timezone.utc)]
    session = BookingSession(context, handler, object(), object(),
        SimpleNamespace(tenant_id="tenant", business_id="business"),
        clock=lambda:clock[0],
        on_new_input=lambda: orchestrator._revoke_verified_speech("call",context,handler))
    context.booking_session=session
    async def reset():
        if arrival in ("reset", "after"):
            entered.set()
            await released.wait()
    stt.reset_for_listening=reset
    wait_for_playback = handler.wait_for_playback
    async def wait(*args, **kwargs):
        if arrival == "before_mark":
            entered.set()
            await released.wait()
        return await wait_for_playback(*args, **kwargs)
    handler.wait_for_playback = wait
    assert session.admit(final(1,"check a time",when=clock[0]))
    token=take_pending(session).token
    session.offer(selected,CustomerSnapshot("Synthetic Customer","+14165550100",""),
        "Rami","appointment","1901 Banff Ave, Ottawa","America/Toronto",timedelta(),timedelta())
    original_send = handler.websocket.send_json
    async def send(message):
        await original_send(message)
        if arrival == "at_mark" and message["event"] == "mark":
            clock[0] += timedelta(seconds=1)
            assert session.admit(final(2, answer, when=clock[0]))
            entered.set()
    handler.websocket.send_json = send
    task=asyncio.create_task(session.present(orchestrator,"call","en",token))
    create=mock.AsyncMock(return_value=BookingResult(BookingOutcome.VERIFIED))
    try:
        await asyncio.wait_for(entered.wait(),2)
        # Real handler already received the exact mark before it reaches reset.
        assert any(message['event']=='mark' for message in handler.websocket.messages) == (arrival != 'before_mark')
        if arrival not in ("after", "at_mark"):
            clock[0] += timedelta(seconds=1)
            assert session.admit(final(2,answer,when=clock[0]))
        released.set()
        await asyncio.wait_for(task,2)
        if arrival == "after":
            clock[0] += timedelta(seconds=1)
            assert session.admit(final(2,answer,when=clock[0]))
        batch=take_pending(session)
        with mock.patch('services.conversation.booking_session.BookingService.create',create):
            outcome=await session.confirm(answer,'en',batch.token)
        if arrival != "before_mark" and answer == "yes":
            assert outcome is BookingOutcome.VERIFIED
            create.assert_awaited_once()
        else:
            assert outcome is None
            create.assert_not_awaited()
    finally:
        released.set()
        await asyncio.gather(task,return_exceptions=True)
        session.close()
        await asyncio.gather(*getattr(orchestrator,'_verified_clear_tasks',()),return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["ack", "clear", "replacement", "exception", "cancel", "retire_in_callback"])
async def test_exact_mark_observer_ownership_once_and_cleanup(action):
    socket = RecordingWebSocket()
    handler = TwilioMediaStreamHandler("synthetic", "stream", socket)
    owner = await handler.begin_playback()
    called = []
    def observe(value):
        called.append(value)
        if action == "exception":
            raise ValueError("private-message-canary")
        if action == "retire_in_callback":
            handler.invalidate_playback(value)
    task = asyncio.create_task(handler.wait_for_playback(
        owner, PlaybackStreamResult(1600, True), 2, on_acknowledged=observe))
    try:
        mark = await wait_for_message(socket, "mark")
        handler._on_mark({"mark": {"name": "foreign"}})
        assert not called
        if action == "clear":
            await handler.clear_audio(owner)
        elif action == "replacement":
            await handler.begin_playback()
        elif action == "cancel":
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        handler._on_mark(mark)
        handler._on_mark(mark)
        if action != "cancel":
            assert await task is (action == "ack")
        assert called == ([owner] if action in ("ack", "exception", "retire_in_callback") else [])
        assert not handler._pending_marks and not handler._mark_observers
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await handler.wait_for_cleanup(timeout=2)


@pytest.mark.asyncio
async def test_speech_success_without_exact_completion_cannot_authorize():
    from tests.conversation.test_safe_booking_flow import setup
    orchestrator, context, session = setup()
    assert session.admit(final(1, "check"))
    token = take_pending(session).token
    session.offer(candidate(), CustomerSnapshot("Private Name", "+14165550100", ""),
                  "Rami", "appointment", "private-location", "America/Toronto", timedelta(), timedelta())
    async def speak(*args, on_booking_playback=None, **kwargs):
        on_booking_playback(SimpleNamespace(session_id="stream", generation=1))
        return True
    orchestrator._speak_to_caller = speak
    try:
        assert not await session.present(orchestrator, "call", "en", token)
        assert session.proposals._presentation_completed is None
    finally:
        session.close()
