"""Enabled phone intake through actual orchestrator, playback and Graph adapter."""
import asyncio
import os
import unittest
from contextlib import nullcontext
from datetime import timedelta
from unittest import mock
from zoneinfo import ZoneInfo

import tests.integration.test_verified_booking_create as journey
from services.calendar.booking_mutations import GraphBookingMutations
from services.conversation.events import UtteranceIdentity
from services.conversation.orchestrator import (ConversationContext,
                                                 ConversationOrchestrator)
from services.conversation.booking_session import BookingSession
from services.llm.llm_base import LLMResponse
from services.stt.stt_base import STTResult
from services.telephony.twilio_service import TwilioMediaStreamHandler


class Calendar(journey.SyntheticCalendar):
    tenant_id = "tenant"
    business_id = "business"

    async def is_available(self):
        return True

    async def get_customer_appointments(self, phone):
        return []


class Socket:
    def __init__(self):
        self.handler = None
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)
        if message["event"] == "mark":
            self.handler._on_mark(message)


class Audio:
    def __init__(self):
        self.sent = False
        self.cancelled = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.sent:
            raise StopAsyncIteration
        self.sent = True
        return b"\xff" * 1600

    def cancel(self):
        self.cancelled = True

    async def aclose(self):
        self.cancel()


class Speech:
    def __init__(self):
        self.texts = []

    def create_stream(self, request):
        self.texts.append(request.text)
        return Audio()


class STT:
    language = "en"

    def __init__(self):
        self.queue = asyncio.Queue()

    async def get_transcript(self):
        while True:
            yield await self.queue.get()

    async def reset_for_listening(self):
        pass


class LLM:
    def __init__(self, search_first=False):
        self.calls = 0
        self.requests = []
        self.search_first = search_first

    async def chat_with_tools(self, request):
        self.calls += 1
        self.requests.append(request)
        if self.search_first and self.calls == 1:
            return LLMResponse("", tool_calls=[{
                "id": "call_search", "name": "search_appointments",
                "arguments": '{"accountant_name":"Synthetic Consultant","date":"2026-09-21"}'
            }])
        if self.calls == (2 if self.search_first else 1):
            return LLMResponse("The appointment is booked", tool_calls=[{
                "id": "call_check", "name": "check_appointment",
                "arguments": ('{"accountant_name":"Synthetic Consultant",'
                              '"date_time":"2026-09-21 10:00",'
                              '"customer_email":"s@example.invalid"}')
            }])
        return LLMResponse("The appointment is booked", tool_calls=[{
            "id": "call_confirm", "name": "confirm_appointment",
            "arguments": '{"confirm":true}'
        }])


async def until(predicate):
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0.01)


class PhoneBookingDatabaseTests(unittest.IsolatedAsyncioTestCase):
    setUpClass = classmethod(journey.VerifiedCreateDatabaseTests.setUpClass.__func__)
    tearDownClass = classmethod(journey.VerifiedCreateDatabaseTests.tearDownClass.__func__)
    asyncSetUp = journey.VerifiedCreateDatabaseTests.asyncSetUp
    connection = journey.VerifiedCreateDatabaseTests.connection

    async def _journey(self, language, *, correction=False, unknown=False,
                       rollback=False, search_first=False, approval_text=None,
                       delayed_reset=False):
        import services.conversation.orchestrator as module
        pool = await self.driver.create_pool(self.dsn, min_size=1, max_size=10)
        self.addAsyncCleanup(pool.close)
        http = journey.SyntheticHttp(
            read_wire=journey.readback(), create_wire={} if unknown else None)
        mutations = GraphBookingMutations(tenant_id="tenant", business_id="business",
            client_id="synthetic", client_secret="synthetic", client=http)
        calendar, clock = Calendar(), journey.Clock()
        socket = Socket()
        handler = TwilioMediaStreamHandler("synthetic-call", "synthetic-stream", socket)
        socket.handler = handler
        handler._is_streaming = True
        handler._is_connected = True
        context = ConversationContext("synthetic-call", "+14165550100", language=language)
        context.caller_name = "Synthetic Customer"
        orchestrator = ConversationOrchestrator.__new__(ConversationOrchestrator)
        orchestrator._verified_phone_booking_enabled = True
        orchestrator._conversations = {"synthetic-call": context}
        orchestrator._twilio_handlers = {"synthetic-call": handler}
        orchestrator._call_state_locks = {"synthetic-call": asyncio.Lock()}
        orchestrator._speech_setup_locks = {}
        orchestrator._active_speech = {}
        orchestrator._verified_clear_tasks = set()
        orchestrator._system_prompt = "Synthetic approved Flexible Accounting context."
        orchestrator._barge_in_reset_done = {}
        orchestrator._barge_in_speech_start = {}
        orchestrator._echo_guard_until = {}
        orchestrator._speech_aware_barge_in_enabled = False
        orchestrator._barge_in_observer = None
        orchestrator._call_stt_instances = {"synthetic-call": STT()}
        orchestrator.tts = Speech()
        orchestrator.llm = LLM(search_first)
        session = BookingSession(context, handler, pool, calendar, mutations,
            clock=clock, selection_loader=journey.SyntheticSelections,
            owner_check=lambda: orchestrator._conversations.get("synthetic-call") is context,
            on_new_input=lambda: orchestrator._revoke_verified_speech(
                "synthetic-call", context, handler))
        context.booking_session = session
        worker = asyncio.create_task(session.run(orchestrator, "synthetic-call"))
        consumer = asyncio.create_task(orchestrator._consume_stt_transcripts("synthetic-call"))
        stt = orchestrator._call_stt_instances["synthetic-call"]
        reset_entered, reset_release = asyncio.Event(), asyncio.Event()
        if delayed_reset:
            async def reset():
                if session.proposals._presentation_completed is not None:
                    reset_entered.set()
                    await reset_release.wait()
            stt.reset_for_listening = reset
        gate = None
        try:
            with (mock.patch.object(module, "business_now",
                                    return_value=journey.NOW.astimezone(ZoneInfo("America/Toronto"))),
                  mock.patch("services.calendar.ms_bookings_service.get_calendar_service",
                             return_value=calendar),
                  mock.patch("services.config.accountants_service.get_accountants_service",
                             return_value=journey.SyntheticSelections()),
                  mock.patch("services.sms.notification_outbox.datetime") as notification_clock,
                  (mock.patch("services.sms.notification_outbox.create_new_booking_jobs",
                              side_effect=RuntimeError("synthetic rollback"))
                   if rollback else nullcontext())):
                # The entire journey is in September 2026; reminder eligibility
                # must use that same test clock rather than today's wall clock.
                notification_clock.now.side_effect = lambda _zone=None: clock.now
                first_text = ("أريد موعدا يوم الاثنين الساعة العاشرة s@example.invalid"
                              if language == "ar" else
                              "I want Monday at ten, s@example.invalid")
                if search_first:
                    first_text = "شو المواعيد المتاحة يوم الاثنين"
                await stt.queue.put(STTResult(first_text, language, None, is_final=True,
                    utterance_id=UtteranceIdentity("stt", "epoch", 1),
                    received_at=clock.now))
                if search_first:
                    await until(lambda: any("أي وقت يناسبك" in text for text in orchestrator.tts.texts)
                                and context.state.value == "listening")
                    self.assertIsNone(session.proposals.current)
                    self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.booking_operations"), 0)
                    self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
                        for method, url, _ in http.calls), 0)
                    clock.now += timedelta(seconds=1)
                    await stt.queue.put(STTResult("الاثنين الساعة العاشرة s@example.invalid", language,
                        None, is_final=True, utterance_id=UtteranceIdentity("stt", "epoch", 4),
                        received_at=clock.now))
                await until(lambda: session.proposals._presentation_completed is not None)
                self.assertEqual(await self.conn.fetchval(
                    "SELECT count(*) FROM public.booking_operations"), 0)
                if correction:
                    class DispatchGate:
                        def __init__(self):
                            self.count = 0
                            self.entered = asyncio.Event()
                            self.release = asyncio.Event()

                        async def __aenter__(self):
                            self.count += 1
                            if self.count == 2:
                                self.entered.set()
                                await self.release.wait()

                        async def __aexit__(self, *args):
                            pass

                    gate = DispatchGate()
                    mutations._slots = gate
                clock.now += timedelta(seconds=1)
                await stt.queue.put(STTResult(approval_text or ("نعم" if language == "ar" else "yes"),
                    language, None, is_final=True,
                    utterance_id=UtteranceIdentity("stt", "epoch", 2),
                    received_at=clock.now))
                if delayed_reset:
                    await asyncio.wait_for(reset_entered.wait(), 2)
                    await until(lambda: session.turn.generation == 2)
                    reset_release.set()
                if correction:
                    await asyncio.wait_for(gate.entered.wait(), 10)
                    await stt.queue.put(STTResult("No, change the time", language,
                        None, is_final=True,
                        utterance_id=UtteranceIdentity("stt", "epoch", 3),
                        received_at=clock.now))
                    await until(lambda: session.turn.generation == 3)
                    gate.release.set()
                    await until(lambda: session.unresolved)
                    self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
                        for method, url, _ in http.calls), 0)
                    self.assertEqual(await self.conn.fetchval(
                        "SELECT count(*) FROM public.bookings"), 0)
                    self.assertFalse(context.booking_just_completed)
                    return
                if unknown or rollback:
                    await until(lambda: session.unresolved)
                    await stt.queue.put(STTResult("yes", language, None, is_final=True,
                        utterance_id=UtteranceIdentity("stt", "epoch", 3),
                        received_at=clock.now))
                    await until(lambda: session.turn.generation == 3)
                    self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
                        for method, url, _ in http.calls), 1)
                    self.assertEqual(await self.conn.fetchval(
                        "SELECT count(*) FROM public.bookings"), 0)
                    self.assertEqual(await self.conn.fetchval(
                        "SELECT count(*) FROM public.booking_notification_outbox"), 0)
                    self.assertFalse(context.booking_just_completed)
                    return
                await until(lambda: context.booking_just_completed)
                expected_marks = 5 if search_first else 3  # search acknowledgement, proposal, success
                await until(lambda: len([m for m in socket.messages
                                          if m["event"] == "mark"]) == expected_marks)
                self.assertEqual(await self.conn.fetchval(
                    "SELECT count(*) FROM public.booking_operations WHERE state='applied'"), 1)
                self.assertEqual(await self.conn.fetchval(
                    "SELECT count(*) FROM public.bookings"), 1)
                self.assertEqual(await self.conn.fetchval(
                    "SELECT count(*) FROM public.booking_notification_outbox WHERE state='held'"), 2)
                self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
                    for method, url, _ in http.calls), 1)
                self.assertEqual(len([m for m in socket.messages if m["event"] == "mark"]), expected_marks)
                self.assertTrue(any("Synthetic Customer" in text
                    for text in orchestrator.tts.texts))
                confirmation_index = 2 if search_first else 1
                self.assertEqual(orchestrator.llm.requests[confirmation_index].messages[-1].role.value,
                                 "user")
                self.assertTrue(any(message.role.value == "tool" for message in
                    orchestrator.llm.requests[confirmation_index].messages))
                self.assertIsNone(context.pending_booking)
        finally:
            reset_release.set()
            if gate is not None:
                gate.release.set()
            session.close()
            for task in (consumer, worker):
                task.cancel()
            await asyncio.gather(consumer, worker, return_exceptions=True)
            await mutations.close()
            await handler.cleanup()

    async def test_enabled_english_full_verified_phone_journey(self):
        await self._journey("en")

    async def test_english_confirmation_during_post_readback_reset_creates_once(self):
        await self._journey("en", delayed_reset=True)

    async def test_arabic_confirmation_during_post_readback_reset_creates_once(self):
        await self._journey("ar", delayed_reset=True)

    async def test_enabled_arabic_full_verified_phone_journey(self):
        await self._journey("ar")

    async def test_arabic_day_search_requires_time_choice_before_proposal_and_create(self):
        await self._journey("ar", search_first=True)

    async def test_english_yes_during_arabic_call_keeps_approval_and_language(self):
        await self._journey("ar", approval_text="yes")

    async def test_correction_during_actual_transport_admission_prevents_post(self):
        await self._journey("en", correction=True)

    async def test_unknown_create_response_does_not_speak_success_or_repost(self):
        await self._journey("en", unknown=True)

    async def test_local_job_rollback_keeps_receipt_without_success(self):
        await self._journey("en", rollback=True)


class PhoneReadinessDatabaseTests(unittest.IsolatedAsyncioTestCase):
    setUpClass = classmethod(journey.VerifiedCreateDatabaseTests.setUpClass.__func__)
    tearDownClass = classmethod(journey.VerifiedCreateDatabaseTests.tearDownClass.__func__)

    async def asyncSetUp(self):
        from migrations import runner
        self.conn = await self.driver.connect(self.dsn, timeout=10, command_timeout=20)
        self.addAsyncCleanup(self.conn.close)
        await self.conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        with mock.patch.dict(os.environ, {"MIGRATION_DATABASE_URL": self.dsn}):
            if self._testMethodName == "test_disabled_0004_ready_enabled_0004_refused":
                await runner.run(prepare=True)
            else:
                await runner.run(prepare_operations=True)

    async def test_disabled_0004_ready_enabled_0004_refused(self):
        from types import SimpleNamespace
        import services.database as database
        pool = await self.driver.create_pool(self.dsn, min_size=1, max_size=2)
        self.addAsyncCleanup(pool.close)
        with mock.patch.object(database, "settings", SimpleNamespace(
                automatic_notifications_enabled=True,
                verified_phone_booking_enabled=False)):
            await database.check_database_compatibility(pool)
        with mock.patch.object(database, "settings", SimpleNamespace(
                automatic_notifications_enabled=True,
                verified_phone_booking_enabled=True)):
            with self.assertRaises(database.DatabaseReadinessError):
                await database.check_database_compatibility(pool)

    async def test_enabled_0005_readiness_and_owned_transport_teardown(self):
        from types import SimpleNamespace
        import services.database as database
        pool = await self.driver.create_pool(self.dsn, min_size=1, max_size=2)
        self.addAsyncCleanup(pool.close)
        with mock.patch.object(database, "settings", SimpleNamespace(
                automatic_notifications_enabled=True,
                verified_phone_booking_enabled=True)):
            await database.check_database_compatibility(pool)
        orchestrator = ConversationOrchestrator.__new__(ConversationOrchestrator)
        orchestrator._verified_phone_booking_enabled = True
        orchestrator._conversations = {}
        orchestrator._booking_pool = None
        orchestrator._booking_mutations = None
        fake_settings = SimpleNamespace(ms_bookings_tenant_id="tenant",
            ms_bookings_business_id="business", ms_bookings_client_id="synthetic",
            ms_bookings_client_secret="synthetic")
        with mock.patch("services.conversation.orchestrator.get_settings",
                        return_value=fake_settings):
            orchestrator.initialize_verified_booking(pool)
        self.assertIs(orchestrator._booking_pool, pool)
        await orchestrator.close_verified_booking()
        self.assertIsNone(orchestrator._booking_pool)
        self.assertIsNone(orchestrator._booking_mutations)
