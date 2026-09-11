"""Real application booking-save regressions; PostgreSQL is explicitly opt-in."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock

from loguru import logger
from services.calendar.calendar_base import BookingResult
from services.conversation.orchestrator import ConversationContext, ConversationOrchestrator
from services.dashboard import dashboard_routes

ROOT = Path(__file__).resolve().parents[2]
UTC = timezone.utc
SENTINEL = "T005-D-private-customer@example.invalid"


class ConfirmationPersistenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calendar = mock.Mock()
        self.calendar.create_booking = mock.AsyncMock(return_value=BookingResult(
            success=True, appointment_id="synthetic-provider", staff_name="Synthetic Consultant"))
        self.enterContext(mock.patch(
            "services.calendar.ms_bookings_service.get_calendar_service",
            return_value=self.calendar))
        self.pool = mock.Mock()
        self.pool.fetchval = mock.AsyncMock(return_value=41)
        self.enterContext(mock.patch("services.database.get_db_pool",
                                    new=mock.AsyncMock(return_value=self.pool)))
        self.agent = ConversationOrchestrator.__new__(ConversationOrchestrator)
        self.agent._speak_to_caller = mock.AsyncMock()
        self.agent._send_booking_sms = mock.AsyncMock()
        self.context = ConversationContext(
            call_sid="synthetic-call", phone_number="+14165550100", language="en")
        self.agent._conversations = {self.context.call_sid: self.context}

    def pending(self, instant):
        self.context.pending_booking = {
            "service_id": "service-1", "staff_id": "staff-1",
            "staff_name": "Synthetic Consultant", "appointment_time": instant,
            "customer_name": "Synthetic Caller", "customer_email": "",
            "client_type": "individual",
        }

    async def test_invalid_pending_time_never_dispatches_provider(self):
        for value in (None, "2030-01-08T11:00:00", datetime(2030, 1, 8, 11)):
            with self.subTest(value=value):
                self.pending(value)
                result = await self.agent._confirm_booking(self.context.call_sid, {"confirm": True})
                self.assertTrue(result.startswith("BOOKING_NEEDS_RECHECK:"), result)
                self.assertIsNone(self.context.pending_booking)
                self.calendar.create_booking.assert_not_awaited()
                self.agent._speak_to_caller.assert_not_awaited()

    async def test_failed_local_save_or_missing_provider_id_never_confirms_or_retries(self):
        for provider_id, database_error, row_id in (
            ("synthetic-provider", RuntimeError(SENTINEL), 41),
            ("", None, 41), ("  ", None, 41), (None, None, 41),
            ("synthetic-provider", None, None),
        ):
            with self.subTest(provider_id=provider_id, database_error=type(database_error).__name__):
                self.pending(datetime(2030, 1, 8, 16, tzinfo=UTC))
                self.context.booking_just_completed = True
                self.context.last_booking_summary = "earlier booking"
                self.calendar.create_booking.reset_mock()
                self.calendar.create_booking.return_value = BookingResult(
                    success=True, appointment_id=provider_id, staff_name="Synthetic Consultant")
                self.pool.fetchval.side_effect = database_error
                self.pool.fetchval.return_value = row_id
                messages = []
                sink = logger.add(lambda message: messages.append(str(message)))
                try:
                    result = await self.agent._confirm_booking(self.context.call_sid, {"confirm": True})
                    await asyncio.sleep(0)
                    second = await self.agent._confirm_booking(self.context.call_sid, {"confirm": True})
                finally:
                    logger.remove(sink)
                self.assertTrue(result.startswith("BOOKING_OUTCOME_UNKNOWN:"), result)
                self.assertTrue(second.startswith("BOOKING_NEEDS_RECHECK:"), second)
                self.assertFalse(self.context.booking_just_completed)
                self.assertIsNone(self.context.last_booking_summary)
                self.assertIsNone(self.context.pending_booking)
                self.calendar.create_booking.assert_awaited_once()
                self.agent._send_booking_sms.assert_not_awaited()
                self.assertNotIn(SENTINEL, result + "".join(messages))

    async def test_cancelled_local_save_propagates_and_invalidates_confirmation(self):
        self.pending(datetime(2030, 1, 8, 16, tzinfo=UTC))
        self.pool.fetchval.side_effect = asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await self.agent._confirm_booking(self.context.call_sid, {"confirm": True})
        self.assertIsNone(self.context.pending_booking)
        self.calendar.create_booking.assert_awaited_once()
        self.agent._send_booking_sms.assert_not_awaited()


@unittest.skipUnless(os.environ.get("T005_RUN_DOCKER_TESTS") == "1",
                     "real PostgreSQL requires T005_RUN_DOCKER_TESTS=1")
class PostgreSQLBookingPersistenceTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "t005d_migration_harness", ROOT / "tests/integration/test_migrations.py")
        cls.harness_module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(cls.harness_module)
        cls.harness = cls.harness_module.PostgreSQLTests
        cls.harness.setUpClass()
        cls.driver = cls.harness.driver
        cls.dsn = cls.harness.dsn

    @classmethod
    def tearDownClass(cls):
        cls.harness.tearDownClass()

    async def asyncSetUp(self):
        self.conn = await self.driver.connect(self.dsn, timeout=10, command_timeout=15)
        self.addAsyncCleanup(self.conn.close)
        await self.conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        with mock.patch.dict(os.environ, {"MIGRATION_DATABASE_URL": self.dsn}):
            await self.harness.runner.run(prepare=True)

    async def test_real_orchestrator_persists_exact_instants_across_session_zones(self):
        from pytz import timezone as pytz_timezone
        from services.calendar.calendar_base import BookingResult
        from services.conversation.orchestrator import ConversationContext, ConversationOrchestrator
        toronto = pytz_timezone("America/Toronto")
        cases = (
            ("UTC", toronto.localize(datetime(2030, 1, 8, 11)),
             datetime(2030, 1, 8, 16, tzinfo=UTC)),
            ("Asia/Kolkata", datetime(2030, 1, 8, 16, tzinfo=UTC),
             datetime(2030, 1, 8, 16, tzinfo=UTC)),
            ("UTC", toronto.localize(datetime(2030, 9, 10, 11)),
             datetime(2030, 9, 10, 15, tzinfo=UTC)),
            ("Asia/Kolkata", datetime(2030, 9, 10, 15, tzinfo=UTC),
             datetime(2030, 9, 10, 15, tzinfo=UTC)),
        )
        for number, (session_zone, pending_time, expected_utc) in enumerate(cases, 1):
            with self.subTest(session_zone=session_zone):
                await self.conn.execute("TRUNCATE public.bookings RESTART IDENTITY CASCADE")
                await self.conn.execute("SET TIME ZONE '" + session_zone + "'")
                sid = "synthetic-real-" + str(number)
                context = ConversationContext(call_sid=sid, phone_number="+14165550100", language="en")
                context.pending_booking = {
                    "service_id": "service-1", "staff_id": "staff-1",
                    "staff_name": "Synthetic Consultant", "appointment_time": pending_time,
                    "customer_name": "Synthetic Caller", "customer_email": "",
                    "client_type": "individual",
                }
                agent = ConversationOrchestrator.__new__(ConversationOrchestrator)
                agent._conversations = {sid: context}
                agent._speak_to_caller = mock.AsyncMock()
                agent._send_booking_sms = mock.AsyncMock()
                calendar = mock.Mock()
                calendar.create_booking = mock.AsyncMock(return_value=BookingResult(
                    success=True, appointment_id="provider-" + str(number),
                    staff_name="Synthetic Consultant"))
                with mock.patch("services.calendar.ms_bookings_service.get_calendar_service",
                                return_value=calendar), mock.patch(
                    "services.database.get_db_pool", new=mock.AsyncMock(return_value=self.conn)):
                    result = await agent._confirm_booking(sid, {"confirm": True})
                    await asyncio.sleep(0)
                self.assertTrue(result.startswith("BOOKING_SUCCESS:"), result)
                calendar.create_booking.assert_awaited_once()
                agent._send_booking_sms.assert_awaited_once()
                row = await self.conn.fetchrow(
                    "SELECT appointment_time,appointment_time_utc,ms_booking_id FROM public.bookings")
                self.assertEqual(row["appointment_time"], datetime(2030, pending_time.month,
                                                                   pending_time.day, 11))
                self.assertEqual(row["appointment_time_utc"], expected_utc)
                self.assertEqual(row["ms_booking_id"], "provider-" + str(number))
                self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.bookings"), 1)
                self.assertIn("11:00 AM", result)
                sent_time = agent._send_booking_sms.call_args.kwargs["appointment_dt"]
                self.assertEqual(sent_time.hour, 11)
                self.assertEqual(sent_time, expected_utc)
                dispatched_time = calendar.create_booking.call_args.kwargs["start_time"]
                self.assertEqual(dispatched_time.hour, 11)
                self.assertEqual(dispatched_time, expected_utc)

    async def test_real_dashboard_counts_only_verified_upcoming_across_session_zones(self):
        from services.scheduling import booking_records
        await booking_records.persist_booking_record(
            self.conn,
            call_sid="verified-call",
            phone_number="+14165550100",
            client_name="Verified Synthetic",
            client_email="",
            accountant_name="Synthetic Consultant",
            appointment_time=datetime(2099, 1, 8, 16, tzinfo=UTC),
            client_type="individual",
            language="en",
            provider_appointment_id="provider-verified",
            notes="synthetic",
        )
        await self.conn.execute("""
            INSERT INTO public.bookings
                (call_sid,client_name,appointment_time,status,ms_booking_id)
            VALUES ('legacy-call','Legacy Synthetic',
                    TIMESTAMP '2099-01-08 11:00:00','confirmed','provider-legacy')
        """)
        dashboard = dashboard_routes
        results = []
        for session_zone in ("UTC", "Asia/Kolkata"):
            await self.conn.execute("SET TIME ZONE '" + session_zone + "'")
            with mock.patch.object(dashboard, "get_db_pool", new=mock.AsyncMock(return_value=self.conn)):
                results.append(await dashboard.get_bookings(user={}))
        self.assertEqual(results[0], results[1])
        result = results[0]
        self.assertEqual(result["upcoming"], 1)
        self.assertEqual(result["unresolved_time"], 1)
        verified, legacy = result["bookings"]
        self.assertTrue(verified["time_verified"])
        self.assertEqual(verified["time"], "2099-01-08T11:00:00-05:00")
        self.assertFalse(legacy["time_verified"])
        self.assertEqual(legacy["time"], "2099-01-08T11:00:00")


if __name__ == "__main__":
    unittest.main()
