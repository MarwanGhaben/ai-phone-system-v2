"""T020-A regressions for stale availability and strict booking authorization."""
from __future__ import annotations

import asyncio
from datetime import date, datetime as RealDateTime, time, timedelta, timezone
import unittest
from unittest import mock
from zoneinfo import ZoneInfo

from services.conversation import orchestrator as orchestrator_module
from services.calendar.calendar_base import TimeSlot, BookingResult
from services.config.accountants_service import BookingSelection, BookingSelectionResult
from services.calendar.service_facts import decode_service_facts
from services.scheduling.booking_check import BookingPolicySnapshot
from services.scheduling.models import (
    AvailabilityFailure, AvailabilityFailureCategory, AvailabilityResult,
    AvailabilityStatus, FreeInterval, TimeInterval,
)
from services.scheduling.policy import AppointmentFormat, ClosureCalendar, SchedulingPolicy

TORONTO = ZoneInfo("America/Toronto")
UTC = timezone.utc
NOW = RealDateTime(2030, 1, 7, 9, 0, tzinfo=TORONTO)  # Monday


class FrozenDateTime(RealDateTime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is None else NOW.astimezone(tz)


class BookingSafetyGuardsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.module = orchestrator_module
        self.enterContext(mock.patch.object(self.module, "datetime", FrozenDateTime))
        self.enterContext(mock.patch.object(self.module, "business_now", return_value=NOW))
        self.calendar = mock.Mock()
        self.calendar.tenant_id = "tenant"
        self.calendar.business_id = "business"
        self.calendar.is_available = mock.AsyncMock(return_value=True)
        self.calendar.get_staff_members = mock.AsyncMock(return_value=[])
        self.calendar.get_services = mock.AsyncMock(return_value=[])
        self.calendar.get_customer_appointments = mock.AsyncMock(return_value=[])
        self.calendar.get_available_slots = mock.AsyncMock(return_value=[])
        self.calendar.synthetic_slots = []
        self.calendar.get_service_facts = mock.AsyncMock(side_effect=self._facts)
        self.calendar.get_availability = mock.AsyncMock(side_effect=self._availability)
        self.calendar.create_booking = mock.AsyncMock(
            return_value=BookingResult(success=False, error_message="synthetic rejection"))
        self.accountants = mock.Mock()
        self.accountants.resolve_booking_accountant.return_value = BookingSelectionResult(
            "resolved", BookingSelection("Hussam", "staff-1", "service-1"))
        self.enterContext(mock.patch(
            "services.calendar.ms_bookings_service.get_calendar_service",
            return_value=self.calendar))
        self.enterContext(mock.patch(
            "services.config.accountants_service.get_accountants_service",
            return_value=self.accountants))
        self.enterContext(mock.patch(
            "services.scheduling.booking_check.load_booking_policy",
            side_effect=self._policy))
        self.orchestrator = self.module.ConversationOrchestrator.__new__(
            self.module.ConversationOrchestrator)
        self.orchestrator._speak_to_caller = mock.AsyncMock()
        self.orchestrator._conversations = {}
        self.context = self._context("call-a")

    def _policy(self, scope, staff, now):
        today = now.astimezone(TORONTO).date()
        horizon = today
        for _ in range(2):
            horizon += timedelta(days=1)
            while horizon.weekday() >= 5:
                horizon += timedelta(days=1)
        policy = SchedulingPolicy(
            "synthetic-v1", scope, (staff,), timedelta(minutes=30),
            timedelta(minutes=30), timedelta(0), timedelta(0),
            "America/Toronto", (0, 1, 2, 3, 4), time(10), time(17),
            timedelta(minutes=30), 2,
            ClosureCalendar(date(2030, 1, 1), date(2031, 12, 31), frozenset()),
            AppointmentFormat.IN_PERSON, 1)
        return BookingPolicySnapshot(policy, horizon, "Synthetic office")

    def _facts(self, scope):
        return decode_service_facts({
            "id": scope.service_id, "defaultDuration": "PT30M",
            "preBuffer": "PT0S", "postBuffer": "PT0S",
            "staffMemberIds": ["staff-1"], "isLocationOnline": False,
            "maximumAttendeesCount": 1, "isHiddenFromCustomers": False,
            "schedulingPolicy": {
                "allowStaffSelection": True, "timeSlotInterval": "PT30M",
                "minimumLeadTime": "PT0S", "maximumAdvance": "P365D",
                "generalAvailability": {"availabilityType": "bookWhenStaffAreFree"},
                "customAvailabilities": [],
            },
        }, scope, self.module.business_now())

    async def _availability(self, query):
        slots = self.calendar.synthetic_slots
        if isinstance(slots, Exception):
            raise slots
        if not isinstance(slots, list) or any(
                not isinstance(slot, TimeSlot) or slot.staff_id != query.staff_ids[0]
                or slot.start_time.tzinfo is None or slot.end_time.tzinfo is None
                or slot.end_time <= slot.start_time for slot in slots):
            return AvailabilityResult(query, AvailabilityStatus.INVALID_RESPONSE,
                self.module.business_now(), failure=AvailabilityFailure(
                    AvailabilityFailureCategory.INVALID_RESPONSE))
        intervals = tuple(FreeInterval(query.scope, slot.staff_id,
                                       TimeInterval(slot.start_time, slot.end_time))
                          for slot in slots)
        return AvailabilityResult(query, AvailabilityStatus.AVAILABLE if intervals else
                                  AvailabilityStatus.NO_AVAILABILITY,
                                  self.module.business_now(), intervals)

    def _context(self, call_sid):
        context = self.module.ConversationContext(
            call_sid=call_sid, phone_number="+14165550100", language="en")
        self.orchestrator._conversations[call_sid] = context
        return context

    @staticmethod
    def _arguments(date_time="2030-01-08 10:00"):
        return {
            "date_time": date_time,
            "accountant_name": "Hussam",
            "client_type": "individual",
            "customer_name": "Synthetic Caller",
            "customer_email": "caller@example.invalid",
        }

    @staticmethod
    def _slot(start, *, staff_id="staff-1", end=None):
        return TimeSlot(
            staff_id=staff_id,
            staff_name="Hussam",
            start_time=start,
            end_time=end if end is not None else start + timedelta(minutes=30),
            formatted="synthetic",
        )

    async def _confirm_after_rejection(self):
        result = await self.orchestrator._confirm_booking("call-a", {"confirm": True})
        self.assertTrue(result.startswith("BOOKING_NEEDS_RECHECK:"))
        self.calendar.create_booking.assert_not_awaited()
        self.orchestrator._speak_to_caller.assert_not_awaited()

    async def test_all_rejections_invalidate_seeded_pending_and_cannot_confirm(self):
        cases = (
            ("empty", {}, self._arguments(), "NO_ELIGIBLE_SLOT:"),
            ("adapter-error", {"slots": RuntimeError("private provider body")},
             self._arguments(), "AVAILABILITY_UNVERIFIED:"),
            ("invalid-result", {"slots": [object()]},
             self._arguments(), "AVAILABILITY_UNVERIFIED:"),
            ("not-configured", {"available": False},
             self._arguments(), "AVAILABILITY_UNVERIFIED:"),
            ("missing-service", {"accountant": {"name": "Hussam"},
                                 "staff": [], "services": []},
             self._arguments(), "BOOKING_ERROR:"),
            ("invalid-date", {}, self._arguments("not-a-date"), "INVALID_DATE_TIME:"),
            ("weekend", {}, self._arguments("2030-01-12 10:00"), "NO_ELIGIBLE_SLOT:"),
            ("horizon", {}, self._arguments("2030-01-15 10:00"), "NO_ELIGIBLE_SLOT:"),
            ("duplicate", {"existing": [{
                "start_time": RealDateTime(2030, 1, 8, 9, 0, tzinfo=TORONTO),
                "staff_name": "Other Staff",
            }]}, self._arguments(), "DUPLICATE_WARNING:"),
        )
        for name, setup, arguments, prefix in cases:
            with self.subTest(name=name):
                self.context.pending_booking = {"appointment_time": "stale"}
                self.context.duplicate_warned = False
                self.calendar.reset_mock()
                self.orchestrator._speak_to_caller.reset_mock()
                self.calendar.is_available = mock.AsyncMock(
                    return_value=setup.get("available", True))
                self.calendar.synthetic_slots = setup.get("slots", [])
                self.calendar.get_customer_appointments = mock.AsyncMock(
                    return_value=setup.get("existing", []))
                self.calendar.get_staff_members = mock.AsyncMock(
                    return_value=setup.get("staff", []))
                self.calendar.get_services = mock.AsyncMock(
                    return_value=setup.get("services", []))
                self.accountants.resolve_booking_accountant.return_value = (
                    BookingSelectionResult("invalid_configuration")
                    if "accountant" in setup else
                    BookingSelectionResult("resolved", BookingSelection(
                        "Hussam", "staff-1", "service-1")))

                result = await self.orchestrator._check_booking("call-a", arguments)

                self.assertTrue(result.startswith(prefix), result)
                self.assertIsNone(self.context.pending_booking)
                await self._confirm_after_rejection()

    async def test_missing_context_returns_without_provider_work(self):
        result = await self.orchestrator._check_booking(
            "missing-call", self._arguments())
        self.assertTrue(result.startswith("BOOKING_ERROR:"))
        self.calendar.is_available.assert_not_awaited()
        self.calendar.get_available_slots.assert_not_awaited()

    async def test_cancelled_check_propagates_after_old_pending_is_cleared(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def suspended():
            entered.set()
            await release.wait()
            return True

        self.context.pending_booking = {"appointment_time": "stale"}
        self.calendar.is_available = mock.AsyncMock(side_effect=suspended)
        task = asyncio.create_task(
            self.orchestrator._check_booking("call-a", self._arguments()))
        await entered.wait()
        self.assertIsNone(self.context.pending_booking)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNone(self.context.pending_booking)

    async def test_alternatives_are_suggestions_until_checked_again(self):
        requested = RealDateTime(2030, 1, 8, 10, 0, tzinfo=TORONTO)
        same_day = self._slot(RealDateTime(2030, 1, 8, 11, 0, tzinfo=TORONTO))
        later_day = self._slot(RealDateTime(2030, 1, 9, 13, 30, tzinfo=TORONTO))
        for name, slots, text in (
            ("same-day", [same_day], "Tuesday, January 08 at 11:00 AM"),
            ("later-day", [later_day], "Wednesday, January 09 at 01:30 PM"),
        ):
            with self.subTest(name=name):
                self.calendar.synthetic_slots = slots
                result = await self.orchestrator._check_booking(
                    "call-a", self._arguments(requested.isoformat()))
                self.assertIn(text, result)
                self.assertIn("call check_appointment again", result.lower())
                self.assertIn("fresh confirmation", result.lower())
                self.assertNotIn("confirm=true directly", result.lower())
                self.assertIsNone(self.context.pending_booking)
                await self._confirm_after_rejection()

        self.calendar.synthetic_slots = [later_day]
        result = await self.orchestrator._check_booking(
            "call-a", self._arguments(later_day.start_time.isoformat()))
        self.assertTrue(result.startswith("SLOT_AVAILABLE:"), result)
        self.assertEqual(self.context.pending_booking["appointment_time"],
                         later_day.start_time)
        self.calendar.create_booking.assert_not_awaited()

    async def test_malformed_or_nonmatching_slots_never_authorize(self):
        requested = RealDateTime(2030, 1, 8, 10, 0, tzinfo=TORONTO)
        cases = (
            ("wrong-staff", [self._slot(requested, staff_id="staff-2")]),
            ("same-wall-different-instant", [
                self._slot(RealDateTime(2030, 1, 8, 10, 0, tzinfo=UTC))]),
            ("seconds-mismatch", [self._slot(requested.replace(second=30))]),
            ("naive", [self._slot(RealDateTime(2030, 1, 8, 10, 0))]),
            ("reversed", [self._slot(
                requested, end=requested - timedelta(minutes=30))]),
            ("malformed", [object()]),
        )
        for name, slots in cases:
            with self.subTest(name=name):
                self.context.pending_booking = {"appointment_time": "stale"}
                self.calendar.synthetic_slots = slots
                result = await self.orchestrator._check_booking(
                    "call-a", self._arguments(requested.isoformat()))
                self.assertFalse(result.startswith("SLOT_AVAILABLE:"), result)
                self.assertIsNone(self.context.pending_booking)
                self.calendar.create_booking.assert_not_awaited()

    async def test_exact_and_equivalent_instants_authorize_selected_staff(self):
        requested = RealDateTime(2030, 1, 8, 10, 0, tzinfo=TORONTO)
        for name, start in (
            ("exact", requested),
            ("equivalent-utc", requested.astimezone(UTC)),
        ):
            with self.subTest(name=name):
                slot = self._slot(start)
                self.calendar.synthetic_slots = [slot]
                result = await self.orchestrator._check_booking(
                    "call-a", self._arguments(requested.isoformat()))
                self.assertTrue(result.startswith("SLOT_AVAILABLE:"), result)
                self.assertEqual(self.context.pending_booking["staff_id"], "staff-1")
                pending_time = self.context.pending_booking["appointment_time"]
                self.assertEqual(pending_time, start)
                self.assertEqual(pending_time.hour, 10)
                self.assertEqual(pending_time.utcoffset(), timedelta(hours=-5))
                self.calendar.create_booking.assert_not_awaited()

    async def test_only_literal_true_can_reach_hold_speech_or_create(self):
        invalid_values = (None, "false", "true", 0, 1, [], {}, [True])
        for value in invalid_values:
            with self.subTest(value=value):
                self.context.pending_booking = {
                    "service_id": "service-1", "staff_id": "staff-1",
                    "staff_name": "Hussam",
                    "appointment_time": RealDateTime(2030, 1, 8, 10, 0, tzinfo=TORONTO),
                    "customer_name": "Synthetic", "customer_email": "",
                    "client_type": "individual",
                }
                self.calendar.create_booking.reset_mock()
                self.orchestrator._speak_to_caller.reset_mock()
                result = await self.orchestrator._confirm_booking(
                    "call-a", {"confirm": value})
                self.assertTrue(result.startswith("BOOKING_NEEDS_RECHECK:"), result)
                self.assertIsNone(self.context.pending_booking)
                self.calendar.create_booking.assert_not_awaited()
                self.orchestrator._speak_to_caller.assert_not_awaited()

        self.context.pending_booking = {"appointment_time": "stale"}
        result = await self.orchestrator._confirm_booking(
            "call-a", {"confirm": False})
        self.assertTrue(result.startswith("BOOKING_CANCELLED:"))
        self.assertIsNone(self.context.pending_booking)
        self.calendar.create_booking.assert_not_awaited()
        self.orchestrator._speak_to_caller.assert_not_awaited()

    async def test_summer_utc_slot_reaches_create_as_toronto_wall_time(self):
        summer_now = RealDateTime(2030, 7, 8, 9, 0, tzinfo=TORONTO)  # Monday
        requested = RealDateTime(2030, 7, 9, 10, 0, tzinfo=TORONTO)

        class SummerClock(RealDateTime):
            @classmethod
            def now(cls, tz=None):
                return summer_now if tz is None else summer_now.astimezone(tz)

        with mock.patch.object(self.module, "datetime", SummerClock), mock.patch.object(
            self.module, "business_now", return_value=summer_now
        ):
            self.calendar.synthetic_slots = [self._slot(requested.astimezone(UTC))]
            checked = await self.orchestrator._check_booking(
                "call-a", self._arguments(requested.isoformat()))
            self.assertTrue(checked.startswith("SLOT_AVAILABLE:"), checked)
            await self.orchestrator._confirm_booking("call-a", {"confirm": True})

        self.calendar.create_booking.assert_awaited_once()
        sent_time = self.calendar.create_booking.await_args.kwargs["start_time"]
        self.assertEqual(sent_time, requested)
        self.assertEqual(sent_time.strftime("%Y-%m-%dT%H:%M:%S"), "2030-07-09T10:00:00")
        self.assertEqual(sent_time.utcoffset(), timedelta(hours=-4))

    async def test_late_check_exception_clears_new_pending_authorization(self):
        requested = RealDateTime(2030, 1, 8, 10, 0, tzinfo=TORONTO)
        self.calendar.synthetic_slots = [self._slot(requested)]

        def fail_final_log(message):
            if message == "Orchestrator: exact policy-approved slot pending confirmation":
                raise OSError("synthetic log sink failure")

        with mock.patch.object(self.module.logger, "info", side_effect=fail_final_log):
            response = await self.orchestrator._check_booking("call-a", self._arguments())
        self.assertTrue(response.startswith("AVAILABILITY_UNVERIFIED:"), response)
        self.assertIsNone(self.context.pending_booking)
        await self._confirm_after_rejection()

    async def test_literal_true_after_exact_check_reaches_one_expected_create(self):
        requested = RealDateTime(2030, 1, 8, 10, 0, tzinfo=TORONTO)
        self.calendar.synthetic_slots = [self._slot(requested)]
        checked = await self.orchestrator._check_booking(
            "call-a", self._arguments(requested.isoformat()))
        self.assertTrue(checked.startswith("SLOT_AVAILABLE:"), checked)
        self.calendar.create_booking.assert_not_awaited()

        result = await self.orchestrator._confirm_booking(
            "call-a", {"confirm": True})

        self.assertTrue(result.startswith("BOOKING_OUTCOME_UNKNOWN:"), result)
        self.calendar.create_booking.assert_awaited_once()
        kwargs = self.calendar.create_booking.await_args.kwargs
        self.assertEqual(kwargs["service_id"], "service-1")
        self.assertEqual(kwargs["staff_id"], "staff-1")
        self.assertEqual(kwargs["start_time"], requested)
        self.orchestrator._speak_to_caller.assert_awaited_once()

    async def test_uncertain_create_does_not_advise_retry_or_claim_failure(self):
        requested = RealDateTime(2030, 1, 8, 10, 0, tzinfo=TORONTO)
        for failure in (BookingResult(success=False, error_message="private body"),
                        TimeoutError("private body")):
            with self.subTest(failure=type(failure).__name__):
                self.calendar.synthetic_slots = [self._slot(requested)]
                self.calendar.create_booking.reset_mock()
                self.calendar.create_booking.side_effect = (
                    failure if isinstance(failure, Exception) else None)
                self.calendar.create_booking.return_value = failure
                await self.orchestrator._check_booking("call-a", self._arguments())
                response = await self.orchestrator._confirm_booking("call-a", {"confirm": True})
                self.assertTrue(response.startswith("BOOKING_OUTCOME_UNKNOWN:"), response)
                self.assertNotIn("private body", response)
                self.assertNotIn("could not be created", response)
                self.assertNotIn("offer to retry", response)
                self.assertIsNone(self.context.pending_booking)
                self.calendar.create_booking.assert_awaited_once()
                await self.orchestrator._confirm_booking("call-a", {"confirm": True})
                self.calendar.create_booking.assert_awaited_once()

    async def test_rejection_in_one_call_does_not_clear_another_call(self):
        context_b = self._context("call-b")
        pending_b = {"appointment_time": "independent"}
        context_b.pending_booking = pending_b
        self.context.pending_booking = {"appointment_time": "stale-a"}
        self.calendar.synthetic_slots = []

        await self.orchestrator._check_booking("call-a", self._arguments())

        self.assertIsNone(self.context.pending_booking)
        self.assertIs(context_b.pending_booking, pending_b)


if __name__ == "__main__":
    unittest.main()
