"""Exact configured consultant selection through the real booking check."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
from zoneinfo import ZoneInfo

import yaml

from services.calendar.calendar_base import TimeSlot
from services.config.accountants_service import AccountantsService
from services.calendar.service_facts import decode_service_facts
from services.conversation import orchestrator as module
from services.scheduling.booking_check import BookingPolicySnapshot
from services.scheduling.models import (
    AvailabilityFailure, AvailabilityFailureCategory, AvailabilityResult,
    AvailabilityStatus, FreeInterval, TimeInterval,
)
from services.scheduling.policy import AppointmentFormat, ClosureCalendar, SchedulingPolicy


TORONTO = ZoneInfo("America/Toronto")
NOW = datetime(2030, 1, 7, 9, tzinfo=TORONTO)
REQUESTED = datetime(2030, 1, 8, 10, tzinfo=TORONTO)


class FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is None else NOW.astimezone(tz)


def row(name, staff, service="service-1", *, arabic=None, aliases=None):
    return {"name": name, "name_ar": arabic or name,
            "booking_aliases": aliases or [], "staff_id": staff,
            "service_id": service}


ROWS = [row("Hussam Saadaldin", "staff-1", arabic="حسام سعد الدين",
            aliases=["Hussam", "Hossam", "حسام"]),
        row("Rami Kahwaji", "staff-2", arabic="رامي قهوجي",
            aliases=["Rami", "رامي"]),
        row("Abdul ElFarra", "staff-3", arabic="عبدول الفرا",
            aliases=["Abdul", "عبدول", "عبد"])]


class BookingSelectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "accountants.yaml"
        self.write_rows(ROWS)
        self.service = AccountantsService(str(self.path))
        self.enterContext(mock.patch.object(module, "datetime", FrozenDateTime))
        self.enterContext(mock.patch.object(module, "business_now", return_value=NOW))
        self.calendar = mock.Mock()
        self.calendar.tenant_id = "tenant"
        self.calendar.business_id = "business"
        self.calendar.is_available = mock.AsyncMock(return_value=True)
        self.calendar.get_staff_members = mock.AsyncMock(return_value=[
            mock.Mock(id="first-wrong", name="Hussam")])
        self.calendar.get_services = mock.AsyncMock(return_value=[
            mock.Mock(id="first-wrong-service", name="individual", description="individual")])
        self.calendar.get_customer_appointments = mock.AsyncMock(return_value=[])
        self.calendar.get_available_slots = mock.AsyncMock(return_value=[])
        self.calendar.synthetic_slots = []
        self.calendar.get_service_facts = mock.AsyncMock(side_effect=self._facts)
        self.calendar.get_availability = mock.AsyncMock(side_effect=self._availability)
        self.calendar.create_booking = mock.AsyncMock()
        self.enterContext(mock.patch(
            "services.calendar.ms_bookings_service.get_calendar_service",
            return_value=self.calendar))
        self.enterContext(mock.patch(
            "services.config.accountants_service.get_accountants_service",
            return_value=self.service))
        self.enterContext(mock.patch(
            "services.scheduling.booking_check.load_booking_policy",
            side_effect=self._policy))
        self.orchestrator = module.ConversationOrchestrator.__new__(
            module.ConversationOrchestrator)
        self.orchestrator._conversations = {}
        self.orchestrator._speak_to_caller = mock.AsyncMock()
        self.context("call-a")

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
            "staffMemberIds": ["staff-1", "staff-2", "staff-3"],
            "isLocationOnline": False, "maximumAttendeesCount": 1,
            "isHiddenFromCustomers": False,
            "schedulingPolicy": {
                "allowStaffSelection": True, "timeSlotInterval": "PT30M",
                "minimumLeadTime": "PT0S", "maximumAdvance": "P365D",
                "generalAvailability": {"availabilityType": "bookWhenStaffAreFree"},
                "customAvailabilities": [],
            },
        }, scope, NOW)

    async def _availability(self, query):
        slots = self.calendar.synthetic_slots
        if not isinstance(slots, list) or any(
                not isinstance(slot, TimeSlot) or slot.staff_id != query.staff_ids[0]
                for slot in slots):
            return AvailabilityResult(query, AvailabilityStatus.INVALID_RESPONSE, NOW,
                failure=AvailabilityFailure(AvailabilityFailureCategory.INVALID_RESPONSE))
        intervals = tuple(FreeInterval(query.scope, slot.staff_id,
                                       TimeInterval(slot.start_time, slot.end_time))
                          for slot in slots)
        return AvailabilityResult(query, AvailabilityStatus.AVAILABLE if intervals else
                                  AvailabilityStatus.NO_AVAILABILITY, NOW, intervals)

    def write_rows(self, rows):
        self.path.write_text(yaml.safe_dump({"accountants": rows}, allow_unicode=True),
                             encoding="utf-8")

    def context(self, sid):
        context = module.ConversationContext(
            call_sid=sid, phone_number="+14165550100", language="en")
        self.orchestrator._conversations[sid] = context
        return context

    def arguments(self, name):
        return {"date_time": REQUESTED.isoformat(), "accountant_name": name,
                "client_type": "individual", "customer_name": "Caller"}

    def slot(self, staff, display="Provider Display"):
        return TimeSlot(staff_id=staff, staff_name=display, start_time=REQUESTED,
                        end_time=REQUESTED + timedelta(minutes=30), formatted="slot")

    async def reject(self, name, prefix):
        context = self.orchestrator._conversations["call-a"]
        context.pending_booking = {"appointment_time": "stale"}
        result = await self.orchestrator._check_booking("call-a", self.arguments(name))
        self.assertTrue(result.startswith(prefix), result)
        self.assertIsNone(context.pending_booking)
        confirmed = await self.orchestrator._confirm_booking("call-a", {"confirm": True})
        self.assertTrue(confirmed.startswith("BOOKING_NEEDS_RECHECK:"), confirmed)
        self.calendar.get_available_slots.assert_not_awaited()
        self.calendar.is_available.assert_not_awaited()
        self.calendar.get_staff_members.assert_not_awaited()
        self.calendar.get_services.assert_not_awaited()
        self.calendar.create_booking.assert_not_awaited()

    async def test_exact_names_aliases_and_normalization(self):
        for name, staff in (("Hussam Saadaldin", "staff-1"),
                            ("Hussam", "staff-1"),
                            ("  HOSSAM  ", "staff-1"),
                            ("حسام", "staff-1"),
                            ("حسام سعد الدين", "staff-1"),
                            ("Rami Kahwaji", "staff-2"),
                            ("  رامي  ", "staff-2"),
                            ("رامي قهوجي", "staff-2"),
                            ("Abdul ElFarra", "staff-3"),
                            ("عبدول الفرا", "staff-3"),
                            ("Abdul", "staff-3"),
                            ("عبدول", "staff-3"), ("عبد", "staff-3")):
            with self.subTest(name=name):
                self.calendar.get_availability.reset_mock()
                self.calendar.synthetic_slots = [self.slot(staff)]
                result = await self.orchestrator._check_booking(
                    "call-a", self.arguments(name))
                self.assertTrue(result.startswith("SLOT_AVAILABLE:"), result)
                self.calendar.get_availability.assert_awaited_once()
                query = self.calendar.get_availability.await_args.args[0]
                self.assertEqual(query.scope.service_id, "service-1")
                self.assertEqual(query.staff_ids, (staff,))
                self.calendar.get_available_slots.assert_not_awaited()
                pending = self.orchestrator._conversations["call-a"].pending_booking
                self.assertEqual(pending["staff_id"], staff)
                self.assertEqual(pending["service_id"], "service-1")
                self.assertEqual(pending["staff_name"], next(
                    r["name"] for r in ROWS if r["staff_id"] == staff))
                self.calendar.get_staff_members.assert_not_awaited()
                self.calendar.get_services.assert_not_awaited()

    async def test_unknown_missing_fuzzy_prefix_and_wrong_types_reject(self):
        for name in (None, "", "anyone", "accountant", "Hus", "Huss",
                     "Rami!", "حسَام", 12, []):
            with self.subTest(name=name):
                await self.reject(name, "BOOKING_NEEDS_CLARIFICATION:")

    async def test_alias_collision_is_ambiguous_in_either_row_order(self):
        for rows in (ROWS, list(reversed(ROWS))):
            changed = [dict(r) for r in rows]
            for r in changed:
                if r["staff_id"] in ("staff-1", "staff-2"):
                    r["booking_aliases"] = [*r["booking_aliases"], "Shared"]
            self.write_rows(changed)
            self.service.reload()
            await self.reject("Shared", "BOOKING_NEEDS_CLARIFICATION:")

    async def test_invalid_configuration_never_falls_back(self):
        cases = [
            [dict(ROWS[0], staff_id="")],
            [dict(ROWS[0], staff_id=" staff-1")],
            [dict(ROWS[0], service_id=None)],
            [dict(ROWS[0], staff_id=17)],
            [dict(ROWS[0], name=42)],
            [dict(ROWS[0], name_ar=42)],
            [dict(ROWS[0], aliases="Hussam")],
            [dict(ROWS[0], booking_aliases="Hussam")],
            [dict(ROWS[0], booking_aliases=[42])],
            [ROWS[0], dict(ROWS[0])],
            [ROWS[0], dict(ROWS[0], name="Other", service_id="service-2")],
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                self.write_rows(rows)
                self.service.reload()
                self.calendar.get_availability.reset_mock()
                await self.reject("Hussam", "BOOKING_ERROR:")

    async def test_missing_and_malformed_yaml_fail_closed(self):
        self.path.unlink()
        self.service.reload()
        await self.reject("Hussam", "BOOKING_ERROR:")
        self.path.write_text("accountants: [", encoding="utf-8")
        self.service.reload()
        await self.reject("Hussam", "BOOKING_ERROR:")

    def test_unicode_nfc_and_collapsed_whitespace(self):
        changed = [dict(r) for r in ROWS]
        changed[0]["booking_aliases"] = [*changed[0]["booking_aliases"], "Café"]
        self.write_rows(changed)
        self.service.reload()
        self.assertEqual(self.service.resolve_booking_accountant("Cafe\u0301").status,
                         "resolved")
        self.assertEqual(self.service.resolve_booking_accountant(
            "  HUSSAM\t SAADALDIN  ").selection.staff_id, "staff-1")

    async def test_wrong_slot_staff_cannot_authorize_and_calls_stay_independent(self):
        self.orchestrator._conversations["call-a"].pending_booking = {
            "appointment_time": "stale"}
        self.calendar.synthetic_slots = [self.slot("staff-2")]
        result = await self.orchestrator._check_booking(
            "call-a", self.arguments("Hussam"))
        self.assertFalse(result.startswith("SLOT_AVAILABLE:"), result)
        self.assertIsNone(self.orchestrator._conversations["call-a"].pending_booking)
        confirmed = await self.orchestrator._confirm_booking("call-a", {"confirm": True})
        self.assertTrue(confirmed.startswith("BOOKING_NEEDS_RECHECK:"), confirmed)
        self.calendar.create_booking.assert_not_awaited()
        self.context("call-b")
        self.calendar.synthetic_slots = [self.slot("staff-2")]
        result = await self.orchestrator._check_booking(
            "call-b", self.arguments("Rami"))
        self.assertTrue(result.startswith("SLOT_AVAILABLE:"), result)
        self.assertEqual(self.orchestrator._conversations["call-b"].pending_booking["staff_id"],
                         "staff-2")
        self.assertIsNone(self.orchestrator._conversations["call-a"].pending_booking)

    async def test_two_successful_calls_keep_distinct_selected_ids(self):
        self.context("call-b")
        for sid, name, staff in (("call-a", "Hussam", "staff-1"),
                                 ("call-b", "Rami", "staff-2")):
            self.calendar.synthetic_slots = [self.slot(staff)]
            result = await self.orchestrator._check_booking(sid, self.arguments(name))
            self.assertTrue(result.startswith("SLOT_AVAILABLE:"), result)
        self.assertEqual(self.orchestrator._conversations["call-a"].pending_booking["staff_id"],
                         "staff-1")
        self.assertEqual(self.orchestrator._conversations["call-b"].pending_booking["staff_id"],
                         "staff-2")

    def test_repository_mapping_has_three_approved_staff_only(self):
        repo = AccountantsService()
        for name, staff in (("Hussam", "93ee7133-8b0c-42c4-a886-a368b998de4b"),
                            ("Rami", "d3d0fa56-b0ef-4267-a15d-85e3af42db38"),
                            ("Abdul", "91142a93-3c60-4127-bdbd-efde7fd61b75")):
            result = repo.resolve_booking_accountant(name)
            self.assertEqual(result.status, "resolved")
            self.assertEqual(result.selection.staff_id, staff)
            self.assertEqual(result.selection.service_id,
                             "357dc857-4360-4801-8bc4-12d3ed63afa3")
        self.assertEqual(repo.resolve_booking_accountant("Diana").status, "not_found")
        self.assertEqual({selection.staff_id for matches in repo._booking_index.values()
                          for selection in matches},
                         {"93ee7133-8b0c-42c4-a886-a368b998de4b",
                          "d3d0fa56-b0ef-4267-a15d-85e3af42db38",
                          "91142a93-3c60-4127-bdbd-efde7fd61b75"})
