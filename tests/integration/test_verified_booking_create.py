"""Real local PostgreSQL and synthetic HTTP journey for the unwired create path."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import unittest
from unittest import mock

from services.calendar.booking_mutations import GraphBookingMutations
from services.calendar.service_facts import ServiceFacts, ServiceFactsRead
from services.config.accountants_service import BookingSelection, BookingSelectionResult
from services.conversation.events import UtteranceIdentity
from services.scheduling.booking_service import (BookingOutcome, BookingService,
                                                 TrustedBookingContext)
from services.scheduling.models import (AvailabilityQuery, AvailabilityResult,
                                        AvailabilityStatus, CalendarScope, FreeInterval,
                                        TimeInterval)
from services.scheduling.policy import AppointmentCandidate
from services.scheduling.proposals import (ApprovedProposal, AppointmentProposal,
                                           CustomerSnapshot, SessionScope)
from tests.calendar.test_graph_booking_mutations import SyntheticHttp, wire
import tests.integration.test_booking_operations as operations


NOW = datetime(2026, 9, 18, 14, tzinfo=timezone.utc)
START = datetime(2026, 9, 21, 14, tzinfo=timezone.utc)
POLICY = "2026-09-18-owner-approved-v1"
LOCATION = "1901 Banff Ave, Ottawa, ON K1V 7W9"


def approved(*, proposal_id="verified-a", staff="staff", start=START,
             customer="Synthetic Customer"):
    scope = CalendarScope("tenant", "business", "service")
    window = TimeInterval(start - timedelta(days=3), start + timedelta(days=3))
    query = AvailabilityQuery(scope, window, (staff,), "proposal-query")
    candidate = AppointmentCandidate(scope, staff,
        TimeInterval(start, start + timedelta(minutes=30)), POLICY, query, NOW)
    proposal = AppointmentProposal(proposal_id, 1, "a" * 64,
        SessionScope("session", "stream"), candidate,
        CustomerSnapshot(customer, "+14165550100", "s@example.invalid"),
        "Synthetic Consultant", "Consultation", LOCATION, "America/Toronto",
        timedelta(0), timedelta(0), NOW - timedelta(seconds=30),
        NOW + timedelta(seconds=90))
    return ApprovedProposal(proposal, UtteranceIdentity("stt", "epoch", 1),
                            NOW - timedelta(seconds=10))


def readback(*, name="Synthetic Customer"):
    result = wire()
    result["start"] = {"dateTime": START.isoformat(), "timeZone": "UTC"}
    result["end"] = {"dateTime": (START + timedelta(minutes=30)).isoformat(), "timeZone": "UTC"}
    result["serviceLocation"] = {"displayName": LOCATION}
    result["customerName"] = name
    result["customers"] = [{"name": name, "phone": "+14165550100",
                            "emailAddress": "s@example.invalid"}]
    return result


class SyntheticCalendar:
    def __init__(self, *, busy=False, stale=False, changed=False):
        self.busy, self.stale, self.changed = busy, stale, changed
        self.after_availability = None

    async def get_service_facts(self, scope):
        facts = ServiceFacts(scope, timedelta(minutes=30), timedelta(0), timedelta(0),
            (("other",) if self.changed else ("staff",)), False, 1, False, timedelta(minutes=30),
            timedelta(minutes=30), timedelta(days=60), True, "bookWhenStaffAreFree", ())
        return ServiceFactsRead("verified", scope,
            NOW - timedelta(minutes=2) if self.stale else NOW, facts)

    async def get_availability(self, query):
        if self.after_availability is not None:
            self.after_availability()
        intervals = () if self.busy else (FreeInterval(query.scope, "staff",
            TimeInterval(START, START + timedelta(minutes=30))),)
        return AvailabilityResult(query,
            AvailabilityStatus.NO_AVAILABILITY if self.busy else AvailabilityStatus.AVAILABLE,
            NOW, intervals)


class Authority:
    allowed = True

    async def valid(self, context, approval):
        return self.allowed

    def current(self, context, approval):
        return self.allowed


class SyntheticSelections:
    def resolve_booking_accountant(self, name):
        return BookingSelectionResult("resolved", BookingSelection(name, "staff", "service"))


class Clock:
    now = NOW

    def __call__(self):
        return self.now


class VerifiedCreateDatabaseTests(unittest.IsolatedAsyncioTestCase):
    # Reuse the accepted isolated PostgreSQL 16 lifecycle without inheriting its
    # test cases or touching an application DSN.
    setUpClass = classmethod(operations.BookingOperationDatabaseTests.setUpClass.__func__)
    tearDownClass = classmethod(operations.BookingOperationDatabaseTests.tearDownClass.__func__)
    asyncSetUp = operations.BookingOperationDatabaseTests.asyncSetUp
    connection = operations.BookingOperationDatabaseTests.connection

    async def _service(self, *, busy=False, stale=False, changed=False, read_wire=None):
        pool = await self.driver.create_pool(self.dsn, min_size=1, max_size=25)
        self.addAsyncCleanup(pool.close)
        http = SyntheticHttp(read_wire=read_wire or readback())
        mutations = GraphBookingMutations(tenant_id="tenant", business_id="business",
            client_id="synthetic", client_secret="synthetic", client=http)
        clock, authority = Clock(), Authority()
        service = BookingService(pool, SyntheticCalendar(busy=busy, stale=stale, changed=changed),
                                 mutations, authority, clock=clock,
                                 selection_loader=SyntheticSelections)
        return service, http, clock, authority

    @staticmethod
    def _context():
        return TrustedBookingContext("tenant", "business", "session", "stream",
                                     "synthetic-call", "en")

    async def test_one_verified_create_and_twenty_identical_invocations(self):
        service, http, _, _ = await self._service()
        intent = approved()
        gate = asyncio.Event()

        async def attempt():
            await gate.wait()
            return await service.create(intent, self._context())

        tasks = [asyncio.create_task(attempt()) for _ in range(20)]
        try:
            gate.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks), 30)
            self.assertGreaterEqual(sum(x.outcome is BookingOutcome.VERIFIED for x in results), 1)
            self.assertTrue(all(x.outcome in (BookingOutcome.VERIFIED, BookingOutcome.PENDING)
                                for x in results))
            self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
                for method, url, _ in http.calls), 1)
            self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.bookings"), 1)
            self.assertEqual(await self.conn.fetchval(
                "SELECT count(*) FROM public.booking_notification_reconciliation"), 1)
            self.assertEqual(await self.conn.fetchval(
                "SELECT count(*) FROM public.booking_notification_outbox WHERE state='held'"), 2)
            row = await self.conn.fetchrow("SELECT state,booking_id,receipt_provider_id "
                                           "FROM public.booking_operations")
            self.assertEqual(row["state"], "applied")
            self.assertIsNotNone(row["booking_id"])
            self.assertEqual(row["receipt_provider_id"], "opaque-provider-id")
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_twenty_competing_slot_intents_have_one_create(self):
        service, http, _, _ = await self._service()
        gate = asyncio.Event()

        async def attempt(index):
            await gate.wait()
            return await service.create(approved(proposal_id=f"competing-{index}"),
                                        self._context())

        tasks = [asyncio.create_task(attempt(index)) for index in range(20)]
        try:
            gate.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks), 30)
            self.assertEqual(sum(item.outcome is BookingOutcome.VERIFIED for item in results), 1)
            self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
                for method, url, _ in http.calls), 1)
            self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.bookings"), 1)
            self.assertEqual(await self.conn.fetchval(
                "SELECT count(*) FROM public.booking_notification_reconciliation"), 1)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_busy_stale_and_revoked_authority_have_no_create(self):
        for mode in ("busy", "stale", "changed", "revoked", "revoked_during_preparation"):
            with self.subTest(mode=mode):
                service, http, _, authority = await self._service(
                    busy=mode == "busy", stale=mode == "stale", changed=mode == "changed")
                if mode == "revoked":
                    authority.allowed = False
                if mode == "revoked_during_preparation":
                    service.calendar.after_availability = lambda: setattr(authority, "allowed", False)
                result = await service.create(approved(proposal_id=mode), self._context())
                self.assertIsNot(result.outcome, BookingOutcome.VERIFIED)
                self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
                    for method, url, _ in http.calls), 0)
                self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.bookings"), 0)

    async def test_expired_proposal_has_zero_provider_posts(self):
        service, http, clock, _ = await self._service()
        clock.now = NOW + timedelta(minutes=2)
        result = await service.create(approved(), self._context())
        self.assertIsNot(result.outcome, BookingOutcome.VERIFIED)
        self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
            for method, url, _ in http.calls), 0)
        self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.bookings"), 0)

    async def test_saved_receipt_recovers_by_get_without_second_post(self):
        mismatched = readback(name="Other")
        service, http, clock, _ = await self._service(read_wire=mismatched)
        intent = approved()
        first = await service.create(intent, self._context())
        self.assertIs(first.outcome, BookingOutcome.PENDING)
        self.assertEqual(await self.conn.fetchval(
            "SELECT receipt_provider_id FROM public.booking_operations"), "opaque-provider-id")
        self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.bookings"), 0)
        http.read_wire = readback()
        clock.now = NOW + timedelta(minutes=2)
        recovered = await service.recover(intent, self._context())
        self.assertIs(recovered.outcome, BookingOutcome.VERIFIED)
        self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
            for method, url, _ in http.calls), 1)
        self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.bookings"), 1)

    async def test_exact_get_404_does_not_release_known_receipt(self):
        service, http, clock, _ = await self._service()
        http.read_status = 404
        intent = approved()
        first = await service.create(intent, self._context())
        self.assertIs(first.outcome, BookingOutcome.PENDING)
        self.assertEqual(await self.conn.fetchval(
            "SELECT receipt_provider_id FROM public.booking_operations"), "opaque-provider-id")
        clock.now = NOW + timedelta(minutes=2)
        second = await service.recover(intent, self._context())
        self.assertIs(second.outcome, BookingOutcome.PENDING)
        self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.bookings"), 0)
        self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
            for method, url, _ in http.calls), 1)

    async def test_arabic_name_is_preserved_without_changing_utc_instant(self):
        name = "عميل تجريبي"
        service, http, _, _ = await self._service(read_wire=readback(name=name))
        result = await service.create(approved(customer=name), self._context())
        self.assertIs(result.outcome, BookingOutcome.VERIFIED)
        booking = await self.conn.fetchrow(
            "SELECT client_name,appointment_time_utc FROM public.bookings")
        self.assertEqual(booking["client_name"], name)
        self.assertEqual(booking["appointment_time_utc"], START)
        import json
        graph_post = next(json.loads(args["content"]) for method, url, args in http.calls
                          if method == "POST" and "graph.microsoft.com" in url)
        self.assertEqual(graph_post["customerName"], name)
        self.assertEqual(graph_post["end"]["dateTime"], "2026-09-21T14:30:00Z")

    async def test_201_without_id_retains_unknown_claim(self):
        service, http, clock, _ = await self._service()
        http.create_wire = {}
        intent = approved()
        result = await service.create(intent, self._context())
        self.assertIs(result.outcome, BookingOutcome.PENDING)
        self.assertEqual(await self.conn.fetchval(
            "SELECT receipt_provider_id FROM public.booking_operations"), None)
        self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.bookings"), 0)
        clock.now = NOW + timedelta(minutes=2)
        recovered = await service.recover(intent, self._context())
        self.assertIs(recovered.outcome, BookingOutcome.PENDING)
        self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
            for method, url, _ in http.calls), 1)

    async def test_receipt_write_failure_keeps_dispatched_without_booking(self):
        service, http, _, _ = await self._service()
        with mock.patch.object(service.store, "save_receipt", return_value=False):
            result = await service.create(approved(), self._context())
        self.assertIs(result.outcome, BookingOutcome.PENDING)
        row = await self.conn.fetchrow(
            "SELECT state,receipt_provider_id FROM public.booking_operations")
        self.assertEqual(row["state"], "dispatched")
        self.assertIsNone(row["receipt_provider_id"])
        self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.bookings"), 0)
        self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
            for method, url, _ in http.calls), 1)

    async def test_atomic_job_failure_rolls_back_booking_but_keeps_receipt(self):
        service, http, clock, _ = await self._service()
        intent = approved()
        with mock.patch("services.sms.notification_outbox.create_new_booking_jobs",
                        side_effect=RuntimeError("synthetic failure")):
            first = await service.create(intent, self._context())
        self.assertIs(first.outcome, BookingOutcome.PENDING)
        self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.bookings"), 0)
        self.assertEqual(await self.conn.fetchval(
            "SELECT count(*) FROM public.booking_notification_outbox"), 0)
        self.assertEqual(await self.conn.fetchval(
            "SELECT receipt_provider_id FROM public.booking_operations"), "opaque-provider-id")
        clock.now = NOW + timedelta(minutes=2)
        recovered = await service.recover(intent, self._context())
        self.assertIs(recovered.outcome, BookingOutcome.VERIFIED)
        self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.bookings"), 1)
        self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
            for method, url, _ in http.calls), 1)

    async def test_atomic_operation_link_failure_rolls_back_booking_and_jobs(self):
        service, http, clock, _ = await self._service()
        intent = approved()
        with mock.patch.object(service.store, "finalize_verified", return_value=False):
            first = await service.create(intent, self._context())
        self.assertIs(first.outcome, BookingOutcome.PENDING)
        self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.bookings"), 0)
        self.assertEqual(await self.conn.fetchval(
            "SELECT count(*) FROM public.booking_notification_outbox"), 0)
        self.assertEqual(await self.conn.fetchval(
            "SELECT receipt_provider_id FROM public.booking_operations"), "opaque-provider-id")
        clock.now = NOW + timedelta(minutes=2)
        recovered = await service.recover(intent, self._context())
        self.assertIs(recovered.outcome, BookingOutcome.VERIFIED)
        self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
            for method, url, _ in http.calls), 1)

    async def test_lease_transfer_fences_delayed_readback_owner(self):
        entered, release = asyncio.Event(), asyncio.Event()

        class GatedHttp(SyntheticHttp):
            @asynccontextmanager
            async def stream(self, method, url, **kwargs):
                if method == "GET" and "graph.microsoft.com" in url:
                    entered.set()
                    await release.wait()
                async with super().stream(method, url, **kwargs) as response:
                    yield response

        service, _, clock, _ = await self._service()
        http = GatedHttp(read_wire=readback())
        service.mutations._client = http
        intent = approved()
        task = asyncio.create_task(service.create(intent, self._context()))
        try:
            await asyncio.wait_for(entered.wait(), 10)
            operation_id = await self.conn.fetchval(
                "SELECT operation_id FROM public.booking_operations")
            clock.now = NOW + timedelta(minutes=2)
            claim = await service.store.claim_reconciliation(
                self.conn, operation_id, now=clock.now)
            self.assertEqual(claim.status.value, "granted")
            release.set()
            result = await asyncio.wait_for(task, 10)
            self.assertIs(result.outcome, BookingOutcome.PENDING)
            self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.bookings"), 0)
            clock.now = NOW + timedelta(minutes=4)
            recovered = await service.recover(intent, self._context())
            self.assertIs(recovered.outcome, BookingOutcome.VERIFIED)
            self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
                for method, url, _ in http.calls), 1)
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_receipt_is_unique_and_survives_inspection_fence(self):
        first = await self.store.admit(self.conn, approved(), now=NOW)
        second = await self.store.admit(self.conn,
            approved(proposal_id="second", staff="other"), now=NOW)
        self.assertIsNotNone(first.operation)
        self.assertIsNotNone(second.operation)
        first_grant = await self.store.claim_dispatch(
            self.conn, first.operation.operation_id, now=NOW)
        second_grant = await self.store.claim_dispatch(
            self.conn, second.operation.operation_id, now=NOW)
        self.assertTrue(await self.store.save_receipt(
            self.conn, first.operation.operation_id, first_grant.fence,
            first_grant.owner_token, "shared-provider-id", now=NOW))
        self.assertFalse(await self.store.save_receipt(
            self.conn, second.operation.operation_id, second_grant.fence,
            second_grant.owner_token, "shared-provider-id", now=NOW))
        self.assertFalse(await self.store.save_receipt(
            self.conn, first.operation.operation_id, first_grant.fence,
            second_grant.owner_token, "other-provider-id", now=NOW))
        inspection = await self.store.claim_reconciliation(
            self.conn, first.operation.operation_id, now=NOW + timedelta(minutes=2))
        self.assertEqual(inspection.status.value, "granted")
        current = await self.store.get(self.conn, first.operation.operation_id)
        self.assertEqual(current.receipt_provider_id, "shared-provider-id")
        self.assertFalse(await self.store.save_receipt(
            self.conn, first.operation.operation_id, first_grant.fence,
            first_grant.owner_token, "other-provider-id", now=NOW + timedelta(minutes=2)))

    async def test_recovery_missing_intent_does_not_reserve_interval(self):
        service, http, _, _ = await self._service()
        result = await service.recover(approved(), self._context())
        self.assertIs(result.outcome, BookingOutcome.INVALID)
        self.assertEqual(http.calls, [])
        self.assertEqual(await self.conn.fetchval(
            "SELECT count(*) FROM public.booking_operations"), 0)

    async def test_expiry_during_last_authority_await_has_no_post(self):
        service, http, clock, authority = await self._service()
        intent = approved()
        checks = 0

        async def delayed(context, approval):
            nonlocal checks
            checks += 1
            if checks == 3:
                await asyncio.sleep(0)
                clock.now = intent.proposal.expires_at + timedelta(seconds=1)
            return True

        authority.valid = delayed
        result = await service.create(intent, self._context())
        self.assertIs(result.outcome, BookingOutcome.PENDING)
        self.assertEqual(checks, 3)
        self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
            for method, url, _ in http.calls), 0)

    async def test_observation_ages_out_during_token_preparation(self):
        service, http, clock, _ = await self._service()
        facts, availability = service.calendar.get_service_facts, service.calendar.get_availability

        async def old_facts(scope):
            return replace(await facts(scope), observed_at=NOW - timedelta(seconds=59))

        async def old_availability(query):
            return replace(await availability(query), observed_at=NOW - timedelta(seconds=59))

        service.calendar.get_service_facts = old_facts
        service.calendar.get_availability = old_availability
        original = http.stream

        @asynccontextmanager
        async def delayed_token(method, url, **kwargs):
            if "login.microsoftonline.com" in url:
                clock.now = NOW + timedelta(seconds=2)
            async with original(method, url, **kwargs) as response:
                yield response

        http.stream = delayed_token
        result = await service.create(approved(), self._context())
        self.assertIs(result.outcome, BookingOutcome.PENDING)
        self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
            for method, url, _ in http.calls), 0)

    async def _admission_gate_scenario(self, mode):
        service, http, clock, authority = await self._service()
        entered, release = asyncio.Event(), asyncio.Event()

        class Gate:
            async def __aenter__(self):
                entered.set()
                await release.wait()

            async def __aexit__(self, *args):
                pass

        checks = 0

        async def current(context, approval):
            nonlocal checks
            checks += 1
            if checks == 3:
                service.mutations._slots = Gate()
            return authority.allowed

        authority.valid = current
        task = asyncio.create_task(service.create(approved(), self._context()))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            if mode == "revoked":
                authority.allowed = False
            elif mode == "transferred":
                operation_id = await self.conn.fetchval(
                    "SELECT operation_id FROM public.booking_operations")
                clock.now = NOW + timedelta(minutes=2)
                inspection = await service.store.claim_reconciliation(
                    self.conn, operation_id, now=clock.now)
                self.assertEqual(inspection.status.value, "granted")
            else:
                task.cancel()
            release.set()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
            if service.mutations._retained:
                await asyncio.wait_for(asyncio.gather(
                    *tuple(service.mutations._retained), return_exceptions=True), 5)
            self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
                for method, url, _ in http.calls), 0)
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_transport_admission_revocation_cannot_post(self):
        await self._admission_gate_scenario("revoked")

    async def test_transport_admission_lease_transfer_cannot_post(self):
        await self._admission_gate_scenario("transferred")

    async def test_transport_admission_cancellation_cannot_post(self):
        await self._admission_gate_scenario("cancelled")

    async def test_pre_and_post_send_pool_failures_are_sanitized(self):
        service, http, _, _ = await self._service()

        class BrokenPool:
            @asynccontextmanager
            async def acquire(self):
                raise RuntimeError("T018_PRIVATE_database_details")
                yield

        real_pool = service.pool
        service.pool = BrokenPool()
        first = await service.create(approved(), self._context())
        recovery = await service.recover(approved(), self._context())
        self.assertIs(first.outcome, BookingOutcome.STORE_ERROR)
        self.assertIs(recovery.outcome, BookingOutcome.STORE_ERROR)
        self.assertNotIn("T018_PRIVATE", repr(first) + repr(recovery))
        self.assertEqual(http.calls, [])

        class FourthAcquireFails:
            def __init__(self):
                self.count = 0

            @asynccontextmanager
            async def acquire(self):
                self.count += 1
                if self.count == 4:
                    raise RuntimeError("T018_PRIVATE_receipt_details")
                async with real_pool.acquire() as conn:
                    yield conn

        service.pool = FourthAcquireFails()
        second = await service.create(approved(), self._context())
        self.assertIs(second.outcome, BookingOutcome.PENDING)
        self.assertNotIn("T018_PRIVATE", repr(second))
        self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
            for method, url, _ in http.calls), 1)
        self.assertEqual(await self.conn.fetchval(
            "SELECT count(*) FROM public.bookings"), 0)

    async def test_dispatch_ownership_read_failure_prevents_post(self):
        service, http, _, _ = await self._service()
        with mock.patch.object(service.store, "inspect_dispatch_owner",
                               side_effect=RuntimeError("T018_PRIVATE_owner_read")):
            result = await service.create(approved(), self._context())
        self.assertIs(result.outcome, BookingOutcome.STORE_ERROR)
        self.assertNotIn("T018_PRIVATE", repr(result))
        self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
            for method, url, _ in http.calls), 0)
        self.assertEqual(await self.conn.fetchval(
            "SELECT state FROM public.booking_operations"), "dispatched")

    async def test_recovery_read_failure_after_receipt_stays_pending(self):
        service, http, clock, _ = await self._service(read_wire=readback(name="Other"))
        intent = approved()
        first = await service.create(intent, self._context())
        self.assertIs(first.outcome, BookingOutcome.PENDING)
        clock.now = NOW + timedelta(minutes=2)
        with mock.patch.object(service.store, "get",
                               side_effect=RuntimeError("T018_PRIVATE_operation_read")):
            result = await service.recover(intent, self._context())
        self.assertIs(result.outcome, BookingOutcome.PENDING)
        self.assertNotIn("T018_PRIVATE", repr(result))
        self.assertEqual(await self.conn.fetchval(
            "SELECT receipt_provider_id FROM public.booking_operations"), "opaque-provider-id")
        self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.bookings"), 0)
        self.assertEqual(sum(method == "POST" and "graph.microsoft.com" in url
            for method, url, _ in http.calls), 1)
