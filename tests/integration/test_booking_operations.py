"""T017-A real PostgreSQL 16 concurrency and restart contracts.

Uses only an isolated, loopback-bound, disposable local container. No application
DSN, provider, or mock database participates in the concurrency assertions.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import unittest
from uuid import uuid4

from services.conversation.events import UtteranceIdentity
from services.scheduling.models import AvailabilityQuery, CalendarScope, TimeInterval
from services.scheduling.policy import AppointmentCandidate
from services.scheduling.proposals import ApprovedProposal, CustomerSnapshot, SessionScope, AppointmentProposal


NOW = datetime(2030, 1, 7, 14, tzinfo=timezone.utc)
START = datetime(2030, 1, 8, 15, tzinfo=timezone.utc)


def approved(*, proposal_id="proposal-a", staff="staff-a", service="service-a",
             business="business-a", start=START, pre=timedelta(0), post=timedelta(0),
             revision=1, phone="+14165550100", expires=NOW + timedelta(minutes=2)):
    scope = CalendarScope("tenant-a", business, service)
    window = TimeInterval(start - timedelta(days=1), start + timedelta(days=1))
    query = AvailabilityQuery(scope, window, (staff,), "query-a")
    candidate = AppointmentCandidate(scope, staff, TimeInterval(start, start + timedelta(minutes=30)),
                                     "policy-a", query, NOW - timedelta(seconds=10))
    proposal = AppointmentProposal(
        proposal_id, revision, "a" * 64, SessionScope("session-a", "stream-a"),
        candidate, CustomerSnapshot("Synthetic Customer", phone, "s@example.invalid"),
        "Synthetic Consultant", "Consultation", "Synthetic office", "America/Toronto",
        pre, post, NOW, expires,
    )
    return ApprovedProposal(proposal, UtteranceIdentity("stt", "epoch-a", 1),
                            NOW + timedelta(seconds=20))


def docker(*args, input=None, timeout=30):
    result = subprocess.run(["docker", "--context", "desktop-linux", *args],
                            input=input, capture_output=True, timeout=timeout,
                            env={key: value for key, value in os.environ.items()
                                 if key not in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY")})
    if result.returncode:
        raise RuntimeError("local docker command failed")
    return result.stdout.decode("utf-8", "replace")


class OperationStoreReadBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_failure_is_sanitized_and_distinct_from_missing(self):
        from services.scheduling.operation_store import OperationStore

        class BrokenConnection:
            async def fetchrow(self, *args):
                raise RuntimeError("T017_PRIVATE_customer@example.invalid")

        with self.assertRaisesRegex(Exception, "operation read failed") as captured:
            await OperationStore().get(BrokenConnection(), uuid4())
        self.assertNotIn("T017_PRIVATE", str(captured.exception))
        self.assertIsNone(captured.exception.__cause__)
        rendered = "".join(traceback.format_exception(captured.exception))
        self.assertNotIn("T017_PRIVATE", rendered)

    async def test_read_cancellation_is_not_converted_to_failure(self):
        from services.scheduling.operation_store import OperationStore

        class CancelledConnection:
            async def fetchrow(self, *args):
                raise asyncio.CancelledError

        with self.assertRaises(asyncio.CancelledError):
            await OperationStore().get(CancelledConnection(), uuid4())


class BookingOperationDatabaseTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("docker") is None:
            raise unittest.SkipTest("local Docker unavailable")
        try:
            if docker("info", "--format", "{{.OSType}}").strip() != "linux":
                raise unittest.SkipTest("local Linux Docker unavailable")
            images = docker("image", "ls", "--format", "{{.Repository}}:{{.Tag}}")
            if "postgres:16-alpine" not in images.splitlines():
                raise unittest.SkipTest("local PostgreSQL 16 image unavailable")
        except RuntimeError:
            raise unittest.SkipTest("local Docker daemon unavailable") from None
        import asyncpg
        cls.driver = asyncpg
        cls.name = "t017-ops-" + uuid4().hex
        cls.database = "t017_" + uuid4().hex
        cls.created = False
        try:
            docker("create", "--pull", "never", "--name", cls.name,
                   "--memory", "256m", "--memory-swap", "256m", "--cpus", "1",
                   "--pids-limit", "128", "--network", "bridge",
                   "--publish", "127.0.0.1::5432",
                   "--tmpfs", "/var/lib/postgresql/data:rw,noexec,nosuid,size=160m",
                   "--env", "PGDATA=/var/lib/postgresql/data/pgdata",
                   "--env", "POSTGRES_USER=t017", "--env", "POSTGRES_PASSWORD=t017-synthetic",
                   "--env", "POSTGRES_DB=" + cls.database, "postgres:16-alpine",
                   "-c", "max_connections=48", "-c", "shared_buffers=16MB")
            cls.created = True
            docker("start", cls.name)
            ports = json.loads(docker("inspect", "--format", "{{json .NetworkSettings.Ports}}", cls.name))
            binding, = ports["5432/tcp"]
            if binding["HostIp"] != "127.0.0.1":
                raise RuntimeError("local port binding rejected")
            cls.dsn = ("postgresql://t017:t017-synthetic@127.0.0.1:"
                       + binding["HostPort"] + "/" + cls.database)
            for _ in range(60):
                try:
                    docker("exec", cls.name, "pg_isready", "-h", "127.0.0.1", "-U", "t017",
                           "-d", cls.database, timeout=3)
                    break
                except RuntimeError:
                    import time
                    time.sleep(0.25)
            else:
                raise RuntimeError("local PostgreSQL startup failed")
        except BaseException:
            if cls.created:
                subprocess.run(["docker", "--context", "desktop-linux", "rm", "-f", cls.name],
                               capture_output=True, timeout=30)
            raise

    @classmethod
    def tearDownClass(cls):
        if cls.created:
            docker("rm", "-f", cls.name)

    async def asyncSetUp(self):
        self.conn = await self.driver.connect(self.dsn, timeout=10, command_timeout=20)
        self.addAsyncCleanup(self.conn.close)
        self.assertTrue(160000 <= int(await self.conn.fetchval("SHOW server_version_num")) < 170000)
        await self.conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        from migrations import runner
        from unittest import mock
        with mock.patch.dict(os.environ, {"MIGRATION_DATABASE_URL": self.dsn}):
            if self._testMethodName == "test_0004_baseline_has_no_staff_interval_exclusion":
                await runner.run(prepare=True)
            else:
                await runner.run(prepare_operations=True)
        if self._testMethodName == "test_0004_baseline_has_no_staff_interval_exclusion":
            return
        from services.scheduling.operation_store import OperationStore
        self.store = OperationStore()

    async def connection(self):
        conn = await self.driver.connect(self.dsn, timeout=10, command_timeout=20)
        self.addAsyncCleanup(conn.close)
        return conn

    async def test_0004_baseline_has_no_staff_interval_exclusion(self):
        for index in range(20):
            await self.conn.execute("""
                INSERT INTO public.bookings (call_sid,appointment_time_utc,status)
                VALUES ($1,$2,'confirmed')
            """, f"synthetic-{index}", START)
        self.assertEqual(await self.conn.fetchval(
            "SELECT count(*) FROM public.bookings WHERE appointment_time_utc=$1", START), 20)

    async def test_twenty_distinct_simultaneous_intents_one_interval_owner(self):
        from services.scheduling.operation_store import AdmissionStatus
        connections = [await self.connection() for _ in range(20)]
        ready = asyncio.Event()
        async def attempt(index):
            await ready.wait()
            return await self.store.admit(connections[index],
                approved(proposal_id=f"proposal-{index}"), now=NOW + timedelta(seconds=30))
        tasks = [asyncio.create_task(attempt(i)) for i in range(20)]
        try:
            ready.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks), 20)
            self.assertEqual(sum(x.status is AdmissionStatus.CREATED for x in results), 1)
            self.assertEqual(sum(x.status is AdmissionStatus.INTERVAL_CONFLICT for x in results), 19)
            self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.booking_operations"), 1)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_conflicting_admission_does_not_wait_for_receipt_writer(self):
        from services.scheduling.operation_store import AdmissionStatus, DispatchStatus

        intent = approved()
        now = NOW + timedelta(seconds=30)
        admitted = await self.store.admit(self.conn, intent, now=now)
        grant = await self.store.claim_dispatch(self.conn, admitted.operation.operation_id, now=now)
        self.assertIs(grant.status, DispatchStatus.GRANTED)
        writer = await self.connection()
        contender = await self.connection()
        transaction = writer.transaction()
        await transaction.start()
        task = None
        try:
            self.assertTrue(await self.store.save_receipt(
                writer, admitted.operation.operation_id, grant.fence, grant.owner_token,
                "synthetic-provider-receipt", now=now))
            task = asyncio.create_task(self.store.admit(
                contender, approved(proposal_id="competing-receipt"), now=now))
            # The committed interval already excludes this contender. It must
            # return conflict without entering GiST's wait on the receipt writer.
            done, _ = await asyncio.wait({task}, timeout=2)
            self.assertIn(task, done, "conflicting admission waited on receipt persistence")
            self.assertIs(task.result().status, AdmissionStatus.INTERVAL_CONFLICT)
        finally:
            await transaction.rollback()
            if task is not None:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.booking_operations"), 1)

    async def test_admission_wait_is_staff_scoped_and_cancellation_releases_transaction(self):
        from services.scheduling.operation_store import AdmissionStatus

        now = NOW + timedelta(seconds=30)
        waiter = await self.connection()
        other_staff = await self.connection()
        transaction = self.conn.transaction()
        await transaction.start()
        task = None
        try:
            first = await self.store.admit(self.conn, approved(), now=now)
            self.assertIs(first.status, AdmissionStatus.CREATED)
            task = asyncio.create_task(self.store.admit(
                waiter, approved(proposal_id="blocked"), now=now))
            result = await asyncio.wait_for(self.store.admit(other_staff,
                approved(proposal_id="other-staff", staff="staff-b"), now=now), 2)
            self.assertIs(result.status, AdmissionStatus.CREATED)
            self.assertFalse(task.done())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(waiter.is_in_transaction())
        finally:
            if task is not None:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await transaction.rollback()
        retry = await self.store.admit(waiter, approved(proposal_id="after-rollback"), now=now)
        self.assertIs(retry.status, AdmissionStatus.CREATED)

    async def test_get_distinguishes_existing_missing_and_closed_connection(self):
        from services.scheduling.operation_store import OperationReadError
        intent = approved()
        created = await self.store.admit(self.conn, intent, now=intent.approved_at)
        self.assertEqual((await self.store.get(
            self.conn, created.operation.operation_id)).operation_id,
            created.operation.operation_id)
        self.assertIsNone(await self.store.get(self.conn, uuid4()))
        broken = await self.connection()
        await broken.close()
        with self.assertRaisesRegex(OperationReadError, "operation read failed"):
            await self.store.get(broken, created.operation.operation_id)

    async def test_exact_existing_intent_lookup_is_read_only_and_scoped(self):
        from services.scheduling.operation_store import AdmissionStatus
        intent = approved()
        missing = await self.store.find_existing_intent(self.conn, intent)
        self.assertIs(missing.status, AdmissionStatus.INVALID)
        self.assertEqual(await self.conn.fetchval(
            "SELECT count(*) FROM public.booking_operations"), 0)
        created = await self.store.admit(self.conn, intent, now=intent.approved_at)
        found = await self.store.find_existing_intent(self.conn, intent)
        self.assertIs(found.status, AdmissionStatus.EXISTING)
        self.assertEqual(found.operation.operation_id, created.operation.operation_id)
        changed = await self.store.find_existing_intent(
            self.conn, approved(phone="+14165550199"))
        self.assertIs(changed.status, AdmissionStatus.PAYLOAD_CONFLICT)
        foreign = await self.store.find_existing_intent(
            self.conn, approved(business="other-business"))
        self.assertIs(foreign.status, AdmissionStatus.INVALID)
        self.assertEqual(await self.conn.fetchval(
            "SELECT count(*) FROM public.booking_operations"), 1)

    async def test_admission_and_dispatch_causal_clock_boundaries(self):
        from services.scheduling.operation_store import (AdmissionStatus, DispatchStatus,
            TransitionStatus)
        intent = approved()
        before = await self.store.admit(self.conn, intent, now=intent.approved_at - timedelta(microseconds=1))
        self.assertIs(before.status, AdmissionStatus.INVALID)
        self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.booking_operations"), 0)
        exact = await self.store.admit(self.conn, intent, now=intent.approved_at)
        self.assertIs(exact.status, AdmissionStatus.CREATED)
        admission = await self.conn.fetchval(
            "SELECT admitted_at FROM public.booking_operations WHERE operation_id=$1",
            exact.operation.operation_id)
        self.assertEqual(admission, intent.approved_at)
        early = await self.store.claim_dispatch(
            self.conn, exact.operation.operation_id,
            now=intent.approved_at - timedelta(microseconds=1))
        self.assertIsNot(early.status, DispatchStatus.GRANTED)
        self.assertEqual((await self.store.get(self.conn, exact.operation.operation_id)).state,
                         "pending")
        granted = await self.store.claim_dispatch(
            self.conn, exact.operation.operation_id, now=intent.approved_at)
        self.assertIs(granted.status, DispatchStatus.GRANTED)

        cancel_intent = approved(proposal_id="exact-cancel", staff="staff-b")
        cancel_created = await self.store.admit(
            self.conn, cancel_intent, now=cancel_intent.approved_at)
        cancelled = await self.store.cancel_before_dispatch(
            self.conn, cancel_created.operation.operation_id,
            now=cancel_intent.approved_at)
        self.assertIs(cancelled.status, TransitionStatus.CANCELLED)

    async def test_existing_exact_intent_remains_inspectable_after_expiry(self):
        from services.scheduling.operation_store import AdmissionStatus
        intent = approved()
        created = await self.store.admit(self.conn, intent, now=intent.approved_at)
        self.assertIs(created.status, AdmissionStatus.CREATED)
        existing = await self.store.admit(self.conn, intent,
                                          now=intent.proposal.expires_at + timedelta(days=1))
        self.assertIs(existing.status, AdmissionStatus.EXISTING)
        self.assertEqual(existing.operation.operation_id, created.operation.operation_id)

    async def test_cancel_and_reconciliation_reject_backwards_clocks(self):
        from services.scheduling.operation_store import (DispatchStatus, ReconcileStatus,
            TransitionStatus)
        intent = approved()
        created = await self.store.admit(self.conn, intent, now=intent.approved_at)
        cancelled = await self.store.cancel_before_dispatch(
            self.conn, created.operation.operation_id,
            now=intent.approved_at - timedelta(microseconds=1))
        self.assertIs(cancelled.status, TransitionStatus.INVALID_STATE)
        grant = await self.store.claim_dispatch(
            self.conn, created.operation.operation_id, now=intent.approved_at)
        self.assertIs(grant.status, DispatchStatus.GRANTED)
        backwards = await self.store.claim_reconciliation(
            self.conn, created.operation.operation_id,
            now=intent.approved_at - timedelta(microseconds=1))
        self.assertIs(backwards.status, ReconcileStatus.UNAVAILABLE)
        self.assertEqual((await self.store.get(self.conn, created.operation.operation_id)).state,
                         "dispatched")

    async def test_reconciliation_cannot_precede_latest_manual_review(self):
        from services.scheduling.operation_store import (
            ReconcileStatus, SettlementEvidence, SettlementKind, TransitionStatus)
        admitted = await self.store.admit(
            self.conn, approved(), now=NOW + timedelta(seconds=30))
        operation_id = admitted.operation.operation_id
        grant = await self.store.claim_dispatch(
            self.conn, operation_id, now=NOW + timedelta(seconds=31))
        reviewed_at = NOW + timedelta(minutes=3)
        reviewed = await self.store.settle(
            self.conn, operation_id, grant.fence, grant.owner_token,
            SettlementEvidence(SettlementKind.MANUAL_REVIEW,
                               "trusted-review", reviewed_at))
        self.assertIs(reviewed.status, TransitionStatus.MANUAL_REVIEW)
        # The lease expired, but this clock predates the latest committed result.
        stale = await self.store.claim_reconciliation(
            self.conn, operation_id, now=NOW + timedelta(minutes=2))
        self.assertIs(stale.status, ReconcileStatus.UNAVAILABLE)
        preserved = await self.conn.fetchrow(
            "SELECT state,fence,settlement_observed_at FROM public.booking_operations "
            "WHERE operation_id=$1", operation_id)
        self.assertEqual(preserved["state"], "manual_review")
        self.assertEqual(preserved["fence"], grant.fence)
        self.assertEqual(preserved["settlement_observed_at"], reviewed_at)
        current = await self.store.claim_reconciliation(
            self.conn, operation_id, now=reviewed_at)
        self.assertIs(current.status, ReconcileStatus.GRANTED)
        self.assertFalse(current.create_permission)

    async def test_twenty_identical_retries_one_operation_and_dispatch_grant(self):
        from services.scheduling.operation_store import AdmissionStatus, DispatchStatus
        connections = [await self.connection() for _ in range(20)]
        start = asyncio.Event()
        intent = approved()
        async def attempt(index):
            await start.wait()
            return await self.store.admit(connections[index], intent,
                                          now=NOW + timedelta(seconds=30))
        tasks = [asyncio.create_task(attempt(i)) for i in range(20)]
        try:
            start.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks), 20)
            self.assertEqual(sum(x.status is AdmissionStatus.CREATED for x in results), 1)
            self.assertEqual(sum(x.status is AdmissionStatus.EXISTING for x in results), 19)
            ids = {x.operation.operation_id for x in results}
            self.assertEqual(len(ids), 1)
            claims = await asyncio.wait_for(asyncio.gather(*[
                self.store.claim_dispatch(connections[i], next(iter(ids)),
                    now=NOW + timedelta(seconds=31)) for i in range(20)]), 20)
            self.assertEqual(sum(x.status is DispatchStatus.GRANTED for x in claims), 1)
            self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.booking_operations"), 1)
            mismatch = await self.store.admit(self.conn, approved(phone="+14165550199"),
                                               now=NOW + timedelta(minutes=5))
            self.assertIs(mismatch.status, AdmissionStatus.PAYLOAD_CONFLICT)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_direct_sql_exclusion_buffers_adjacency_and_scopes(self):
        from services.scheduling.operation_store import AdmissionStatus
        first = await self.store.admit(self.conn, approved(pre=timedelta(minutes=5)),
                                       now=NOW + timedelta(seconds=30))
        self.assertIs(first.status, AdmissionStatus.CREATED)
        # Direct SQL cannot bypass the cross-service staff exclusion.
        with self.assertRaises(self.driver.ExclusionViolationError):
            await self.conn.execute("""
                INSERT INTO public.booking_operations
                    (operation_id,tenant_id,business_id,staff_id,service_id,action,
                     proposal_id,proposal_revision,payload_hash,payload_snapshot,
                     confirmation_identity,confirmed_at,issued_at,expires_at,admitted_at,
                     starts_at,ends_at,pre_buffer,post_buffer,claim_span)
                SELECT $1,tenant_id,business_id,staff_id,'other-service',action,
                       'other-proposal',proposal_revision,payload_hash,payload_snapshot,
                       confirmation_identity,confirmed_at,issued_at,expires_at,admitted_at,
                       starts_at,ends_at,pre_buffer,post_buffer,claim_span
                FROM public.booking_operations WHERE operation_id=$2
            """, uuid4(), first.operation.operation_id)
        adjacent = await self.store.admit(self.conn, approved(proposal_id="adjacent",
            start=START + timedelta(minutes=30)), now=NOW + timedelta(seconds=30))
        self.assertIs(adjacent.status, AdmissionStatus.CREATED)
        other_staff = await self.store.admit(self.conn, approved(proposal_id="staff-b", staff="staff-b"),
                                             now=NOW + timedelta(seconds=30))
        self.assertIs(other_staff.status, AdmissionStatus.CREATED)
        other_business = await self.store.admit(self.conn,
            approved(proposal_id="business-b", business="business-b"),
            now=NOW + timedelta(seconds=30))
        self.assertIs(other_business.status, AdmissionStatus.CREATED)
        across_service = await self.store.admit(self.conn,
            approved(proposal_id="service-b", service="service-b"),
            now=NOW + timedelta(seconds=30))
        self.assertIs(across_service.status, AdmissionStatus.INTERVAL_CONFLICT)
        with self.assertRaises(self.driver.CheckViolationError):
            await self.conn.execute("""
                INSERT INTO public.booking_operations
                    (operation_id,tenant_id,business_id,staff_id,service_id,action,
                     proposal_id,proposal_revision,payload_hash,payload_snapshot,
                     confirmation_identity,confirmed_at,issued_at,expires_at,admitted_at,
                     starts_at,ends_at,pre_buffer,post_buffer,claim_span)
                SELECT $1,tenant_id,business_id,'other-staff',service_id,action,
                       'bad-span',proposal_revision,payload_hash,payload_snapshot,
                       confirmation_identity,confirmed_at,issued_at,expires_at,admitted_at,
                       starts_at,ends_at,pre_buffer,post_buffer,
                       tstzrange(starts_at,ends_at + INTERVAL '1 day','[)')
                FROM public.booking_operations WHERE operation_id=$2
            """, uuid4(), first.operation.operation_id)

    async def test_restart_dispatch_uncertainty_fence_and_settlement(self):
        from services.scheduling.operation_store import (AdmissionStatus, DispatchStatus,
            ReconcileStatus, SettlementKind, SettlementEvidence, TransitionStatus, OperationStore)
        created = await self.store.admit(self.conn, approved(), now=NOW + timedelta(seconds=30))
        self.assertIs(created.status, AdmissionStatus.CREATED)
        other = OperationStore()
        new_conn = await self.connection()
        self.assertEqual((await other.get(new_conn, created.operation.operation_id)).state, "pending")
        grant = await other.claim_dispatch(new_conn, created.operation.operation_id,
                                           now=NOW + timedelta(seconds=31))
        self.assertIs(grant.status, DispatchStatus.GRANTED)
        self.assertIs((await self.store.claim_dispatch(self.conn, created.operation.operation_id,
            now=NOW + timedelta(hours=1))).status, DispatchStatus.UNAVAILABLE)
        inspection = await other.claim_reconciliation(new_conn, created.operation.operation_id,
                                                      now=NOW + timedelta(hours=1))
        self.assertIs(inspection.status, ReconcileStatus.GRANTED)
        self.assertFalse(inspection.create_permission)
        evidence = SettlementEvidence(SettlementKind.VERIFIED_APPLIED, "trusted-readback",
                                      NOW + timedelta(hours=1, seconds=1), "provider-a")
        stale = await self.store.settle(self.conn, created.operation.operation_id, grant.fence,
                                        grant.owner_token, evidence)
        self.assertIs(stale.status, TransitionStatus.STALE_FENCE)
        settled = await other.settle(new_conn, created.operation.operation_id, inspection.fence,
                                     inspection.owner_token, evidence)
        self.assertIs(settled.status, TransitionStatus.APPLIED)
        self.assertIs((await other.settle(new_conn, created.operation.operation_id,
            inspection.fence, inspection.owner_token, evidence)).status, TransitionStatus.INVALID_STATE)
        self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.booking_operations"), 1)

    async def test_settlement_evidence_must_belong_to_current_ownership(self):
        from services.scheduling.operation_store import (DispatchStatus, ReconcileStatus,
            SettlementKind, SettlementEvidence, TransitionStatus)
        for index, kind in enumerate((SettlementKind.VERIFIED_NOT_APPLIED,
                                      SettlementKind.VERIFIED_APPLIED,
                                      SettlementKind.MANUAL_REVIEW), 1):
            with self.subTest(kind=kind.value):
                start = START + timedelta(hours=index)
                intent = approved(proposal_id=f"causal-{index}", start=start)
                created = await self.store.admit(self.conn, intent, now=intent.approved_at)
                dispatched_at = intent.approved_at + timedelta(seconds=1)
                grant = await self.store.claim_dispatch(
                    self.conn, created.operation.operation_id, now=dispatched_at)
                self.assertIs(grant.status, DispatchStatus.GRANTED)
                stale = SettlementEvidence(
                    kind, "trusted-stale", dispatched_at - timedelta(microseconds=1),
                    "provider-a" if kind is SettlementKind.VERIFIED_APPLIED else None)
                rejected = await self.store.settle(
                    self.conn, created.operation.operation_id, grant.fence,
                    grant.owner_token, stale)
                self.assertIs(rejected.status, TransitionStatus.INVALID_STATE)
                self.assertEqual((await self.store.get(
                    self.conn, created.operation.operation_id)).state, "dispatched")
                exact = SettlementEvidence(
                    kind, "trusted-exact", dispatched_at,
                    "provider-a" if kind is SettlementKind.VERIFIED_APPLIED else None)
                accepted = await self.store.settle(
                    self.conn, created.operation.operation_id, grant.fence,
                    grant.owner_token, exact)
                self.assertIn(accepted.status, {
                    TransitionStatus.RELEASED, TransitionStatus.APPLIED,
                    TransitionStatus.MANUAL_REVIEW})

        transfer_intent = approved(proposal_id="transfer", start=START + timedelta(hours=5))
        transfer = await self.store.admit(self.conn, transfer_intent,
                                          now=transfer_intent.approved_at)
        dispatch_at = transfer_intent.approved_at + timedelta(seconds=1)
        grant = await self.store.claim_dispatch(
            self.conn, transfer.operation.operation_id, now=dispatch_at,
            lease=timedelta(seconds=1))
        reconcile_at = dispatch_at + timedelta(seconds=1)
        inspection = await self.store.claim_reconciliation(
            self.conn, transfer.operation.operation_id, now=reconcile_at)
        self.assertIs(inspection.status, ReconcileStatus.GRANTED)
        old_evidence = SettlementEvidence(
            SettlementKind.VERIFIED_NOT_APPLIED, "trusted-old-owner", dispatch_at)
        rejected = await self.store.settle(
            self.conn, transfer.operation.operation_id, inspection.fence,
            inspection.owner_token, old_evidence)
        self.assertIs(rejected.status, TransitionStatus.INVALID_STATE)
        self.assertEqual((await self.store.get(
            self.conn, transfer.operation.operation_id)).state, "unresolved")
        fresh_evidence = SettlementEvidence(
            SettlementKind.VERIFIED_NOT_APPLIED, "trusted-new-owner", reconcile_at)
        self.assertIs((await self.store.settle(
            self.conn, transfer.operation.operation_id, inspection.fence,
            inspection.owner_token, fresh_evidence)).status, TransitionStatus.RELEASED)

    async def test_failed_finalization_transaction_keeps_unresolved_claim(self):
        from services.scheduling.operation_store import SettlementKind, SettlementEvidence
        created = await self.store.admit(self.conn, approved(), now=NOW + timedelta(seconds=30))
        grant = await self.store.claim_dispatch(self.conn, created.operation.operation_id,
                                                now=NOW + timedelta(seconds=31))
        with self.assertRaises(RuntimeError):
            async with self.conn.transaction():
                await self.store.settle(self.conn, created.operation.operation_id,
                    grant.fence, grant.owner_token,
                    SettlementEvidence(SettlementKind.VERIFIED_APPLIED, "trusted-readback",
                                       NOW + timedelta(seconds=40), "provider-a"))
                raise RuntimeError("synthetic finalization failure")
        self.assertEqual(await self.conn.fetchval(
            "SELECT state FROM public.booking_operations WHERE operation_id=$1",
            created.operation.operation_id), "dispatched")
        recovered = await self.store.settle(self.conn, created.operation.operation_id,
            grant.fence, grant.owner_token,
            SettlementEvidence(SettlementKind.VERIFIED_APPLIED, "trusted-readback",
                               NOW + timedelta(seconds=41), "provider-a"))
        self.assertEqual(recovered.operation.operation_id, created.operation.operation_id)
        self.assertEqual(await self.conn.fetchval(
            "SELECT state FROM public.booking_operations WHERE operation_id=$1",
            created.operation.operation_id), "applied")

    async def test_dispatch_refuses_outer_transaction_and_second_process_cannot_redispatch(self):
        from services.scheduling.operation_store import DispatchStatus
        created = await self.store.admit(self.conn, approved(), now=NOW + timedelta(seconds=30))
        async with self.conn.transaction():
            self.assertIs((await self.store.claim_dispatch(self.conn, created.operation.operation_id,
                now=NOW + timedelta(seconds=31))).status, DispatchStatus.INVALID)
        grant = await self.store.claim_dispatch(self.conn, created.operation.operation_id,
                                                now=NOW + timedelta(seconds=31))
        self.assertIs(grant.status, DispatchStatus.GRANTED)
        code = """import asyncio, os, sys
from datetime import datetime, timezone
from uuid import UUID
import asyncpg
from services.scheduling.operation_store import OperationStore

async def run():
    conn = await asyncpg.connect(os.environ['T017_TEST_DSN'])
    try:
        result = await OperationStore().claim_dispatch(
            conn, UUID(sys.argv[1]), now=datetime(2030, 1, 7, 15, tzinfo=timezone.utc))
        print(result.status.value)
    finally:
        await conn.close()

asyncio.run(run())"""
        child = await asyncio.create_subprocess_exec(sys.executable, "-B", "-c", code,
            str(created.operation.operation_id), stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "T017_TEST_DSN": self.dsn})
        output, errors = await asyncio.wait_for(child.communicate(), 12)
        self.assertEqual(child.returncode, 0, errors.decode("utf-8", "replace"))
        self.assertEqual(output.strip(), b"unavailable")

    async def test_manual_review_is_not_create_permission_and_fences_old_owner(self):
        from services.scheduling.operation_store import (SettlementKind, SettlementEvidence,
            TransitionStatus, ReconcileStatus, DispatchStatus)
        created = await self.store.admit(self.conn, approved(), now=NOW + timedelta(seconds=30))
        grant = await self.store.claim_dispatch(self.conn, created.operation.operation_id,
                                                now=NOW + timedelta(seconds=31))
        self.assertIs(grant.status, DispatchStatus.GRANTED)
        manual = SettlementEvidence(SettlementKind.MANUAL_REVIEW, "trusted-inspector",
                                    NOW + timedelta(seconds=40))
        self.assertIs((await self.store.settle(self.conn, created.operation.operation_id,
            grant.fence, grant.owner_token, manual)).status, TransitionStatus.MANUAL_REVIEW)
        self.assertIs((await self.store.claim_dispatch(self.conn, created.operation.operation_id,
            now=NOW + timedelta(hours=1))).status, DispatchStatus.UNAVAILABLE)
        inspection = await self.store.claim_reconciliation(self.conn, created.operation.operation_id,
                                                            now=NOW + timedelta(hours=1))
        self.assertIs(inspection.status, ReconcileStatus.GRANTED)
        self.assertFalse(inspection.create_permission)
        self.assertIs((await self.store.settle(self.conn, created.operation.operation_id,
            grant.fence, grant.owner_token, manual)).status, TransitionStatus.STALE_FENCE)

    async def test_predispatch_cancel_release_and_verified_not_applied_release(self):
        from services.scheduling.operation_store import (AdmissionStatus, TransitionStatus,
            SettlementKind, SettlementEvidence, DispatchStatus)
        first = await self.store.admit(self.conn, approved(), now=NOW + timedelta(seconds=30))
        self.assertIs((await self.store.cancel_before_dispatch(self.conn, first.operation.operation_id,
            now=NOW + timedelta(seconds=31))).status, TransitionStatus.CANCELLED)
        self.assertIs((await self.store.claim_dispatch(self.conn, first.operation.operation_id,
            now=NOW + timedelta(seconds=32))).status, DispatchStatus.UNAVAILABLE)
        second = await self.store.admit(self.conn, approved(proposal_id="second"),
                                        now=NOW + timedelta(seconds=32))
        self.assertIs(second.status, AdmissionStatus.CREATED)
        grant = await self.store.claim_dispatch(self.conn, second.operation.operation_id,
                                                now=NOW + timedelta(seconds=33))
        self.assertIs((await self.store.cancel_before_dispatch(self.conn, second.operation.operation_id,
            now=NOW + timedelta(seconds=34))).status, TransitionStatus.INVALID_STATE)
        evidence = SettlementEvidence(SettlementKind.VERIFIED_NOT_APPLIED,
                                      "trusted-provider-audit", NOW + timedelta(seconds=35))
        self.assertIs((await self.store.settle(self.conn, second.operation.operation_id,
            grant.fence, grant.owner_token, evidence)).status, TransitionStatus.RELEASED)
        third = await self.store.admit(self.conn, approved(proposal_id="third"),
                                       now=NOW + timedelta(seconds=36))
        self.assertIs(third.status, AdmissionStatus.CREATED)
        self.assertIs((await self.store.admit(self.conn, approved(proposal_id="second"),
            now=NOW + timedelta(minutes=5))).status, AdmissionStatus.EXISTING)

    async def test_invalid_approval_and_expiry_do_not_create_intents(self):
        from dataclasses import replace
        from services.scheduling.operation_store import AdmissionStatus, DispatchStatus
        base = approved()
        self.assertIs((await self.store.admit(self.conn, base, now=base.proposal.expires_at)).status,
                      AdmissionStatus.EXPIRED)
        self.assertIs((await self.store.admit(self.conn,
            replace(base, approved_at=NOW - timedelta(seconds=1)), now=NOW)).status,
            AdmissionStatus.INVALID)
        self.assertIs((await self.store.admit(self.conn, base,
            now=NOW.replace(tzinfo=None))).status, AdmissionStatus.INVALID)
        self.assertIs((await self.store.admit(self.conn,
            replace(base, proposal=replace(base.proposal, pre_buffer=timedelta(hours=3))),
            now=NOW)).status, AdmissionStatus.INVALID)
        self.assertIs((await self.store.admit(self.conn,
            replace(base, proposal=replace(base.proposal, customer=None)),
            now=NOW)).status, AdmissionStatus.INVALID)
        self.assertEqual(await self.conn.fetchval("SELECT count(*) FROM public.booking_operations"), 0)
        for value in (base, self.store):
            self.assertNotIn("Synthetic Customer", repr(value))
        created = await self.store.admit(self.conn, base, now=NOW + timedelta(seconds=30))
        self.assertIs((await self.store.claim_dispatch(self.conn, created.operation.operation_id,
            now=base.proposal.expires_at)).status, DispatchStatus.UNAVAILABLE)

    async def test_direct_sql_rejects_impossible_state_and_null_bypasses(self):
        created = await self.store.admit(self.conn, approved(),
                                         now=NOW + timedelta(seconds=20))
        impossible = (
            "state='dispatched',dispatch_count=1,fence=0,owner_token=NULL,lease_until=NULL,"
            "ownership_started_at=NULL")
        with self.assertRaises(self.driver.CheckViolationError):
            await self.conn.execute(
                "UPDATE public.booking_operations SET " + impossible + " WHERE operation_id=$1",
                created.operation.operation_id)
        for statement, parameters in (
            ("UPDATE public.booking_operations SET settlement_source='fabricated' "
             "WHERE operation_id=$1", (created.operation.operation_id,)),
            ("UPDATE public.booking_operations SET state='applied',dispatch_count=1,fence=1,"
             "owner_token=$2,lease_until=$3,ownership_started_at=$3 WHERE operation_id=$1",
             (created.operation.operation_id, uuid4(), NOW + timedelta(minutes=1))),
        ):
            with self.assertRaises(self.driver.CheckViolationError):
                await self.conn.execute(statement, *parameters)

    async def test_0005_upgrade_repeat_history_preservation_and_drift(self):
        from migrations import runner
        from migrations.schema_contract import (check_runtime_compatibility,
            SchemaCompatibilityError, OPERATION_REVISION_CHECKSUM)
        from unittest import mock
        await self.conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        with mock.patch.dict(os.environ, {"MIGRATION_DATABASE_URL": self.dsn}):
            self.assertIn("0004", await runner.run(prepare=True))
        await self.conn.execute("""
            INSERT INTO public.bookings (call_sid,appointment_time_utc,status)
            VALUES ('synthetic-preserve',TIMESTAMPTZ '2030-01-08 15:00+00','confirmed')
        """)
        before = await self.conn.fetch("SELECT version,checksum FROM public.schema_migrations ORDER BY version")
        with mock.patch.dict(os.environ, {"MIGRATION_DATABASE_URL": self.dsn}):
            self.assertEqual(await runner.run(prepare_operations=True), "applied 0005")
            self.assertEqual(await runner.run(prepare_operations=True), "up-to-date 0005")
            self.assertEqual(await runner.run(), "up-to-date 0005")
        after = await self.conn.fetch("SELECT version,checksum FROM public.schema_migrations ORDER BY version")
        self.assertEqual({row["version"]: row["checksum"] for row in before}.items()
                         <= {row["version"]: row["checksum"] for row in after}.items(), True)
        self.assertEqual({row["version"]: row["checksum"] for row in after}["0005"],
                         OPERATION_REVISION_CHECKSUM)
        self.assertEqual(await self.conn.fetchval(
            "SELECT count(*) FROM public.bookings WHERE call_sid='synthetic-preserve'"), 1)
        async with self.conn.transaction(readonly=True):
            await check_runtime_compatibility(self.conn, require_notification=True)
        transaction = self.conn.transaction()
        await transaction.start()
        try:
            await self.conn.execute("ALTER TABLE public.booking_operations DROP CONSTRAINT "
                                    "booking_operations_staff_exclusion")
            with self.assertRaises(SchemaCompatibilityError):
                await check_runtime_compatibility(self.conn, require_notification=True)
        finally:
            await transaction.rollback()
        for name in ("booking_operations_interval_check", "booking_operations_intent_key"):
            transaction = self.conn.transaction()
            await transaction.start()
            try:
                await self.conn.execute("ALTER TABLE public.booking_operations DROP CONSTRAINT " + name)
                with self.assertRaises(SchemaCompatibilityError):
                    await check_runtime_compatibility(self.conn, require_notification=True)
            finally:
                await transaction.rollback()
        transaction = self.conn.transaction()
        await transaction.start()
        try:
            await self.conn.execute("ALTER TABLE public.sms_logs DROP CONSTRAINT "
                                    "sms_logs_booking_id_fkey")
            with self.assertRaises(SchemaCompatibilityError):
                await check_runtime_compatibility(self.conn, require_notification=True)
        finally:
            await transaction.rollback()

    async def test_0005_sql_transaction_rollback_leaves_no_partial_schema(self):
        from migrations import runner
        from unittest import mock
        await self.conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        with mock.patch.dict(os.environ, {"MIGRATION_DATABASE_URL": self.dsn}):
            await runner.run(prepare=True)
        sql = (Path(__file__).resolve().parents[2] /
               "migrations/0005_booking_operations.sql").read_text(encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "synthetic rollback"):
            async with self.conn.transaction():
                await self.conn.execute(sql)
                raise RuntimeError("synthetic rollback")
        self.assertIsNone(await self.conn.fetchval(
            "SELECT pg_catalog.to_regclass('public.booking_operations')"))
        self.assertEqual(await self.conn.fetchval(
            "SELECT count(*) FROM public.schema_migrations WHERE version='0005'"), 0)
        with mock.patch.dict(os.environ, {"MIGRATION_DATABASE_URL": self.dsn}):
            self.assertEqual(await runner.run(prepare_operations=True), "applied 0005")

    async def test_dump_restore_preserves_exclusion_and_history(self):
        from migrations import runner
        from unittest import mock
        archive = subprocess.run(["docker", "--context", "desktop-linux", "exec", self.name,
            "pg_dump", "-U", "t017", "-d", self.database, "-Fc"],
            capture_output=True, timeout=60, check=True).stdout
        await self.conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        subprocess.run(["docker", "--context", "desktop-linux", "exec", "-i", self.name,
            "pg_restore", "--exit-on-error", "--single-transaction", "--no-owner", "--no-acl",
            "-U", "t017", "-d", self.database], input=archive,
            capture_output=True, timeout=60, check=True)
        with mock.patch.dict(os.environ, {"MIGRATION_DATABASE_URL": self.dsn}):
            self.assertEqual(await runner.run(), "up-to-date 0005")
        first = await self.store.admit(self.conn, approved(), now=NOW + timedelta(seconds=30))
        from services.scheduling.operation_store import AdmissionStatus
        self.assertIs(first.status, AdmissionStatus.CREATED)
        self.assertIs((await self.store.admit(self.conn, approved(proposal_id="overlap"),
            now=NOW + timedelta(seconds=30))).status, AdmissionStatus.INTERVAL_CONFLICT)


if __name__ == "__main__":
    unittest.main()
