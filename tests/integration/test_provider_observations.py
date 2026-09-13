"""T018-B offline contracts; real PostgreSQL cases run in the local DB suite."""
from __future__ import annotations

import asyncio
import logging
import importlib
from pathlib import Path
import sys
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace
from unittest import mock
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


UTC = timezone.utc
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def payload(**changes):
    result = {
        "id": "synthetic/provider id",
        "startDateTime": {"dateTime": "2026-09-14T15:00:00.0000000", "timeZone": "UTC"},
        "endDateTime": {"dateTime": "2026-09-14T15:30:00.0000000Z", "timeZone": "UTC"},
        "staffMemberIds": ["synthetic-staff"],
        "serviceId": "synthetic-service",
        "isLocationOnline": False,
        "customerName": "PRIVATE_CONTACT",
    }
    result.update(changes)
    return result


class ReadbackParsingTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("services.calendar.booking_readback")

    def test_observed_official_and_equal_dual_pairs(self):
        observed = self.module.parse_appointment(payload(), "synthetic/provider id")
        self.assertEqual(datetime(2026, 9, 14, 15, tzinfo=UTC), observed.start)
        self.assertEqual(datetime(2026, 9, 14, 15, 30, tzinfo=UTC), observed.end)
        official = payload()
        official["start"] = official.pop("startDateTime")
        official["end"] = official.pop("endDateTime")
        self.assertEqual(observed, self.module.parse_appointment(official, "synthetic/provider id"))
        dual = payload(start=official["start"], end=official["end"])
        self.assertEqual(observed, self.module.parse_appointment(dual, "synthetic/provider id"))

    def test_partial_conflicting_and_invalid_utc_pairs_fail_closed(self):
        cases = [
            payload(endDateTime=None),
            payload(startDateTime={"dateTime": "2026-09-14T11:00:00-04:00", "timeZone": "UTC"}),
            payload(startDateTime={"dateTime": "2026-09-14T15:00:00", "timeZone": "PRIVATE_ZONE"}),
            payload(startDateTime={"dateTime": "2026-09-14", "timeZone": "UTC"}),
            payload(endDateTime={"dateTime": "2026-09-14T15:00:00", "timeZone": "UTC"}),
            payload(start={"dateTime": "2026-09-14T16:00:00", "timeZone": "UTC"},
                    end={"dateTime": "2026-09-14T16:30:00", "timeZone": "UTC"}),
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(self.module.ReadbackFailure):
                self.module.parse_appointment(case, "synthetic/provider id")
        with self.assertRaises(self.module.ReadbackFailure) as mismatch:
            self.module.parse_appointment(payload(id="different"), "synthetic/provider id")
        self.assertEqual("identity_mismatch", mismatch.exception.category)


class ObservationStateTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("services.scheduling.provider_observations")
        self.now = datetime(2026, 9, 14, 15, tzinfo=UTC)
        self.row = {
            "id": 18, "status": "confirmed", "ms_booking_id": "opaque",
            "appointment_time_utc": self.now + timedelta(days=1),
            "snapshot_provider_id": "opaque",
            "snapshot_start": self.now + timedelta(days=1),
            "snapshot_status": "confirmed",
            "checked_at": self.now,
            "outcome": "present",
            "provider_start": self.now + timedelta(days=1),
            "provider_end": self.now + timedelta(days=1, minutes=30),
        }

    def test_fresh_present_and_stale_or_mismatched_snapshot(self):
        self.assertEqual("present", self.module.provider_state(self.row, self.now, 180))
        for change in (
            {"checked_at": self.now - timedelta(seconds=181)},
            {"snapshot_provider_id": "different"},
            {"snapshot_start": self.now},
            {"snapshot_status": "cancelled"},
            {"outcome": None},
        ):
            with self.subTest(change=change):
                row = dict(self.row, **change)
                self.assertNotEqual("present", self.module.provider_state(row, self.now, 180))

    def test_changed_unavailable_and_failed_are_never_verified(self):
        for outcome in ("changed", "unavailable", "check_failed"):
            with self.subTest(outcome=outcome):
                row = dict(self.row, outcome=outcome)
                self.assertEqual(outcome, self.module.provider_state(row, self.now, 180))

    def test_real_toronto_summer_and_winter_offsets_when_tzdata_available(self):
        try:
            ZoneInfo("America/Toronto")
        except ZoneInfoNotFoundError:
            self.skipTest("local Python tzdata unavailable; normal image must run this")
        from services.scheduling.booking_records import booking_time_for_dashboard
        self.assertEqual(
            booking_time_for_dashboard(datetime(2026, 9, 14, 15, tzinfo=UTC)),
            "2026-09-14T11:00:00-04:00",
        )
        self.assertEqual(
            booking_time_for_dashboard(datetime(2027, 1, 14, 16, tzinfo=UTC)),
            "2027-01-14T11:00:00-05:00",
        )

    def test_dashboard_provider_cells_use_text_nodes_and_warning_badge(self):
        html = (Path(__file__).resolve().parents[2] / "templates/dashboard.html").read_text(
            encoding="utf-8")
        booking_section = html.split("async function loadBookings()", 1)[1].split(
            "// Delete booking", 1)[0]
        self.assertNotIn("innerHTML", booking_section)
        self.assertIn("appendTextCell(row, providerLabels[booking.provider_state]", booking_section)
        self.assertIn("status-badge warning", booking_section)
        self.assertIn("Unavailable in Microsoft: needs review", booking_section)
        self.assertIn("Stale: needs recheck", booking_section)
        self.assertIn("element.textContent = text", html)


class FakeResponse:
    def __init__(self, code, body=None, headers=None):
        self.status_code = code
        self.body = body
        self.headers = headers or {}

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


class FakeTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return FakeResponse(200, {"access_token": "PRIVATE_SECRET", "expires_in": 300})

    async def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class ReadbackClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.module = importlib.import_module("services.calendar.booking_readback")
        self.settings = SimpleNamespace(
            ms_bookings_tenant_id="tenant/space", ms_bookings_client_id="synthetic",
            ms_bookings_client_secret="PRIVATE_SECRET", ms_bookings_business_id="business/space",
        )
        self.local_start = datetime(2026, 9, 14, 15, tzinfo=UTC)

    def client(self, replies):
        transport = FakeTransport(replies)
        return self.module.BookingReadbackClient(self.settings, client=transport), transport

    async def test_present_404_present_recovery_without_mutation_or_secret_output(self):
        client, transport = self.client([
            FakeResponse(200, payload()), FakeResponse(404, {"error": "PRIVATE_CONTACT"}),
            FakeResponse(200, payload()),
        ])
        results = [await client.observe("synthetic/provider id", self.local_start)
                   for _ in range(3)]
        self.assertEqual([r.outcome for r in results], ["present", "unavailable", "present"])
        self.assertEqual([r.http_status for r in results], [200, 404, 200])
        self.assertEqual([c[0] for c in transport.calls], ["POST", "GET", "GET", "GET"])
        self.assertTrue(all("graph.microsoft.com" in c[1] for c in transport.calls[1:]))
        self.assertIn("synthetic%2Fprovider%20id", transport.calls[1][1])
        self.assertIn("business%2Fspace", transport.calls[1][1])
        self.assertNotIn("PRIVATE_CONTACT", repr(results))
        self.assertNotIn("PRIVATE_SECRET", repr(results))

    async def test_changed_time_and_original_recovery(self):
        changed = payload(
            startDateTime={"dateTime": "2026-09-14T16:00:00", "timeZone": "UTC"},
            endDateTime={"dateTime": "2026-09-14T16:30:00", "timeZone": "UTC"},
        )
        client, _ = self.client([FakeResponse(200, changed), FakeResponse(200, payload())])
        self.assertEqual("changed", (await client.observe("synthetic/provider id", self.local_start)).outcome)
        self.assertEqual("present", (await client.observe("synthetic/provider id", self.local_start)).outcome)

    async def test_auth_throttle_server_timeout_malformed_identity_are_fixed(self):
        cases = (
            (FakeResponse(401, {"error": "PRIVATE_CONTACT"},
                          {"Retry-After": "3600"}), "authentication", True),
            (FakeResponse(403, {"error": "PRIVATE_CONTACT"}), "authorization", True),
            (FakeResponse(429, {}, {"Retry-After": "9999"}), "throttled", True),
            (FakeResponse(503, {"error": "PRIVATE_CONTACT"},
                          {"Retry-After": "120"}), "server_error", True),
            (TimeoutError("PRIVATE_CONTACT"), "timeout", False),
            (FakeResponse(200, ValueError("PRIVATE_CONTACT")), "malformed", False),
            (FakeResponse(200, payload(id="different")), "identity_mismatch", False),
        )
        for reply, category, stop in cases:
            with self.subTest(category=category):
                client, transport = self.client([reply])
                result = await client.observe("synthetic/provider id", self.local_start)
                self.assertEqual("check_failed", result.outcome)
                self.assertEqual(category, result.error_category)
                self.assertEqual(stop, result.stop_batch)
                self.assertNotIn("PRIVATE_CONTACT", repr(result))
                self.assertTrue(all(method in ("POST", "GET") for method, *_ in transport.calls))
                if category == "throttled":
                    self.assertEqual(9999, result.retry_after)
                if category == "authentication":
                    self.assertEqual(3600, result.retry_after)
                if category == "server_error":
                    self.assertEqual(120, result.retry_after)

    async def test_retry_after_numeric_http_date_and_invalid_fallback(self):
        self.assertEqual(3600, self.module._retry_after({"Retry-After": "3600"}))
        future = format_datetime(datetime.now(UTC) + timedelta(hours=1), usegmt=True)
        self.assertGreaterEqual(self.module._retry_after({"Retry-After": future}), 3590)
        self.assertEqual(60, self.module._retry_after({"Retry-After": "PRIVATE_BAD"}))

    async def test_rejected_cached_token_is_refreshed_only_on_later_observation(self):
        class RotatingTransport(FakeTransport):
            def __init__(self):
                super().__init__([FakeResponse(401), FakeResponse(200, payload())])
                self.issued = 0

            async def post(self, url, **kwargs):
                self.issued += 1
                self.calls.append(("POST", url, kwargs))
                return FakeResponse(200, {"access_token": f"PRIVATE_TOKEN_{self.issued}",
                                          "expires_in": 3600})

        transport = RotatingTransport()
        client = self.module.BookingReadbackClient(self.settings, client=transport)
        self.assertEqual("check_failed", (await client.observe(
            "synthetic/provider id", self.local_start)).outcome)
        self.assertEqual(["POST", "GET"], [call[0] for call in transport.calls])
        self.assertEqual("present", (await client.observe(
            "synthetic/provider id", self.local_start)).outcome)
        self.assertEqual(["POST", "GET", "POST", "GET"],
                         [call[0] for call in transport.calls])
        self.assertEqual("Bearer PRIVATE_TOKEN_2", transport.calls[-1][2]["headers"]["Authorization"])

    async def test_nonfinite_token_lifetime_is_rejected(self):
        for lifetime in (float("nan"), float("inf"), -1):
            with self.subTest(lifetime=lifetime):
                class BadTransport(FakeTransport):
                    async def post(self, url, **kwargs):
                        return FakeResponse(200, {"access_token": "PRIVATE_TOKEN",
                                                  "expires_in": lifetime})
                client = self.module.BookingReadbackClient(
                    self.settings, client=BadTransport([]))
                result = await client.observe("synthetic/provider id", self.local_start)
                self.assertEqual(("check_failed", "malformed"),
                                 (result.outcome, result.error_category))

    async def test_graph_403_does_not_force_token_refresh_loop(self):
        client, transport = self.client([FakeResponse(403), FakeResponse(200, payload())])
        first = await client.observe("synthetic/provider id", self.local_start)
        second = await client.observe("synthetic/provider id", self.local_start)
        self.assertEqual(("check_failed", "authorization"),
                         (first.outcome, first.error_category))
        self.assertEqual("present", second.outcome)
        self.assertEqual(["POST", "GET", "GET"], [call[0] for call in transport.calls])


class FakeConnection:
    def __init__(self, row, *, lock=True):
        self.row = row
        self.lock = lock
        self.calls = []
        self.closed = False
        self.paused = False

    def is_closed(self):
        return self.closed

    async def fetchval(self, sql, *args):
        self.calls.append(("fetchval", sql, args))
        if "pg_try_advisory_lock" in sql:
            return self.lock
        if "pg_advisory_unlock" in sql:
            return True
        if "pg_backend_pid" in sql:
            return 1234
        if "next_request_at > CURRENT_TIMESTAMP" in sql:
            return self.paused
        if "UPDATE public.booking_provider_observation_control" in sql:
            self.paused = True
            return datetime.now(UTC) + timedelta(hours=1)
        if "INSERT INTO public.booking_provider_observations" in sql:
            return args[0]
        raise AssertionError("unexpected SQL")

    async def fetch(self, sql, *args):
        self.calls.append(("fetch", sql, args))
        return self.row if isinstance(self.row, list) else [self.row]


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        pool = self

        class Context:
            async def __aenter__(self):
                return pool.conn

            async def __aexit__(self, *args):
                return None

        return Context()


class PollerOfflineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.module = importlib.import_module("services.scheduling.provider_observations")
        self.calendar = importlib.import_module("services.calendar.booking_readback")
        self.settings = SimpleNamespace(
            booking_observation_enabled=True, booking_observation_interval_seconds=60,
            ms_bookings_tenant_id="tenant", ms_bookings_client_id="client",
            ms_bookings_client_secret="PRIVATE_SECRET", ms_bookings_business_id="business",
        )
        self.row = {"id": 18, "ms_booking_id": "synthetic/provider id",
                    "appointment_time_utc": datetime(2026, 9, 14, 15, tzinfo=UTC),
                    "status": "confirmed"}

    async def test_disabled_and_busy_workers_make_no_provider_request(self):
        conn = FakeConnection(self.row, lock=False)
        client = self.calendar.BookingReadbackClient(self.settings, client=FakeTransport([]))
        poller = self.module.ObservationPoller(FakePool(conn), self.settings, client=client)
        self.assertEqual(0, await poller.tick())
        self.assertEqual([], client._client.calls)
        self.settings.booking_observation_enabled = False
        self.assertEqual(0, await poller.tick())
        self.assertEqual([], client._client.calls)

    async def test_one_tick_writes_same_snapshot_and_unlocks(self):
        conn = FakeConnection(self.row)
        client = self.calendar.BookingReadbackClient(
            self.settings, client=FakeTransport([FakeResponse(200, payload())]))
        poller = self.module.ObservationPoller(FakePool(conn), self.settings, client=client)
        self.assertEqual(1, await poller.tick())
        self.assertTrue(any("pg_advisory_unlock" in call[1] for call in conn.calls))
        writes = [call for call in conn.calls if "INSERT INTO public.booking_provider_observations" in call[1]]
        self.assertEqual(1, len(writes))
        self.assertEqual((18, "synthetic/provider id", self.row["appointment_time_utc"],
                          "confirmed"), writes[0][2][:4])
        self.assertIn("FOR UPDATE", writes[0][1])
        self.assertIn("ON CONFLICT", writes[0][1])
        self.assertIn("ORDER BY o.checked_at ASC NULLS FIRST,b.id ASC", self.module.SELECT_DUE)
        self.assertIn("LIMIT 20", self.module.SELECT_DUE)

    async def test_throttle_stamps_attempt_and_shares_backoff_with_replacement(self):
        second = dict(self.row, id=19, ms_booking_id="other-id")
        conn = FakeConnection([self.row, second])
        transport = FakeTransport([FakeResponse(429, {}, {"Retry-After": "3600"})])
        client = self.calendar.BookingReadbackClient(self.settings, client=transport)
        poller = self.module.ObservationPoller(FakePool(conn), self.settings, client=client)
        self.assertEqual(1, await poller.tick())
        replacement = self.module.ObservationPoller(FakePool(conn), self.settings, client=client)
        self.assertEqual(0, await replacement.tick())
        self.assertEqual(["POST", "GET"], [call[0] for call in transport.calls])
        writes = [call for call in conn.calls if "INSERT INTO public.booking_provider_observations" in call[1]]
        self.assertEqual(1, len(writes))
        self.assertEqual("check_failed", writes[0][2][4])
        self.assertEqual("throttled", writes[0][2][5])
        pauses = [call for call in conn.calls if "UPDATE public.booking_provider_observation_control" in call[1]]
        self.assertEqual(1, len(pauses))
        self.assertEqual((3600,), pauses[0][2])

    async def test_lost_session_prevents_late_write_and_cancel_releases_lock(self):
        conn = FakeConnection(self.row)

        class LostTransport(FakeTransport):
            async def get(self, url, **kwargs):
                conn.closed = True
                return await super().get(url, **kwargs)

        transport = LostTransport([FakeResponse(200, payload())])
        client = self.calendar.BookingReadbackClient(self.settings, client=transport)
        poller = self.module.ObservationPoller(FakePool(conn), self.settings, client=client)
        self.assertEqual(0, await poller.tick())
        self.assertFalse(any("INSERT INTO public.booking_provider_observations" in call[1]
                             for call in conn.calls))

        conn = FakeConnection(self.row)

        class CancelTransport(FakeTransport):
            async def get(self, url, **kwargs):
                raise asyncio.CancelledError

        client = self.calendar.BookingReadbackClient(self.settings, client=CancelTransport([]))
        poller = self.module.ObservationPoller(FakePool(conn), self.settings, client=client)
        with self.assertRaises(asyncio.CancelledError):
            await poller.tick()
        self.assertTrue(any("pg_advisory_unlock" in call[1] for call in conn.calls))

    async def test_slow_first_row_records_timeout_then_second_progresses(self):
        second = dict(self.row, id=19, ms_booking_id="other-id")
        conn = FakeConnection([self.row, second])

        class SlowTransport(FakeTransport):
            async def get(self, url, **kwargs):
                self.calls.append(("GET", url, kwargs))
                if "other-id" not in url:
                    await asyncio.sleep(1)
                return FakeResponse(200, payload(id="other-id"))

        client = self.calendar.BookingReadbackClient(self.settings, client=SlowTransport([]))
        poller = self.module.ObservationPoller(FakePool(conn), self.settings, client=client)
        with mock.patch.object(self.module, "TICK_DEADLINE_SECONDS", 1), \
             mock.patch.object(self.module, "ROW_TIMEOUT_SECONDS", 0.01), \
             mock.patch.object(self.module, "PERSIST_ALLOWANCE_SECONDS", 0.01), \
             mock.patch.object(self.module, "CLEANUP_ALLOWANCE_SECONDS", 0.01):
            self.assertEqual(2, await poller.tick())
        writes = [call for call in conn.calls if "INSERT INTO public.booking_provider_observations" in call[1]]
        self.assertEqual(["timeout", None], [call[2][5] for call in writes])

    async def test_internal_failure_logs_only_fixed_category(self):
        class BrokenConnection(FakeConnection):
            async def fetch(self, sql, *args):
                raise RuntimeError("PRIVATE_SECRET PRIVATE_CONTACT")

        poller = self.module.ObservationPoller(FakePool(BrokenConnection(self.row)), self.settings)
        with self.assertLogs(self.module.__name__, logging.WARNING) as captured:
            self.assertEqual(0, await poller.tick())
        self.assertEqual("internal_error", poller.last_error_category)
        self.assertIn("internal_error", str(captured.output))
        self.assertNotIn("PRIVATE_SECRET", str(captured.output))
        self.assertNotIn("PRIVATE_CONTACT", str(captured.output))



if __name__ == "__main__":
    unittest.main()
