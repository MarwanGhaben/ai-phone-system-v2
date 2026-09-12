"""Contract tests for the owner-run, read-only booking readback probe."""
from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

import asyncpg
import httpx
import config.settings as settings_module


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check-latest-booking.py"
PRIVATE_SENTINELS = (
    "PRIVATE_SECRET",
    "PRIVATE_PROVIDER_ID",
    "PRIVATE_CONTACT",
    "PRIVATE_PAYLOAD",
)
DEFAULT_ROW = object()


def load_diagnostic():
    spec = importlib.util.spec_from_file_location("booking_readback_diagnostic", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, status_code=200, body=None, json_error=None):
        self.status_code = status_code
        self._body = body
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise self._json_error
        return self._body


class FakeConnection:
    def __init__(self, row, calls):
        self.row = row
        self.calls = calls

    def transaction(self, **kwargs):
        self.calls.append(("transaction", kwargs))
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def fetchrow(self, sql, *args, **kwargs):
        normalized = " ".join(sql.split()).upper()
        self.calls.append(("sql", normalized, args, kwargs))
        forbidden = ("INSERT ", "UPDATE ", "DELETE ", "CREATE ", "ALTER ", "DROP ")
        assert normalized.startswith("SELECT ")
        assert not any(word in normalized for word in forbidden)
        return self.row

    async def close(self, **kwargs):
        self.calls.append(("close", kwargs))


class FakeClient:
    def __init__(self, token_response, appointment_response, calls, **kwargs):
        self.token_response = token_response
        self.appointment_response = appointment_response
        self.calls = calls
        self.calls.append(("client", kwargs))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, **kwargs):
        self.calls.append(("http", "POST", url, kwargs))
        if isinstance(self.token_response, BaseException):
            raise self.token_response
        return self.token_response

    async def get(self, url, **kwargs):
        self.calls.append(("http", "GET", url, kwargs))
        if isinstance(self.appointment_response, BaseException):
            raise self.appointment_response
        return self.appointment_response


def expected_appointment(**overrides):
    appointment = {
        "id": "PRIVATE_PROVIDER_ID/with space",
        "startDateTime": {
            "dateTime": "2026-09-14T15:00:00.0000000",
            "timeZone": "UTC",
        },
        "endDateTime": {
            "dateTime": "2026-09-14T15:30:00.0000000",
            "timeZone": "UTC",
        },
        "staffMemberIds": ["93ee7133-8b0c-42c4-a886-a368b998de4b"],
        "serviceId": "357dc857-4360-4801-8bc4-12d3ed63afa3",
        "isLocationOnline": False,
        "customerEmailAddress": "PRIVATE_CONTACT",
        "unexpected": "PRIVATE_PAYLOAD",
    }
    appointment.update(overrides)
    return appointment


class BookingReadbackDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.diagnostic = load_diagnostic()

    async def run_report(
        self,
        *,
        row_id=None,
        row=DEFAULT_ROW,
        appointment=None,
        appointment_status=200,
        token_body=None,
        token_status=200,
        token_error=None,
        appointment_error=None,
        appointment_json_error=None,
    ):
        if row is DEFAULT_ROW:
            row = {
                "id": 17,
                "ms_booking_id": "PRIVATE_PROVIDER_ID/with space",
                "appointment_time_utc": datetime(2026, 9, 14, 15, tzinfo=timezone.utc),
                "status": "confirmed",
            }
        if appointment is None:
            appointment = expected_appointment()
        calls = []
        connection = FakeConnection(row, calls)

        async def connect(*args, **kwargs):
            calls.append(("connect", args, kwargs))
            return connection

        token_response = token_error or FakeResponse(
            token_status,
            {"access_token": "PRIVATE_SECRET"} if token_body is None else token_body,
        )
        appointment_response = appointment_error or FakeResponse(
            appointment_status, appointment, appointment_json_error
        )
        synthetic_settings = SimpleNamespace(
            database_url="PRIVATE_DSN",
            ms_bookings_tenant_id="tenant/with space",
            ms_bookings_client_id="synthetic-client",
            ms_bookings_client_secret="PRIVATE_SECRET",
            ms_bookings_business_id="business/with space",
        )
        stream = io.StringIO()
        with patch.object(asyncpg, "connect", connect), patch.object(
            httpx, "AsyncClient", lambda **kwargs: FakeClient(
                token_response, appointment_response, calls, **kwargs
            )
        ), patch.object(settings_module, "settings", synthetic_settings), redirect_stdout(stream):
            await self.diagnostic.report(row_id=row_id)
        raw = stream.getvalue()
        for sentinel in PRIVATE_SENTINELS:
            self.assertNotIn(sentinel, raw)
        return json.loads(raw), calls

    async def test_observed_pair_returns_canonical_matching_readback(self):
        report, calls = await self.run_report(row_id=17)
        self.assertEqual("READBACK_MATCH", report["status"])
        self.assertEqual(17, report["local_row_id"])
        self.assertEqual("2026-09-14T15:00:00+00:00", report["local_appointment_time_utc"])
        self.assertEqual("2026-09-14T11:00:00-04:00", report["provider_start_toronto"])
        self.assertEqual("2026-09-14T11:30:00-04:00", report["provider_end_toronto"])
        self.assertEqual(30, report["duration_minutes"])
        self.assertIs(True, report["provider_id_matches_local"])
        self.assertIs(True, report["local_provider_start_matches"])
        self.assertEqual("fixed_test_expectations", report["expected_test"]["label"])
        self.assertEqual("Hussam", report["expected_test"]["consultant"])
        self.assertEqual("match", report["expected_test"]["comparisons"]["consultant"])
        self.assertIn(("transaction", {"readonly": True}), calls)
        connect_call = next(call for call in calls if call[0] == "connect")
        self.assertEqual(
            {"timeout": 10, "command_timeout": 10}, connect_call[2]
        )
        sql_call = next(call for call in calls if call[0] == "sql")
        for private_column in ("CALL_SID", "PHONE", "EMAIL", "NAME", "NOTES", "TRANSCRIPT"):
            self.assertNotIn(private_column, sql_call[1])
        self.assertIn("WHERE ID = $1", sql_call[1])
        self.assertEqual((17,), sql_call[2])
        self.assertEqual({"timeout": 10}, sql_call[3])
        methods = [call[1] for call in calls if call[0] == "http"]
        self.assertEqual(["POST", "GET"], methods)
        client_options = next(call[1] for call in calls if call[0] == "client")
        self.assertFalse(client_options["follow_redirects"])
        get_url = next(call[2] for call in calls if call[:2] == ("http", "GET"))
        self.assertEqual("graph.microsoft.com", urlsplit(get_url).hostname)
        self.assertIn("business%2Fwith%20space", get_url)
        self.assertIn("PRIVATE_PROVIDER_ID%2Fwith%20space", get_url)
        token_url = next(call[2] for call in calls if call[:2] == ("http", "POST"))
        self.assertIn("tenant%2Fwith%20space", token_url)

    async def test_official_pair_and_equal_dual_pairs_are_accepted(self):
        official = expected_appointment()
        start = official.pop("startDateTime")
        end = official.pop("endDateTime")
        start["dateTime"] = "2026-09-14T15:00:00+00:00"
        end["dateTime"] = "2026-09-14T15:30:00Z"
        official.update(start=start, end=end)
        report, _ = await self.run_report(appointment=official)
        self.assertEqual("READBACK_MATCH", report["status"])

        dual = expected_appointment(
            start={"dateTime": "2026-09-14T15:00:00.0000000Z", "timeZone": "UTC"},
            end={"dateTime": "2026-09-14T15:30:00.0000000+00:00", "timeZone": "UTC"},
        )
        report, _ = await self.run_report(appointment=dual)
        self.assertEqual("READBACK_MATCH", report["status"])

    async def test_conflicting_dual_pairs_are_malformed(self):
        appointment = expected_appointment(
            start={"dateTime": "2026-09-14T16:00:00", "timeZone": "UTC"},
            end={"dateTime": "2026-09-14T16:30:00", "timeZone": "UTC"},
        )
        report, _ = await self.run_report(appointment=appointment)
        self.assertEqual("MALFORMED_PROVIDER_SUCCESS", report["status"])
        self.assertEqual("validate_provider_response", report["stage"])

    async def test_bad_time_shapes_zones_offsets_and_durations_are_rejected(self):
        cases = {
            "missing": {"id": "PRIVATE_PROVIDER_ID"},
            "partial": expected_appointment(endDateTime=None),
            "structure": expected_appointment(startDateTime=[]),
            "unknown_zone": expected_appointment(startDateTime={"dateTime": "2026-09-14T15:00:00", "timeZone": "PRIVATE_ZONE"}),
            "conflicting_offset": expected_appointment(startDateTime={"dateTime": "2026-09-14T11:00:00-04:00", "timeZone": "UTC"}),
            "invalid_datetime": expected_appointment(startDateTime={"dateTime": "not-a-date-PRIVATE_CONTACT", "timeZone": "UTC"}),
            "date_only": expected_appointment(startDateTime={"dateTime": "2026-09-14", "timeZone": "UTC"}),
            "missing_pair_member": {k: v for k, v in expected_appointment().items() if k != "endDateTime"},
            "zero_duration": expected_appointment(endDateTime={"dateTime": "2026-09-14T15:00:00.0000000", "timeZone": "UTC"}),
            "negative_duration": expected_appointment(endDateTime={"dateTime": "2026-09-14T14:59:00", "timeZone": "UTC"}),
        }
        for name, appointment in cases.items():
            with self.subTest(name=name):
                report, _ = await self.run_report(appointment=appointment)
                self.assertEqual("MALFORMED_PROVIDER_SUCCESS", report["status"])

    async def test_identity_time_and_expected_test_mismatches_are_distinct(self):
        identity, _ = await self.run_report(
            appointment=expected_appointment(id="different-private-provider-id")
        )
        self.assertEqual("PROVIDER_IDENTITY_MISMATCH", identity["status"])
        self.assertIs(False, identity["provider_id_matches_local"])

        local_time, _ = await self.run_report(
            row={
                "id": 17,
                "ms_booking_id": "PRIVATE_PROVIDER_ID/with space",
                "appointment_time_utc": datetime(2026, 9, 14, 16, tzinfo=timezone.utc),
                "status": "confirmed",
            }
        )
        self.assertEqual("READBACK_MISMATCH", local_time["status"])
        self.assertIs(False, local_time["local_provider_start_matches"])

        absent_expected_fields = expected_appointment()
        del absent_expected_fields["staffMemberIds"]
        del absent_expected_fields["serviceId"]
        del absent_expected_fields["isLocationOnline"]
        unknown, _ = await self.run_report(appointment=absent_expected_fields)
        self.assertEqual("READBACK_MISMATCH", unknown["status"])
        comparisons = unknown["expected_test"]["comparisons"]
        self.assertEqual("unknown", comparisons["consultant"])
        self.assertEqual("unknown", comparisons["service"])
        self.assertEqual("unknown", comparisons["in_person"])

    async def test_explicit_row_never_falls_back_and_default_excludes_legacy(self):
        missing, calls = await self.run_report(row_id=29, row=None)
        self.assertEqual("LOCAL_ROW_NOT_FOUND", missing["status"])
        self.assertEqual(29, missing["requested_row_id"])
        connection_call = next(call for call in calls if call[0] == "sql")
        self.assertIn("WHERE ID = $1", connection_call[1])
        self.assertNotIn("ORDER BY ID DESC", connection_call[1])

        report, calls = await self.run_report(row_id=None)
        self.assertEqual("READBACK_MATCH", report["status"])
        sql_call = next(call for call in calls if call[0] == "sql")
        self.assertIn("APPOINTMENT_TIME_UTC IS NOT NULL", sql_call[1])
        self.assertIn("ORDER BY ID DESC", sql_call[1])
        self.assertEqual((), sql_call[2])

    async def test_missing_row_and_missing_local_identity_fail_before_network(self):
        cases = (
            (False, "LOCAL_ROW_NOT_FOUND"),
            ({"id": 17, "ms_booking_id": "PRIVATE_PROVIDER_ID/with space", "appointment_time_utc": None, "status": "confirmed"}, "LOCAL_CANONICAL_TIME_MISSING"),
            ({"id": 17, "ms_booking_id": "PRIVATE_PROVIDER_ID/with space", "appointment_time_utc": datetime(2026, 9, 14, 15), "status": "confirmed"}, "LOCAL_CANONICAL_TIME_INVALID"),
            ({"id": 17, "ms_booking_id": None, "appointment_time_utc": datetime(2026, 9, 14, 15, tzinfo=timezone.utc), "status": "confirmed"}, "LOCAL_PROVIDER_ID_MISSING"),
        )
        for row, expected_status in cases:
            with self.subTest(expected_status=expected_status):
                actual_row = None if row is False else row
                report, calls = await self.run_report(row_id=17, row=actual_row, appointment=False)
                self.assertEqual(expected_status, report["status"])
                if row is not False:
                    self.assertEqual(17, report["local_row_id"])
                self.assertFalse(any(call[0] == "http" for call in calls))

    async def test_http_and_network_failures_have_allowlisted_classifications(self):
        for code, expected in (
            (404, "PROVIDER_APPOINTMENT_UNAVAILABLE"),
            (401, "PROVIDER_ACCESS_DENIED"),
            (403, "PROVIDER_ACCESS_DENIED"),
            (429, "PROVIDER_THROTTLED"),
            (500, "PROVIDER_SERVER_ERROR"),
            (503, "PROVIDER_SERVER_ERROR"),
        ):
            with self.subTest(code=code):
                report, _ = await self.run_report(
                    appointment_status=code,
                    appointment={"error": "PRIVATE_PAYLOAD"},
                )
                self.assertEqual(expected, report["status"])
                self.assertEqual(code, report["http_status"])
                if code == 404:
                    self.assertEqual("unknown", report["cancellation_state"])

        timeout, _ = await self.run_report(
            appointment_error=httpx.ReadTimeout("PRIVATE_CONTACT")
        )
        self.assertEqual("NETWORK_TIMEOUT", timeout["status"])
        network, _ = await self.run_report(
            appointment_error=httpx.ConnectError("PRIVATE_PAYLOAD")
        )
        self.assertEqual("NETWORK_FAILURE", network["status"])

    async def test_auth_and_malformed_success_failures_are_separate_and_private(self):
        denied, calls = await self.run_report(
            token_status=401, token_body={"error": "PRIVATE_PAYLOAD"}
        )
        self.assertEqual("AUTHENTICATION_DENIED", denied["status"])
        self.assertEqual("authenticate", denied["stage"])
        self.assertEqual(["POST"], [call[1] for call in calls if call[0] == "http"])

        missing_token, _ = await self.run_report(token_body={})
        self.assertEqual("MALFORMED_TOKEN_RESPONSE", missing_token["status"])
        malformed, _ = await self.run_report(appointment=[])
        self.assertEqual("MALFORMED_PROVIDER_SUCCESS", malformed["status"])
        invalid_json, _ = await self.run_report(
            appointment_json_error=ValueError("PRIVATE_PAYLOAD")
        )
        self.assertEqual("MALFORMED_PROVIDER_SUCCESS", invalid_json["status"])

    async def test_same_explicit_row_is_reused_across_changed_provider_response(self):
        baseline, first_calls = await self.run_report(row_id=17)
        changed = expected_appointment(
            startDateTime={"dateTime": "2026-09-14T16:00:00", "timeZone": "UTC"},
            endDateTime={"dateTime": "2026-09-14T16:30:00", "timeZone": "UTC"},
        )
        after_change, second_calls = await self.run_report(row_id=17, appointment=changed)
        self.assertEqual("READBACK_MATCH", baseline["status"])
        self.assertEqual("READBACK_MISMATCH", after_change["status"])
        self.assertEqual(17, baseline["local_row_id"])
        self.assertEqual(17, after_change["local_row_id"])
        for calls in (first_calls, second_calls):
            sql_call = next(call for call in calls if call[0] == "sql")
            self.assertEqual((17,), sql_call[2])

    async def test_winter_readback_uses_toronto_standard_time(self):
        appointment = expected_appointment(
            startDateTime={"dateTime": "2026-12-14T16:00:00", "timeZone": "UTC"},
            endDateTime={"dateTime": "2026-12-14T16:30:00", "timeZone": "UTC"},
        )
        row = {"id": 17, "ms_booking_id": appointment["id"],
               "appointment_time_utc": datetime(2026, 12, 14, 16, tzinfo=timezone.utc),
               "status": "confirmed"}
        report, _ = await self.run_report(row_id=17, row=row, appointment=appointment)
        self.assertEqual("2026-12-14T11:00:00-05:00", report["provider_start_toronto"])
        self.assertEqual("2026-12-14T11:30:00-05:00", report["provider_end_toronto"])
        self.assertTrue(report["local_provider_start_matches"])
        self.assertEqual("mismatch", report["expected_test"]["comparisons"]["provider_start"])

    def test_positive_row_id_parser(self):
        self.assertEqual(17, self.diagnostic.positive_row_id("17"))
        for value in ("0", "-1", "abc", "1.5"):
            with self.subTest(value=value), self.assertRaises(Exception):
                self.diagnostic.positive_row_id(value)


if __name__ == "__main__":
    unittest.main()
