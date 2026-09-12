"""Read one local booking and the exact saved Microsoft Bookings appointment.

This file intentionally remains standalone so an owner can pipe it to ``python``
inside the application container.  It emits one allowlisted JSON document and
never mutates the database or Microsoft Graph.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo


GRAPH_ROOT = "https://graph.microsoft.com/v1.0"
TOKEN_ROOT = "https://login.microsoftonline.com"
EXPECTED_HUSSAM_ID = "93ee7133-8b0c-42c4-a886-a368b998de4b"
EXPECTED_SERVICE_ID = "357dc857-4360-4801-8bc4-12d3ed63afa3"
SAFE_LOCAL_STATUSES = frozenset({"confirmed", "cancelled", "pending"})


class DiagnosticIssue(Exception):
    """An error whose fixed classification is safe to emit."""

    def __init__(
        self,
        status: str,
        *,
        http_status: int | None = None,
        local_row_id: int | None = None,
    ):
        super().__init__(status)
        self.status = status
        self.http_status = http_status
        self.local_row_id = local_row_id


def positive_row_id(value: str) -> int:
    """Argparse converter that accepts only positive local row IDs."""
    try:
        row_id = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("row ID must be a positive integer") from None
    if row_id <= 0:
        raise argparse.ArgumentTypeError("row ID must be a positive integer")
    return row_id


def toronto_timezone() -> ZoneInfo:
    return ZoneInfo("America/Toronto")


def _safe_local_status(value: object) -> str:
    return value if isinstance(value, str) and value in SAFE_LOCAL_STATUSES else "unknown"


def _canonical_local_time(value: object, *, local_row_id: int | None = None) -> datetime:
    if not isinstance(value, datetime):
        raise DiagnosticIssue(
            "LOCAL_CANONICAL_TIME_INVALID", local_row_id=local_row_id
        )
    try:
        offset = value.utcoffset()
    except Exception:
        raise DiagnosticIssue(
            "LOCAL_CANONICAL_TIME_INVALID", local_row_id=local_row_id
        ) from None
    if offset is None:
        raise DiagnosticIssue(
            "LOCAL_CANONICAL_TIME_INVALID", local_row_id=local_row_id
        )
    return value.astimezone(timezone.utc)


def _parse_utc_value(value: object) -> datetime:
    """Strictly parse a Graph dateTimeTimeZone value established as UTC."""
    if not isinstance(value, dict):
        raise DiagnosticIssue("MALFORMED_PROVIDER_SUCCESS")
    timestamp = value.get("dateTime")
    zone = value.get("timeZone")
    if not isinstance(timestamp, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,7})?(?:Z|[+-]\d{2}:\d{2})?",
        timestamp,
    ):
        raise DiagnosticIssue("MALFORMED_PROVIDER_SUCCESS")
    if zone != "UTC":
        raise DiagnosticIssue("MALFORMED_PROVIDER_SUCCESS")
    try:
        parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp)
        offset = parsed.utcoffset()
    except (TypeError, ValueError, OverflowError):
        raise DiagnosticIssue("MALFORMED_PROVIDER_SUCCESS") from None
    if offset is None:
        return parsed.replace(tzinfo=timezone.utc)
    if offset != timezone.utc.utcoffset(None):
        raise DiagnosticIssue("MALFORMED_PROVIDER_SUCCESS")
    return parsed.astimezone(timezone.utc)


def parse_provider_interval(appointment: object) -> tuple[datetime, datetime, str]:
    """Accept the official or observed complete UTC time pair, never a partial pair."""
    if not isinstance(appointment, dict):
        raise DiagnosticIssue("MALFORMED_PROVIDER_SUCCESS")
    parsed_pairs: list[tuple[datetime, datetime, str]] = []
    for start_field, end_field in (
        ("start", "end"),
        ("startDateTime", "endDateTime"),
    ):
        start_present = start_field in appointment
        end_present = end_field in appointment
        if start_present != end_present:
            raise DiagnosticIssue("MALFORMED_PROVIDER_SUCCESS")
        if not start_present:
            continue
        start = _parse_utc_value(appointment[start_field])
        end = _parse_utc_value(appointment[end_field])
        if end <= start:
            raise DiagnosticIssue("MALFORMED_PROVIDER_SUCCESS")
        parsed_pairs.append((start, end, f"{start_field}/{end_field}"))
    if not parsed_pairs:
        raise DiagnosticIssue("MALFORMED_PROVIDER_SUCCESS")
    if len(parsed_pairs) == 2 and parsed_pairs[0][:2] != parsed_pairs[1][:2]:
        raise DiagnosticIssue("MALFORMED_PROVIDER_SUCCESS")
    start, end, fields = parsed_pairs[0]
    if len(parsed_pairs) == 2:
        fields = "start/end+startDateTime/endDateTime"
    return start, end, fields


def _field_comparison(
    appointment: dict[str, Any], field: str, expected: object, *, kind: type
) -> str:
    if field not in appointment:
        return "unknown"
    value = appointment[field]
    if kind is list:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise DiagnosticIssue("MALFORMED_PROVIDER_SUCCESS")
    elif kind is bool:
        if not isinstance(value, bool):
            raise DiagnosticIssue("MALFORMED_PROVIDER_SUCCESS")
    elif not isinstance(value, kind):
        raise DiagnosticIssue("MALFORMED_PROVIDER_SUCCESS")
    return "match" if value == expected else "mismatch"


def _duration_minutes(start: datetime, end: datetime) -> int | float:
    minutes = (end - start).total_seconds() / 60
    return int(minutes) if minutes.is_integer() else minutes


def _expected_comparisons(
    appointment: dict[str, Any], start: datetime, duration: int | float
) -> dict[str, str]:
    expected_start = datetime(2026, 9, 14, 11, 0, tzinfo=toronto_timezone()).astimezone(timezone.utc)
    return {
        "provider_start": "match" if start == expected_start else "mismatch",
        "duration": "match" if duration == 30 else "mismatch",
        "consultant": _field_comparison(
            appointment, "staffMemberIds", [EXPECTED_HUSSAM_ID], kind=list
        ),
        "service": _field_comparison(
            appointment, "serviceId", EXPECTED_SERVICE_ID, kind=str
        ),
        "in_person": _field_comparison(
            appointment, "isLocationOnline", False, kind=bool
        ),
    }


def _http_issue(stage: str, status_code: int) -> DiagnosticIssue:
    if stage == "authenticate":
        if status_code in (401, 403):
            status = "AUTHENTICATION_DENIED"
        elif status_code == 429:
            status = "AUTHENTICATION_THROTTLED"
        elif status_code >= 500:
            status = "AUTHENTICATION_SERVER_ERROR"
        else:
            status = "AUTHENTICATION_HTTP_ERROR"
    elif status_code == 404:
        status = "PROVIDER_APPOINTMENT_UNAVAILABLE"
    elif status_code in (401, 403):
        status = "PROVIDER_ACCESS_DENIED"
    elif status_code == 429:
        status = "PROVIDER_THROTTLED"
    elif status_code >= 500:
        status = "PROVIDER_SERVER_ERROR"
    else:
        status = "PROVIDER_HTTP_ERROR"
    return DiagnosticIssue(status, http_status=status_code)


def _require_http_success(response: object, stage: str) -> None:
    status_code = getattr(response, "status_code", None)
    if not isinstance(status_code, int) or not 200 <= status_code < 300:
        raise _http_issue(stage, status_code if isinstance(status_code, int) else 0)


async def _read_local_row(settings: object, row_id: int | None) -> dict[str, Any]:
    import asyncpg

    connection = await asyncpg.connect(
        settings.database_url, timeout=10, command_timeout=10
    )
    try:
        async with connection.transaction(readonly=True):
            if row_id is None:
                row = await connection.fetchrow(
                    """
                    SELECT id, ms_booking_id, appointment_time_utc, status
                    FROM public.bookings
                    WHERE appointment_time_utc IS NOT NULL
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    timeout=10,
                )
            else:
                row = await connection.fetchrow(
                    """
                    SELECT id, ms_booking_id, appointment_time_utc, status
                    FROM public.bookings
                    WHERE id = $1
                    """,
                    row_id,
                    timeout=10,
                )
    finally:
        await connection.close(timeout=5)
    if row is None:
        raise DiagnosticIssue("LOCAL_ROW_NOT_FOUND")
    try:
        result = {
            "id": row["id"],
            "ms_booking_id": row["ms_booking_id"],
            "appointment_time_utc": row["appointment_time_utc"],
            "status": row["status"],
        }
    except (KeyError, TypeError):
        raise DiagnosticIssue("LOCAL_ROW_INVALID") from None
    if not isinstance(result["id"], int) or result["id"] <= 0:
        raise DiagnosticIssue("LOCAL_ROW_INVALID")
    if result["appointment_time_utc"] is None:
        raise DiagnosticIssue(
            "LOCAL_CANONICAL_TIME_MISSING", local_row_id=result["id"]
        )
    provider_id = result["ms_booking_id"]
    if not isinstance(provider_id, str) or not provider_id.strip():
        raise DiagnosticIssue("LOCAL_PROVIDER_ID_MISSING", local_row_id=result["id"])
    result["appointment_time_utc"] = _canonical_local_time(
        result["appointment_time_utc"], local_row_id=result["id"]
    )
    return result


async def check(diagnostics: dict[str, Any], row_id: int | None = None) -> dict[str, Any]:
    diagnostics["stage"] = "load_settings"
    from config.settings import settings
    import httpx

    diagnostics["stage"] = "database_read"
    local = await _read_local_row(settings, row_id)
    diagnostics["local_row_id"] = local["id"]

    async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
        diagnostics["stage"] = "authenticate"
        token_response = await client.post(
            TOKEN_ROOT
            + "/"
            + quote(settings.ms_bookings_tenant_id, safe="")
            + "/oauth2/v2.0/token",
            data={
                "grant_type": "client_credentials",
                "client_id": settings.ms_bookings_client_id,
                "client_secret": settings.ms_bookings_client_secret,
                "scope": "https://graph.microsoft.com/.default",
            },
        )
        _require_http_success(token_response, diagnostics["stage"])
        try:
            token_data = token_response.json()
        except Exception:
            raise DiagnosticIssue("MALFORMED_TOKEN_RESPONSE") from None
        access_token = token_data.get("access_token") if isinstance(token_data, dict) else None
        if not isinstance(access_token, str) or not access_token:
            raise DiagnosticIssue("MALFORMED_TOKEN_RESPONSE")

        diagnostics["stage"] = "appointment_read"
        appointment_response = await client.get(
            GRAPH_ROOT
            + "/solutions/bookingBusinesses/"
            + quote(settings.ms_bookings_business_id, safe="")
            + "/appointments/"
            + quote(local["ms_booking_id"], safe=""),
            headers={"Authorization": "Bearer " + access_token},
        )
        _require_http_success(appointment_response, diagnostics["stage"])
        try:
            appointment = appointment_response.json()
        except Exception:
            raise DiagnosticIssue("MALFORMED_PROVIDER_SUCCESS") from None

    diagnostics["stage"] = "validate_provider_response"
    start, end, time_fields = parse_provider_interval(appointment)
    provider_id = appointment.get("id") if isinstance(appointment, dict) else None
    if not isinstance(provider_id, str) or not provider_id:
        raise DiagnosticIssue("MALFORMED_PROVIDER_SUCCESS")

    duration = _duration_minutes(start, end)
    expected = _expected_comparisons(appointment, start, duration)
    provider_id_match = provider_id == local["ms_booking_id"]
    local_start_match = start == local["appointment_time_utc"]
    if not provider_id_match:
        status = "PROVIDER_IDENTITY_MISMATCH"
    elif local_start_match and all(value == "match" for value in expected.values()):
        status = "READBACK_MATCH"
    else:
        status = "READBACK_MISMATCH"

    toronto = toronto_timezone()
    return {
        "status": status,
        "local_row_id": local["id"],
        "local_booking_status": _safe_local_status(local["status"]),
        "local_appointment_time_utc": local["appointment_time_utc"].isoformat(),
        "provider_time_fields": time_fields,
        "provider_start_toronto": start.astimezone(toronto).isoformat(),
        "provider_end_toronto": end.astimezone(toronto).isoformat(),
        "duration_minutes": duration,
        "provider_id_matches_local": provider_id_match,
        "local_provider_start_matches": local_start_match,
        "expected_test": {
            "label": "fixed_test_expectations",
            "consultant": "Hussam",
            "start_toronto": "2026-09-14T11:00:00-04:00",
            "duration_minutes": 30,
            "location": "in_person",
            "comparisons": expected,
        },
    }


def _failure(error: BaseException, diagnostics: dict[str, Any]) -> dict[str, Any]:
    stage = diagnostics.get("stage", "starting")
    local_row_id = diagnostics.get("local_row_id")
    if isinstance(error, DiagnosticIssue):
        status = error.status
        http_status = error.http_status
        if local_row_id is None:
            local_row_id = error.local_row_id
    elif isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        status = "DIAGNOSTIC_TIMEOUT"
        http_status = None
    else:
        try:
            import httpx

            if isinstance(error, httpx.TimeoutException):
                status = "NETWORK_TIMEOUT"
            elif isinstance(error, httpx.RequestError):
                status = "NETWORK_FAILURE"
            elif stage == "database_read":
                status = "DATABASE_READ_FAILED"
            elif stage == "load_settings":
                status = "CONFIGURATION_FAILED"
            else:
                status = "CHECK_FAILED"
        except Exception:
            status = "CHECK_FAILED"
        http_status = None
    result: dict[str, Any] = {"status": status, "stage": stage}
    requested_row_id = diagnostics.get("requested_row_id")
    if isinstance(requested_row_id, int):
        result["requested_row_id"] = requested_row_id
    if isinstance(local_row_id, int):
        result["local_row_id"] = local_row_id
    if isinstance(http_status, int) and http_status > 0:
        result["http_status"] = http_status
    if status == "PROVIDER_APPOINTMENT_UNAVAILABLE":
        result["cancellation_state"] = "unknown"
    return result


async def report(row_id: int | None = None) -> None:
    diagnostics: dict[str, Any] = {"stage": "starting"}
    if row_id is not None:
        diagnostics["requested_row_id"] = row_id
    try:
        result = await asyncio.wait_for(check(diagnostics, row_id=row_id), timeout=70)
    except Exception as error:
        result = _failure(error, diagnostics)
    print(json.dumps(result, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Read back one local booking from Microsoft Graph without changes."
    )
    parser.add_argument("--row-id", type=positive_row_id)
    args = parser.parse_args(argv)
    from loguru import logger

    logger.remove()
    logging.disable(logging.CRITICAL)
    asyncio.run(report(row_id=args.row_id))


if __name__ == "__main__":
    main()
