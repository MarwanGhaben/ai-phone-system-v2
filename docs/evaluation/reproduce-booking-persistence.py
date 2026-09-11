"""Demonstrate the deployed booking-save failure using synthetic local PostgreSQL.

Run from the repository root with Python 3.11 and --local-docker. Reuses the
existing isolated PostgreSQL harness; never connects to a configured application
database or a real calendar/SMS service. This is a historical bug demonstration,
not a regression test whose desired behavior is the bug.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


async def demonstrate(harness):
    from loguru import logger
    logger.remove()
    logging.disable(logging.CRITICAL)
    from services.conversation.orchestrator import (
        ConversationContext, ConversationOrchestrator,
    )
    from services.calendar.calendar_base import BookingResult

    connection = await harness.driver.connect(harness.dsn, timeout=10, command_timeout=15)
    try:
        await connection.execute((ROOT / "migrations/bootstrap_schema.sql").read_text("utf-8"))
        appointment_time = datetime(2030, 1, 8, 11, tzinfo=ZoneInfo("America/Toronto"))
        codec_error = None
        try:
            await connection.fetchval("SELECT $1::timestamp", appointment_time)
        except harness.driver.DataError as error:
            codec_error = str(error)

        context = ConversationContext(
            call_sid="synthetic-persistence-proof", phone_number="+14165550100", language="en")
        context.pending_booking = {
            "service_id": "synthetic-service", "staff_id": "synthetic-staff",
            "staff_name": "Synthetic Consultant", "appointment_time": appointment_time,
            "customer_name": "Synthetic Caller", "customer_email": "caller@example.invalid",
            "client_type": "individual",
        }
        agent = ConversationOrchestrator.__new__(ConversationOrchestrator)
        agent._conversations = {context.call_sid: context}
        agent._speak_to_caller = mock.AsyncMock()
        agent._send_booking_sms = mock.AsyncMock()
        calendar = mock.Mock()
        calendar.create_booking = mock.AsyncMock(return_value=BookingResult(
            success=True, appointment_id="synthetic-provider-appointment",
            staff_name="Synthetic Consultant"))
        with mock.patch("services.calendar.ms_bookings_service.get_calendar_service", return_value=calendar), \
                mock.patch("services.database.get_db_pool", new=mock.AsyncMock(return_value=connection)):
            result = await agent._confirm_booking(context.call_sid, {"confirm": True})
            await asyncio.sleep(0)  # Let the mocked SMS task complete.

        rows = await connection.fetchval("SELECT count(*) FROM public.bookings")
        print(json.dumps({
            "database": "isolated synthetic PostgreSQL 16",
            "real_provider_calls": 0,
            "timestamp_codec_error": codec_error,
            "simulated_provider_create_calls": calendar.create_booking.await_count,
            "returned_booking_success": result.startswith("BOOKING_SUCCESS:"),
            "persisted_booking_rows": rows,
            "scheduled_confirmation_sms": agent._send_booking_sms.await_count,
            "bug_reproduced": codec_error is not None and rows == 0
                and result.startswith("BOOKING_SUCCESS:"),
        }, indent=2))
    finally:
        await connection.close(timeout=5)


if __name__ == "__main__":
    if sys.argv[1:] != ["--local-docker"]:
        raise SystemExit("Explicit --local-docker is required. Do not run on the server.")
    spec = importlib.util.spec_from_file_location(
        "migration_harness", ROOT / "tests/integration/test_migrations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    harness = module.PostgreSQLTests
    harness.setUpClass()
    original_directory = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="booking-proof-") as working_directory, \
                mock.patch.dict(os.environ, {
                    "SECRET_KEY": "synthetic-secret", "DATABASE_URL": harness.dsn,
                    "TWILIO_ACCOUNT_SID": "ACtest", "TWILIO_AUTH_TOKEN": "synthetic",
                    "TWILIO_PHONE_NUMBER": "+14165550100", "DEEPGRAM_API_KEY": "synthetic",
                    "ELEVENLABS_API_KEY": "synthetic", "OPENAI_API_KEY": "synthetic",
                }, clear=True):
            try:
                os.chdir(working_directory)  # Do not load the repository .env.
                asyncio.run(demonstrate(harness))
            finally:
                os.chdir(original_directory)
    finally:
        harness.tearDownClass()
