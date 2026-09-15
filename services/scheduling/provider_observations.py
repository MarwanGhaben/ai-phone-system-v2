"""Race-safe, single-tick provider observation for confirmed canonical bookings."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timedelta
from typing import Mapping

from services.calendar.booking_readback import (
    BookingReadbackClient, InventoryResult, ProviderResult)


LOCK_KEY = int.from_bytes(
    hashlib.sha256(b"ai-voice-platform-v2:public:booking-observation-tick").digest()[:8],
    "big", signed=True,
)
BATCH_SIZE = 20
TICK_DEADLINE_SECONDS = 50
LOOKBACK_DAYS = 7
ROW_TIMEOUT_SECONDS = 8
PERSIST_ALLOWANCE_SECONDS = 2
CLEANUP_ALLOWANCE_SECONDS = 4
MAX_FINITE_BACKOFF_SECONDS = 3_153_600_000  # Longer waits become infinity, never earlier.
logger = logging.getLogger(__name__)

SELECT_DUE = """
SELECT b.id,b.ms_booking_id,b.appointment_time_utc,b.status
FROM public.bookings AS b
LEFT JOIN public.booking_provider_observations AS o ON o.booking_id=b.id
WHERE b.status='confirmed'
  AND b.appointment_time_utc IS NOT NULL
  AND b.appointment_time_utc >= CURRENT_TIMESTAMP - INTERVAL '7 days'
  AND b.ms_booking_id IS NOT NULL AND b.ms_booking_id <> ''
  AND (o.booking_id IS NULL
       OR o.snapshot_provider_id IS DISTINCT FROM b.ms_booking_id
       OR o.snapshot_start IS DISTINCT FROM b.appointment_time_utc
       OR o.snapshot_status IS DISTINCT FROM b.status
       OR o.checked_at <= CURRENT_TIMESTAMP - ($1::integer * INTERVAL '1 second'))
ORDER BY o.checked_at ASC NULLS FIRST,b.id ASC
LIMIT 20
"""

UPSERT_OBSERVATION = """
WITH current_booking AS (
    SELECT id FROM public.bookings
    WHERE id=$1 AND ms_booking_id=$2 AND appointment_time_utc=$3
      AND status=$4 AND status='confirmed'
    FOR UPDATE
)
INSERT INTO public.booking_provider_observations (
    booking_id,snapshot_provider_id,snapshot_start,snapshot_status,
    checked_at,outcome,error_category,http_status,provider_start,provider_end,
    staff_member_ids,service_id,is_location_online,observed_provider_id
)
SELECT current_booking.id,$2,$3,$4,CURRENT_TIMESTAMP,$5,$6,$7,$8,$9,$10,$11,$12,$13
FROM current_booking WHERE TRUE
ON CONFLICT (booking_id) DO UPDATE SET
    snapshot_provider_id=EXCLUDED.snapshot_provider_id,
    snapshot_start=EXCLUDED.snapshot_start,
    snapshot_status=EXCLUDED.snapshot_status,
    checked_at=EXCLUDED.checked_at,
    outcome=EXCLUDED.outcome,
    error_category=EXCLUDED.error_category,
    http_status=EXCLUDED.http_status,
    provider_start=EXCLUDED.provider_start,
    provider_end=EXCLUDED.provider_end,
    staff_member_ids=EXCLUDED.staff_member_ids,
    service_id=EXCLUDED.service_id,
    is_location_online=EXCLUDED.is_location_online,
    observed_provider_id=EXCLUDED.observed_provider_id
RETURNING booking_id
"""

SELECT_REMOVED = """
SELECT b.id,b.ms_booking_id,b.appointment_time_utc,b.status
FROM public.bookings AS b
JOIN public.booking_notification_reconciliation AS r ON r.booking_id=b.id
WHERE b.status='removed_externally' AND r.disposition='removed_externally'
  AND r.last_result IS DISTINCT FROM 'reappeared'
  AND b.appointment_time_utc > CURRENT_TIMESTAMP
  AND r.updated_at <= CURRENT_TIMESTAMP - ($1::integer * INTERVAL '1 second')
ORDER BY r.updated_at,b.id LIMIT 5
"""


def provider_state(row: Mapping[str, object], now: datetime, freshness_seconds: int) -> str:
    """A dashboard observation is current only for the exact local snapshot."""
    checked = row.get("checked_at")
    if checked is None:
        return "not_checked"
    if (row.get("snapshot_provider_id") != row.get("ms_booking_id")
            or row.get("snapshot_start") != row.get("appointment_time_utc")
            or row.get("snapshot_status") != row.get("status")):
        return "stale"
    if (not isinstance(checked, datetime) or checked.tzinfo is None
            or not isinstance(now, datetime) or now.tzinfo is None
            or checked > now or checked < now - timedelta(seconds=freshness_seconds)):
        return "stale"
    outcome = row.get("outcome")
    if outcome not in ("present", "changed", "unavailable", "check_failed"):
        return "stale"
    if outcome == "present" and row.get("provider_start") != row.get("appointment_time_utc"):
        return "stale"
    return outcome


def _connection_closed(conn: object) -> bool:
    method = getattr(conn, "is_closed", None)
    return bool(method()) if callable(method) else False


async def record_observation(conn: object, row: Mapping[str, object], result: ProviderResult) -> bool:
    """Conditionally write one attempt only while its local row still matches."""
    if _connection_closed(conn):
        return False
    details = result.details
    saved = await conn.fetchval(
        UPSERT_OBSERVATION,
        row["id"], row["ms_booking_id"], row["appointment_time_utc"], row["status"],
        result.outcome, result.error_category, result.http_status,
        details.start if details else None, details.end if details else None,
        list(details.staff_member_ids) if details and details.staff_member_ids is not None else None,
        details.service_id if details else None,
        details.is_location_online if details else None,
        result.observed_provider_id,
    )
    return saved == row["id"]


class ObservationPoller:
    """At most one worker owns a dedicated session advisory lock per tick."""

    def __init__(self, pool: object, settings: object, *, client: BookingReadbackClient | None = None):
        self.pool = pool
        self.settings = settings
        self.client = client
        self._task: asyncio.Task | None = None
        self.last_error_category: str | None = None

    def _report_failure(self, category: str) -> None:
        # Only call with fixed literals; database/HTTP exceptions may contain
        # provider IDs, URLs, tokens or customer data.
        self.last_error_category = category
        logger.warning("Booking observation poller failure: %s", category)

    def start(self) -> None:
        if not self.settings.booking_observation_enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._loop(), name="booking-provider-observations")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self.client is not None:
            await self.client.close()

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._report_failure("loop_error")
            await asyncio.sleep(self.settings.booking_observation_interval_seconds)

    async def tick(self) -> int:
        if not self.settings.booking_observation_enabled:
            return 0
        deadline = time.monotonic() + TICK_DEADLINE_SECONDS
        try:
            async with asyncio.timeout(TICK_DEADLINE_SECONDS):
                async with self.pool.acquire() as conn:
                    held = False
                    try:
                        held = bool(await conn.fetchval(
                            "SELECT pg_catalog.pg_try_advisory_lock($1::bigint)", LOCK_KEY
                        ))
                        if not held:
                            return 0
                        backend_pid = await conn.fetchval("SELECT pg_catalog.pg_backend_pid()")
                        paused = await conn.fetchval(
                            "SELECT next_request_at > CURRENT_TIMESTAMP "
                            "FROM public.booking_provider_observation_control WHERE singleton=1"
                        )
                        if paused is None:
                            raise RuntimeError("observation control row missing")
                        if paused:
                            return 0
                        rows = await conn.fetch(
                            SELECT_DUE, self.settings.booking_observation_interval_seconds
                        )
                        if getattr(self.settings, 'automatic_notifications_enabled', False):
                            rows = list(rows) + list(await conn.fetch(
                                SELECT_REMOVED,
                                self.settings.booking_observation_interval_seconds))
                        if self.client is None:
                            self.client = BookingReadbackClient(self.settings)
                        processed = 0
                        for row in rows:
                            if (deadline - time.monotonic()
                                    < ROW_TIMEOUT_SECONDS + PERSIST_ALLOWANCE_SECONDS
                                    + CLEANUP_ALLOWANCE_SECONDS):
                                break
                            notifications_enabled = getattr(
                                self.settings, 'automatic_notifications_enabled', False)
                            read_started_at = (await conn.fetchval('SELECT clock_timestamp()')
                                               if notifications_enabled else None)
                            try:
                                async with asyncio.timeout(ROW_TIMEOUT_SECONDS):
                                    if notifications_enabled:
                                        from services.calendar.provider_admission import observe
                                        result, read_started_at = await observe(
                                            conn, self.client, row['ms_booking_id'],
                                            row['appointment_time_utc'])
                                    else:
                                        result = await self.client.observe(
                                            row['ms_booking_id'], row['appointment_time_utc'])
                                        read_started_at = None
                            except TimeoutError:
                                result = ProviderResult("check_failed", "timeout",
                                                        retry_after=60 if notifications_enabled else 0,
                                                        stop_batch=notifications_enabled)
                            if (_connection_closed(conn) or await conn.fetchval(
                                    "SELECT pg_catalog.pg_backend_pid()") != backend_pid):
                                self._report_failure("session_lost")
                                break
                            if notifications_enabled and read_started_at is None:
                                # No new exact-ID observation was admitted.
                                break
                            if notifications_enabled and result.error_category == 'timeout':
                                from services.calendar.provider_admission import publish_pause
                                await publish_pause(conn, result.retry_after)
                            if result.stop_batch and not getattr(
                                    self.settings, 'automatic_notifications_enabled', False):
                                pause = max(1, result.retry_after)
                                if pause > MAX_FINITE_BACKOFF_SECONDS:
                                    updated = await conn.fetchval(
                                        "UPDATE public.booking_provider_observation_control "
                                        "SET next_request_at='infinity'::timestamptz "
                                        "WHERE singleton=1 RETURNING next_request_at")
                                else:
                                    updated = await conn.fetchval(
                                        "UPDATE public.booking_provider_observation_control "
                                        "SET next_request_at=GREATEST(next_request_at, "
                                        "CURRENT_TIMESTAMP + pg_catalog.make_interval("
                                        "secs => $1::double precision)) "
                                        "WHERE singleton=1 RETURNING next_request_at", float(pause))
                                if updated is None:
                                    raise RuntimeError("observation control row missing")
                            if row['status'] == 'confirmed':
                                await record_observation(conn, row, result)
                            if getattr(self.settings, 'automatic_notifications_enabled', False):
                                from services.scheduling.removal_reconciliation import (
                                    finalize_inferred_removal, ingest_observation)
                                missing_at = await ingest_observation(
                                    conn, row, result,
                                    self.settings.ms_bookings_tenant_id,
                                    self.settings.ms_bookings_business_id,
                                    read_started_at=read_started_at)
                                if missing_at is not None:
                                    from services.calendar.provider_admission import (
                                        inventory_absent, publish_pause)
                                    from services.scheduling.removal_reconciliation import (
                                        invalidate_inventory_evidence)
                                    remaining = min(30.0, deadline - time.monotonic()
                                                    - PERSIST_ALLOWANCE_SECONDS
                                                    - CLEANUP_ALLOWANCE_SECONDS)
                                    try:
                                        if remaining <= 0:
                                            raise TimeoutError
                                        async with asyncio.timeout(remaining):
                                            inventory = await inventory_absent(
                                                conn, self.client, row['ms_booking_id'],
                                                self.settings.ms_bookings_tenant_id,
                                                self.settings.ms_bookings_business_id)
                                        manual_failure = False
                                    except asyncio.CancelledError:
                                        raise
                                    except TimeoutError:
                                        inventory = InventoryResult(None, 'timeout', 60, True)
                                        manual_failure = True
                                    except Exception:
                                        inventory = InventoryResult(None, 'network', 60, True)
                                        manual_failure = True
                                    if _connection_closed(conn):
                                        # A cancelled advisory-lock acquisition may have
                                        # terminated this session. Use a fresh short
                                        # transaction to invalidate the old 404 evidence.
                                        async with self.pool.acquire() as recovery:
                                            await publish_pause(recovery, 60)
                                            await invalidate_inventory_evidence(
                                                recovery, row['id'], missing_at,
                                                self.settings.ms_bookings_tenant_id,
                                                self.settings.ms_bookings_business_id,
                                                inventory)
                                        break
                                    if manual_failure:
                                        await publish_pause(conn, 60)
                                    if inventory.absent is True:
                                        await finalize_inferred_removal(
                                            conn, row['id'], missing_at,
                                            self.settings.ms_bookings_tenant_id,
                                            self.settings.ms_bookings_business_id)
                                    else:
                                        await invalidate_inventory_evidence(
                                            conn, row['id'], missing_at,
                                            self.settings.ms_bookings_tenant_id,
                                            self.settings.ms_bookings_business_id,
                                            inventory)
                                    if inventory.stop_batch:
                                        break
                            processed += 1
                            if result.stop_batch:
                                break
                        return processed
                    finally:
                        if held and not _connection_closed(conn):
                            try:
                                unlocked = await conn.fetchval(
                                    "SELECT pg_catalog.pg_advisory_unlock($1::bigint)", LOCK_KEY
                                )
                                if not unlocked:
                                    terminate = getattr(conn, "terminate", None)
                                    if terminate is not None:
                                        terminate()
                                    self._report_failure("unlock_failed")
                            except BaseException:
                                terminate = getattr(conn, "terminate", None)
                                if terminate is not None:
                                    terminate()
                                raise
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            self._report_failure("tick_timeout")
            return 0
        except Exception:
            self._report_failure("internal_error")
            return 0
