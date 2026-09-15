"""Opt-in multiworker-safe notification dispatch from the PostgreSQL outbox."""
from __future__ import annotations

import asyncio
from datetime import timedelta
import logging
import uuid

from services.calendar.booking_readback import BookingReadbackClient
from services.scheduling.removal_reconciliation import ingest_observation
from services.sms.notification_outbox import (
    BOOKING_LOCK_NAMESPACE, CONFIRMATION_VALIDITY_SECONDS, REMINDER_VALIDITY_SECONDS)


logger = logging.getLogger(__name__)
SELECT_DUE = """
SELECT id,booking_id FROM public.booking_notification_outbox
WHERE state IN ('pending','held') AND due_at<=CURRENT_TIMESTAMP
  AND next_attempt_at<=CURRENT_TIMESTAMP
ORDER BY next_attempt_at,due_at,id LIMIT 10
"""
JOB = 'SELECT * FROM public.booking_notification_outbox WHERE id=$1 FOR UPDATE'
BOOKING = """
SELECT id,ms_booking_id,appointment_time_utc,status,phone_number,language,
       accountant_name FROM public.bookings WHERE id=$1 FOR UPDATE
"""
STATE = """
SELECT * FROM public.booking_notification_reconciliation
WHERE booking_id=$1 FOR UPDATE
"""


def _same_snapshot(job: object, booking: object, state: object, settings: object) -> bool:
    return (job['snapshot_tenant_id'] == settings.ms_bookings_tenant_id
            and job['snapshot_business_id'] == settings.ms_bookings_business_id
            and state['snapshot_tenant_id'] == job['snapshot_tenant_id']
            and state['snapshot_business_id'] == job['snapshot_business_id']
            and state['snapshot_version'] == job['snapshot_version']
            and state['snapshot_provider_id'] == job['snapshot_provider_id']
            and state['snapshot_start'] == job['snapshot_start']
            and booking['ms_booking_id'] == job['snapshot_provider_id']
            and booking['appointment_time_utc'] == job['snapshot_start'])


class NotificationWorker:
    def __init__(self, pool: object, settings: object, *, client=None, sms=None):
        self.pool, self.settings = pool, settings
        self.client = client
        self.sms = sms
        self._task: asyncio.Task | None = None
        self.last_error_category: str | None = None

    def start(self) -> None:
        if self.settings.automatic_notifications_enabled and self._task is None:
            self._task = asyncio.create_task(self._loop(), name='booking-notification-outbox')

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
                self.last_error_category = 'worker_error'
                logger.warning('Notification worker failure: worker_error')
            await asyncio.sleep(self.settings.automatic_notifications_interval_seconds)

    async def tick(self) -> int:
        if not self.settings.automatic_notifications_enabled:
            return 0
        async with self.pool.acquire() as conn:
            # Crash/expired dispatch intent is unknown, never automatically pending.
            await conn.execute("""
                UPDATE public.booking_notification_outbox SET state='unknown',
                    error_category='claim_expired',claim_token=NULL,
                    dispatch_started_at=NULL,updated_at=CURRENT_TIMESTAMP
                WHERE state='dispatching'
                  AND dispatch_started_at < CURRENT_TIMESTAMP - INTERVAL '2 minutes'
            """)
            rows = await conn.fetch(SELECT_DUE)
        processed = 0
        for row in rows:
            try:
                if await self.process(row['id'], row['booking_id']):
                    processed += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                self.last_error_category = 'dispatch_error'
                logger.warning('Notification worker failure: dispatch_error')
        return processed

    async def _claim(self, conn, job_id, booking_id):
        async with conn.transaction():
            job = await conn.fetchrow(JOB, job_id)
            booking = await conn.fetchrow(BOOKING, booking_id)
            state = await conn.fetchrow(STATE, booking_id)
            if (job is None or booking is None or state is None
                    or job['booking_id'] != booking_id
                    or job['state'] not in ('pending','held')):
                return None
            now = await conn.fetchval('SELECT CURRENT_TIMESTAMP')
            if job['due_at'] > now:
                return None
            if job['next_attempt_at'] > now:
                return None
            if not _same_snapshot(job, booking, state, self.settings):
                await self._set_state(conn, job_id, 'suppressed', 'snapshot_changed')
                return None
            if (job['kind'] == 'reminder' and (
                    job['due_at'] < now - timedelta(seconds=REMINDER_VALIDITY_SECONDS)
                    or booking['appointment_time_utc'] <= now)):
                await self._set_state(conn, job_id, 'suppressed', 'stale_reminder')
                return None
            if job['kind'] == 'confirmation' and (
                    job['due_at'] < now - timedelta(seconds=CONFIRMATION_VALIDITY_SECONDS)
                    or booking['appointment_time_utc'] <= now):
                await self._set_state(conn, job_id, 'suppressed', 'stale_confirmation')
                return None
            active = booking['status'] == 'confirmed' and state['disposition'] == 'active'
            ended = (booking['status'] in ('removed_externally', 'cancelled')
                     and state['disposition'] in ('removed_externally', 'cancelled_by_caller'))
            if not (ended if job['kind'] == 'removal' else active):
                await self._set_state(conn, job_id, 'suppressed', 'status_changed')
                return None
            paused = await conn.fetchval("""
                SELECT next_request_at > CURRENT_TIMESTAMP
                FROM public.booking_provider_observation_control WHERE singleton=1
            """)
            if paused is not False:
                await self._set_state(conn, job_id, 'held', 'provider_backoff')
                return None
            token = uuid.uuid4()
            await conn.execute("""
                UPDATE public.booking_notification_outbox SET state='dispatching',
                    claim_token=$2,dispatch_started_at=CURRENT_TIMESTAMP,
                    error_category=NULL,updated_at=CURRENT_TIMESTAMP
                WHERE id=$1
            """, job_id, token)
            return dict(job), dict(booking), token

    async def _set_state(self, conn, job_id, state, category, token=None):
        await conn.execute("""
            UPDATE public.booking_notification_outbox SET state=$2::varchar,
                error_category=$3,claim_token=NULL,dispatch_started_at=NULL,
                retry_count=retry_count + CASE WHEN $2::varchar='held' THEN 1 ELSE 0 END,
                next_attempt_at=CASE WHEN $2::varchar='held' THEN
                    GREATEST(
                        CURRENT_TIMESTAMP + pg_catalog.make_interval(secs =>
                            LEAST(3600.0,60.0 * pg_catalog.power(2.0,
                                LEAST(retry_count,6)::double precision))),
                        (SELECT next_request_at FROM public.booking_provider_observation_control
                         WHERE singleton=1))
                    ELSE next_attempt_at END,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=$1 AND ($4::uuid IS NULL OR claim_token=$4)
        """, job_id, state, category, token)

    async def _pause_provider(self, conn, result) -> None:
        if not result.stop_batch:
            return
        seconds = max(1, result.retry_after)
        if seconds > 3_153_600_000:
            await conn.execute("""
                UPDATE public.booking_provider_observation_control
                SET next_request_at='infinity'::timestamptz WHERE singleton=1
            """)
        else:
            await conn.execute("""
                UPDATE public.booking_provider_observation_control
                SET next_request_at=GREATEST(next_request_at,
                    CURRENT_TIMESTAMP + pg_catalog.make_interval(secs => $1::double precision))
                WHERE singleton=1
            """, float(seconds))

    async def _authorize(self, conn, job, booking, token, result) -> bool:
        async with conn.transaction():
            current = await conn.fetchrow(JOB, job['id'])
            actual_booking = await conn.fetchrow(BOOKING, booking['id'])
            state = await conn.fetchrow(STATE, booking['id'])
            if (current is None or actual_booking is None or state is None
                    or current['state'] != 'dispatching' or current['claim_token'] != token
                    or not _same_snapshot(current, actual_booking, state, self.settings)):
                return False
            await self._pause_provider(conn, result)
            unique_identity = await conn.fetchval(
                'SELECT count(*) FROM public.bookings WHERE ms_booking_id=$1',
                current['snapshot_provider_id']) == 1
            if job['kind'] == 'removal':
                valid = (unique_identity and result.outcome == 'unavailable'
                         and result.http_status == 404
                         and actual_booking['status'] in ('removed_externally', 'cancelled')
                         and state['disposition'] in ('removed_externally', 'cancelled_by_caller')
                         and (state['disposition'] == 'cancelled_by_caller'
                              or state['last_result'] == 'unavailable'))
            else:
                valid = (unique_identity and result.outcome == 'present'
                         and result.details is not None
                         and result.details.start == job['snapshot_start']
                         and actual_booking['status'] == 'confirmed'
                         and state['enrolled_at'] is not None
                         and state['last_result'] == 'present'
                         and state['disposition'] == 'active')
            if not valid:
                state_name = ('suppressed' if job['kind'] == 'removal'
                              and result.outcome in ('present', 'changed') else 'held')
                await self._set_state(conn, job['id'], state_name,
                                      ('ambiguous_identity' if not unique_identity else
                                       result.error_category or 'provider_mismatch'), token)
                if state_name == 'suppressed' and state['disposition'] == 'removed_externally':
                    await conn.execute("""
                        UPDATE public.booking_notification_reconciliation
                        SET last_result='reappeared',updated_at=CURRENT_TIMESTAMP
                        WHERE booking_id=$1
                    """, booking['id'])
                return False
            # This is a possible carrier POST, distinct from provider read/claim
            # retries. A crash between this commit and HTTP remains unknown.
            await conn.execute("""
                UPDATE public.booking_notification_outbox
                SET attempts=attempts+1,last_attempt_at=CURRENT_TIMESTAMP,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=$1 AND claim_token=$2
            """, job['id'], token)
            return True

    async def process(self, job_id, booking_id: int) -> bool:
        async with self.pool.acquire() as conn:
            held = False
            try:
                held = bool(await conn.fetchval(
                    'SELECT pg_catalog.pg_try_advisory_lock($1::integer,$2::integer)',
                    BOOKING_LOCK_NAMESPACE, booking_id))
                if not held:
                    return False
                backend_pid = await conn.fetchval('SELECT pg_catalog.pg_backend_pid()')
                claimed = await self._claim(conn, job_id, booking_id)
                if claimed is None:
                    return False
                job, booking, token = claimed
                if self.client is None:
                    self.client = BookingReadbackClient(self.settings)
                read_started_at = await conn.fetchval('SELECT clock_timestamp()')
                try:
                    async with asyncio.timeout(8):
                        from services.calendar.provider_admission import observe
                        observed, read_started_at = await observe(
                            conn, self.client, job['snapshot_provider_id'],
                            job['snapshot_start'])
                except TimeoutError:
                    from services.calendar.booking_readback import ProviderResult
                    observed = ProviderResult('check_failed', 'timeout', retry_after=60,
                                              stop_batch=True)
                if (conn.is_closed() or await conn.fetchval(
                        'SELECT pg_catalog.pg_backend_pid()') != backend_pid):
                    return False
                if read_started_at is not None:
                    await ingest_observation(
                        conn, dict(booking, id=booking_id), observed,
                        self.settings.ms_bookings_tenant_id,
                        self.settings.ms_bookings_business_id,
                        read_started_at=read_started_at, count_missing=False)
                if not await self._authorize(conn, job, booking, token, observed):
                    return False
                if self.sms is None:
                    from services.sms.telnyx_sms_service import get_sms_service
                    self.sms = get_sms_service()
                # Dispatch intent was committed by _claim; the per-booking
                # session lock fences local cancellation until POST completes.
                result = await self.sms.submit_notification(job['recipient'], job['body'])
                if (conn.is_closed() or await conn.fetchval(
                        'SELECT pg_catalog.pg_backend_pid()') != backend_pid):
                    return False
                async with conn.transaction():
                    current = await conn.fetchrow(JOB, job_id)
                    if (current is None or current['state'] != 'dispatching'
                            or current['claim_token'] != token):
                        return False
                    await conn.execute("""
                        UPDATE public.booking_notification_outbox SET state=$2::varchar,
                            error_category=$3,provider_message_id=$4,
                            accepted_at=CASE WHEN $2::varchar='accepted' THEN CURRENT_TIMESTAMP ELSE NULL END,
                            claim_token=NULL,dispatch_started_at=NULL,
                            updated_at=CURRENT_TIMESTAMP WHERE id=$1 AND claim_token=$5
                    """, job_id, result.state, result.category, result.message_id, token)
                return True
            finally:
                if held and not conn.is_closed():
                    try:
                        if not await conn.fetchval(
                                'SELECT pg_catalog.pg_advisory_unlock($1::integer,$2::integer)',
                                BOOKING_LOCK_NAMESPACE, booking_id):
                            conn.terminate()
                    except BaseException:
                        conn.terminate()
                        raise
