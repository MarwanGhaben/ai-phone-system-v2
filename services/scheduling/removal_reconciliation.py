"""Conservative local policy for a Bookings appointment missing from Microsoft.

An exact-ID 404 is not evidence of who cancelled or why. The database-backed
reconciler below serializes every transition for one numeric local booking ID.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone



MIN_MISSING_GAP = timedelta(seconds=60)
MAX_MISSING_GAP = timedelta(minutes=10)


@dataclass(frozen=True)
class Evidence:
    enrolled: bool = False
    last_present: datetime | None = None
    first_missing: datetime | None = None
    last_missing: datetime | None = None
    missing_count: int = 0
    last_result: str | None = None


def advance_evidence(state: Evidence, outcome: str, checked_at: datetime) -> tuple[Evidence, bool]:
    """Return updated evidence and whether a complete inventory is now required."""
    if checked_at.utcoffset() is None:
        raise ValueError('observation timestamp must be aware')
    checked_at = checked_at.astimezone(timezone.utc)
    if outcome == 'present':
        return Evidence(True, checked_at, last_result='present'), False
    if outcome != 'unavailable' or not state.enrolled:
        return replace(state, first_missing=None, last_missing=None,
                       missing_count=0, last_result=outcome), False
    if state.last_missing is not None and checked_at <= state.last_missing:
        return state, False
    gap = checked_at - state.last_missing if state.last_missing else None
    if gap is None or not MIN_MISSING_GAP <= gap <= MAX_MISSING_GAP:
        return replace(state, first_missing=checked_at, last_missing=checked_at,
                       missing_count=1, last_result='unavailable'), False
    return replace(state, last_missing=checked_at,
                   missing_count=min(2, state.missing_count + 1),
                   last_result='unavailable'), True


def snapshot_matches(row: object, tenant_id: str, business_id: str) -> bool:
    """Compare the durable scope, ID and canonical instant, never a phone key."""
    return (row['snapshot_tenant_id'] == tenant_id
            and row['snapshot_business_id'] == business_id
            and row['snapshot_provider_id'] == row['ms_booking_id']
            and row['snapshot_start'] == row['appointment_time_utc'])


BOOKING_SQL = """
SELECT id,ms_booking_id,appointment_time_utc,status,phone_number,
       accountant_name,language FROM public.bookings WHERE id=$1 FOR UPDATE
"""
STATE_SQL = """
SELECT * FROM public.booking_notification_reconciliation
WHERE booking_id=$1 FOR UPDATE
"""


async def _insert_state(conn, booking: object, tenant_id: str, business_id: str) -> dict:
    await conn.execute("""
        INSERT INTO public.booking_notification_reconciliation (
            booking_id,snapshot_tenant_id,snapshot_business_id,snapshot_provider_id,
            snapshot_start,snapshot_version,missing_count,disposition,updated_at)
        VALUES ($1,$2,$3,$4,$5,1,0,'active',CURRENT_TIMESTAMP)
    """, booking['id'], tenant_id, business_id, booking['ms_booking_id'],
        booking['appointment_time_utc'])
    return dict(await conn.fetchrow(STATE_SQL, booking['id']))


async def _reset_snapshot(conn, booking: object, state: dict,
                          tenant_id: str, business_id: str) -> dict:
    from services.sms.notification_outbox import suppress_booking_jobs
    await suppress_booking_jobs(conn, booking['id'])
    await conn.execute("""
        UPDATE public.booking_notification_reconciliation SET
            snapshot_tenant_id=$2,snapshot_business_id=$3,
            snapshot_provider_id=$4,snapshot_start=$5,
            snapshot_version=snapshot_version+1,enrolled_at=NULL,last_present_at=NULL,
            last_evidence_started_at=NULL,
            first_missing_at=NULL,last_missing_at=NULL,missing_count=0,
            last_result=NULL,last_error_category=NULL,updated_at=CURRENT_TIMESTAMP
        WHERE booking_id=$1
    """, booking['id'], tenant_id, business_id, booking['ms_booking_id'],
        booking['appointment_time_utc'])
    return dict(await conn.fetchrow(STATE_SQL, booking['id']))


async def ingest_observation(conn, row: object, result: object,
                             tenant_id: str, business_id: str, *,
                             read_started_at: datetime | None = None,
                             count_missing: bool = True) -> datetime | None:
    """Persist evidence and return qualifying 404 timestamp for inventory.

    The poller calls this only when the opt-in flag is enabled. The caller does
    provider I/O outside this short transaction and holds its existing global
    observation session lock across the tick.
    """
    if not tenant_id or not business_id:
        return None
    from services.sms.notification_outbox import (
        enqueue, hold_booking_jobs, lock_booking, release_current_jobs)
    async with conn.transaction():
        await lock_booking(conn, row['id'])
        booking = await conn.fetchrow(BOOKING_SQL, row['id'])
        if (booking is None or booking['ms_booking_id'] != row['ms_booking_id']
                or booking['appointment_time_utc'] != row['appointment_time_utc']):
            return None
        state_row = await conn.fetchrow(STATE_SQL, row['id'])
        state = dict(state_row) if state_row else await _insert_state(
            conn, booking, tenant_id, business_id)
        if not snapshot_matches(dict(state, **dict(booking)), tenant_id, business_id):
            if state['disposition'] != 'active':
                return None
            state = await _reset_snapshot(conn, booking, state, tenant_id, business_id)
        now = await conn.fetchval('SELECT CURRENT_TIMESTAMP')
        read_started_at = read_started_at or now
        if (read_started_at.utcoffset() is None
                or (state['last_evidence_started_at'] is not None
                    and read_started_at <= state['last_evidence_started_at'])):
            return None
        if booking['status'] != 'confirmed' or state['disposition'] != 'active':
            if state['disposition'] == 'removed_externally':
                await conn.execute("""
                    UPDATE public.booking_notification_reconciliation SET
                        last_result=$2,last_evidence_started_at=$3,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE booking_id=$1
                """, row['id'], 'reappeared' if result.outcome == 'present'
                      else result.outcome, read_started_at)
            return None
        duplicate_count = await conn.fetchval(
            'SELECT count(*) FROM public.bookings WHERE ms_booking_id=$1',
            booking['ms_booking_id'])
        valid_absence = (count_missing and result.outcome == 'unavailable'
                         and result.http_status == 404)
        outcome = (result.outcome if duplicate_count == 1
                   and (result.outcome != 'unavailable' or valid_absence)
                   else 'check_failed')
        category = ('ambiguous_identity' if duplicate_count != 1 else
                    'provider_unavailable' if result.outcome == 'unavailable'
                    and not valid_absence else result.error_category)
        old = Evidence(
            enrolled=state['enrolled_at'] is not None,
            last_present=state['last_present_at'],
            first_missing=state['first_missing_at'],
            last_missing=state['last_missing_at'],
            missing_count=state['missing_count'],
            last_result=state['last_result'])
        updated, needs_inventory = advance_evidence(old, outcome, now)
        await conn.execute("""
            UPDATE public.booking_notification_reconciliation SET
                enrolled_at=CASE WHEN $2::boolean AND enrolled_at IS NULL
                                 THEN $3 ELSE enrolled_at END,
                last_present_at=$4,first_missing_at=$5,last_missing_at=$6,
                missing_count=$7,last_result=$8,last_error_category=$9,
                last_evidence_started_at=$10,
                updated_at=CURRENT_TIMESTAMP WHERE booking_id=$1
        """, row['id'], updated.enrolled, now, updated.last_present,
            updated.first_missing, updated.last_missing, updated.missing_count,
            updated.last_result, category, read_started_at)
        if outcome == 'present':
            reminder_due = booking['appointment_time_utc'] - timedelta(hours=24)
            if reminder_due > now and booking['appointment_time_utc'] > now:
                await enqueue(conn, booking, state, 'reminder', reminder_due)
            await release_current_jobs(conn, row['id'])
        else:
            await hold_booking_jobs(conn, row['id'])
        if (needs_inventory and booking['appointment_time_utc'] > now
                and result.http_status == 404):
            return updated.last_missing
        return None


def _inventory_category(result: object) -> str:
    if result.absent is False:
        return 'inventory_found_id'
    allowed = frozenset({
        'authentication', 'authorization', 'throttled', 'server_error',
        'timeout', 'network', 'malformed', 'identity_mismatch',
        'http_error', 'configuration'})
    category = result.error_category
    return 'inventory_' + category if category in allowed else 'inventory_inconclusive'


async def invalidate_inventory_evidence(conn, booking_id: int, missing_at: datetime,
                                        tenant_id: str, business_id: str,
                                        result: object) -> bool:
    """A failed or contradictory complete-list check breaks the 404 sequence."""
    from services.sms.notification_outbox import hold_booking_jobs, lock_booking
    async with conn.transaction():
        await lock_booking(conn, booking_id)
        booking = await conn.fetchrow(BOOKING_SQL, booking_id)
        state_row = await conn.fetchrow(STATE_SQL, booking_id)
        if booking is None or state_row is None:
            return False
        state = dict(state_row)
        if (booking['status'] != 'confirmed' or state['disposition'] != 'active'
                or state['last_missing_at'] != missing_at
                or state['last_result'] != 'unavailable'
                or not snapshot_matches(dict(state, **dict(booking)), tenant_id, business_id)):
            return False
        await conn.execute("""
            UPDATE public.booking_notification_reconciliation SET
                first_missing_at=NULL,last_missing_at=NULL,missing_count=0,
                last_result='check_failed',last_error_category=$2,
                last_evidence_started_at=clock_timestamp(),
                updated_at=CURRENT_TIMESTAMP WHERE booking_id=$1
        """, booking_id, _inventory_category(result))
        await hold_booking_jobs(conn, booking_id)
        return True


async def finalize_inferred_removal(conn, booking_id: int, missing_at: datetime,
                                    tenant_id: str, business_id: str) -> bool:
    """One transaction: status, evidence, pending-job suppression and one notice."""
    from services.sms.notification_outbox import enqueue, lock_booking, suppress_booking_jobs
    async with conn.transaction():
        await lock_booking(conn, booking_id)
        booking = await conn.fetchrow(BOOKING_SQL, booking_id)
        state_row = await conn.fetchrow(STATE_SQL, booking_id)
        if booking is None or state_row is None:
            return False
        state = dict(state_row)
        now = await conn.fetchval('SELECT CURRENT_TIMESTAMP')
        if (booking['status'] != 'confirmed' or state['disposition'] != 'active'
                or state['enrolled_at'] is None or state['missing_count'] < 2
                or state['last_result'] != 'unavailable'
                or state['last_missing_at'] != missing_at
                or booking['appointment_time_utc'] <= now
                or not snapshot_matches(dict(state, **dict(booking)), tenant_id, business_id)
                or await conn.fetchval(
                    'SELECT count(*) FROM public.bookings WHERE ms_booking_id=$1',
                    booking['ms_booking_id']) != 1):
            return False
        await conn.execute("""
            UPDATE public.bookings SET status='removed_externally' WHERE id=$1
        """, booking_id)
        await conn.execute("""
            UPDATE public.booking_notification_reconciliation SET
                disposition='removed_externally',reason='inferred_provider_removal',
                transitioned_at=CURRENT_TIMESTAMP,updated_at=CURRENT_TIMESTAMP
            WHERE booking_id=$1
        """, booking_id)
        await suppress_booking_jobs(conn, booking_id)
        await enqueue(conn, booking, state, 'removal', now)
        return True


async def finalize_caller_cancellation(pool, provider_id: str, phone: str,
                                       tenant_id: str, business_id: str) -> str:
    """Map only one canonical local booking; never cancel by phone alone."""
    from services.sms.notification_outbox import enqueue, lock_booking, suppress_booking_jobs
    if not provider_id or not phone or not tenant_id or not business_id:
        return 'unresolved'
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch("""
                SELECT id FROM public.bookings WHERE ms_booking_id=$1
            """, provider_id)
            if len(rows) != 1:
                return 'unresolved'
            booking_id = rows[0]['id']
            await lock_booking(conn, booking_id)
            booking = await conn.fetchrow(BOOKING_SQL, booking_id)
            if (booking is None or booking['ms_booking_id'] != provider_id
                    or booking['phone_number'] != phone
                    or booking['appointment_time_utc'] is None
                    or await conn.fetchval(
                        'SELECT count(*) FROM public.bookings WHERE ms_booking_id=$1',
                        provider_id) != 1):
                return 'unresolved'
            state_row = await conn.fetchrow(STATE_SQL, booking_id)
            if state_row is None:
                state = await _insert_state(conn, booking, tenant_id, business_id)
            else:
                state = dict(state_row)
            if not snapshot_matches(dict(state, **dict(booking)), tenant_id, business_id):
                return 'unresolved'
            if booking['status'] != 'confirmed':
                return 'already_finalized'
            now = await conn.fetchval('SELECT CURRENT_TIMESTAMP')
            await conn.execute(
                "UPDATE public.bookings SET status='cancelled' WHERE id=$1", booking_id)
            await conn.execute("""
                UPDATE public.booking_notification_reconciliation SET
                    disposition='cancelled_by_caller',reason='caller_confirmed',
                    transitioned_at=CURRENT_TIMESTAMP,updated_at=CURRENT_TIMESTAMP
                WHERE booking_id=$1
            """, booking_id)
            await suppress_booking_jobs(conn, booking_id)
            await enqueue(conn, booking, state, 'removal', now)
            return 'recorded'
