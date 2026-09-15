"""Booking-ID and snapshot-version scoped PostgreSQL notification storage."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import uuid



BOOKING_LOCK_NAMESPACE = 49049
CONFIRMATION_VALIDITY_SECONDS = 15 * 60
REMINDER_VALIDITY_SECONDS = 5 * 60


async def lock_booking(conn, booking_id: int) -> None:
    """Serialize local finalization with a sender holding the matching session lock."""
    await conn.fetchval(
        'SELECT pg_catalog.pg_advisory_xact_lock($1::integer,$2::integer)',
        BOOKING_LOCK_NAMESPACE, booking_id)


async def enqueue(conn, booking: object, state: object, kind: str,
                  due_at: datetime, *, held: bool = False) -> None:
    """Insert one event per local booking/version/kind inside caller transaction."""
    from services.sms.notification_text import render_notice
    if due_at.utcoffset() is None:
        raise ValueError('notification due time must be aware')
    start = state['snapshot_start']
    recipient = booking['phone_number']
    consultant = booking['accountant_name']
    if not recipient or not consultant:
        # Missing delivery details are a visible failed job, not a fabricated SMS.
        recipient = recipient or 'unavailable'
        consultant = consultant or 'consultant'
    body = render_notice(kind, consultant, start, booking['language'] or 'en')
    await conn.execute("""
        INSERT INTO public.booking_notification_outbox (
            id,booking_id,snapshot_version,snapshot_tenant_id,snapshot_business_id,
            snapshot_provider_id,snapshot_start,event_key,kind,due_at,next_attempt_at,
            state,attempts,retry_count,
            recipient,language,consultant,body,error_category,created_at,updated_at)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$10,$11,0,0,$12,$13,$14,$15,$16,
                CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)
        ON CONFLICT (booking_id,snapshot_version,kind) DO NOTHING
    """, uuid.uuid4(), booking['id'], state['snapshot_version'],
        state['snapshot_tenant_id'], state['snapshot_business_id'],
        state['snapshot_provider_id'], start,
        f"booking:{booking['id']}:v{state['snapshot_version']}:{kind}",
        kind, due_at.astimezone(timezone.utc), 'held' if held else 'pending',
        recipient, booking['language'] or 'en', consultant, body,
        'recipient_unavailable' if recipient == 'unavailable' else None)


async def create_new_booking_jobs(conn, booking: object, *, tenant_id: str,
                                  business_id: str) -> None:
    """Must run in the same transaction that inserted the booking row."""
    if not tenant_id or not business_id:
        raise ValueError('notification provider scope unavailable')
    start = booking['appointment_time_utc']
    await conn.execute("""
        INSERT INTO public.booking_notification_reconciliation (
            booking_id,snapshot_tenant_id,snapshot_business_id,snapshot_provider_id,
            snapshot_start,snapshot_version,missing_count,disposition,updated_at)
        VALUES ($1,$2,$3,$4,$5,1,0,'active',CURRENT_TIMESTAMP)
    """, booking['id'], tenant_id, business_id, booking['ms_booking_id'], start)
    state = dict(booking, snapshot_tenant_id=tenant_id,
                 snapshot_business_id=business_id,
                 snapshot_provider_id=booking['ms_booking_id'],
                 snapshot_start=start, snapshot_version=1)
    now = datetime.now(timezone.utc)
    await enqueue(conn, booking, state, 'confirmation', now, held=True)
    reminder_due = start - timedelta(hours=24)
    if reminder_due > now and start > now:
        await enqueue(conn, booking, state, 'reminder', reminder_due, held=True)


async def hold_booking_jobs(conn, booking_id: int) -> None:
    await conn.execute("""
        UPDATE public.booking_notification_outbox SET state='held',
            next_attempt_at=GREATEST(next_attempt_at,
                CURRENT_TIMESTAMP + INTERVAL '60 seconds'),
            retry_count=retry_count+1,
            error_category='provider_unavailable',updated_at=CURRENT_TIMESTAMP
        WHERE booking_id=$1 AND kind IN ('confirmation','reminder')
          AND state='pending'
    """, booking_id)


async def release_current_jobs(conn, booking_id: int) -> None:
    await conn.execute("""
        UPDATE public.booking_notification_outbox AS j SET state='pending',
            next_attempt_at=GREATEST(j.due_at,CURRENT_TIMESTAMP),retry_count=0,
            error_category=NULL,updated_at=CURRENT_TIMESTAMP
        FROM public.bookings AS b, public.booking_notification_reconciliation AS r
        WHERE j.booking_id=$1 AND b.id=j.booking_id AND r.booking_id=j.booking_id
          AND b.status='confirmed' AND b.appointment_time_utc>CURRENT_TIMESTAMP
          AND r.disposition='active' AND r.enrolled_at IS NOT NULL
          AND j.snapshot_version=r.snapshot_version
          AND j.snapshot_tenant_id=r.snapshot_tenant_id
          AND j.snapshot_business_id=r.snapshot_business_id
          AND j.snapshot_provider_id=r.snapshot_provider_id
          AND j.snapshot_provider_id=b.ms_booking_id
          AND j.snapshot_start=r.snapshot_start
          AND j.snapshot_start=b.appointment_time_utc
          AND j.state='held'
          AND ((j.kind='confirmation' AND j.due_at >= CURRENT_TIMESTAMP
                  - pg_catalog.make_interval(secs => $2::double precision))
            OR (j.kind='reminder' AND j.due_at >= CURRENT_TIMESTAMP
                  - pg_catalog.make_interval(secs => $3::double precision)))
    """, booking_id, float(CONFIRMATION_VALIDITY_SECONDS),
        float(REMINDER_VALIDITY_SECONDS))


async def suppress_booking_jobs(conn, booking_id: int) -> None:
    await conn.execute("""
        UPDATE public.booking_notification_outbox SET state='suppressed',
            error_category=NULL,updated_at=CURRENT_TIMESTAMP
        WHERE booking_id=$1 AND kind IN ('confirmation','reminder')
          AND state IN ('pending','held','failed')
    """, booking_id)
