"""Shared, durable admission for opt-in Microsoft Bookings reads.

The observer owns its global tick lock; the SMS worker owns a booking lock.
Both take this session lock only around provider I/O, then release it before
changing booking evidence. No SQL transaction spans a provider request.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib

from services.calendar.booking_readback import InventoryResult, ProviderResult


ADMISSION_LOCK = int.from_bytes(
    hashlib.sha256(b'ai-voice-platform-v2:booking-provider-admission').digest()[:8],
    'big', signed=True)
MAX_FINITE_BACKOFF_SECONDS = 3_153_600_000


async def is_paused(conn) -> bool:
    paused = await conn.fetchval("""
        SELECT next_request_at > CURRENT_TIMESTAMP
        FROM public.booking_provider_observation_control WHERE singleton=1
    """)
    if paused is None:
        raise RuntimeError('provider admission control unavailable')
    return paused


async def publish_pause(conn, seconds: int) -> None:
    if seconds <= 0:
        return
    if seconds > MAX_FINITE_BACKOFF_SECONDS:
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


@asynccontextmanager
async def admitted(conn, client):
    """Serialize admission and publication using the caller's DB session."""
    locked = False
    try:
        try:
            await conn.fetchval('SELECT pg_catalog.pg_advisory_lock($1::bigint)',
                                ADMISSION_LOCK)
            locked = True
        except BaseException:
            conn.terminate()  # An interrupted lock acquisition is ambiguous.
            raise
        if await is_paused(conn):
            yield False
        else:
            binder = getattr(client, '_bind_admission', None)
            if binder is None:
                yield True
            else:
                with binder(conn):
                    yield True
    finally:
        if locked and not conn.is_closed():
            try:
                if not await conn.fetchval(
                        'SELECT pg_catalog.pg_advisory_unlock($1::bigint)',
                        ADMISSION_LOCK):
                    conn.terminate()
            except BaseException:
                conn.terminate()
                raise


async def observe(conn, client, provider_id, start):
    """Return result and read-start fence; denied reads never touch Graph."""
    async with admitted(conn, client) as allowed:
        if not allowed:
            return ProviderResult('check_failed', 'throttled', stop_batch=True), None
        started_at = await conn.fetchval('SELECT clock_timestamp()')
        result = await client.observe(provider_id, start)
        if result.stop_batch:
            await publish_pause(conn, result.retry_after)
        return result, started_at


async def inventory_absent(conn, client, provider_id, tenant_id, business_id):
    async with admitted(conn, client) as allowed:
        if not allowed:
            return InventoryResult(None, 'throttled', stop_batch=True)
        result = await client.inventory_absent(provider_id, tenant_id, business_id)
        if result.stop_batch:
            await publish_pause(conn, result.retry_after)
        return result
