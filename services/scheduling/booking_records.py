"""Persistence boundary for newly created provider bookings."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from services.calendar.business_time import BUSINESS_TIMEZONE


class BookingPersistenceError(Exception):
    """A fixed booking-record persistence classification."""


def require_aware_booking_time(appointment_time: object) -> datetime:
    """Return an aware datetime or raise the fixed persistence classification."""
    if not isinstance(appointment_time, datetime):
        raise BookingPersistenceError("invalid appointment time")
    try:
        offset = appointment_time.utcoffset()
    except Exception:
        raise BookingPersistenceError("invalid appointment time") from None
    if offset is None:
        raise BookingPersistenceError("invalid appointment time")
    return appointment_time


def _booking_times(appointment_time: object) -> tuple[datetime, datetime]:
    appointment_time = require_aware_booking_time(appointment_time)
    canonical = appointment_time.astimezone(timezone.utc)
    legacy_wall_time = appointment_time.astimezone(
        BUSINESS_TIMEZONE
    ).replace(tzinfo=None)
    return legacy_wall_time, canonical


def booking_time_for_dashboard(appointment_time_utc: object) -> str:
    """Render a verified instant in Toronto with its explicit current offset."""
    _, canonical = _booking_times(appointment_time_utc)
    return canonical.astimezone(BUSINESS_TIMEZONE).isoformat()


async def persist_booking_record(
    pool: Any,
    *,
    call_sid: str,
    phone_number: str,
    client_name: str,
    client_email: str,
    accountant_name: str,
    appointment_time: object,
    client_type: str,
    language: str,
    provider_appointment_id: object,
    notes: str,
    notifications_enabled: bool = False,
    provider_tenant_id: str = "",
    provider_business_id: str = "",
) -> int:
    """Insert one provider-created booking and require its local record ID."""
    if (not isinstance(provider_appointment_id, str)
            or not provider_appointment_id.strip()):
        raise BookingPersistenceError("missing provider appointment id")
    legacy_wall_time, canonical = _booking_times(appointment_time)
    statement = """
        INSERT INTO public.bookings (
            call_sid, phone_number, client_name, client_email,
            accountant_name, appointment_time, appointment_time_utc,
            client_type, language, status, ms_booking_id, notes
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,'confirmed',$10,$11)
        RETURNING id
        """
    values = (
        call_sid,
        phone_number,
        client_name,
        client_email,
        accountant_name,
        legacy_wall_time,
        canonical,
        client_type,
        language,
        provider_appointment_id.strip(),
        notes,
    )
    if not notifications_enabled:
        record_id = await pool.fetchval(statement, *values)
        if record_id is None:
            raise BookingPersistenceError("booking record not inserted")
        return record_id
    if not provider_tenant_id or not provider_business_id:
        raise BookingPersistenceError("notification provider scope unavailable")
    from services.sms.notification_outbox import create_new_booking_jobs
    async with pool.acquire() as conn:
        async with conn.transaction():
            record_id = await conn.fetchval(statement, *values)
            if record_id is None:
                raise BookingPersistenceError("booking record not inserted")
            booking = {
                'id': record_id, 'ms_booking_id': provider_appointment_id.strip(),
                'appointment_time_utc': canonical, 'phone_number': phone_number,
                'accountant_name': accountant_name, 'language': language,
            }
            await create_new_booking_jobs(
                conn, booking, tenant_id=provider_tenant_id,
                business_id=provider_business_id)
            return record_id


async def insert_verified_booking(conn: Any, *, call_sid: str, phone_number: str,
                                  client_name: str, client_email: str,
                                  accountant_name: str, appointment_time: datetime,
                                  language: str, provider_appointment_id: str,
                                  tenant_id: str, business_id: str) -> int:
    """Connection-owned insert and held-job creation for the operation transaction."""
    if (not isinstance(provider_appointment_id, str) or not provider_appointment_id
            or not tenant_id or not business_id):
        raise BookingPersistenceError("invalid verified booking")
    legacy, canonical = _booking_times(appointment_time)
    booking_id = await conn.fetchval("""
        INSERT INTO public.bookings (
            call_sid,phone_number,client_name,client_email,accountant_name,
            appointment_time,appointment_time_utc,client_type,language,status,
            ms_booking_id,notes)
        VALUES ($1,$2,$3,$4,$5,$6,$7,'new',$8,'confirmed',$9,'') RETURNING id
    """, call_sid, phone_number, client_name, client_email, accountant_name,
        legacy, canonical, language, provider_appointment_id)
    if type(booking_id) is not int:
        raise BookingPersistenceError("booking record not inserted")
    from services.sms.notification_outbox import create_new_booking_jobs
    await create_new_booking_jobs(conn, {
        "id": booking_id, "ms_booking_id": provider_appointment_id,
        "appointment_time_utc": canonical, "phone_number": phone_number,
        "accountant_name": accountant_name, "language": language,
    }, tenant_id=tenant_id, business_id=business_id)
    return booking_id
