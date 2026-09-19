"""Bounded exact-known-ID create recovery; no search, retry worker or POST."""
from __future__ import annotations

from services.scheduling.booking_service import BookingService, BookingResult, TrustedBookingContext
from services.scheduling.proposals import ApprovedProposal


async def recover_known_id(service: BookingService, approval: ApprovedProposal,
                           context: TrustedBookingContext) -> BookingResult:
    return await service.recover(approval, context)
