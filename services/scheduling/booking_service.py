"""Unwired, trusted create coordinator. No tool or phone route imports this module."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Callable, Protocol
from uuid import UUID

from services.calendar.booking_mutations import (
    BookingMutationRequest, GraphBookingMutations, record_create_diagnostic,
)
from services.config.accountants_service import AccountantsService
from services.scheduling.booking_check import (assess_booking, load_booking_policy,
                                                make_booking_query, EVIDENCE_AGE)
from services.scheduling.booking_records import insert_verified_booking
from services.scheduling.operation_store import (AdmissionStatus, DispatchStatus,
                                                  OperationStore, OwnershipStatus,
                                                  ReconcileStatus)
from services.scheduling.proposals import ApprovedProposal


class BookingOutcome(Enum):
    VERIFIED = "verified"
    INELIGIBLE = "ineligible"
    CONFLICT = "conflict"
    PENDING = "pending"
    INVALID = "invalid"
    STORE_ERROR = "store_error"


@dataclass(frozen=True, slots=True, repr=False)
class BookingResult:
    outcome: BookingOutcome
    operation_id: UUID | None = None
    booking_id: int | None = None


@dataclass(frozen=True, slots=True, repr=False)
class TrustedBookingContext:
    tenant_id: str
    business_id: str
    session_id: str
    caller_source_id: str
    call_sid: str
    language: str

    def __post_init__(self) -> None:
        if (not all(type(item) is str and item and item == item.strip() for item in
                    (self.tenant_id, self.business_id, self.session_id,
                     self.caller_source_id, self.call_sid))
                or len(self.call_sid) > 100 or self.language not in ("en", "ar")):
            raise ValueError("invalid trusted booking context")


class TurnAuthority(Protocol):
    async def valid(self, context: TrustedBookingContext,
                    approval: ApprovedProposal) -> bool: ...

    def current(self, context: TrustedBookingContext,
                approval: ApprovedProposal) -> bool: ...


class BookingService:
    def __init__(self, pool, calendar, mutations: GraphBookingMutations,
                 authority: TurnAuthority, *, clock: Callable[[], datetime] | None = None,
                 selection_loader: Callable[[], object] | None = None):
        self.pool, self.calendar, self.mutations = pool, calendar, mutations
        self.authority = authority
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.selection_loader = selection_loader or AccountantsService
        self.store = OperationStore()

    def _now(self) -> datetime:
        try:
            value = self.clock()
        except Exception:
            raise ValueError("invalid clock") from None
        if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("invalid clock")
        return value.astimezone(timezone.utc)

    def _authorized_now(self, context: TrustedBookingContext,
                        approval: ApprovedProposal) -> bool:
        try:
            return self.authority.current(context, approval) is True
        except Exception:
            return False

    async def _authorized(self, context: TrustedBookingContext,
                          approval: ApprovedProposal) -> bool:
        try:
            value = await self.authority.valid(context, approval)
            return value is True
        except asyncio.CancelledError:
            raise
        except Exception:
            return False

    def _scope_matches(self, approval: ApprovedProposal,
                       context: TrustedBookingContext) -> bool:
        try:
            proposal = approval.proposal
            scope = proposal.candidate.scope
            return (type(approval) is ApprovedProposal
                    and scope.tenant_id == context.tenant_id
                    and scope.business_id == context.business_id
                    and proposal.session_scope.session_id == context.session_id
                    and proposal.session_scope.caller_source_id == context.caller_source_id
                    and self.mutations.tenant_id == context.tenant_id
                    and self.mutations.business_id == context.business_id)
        except (AttributeError, TypeError):
            return False

    @staticmethod
    def _request(approval: ApprovedProposal,
                 context: TrustedBookingContext) -> BookingMutationRequest:
        proposal = approval.proposal
        candidate = proposal.candidate
        if (not 0 < len(proposal.consultant_display) <= 255
                or not 0 < len(proposal.service_display) <= 255):
            raise ValueError("invalid booking display")
        return BookingMutationRequest(
            context.tenant_id, context.business_id, candidate.scope.service_id,
            candidate.staff_id, candidate.interval.start, candidate.interval.end,
            proposal.customer.name, proposal.customer.phone, proposal.customer.email,
            proposal.location, int(proposal.pre_buffer.total_seconds()),
            int(proposal.post_buffer.total_seconds()))

    async def create(self, approval: ApprovedProposal,
                     context: TrustedBookingContext) -> BookingResult:
        if (type(context) is not TrustedBookingContext
                or not self._scope_matches(approval, context)
                or not await self._authorized(context, approval)):
            return BookingResult(BookingOutcome.INVALID)
        try:
            request = self._request(approval, context)
            now = self._now()
        except (AttributeError, TypeError, ValueError, OverflowError):
            return BookingResult(BookingOutcome.INVALID)
        try:
            async with self.pool.acquire() as conn:
                admitted = await self.store.admit(conn, approval, now=now)
        except asyncio.CancelledError:
            raise
        except Exception:
            return BookingResult(BookingOutcome.STORE_ERROR)
        if admitted.status is AdmissionStatus.INTERVAL_CONFLICT or admitted.status is AdmissionStatus.PAYLOAD_CONFLICT:
            return BookingResult(BookingOutcome.CONFLICT)
        if admitted.status is AdmissionStatus.STORE_ERROR:
            return BookingResult(BookingOutcome.STORE_ERROR)
        if admitted.operation is None:
            return BookingResult(BookingOutcome.INVALID)
        op = admitted.operation
        if op.state == "applied" and op.booking_id is not None:
            return BookingResult(BookingOutcome.VERIFIED, op.operation_id, op.booking_id)
        if op.state != "pending":
            return BookingResult(BookingOutcome.PENDING, op.operation_id)
        proposal = approval.proposal
        candidate = proposal.candidate
        try:
            selection = self.selection_loader().resolve_booking_accountant(
                proposal.consultant_display)
            if (selection.status != "resolved" or selection.selection is None
                    or selection.selection.staff_id != candidate.staff_id
                    or selection.selection.service_id != candidate.scope.service_id):
                raise ValueError
            snapshot = load_booking_policy(candidate.scope, candidate.staff_id, self._now())
            if (snapshot is None or snapshot.policy.policy_version != candidate.policy_version
                    or snapshot.location != proposal.location
                    or snapshot.policy.business_timezone != proposal.display_zone
                    or snapshot.policy.pre_buffer != proposal.pre_buffer
                    or snapshot.policy.post_buffer != proposal.post_buffer):
                raise ValueError
            query = make_booking_query(snapshot, candidate.staff_id, self._now())
            facts = await self.calendar.get_service_facts(candidate.scope)
            availability = await self.calendar.get_availability(query)
            checked_at = self._now()
            assessment = assess_booking(snapshot, query, facts, availability, checked_at)
            if (assessment.status != "available" or not any(
                    item.scope == candidate.scope and item.staff_id == candidate.staff_id
                    and item.interval == candidate.interval for item in assessment.candidates)):
                raise ValueError
            if checked_at >= proposal.expires_at or not await self._authorized(context, approval):
                raise ValueError
            prepared = await self.mutations.prepare(request)
            if prepared is None:
                raise ValueError
        except asyncio.CancelledError:
            raise
        except Exception:
            try:
                async with self.pool.acquire() as conn:
                    cancelled = await self.store.cancel_before_dispatch(
                        conn, op.operation_id, now=self._now())
            except asyncio.CancelledError:
                raise
            except Exception:
                return BookingResult(BookingOutcome.STORE_ERROR, op.operation_id)
            if cancelled.status.value == "store_error":
                return BookingResult(BookingOutcome.STORE_ERROR, op.operation_id)
            return BookingResult(BookingOutcome.INELIGIBLE, op.operation_id)
        try:
            async with self.pool.acquire() as conn:
                grant = await self.store.claim_dispatch(conn, op.operation_id, now=self._now())
        except asyncio.CancelledError:
            raise
        except Exception:
            return BookingResult(BookingOutcome.STORE_ERROR, op.operation_id)
        if grant.status is not DispatchStatus.GRANTED:
            return BookingResult(BookingOutcome.PENDING if grant.status is DispatchStatus.UNAVAILABLE
                                 else BookingOutcome.STORE_ERROR, op.operation_id)
        record_create_diagnostic("dispatch", "claimed")
        # The claim has committed. This check also prevents entering transport
        # admission when the final authority await itself crosses expiry.
        if not await self._authorized(context, approval):
            record_create_diagnostic("dispatch", "authority_lost")
            return BookingResult(BookingOutcome.PENDING, op.operation_id)
        try:
            dispatch_check = self._now()
        except ValueError:
            record_create_diagnostic("dispatch", "invalid_clock")
            return BookingResult(BookingOutcome.PENDING, op.operation_id)
        if (dispatch_check < checked_at or dispatch_check < approval.approved_at
                or dispatch_check >= proposal.expires_at):
            record_create_diagnostic("dispatch", "expired_or_backwards_clock")
            return BookingResult(BookingOutcome.PENDING, op.operation_id)

        denial = BookingOutcome.PENDING

        async def before_send() -> bool:
            nonlocal denial
            if not await self._authorized(context, approval):
                return False
            try:
                observed = self._now()
                if observed < dispatch_check or observed >= proposal.expires_at:
                    return False
                async with self.pool.acquire() as conn:
                    ownership = await self.store.inspect_dispatch_owner(
                        conn, op.operation_id, grant.fence, grant.owner_token, now=observed)
                if ownership.status is OwnershipStatus.STORE_ERROR:
                    denial = BookingOutcome.STORE_ERROR
                    return False
                if ownership.status is not OwnershipStatus.CURRENT:
                    return False
            except asyncio.CancelledError:
                raise
            except Exception:
                denial = BookingOutcome.STORE_ERROR
                return False
            try:
                instant = self._now()
                if (not self._authorized_now(context, approval)
                        or instant < observed or instant < approval.approved_at
                        or instant >= proposal.expires_at
                        or ownership.lease_until is None or instant >= ownership.lease_until
                        or facts.observed_at > instant or availability.observed_at > instant
                        or instant - facts.observed_at > EVIDENCE_AGE
                        or instant - availability.observed_at > EVIDENCE_AGE):
                    return False
                selection = self.selection_loader().resolve_booking_accountant(
                    proposal.consultant_display)
                if (selection.status != "resolved" or selection.selection is None
                        or selection.selection.staff_id != candidate.staff_id
                        or selection.selection.service_id != candidate.scope.service_id):
                    return False
                current_policy = load_booking_policy(candidate.scope, candidate.staff_id, instant)
                if (current_policy is None
                        or current_policy.policy.policy_version != candidate.policy_version
                        or current_policy.location != proposal.location
                        or current_policy.policy.business_timezone != proposal.display_zone
                        or current_policy.policy.pre_buffer != proposal.pre_buffer
                        or current_policy.policy.post_buffer != proposal.post_buffer):
                    return False
                assessment = assess_booking(current_policy, query, facts, availability, instant)
                return (assessment.status == "available" and any(
                    item.scope == candidate.scope and item.staff_id == candidate.staff_id
                    and item.interval == candidate.interval for item in assessment.candidates))
            except Exception:
                return False

        try:
            response = await self.mutations.create_once(prepared, before_send=before_send)
        except asyncio.CancelledError:
            record_create_diagnostic("create", "cancelled")
            raise
        if response.status == "not_sent":
            return BookingResult(denial, op.operation_id)
        if response.status != "receipt" or response.provider_id is None:
            return BookingResult(BookingOutcome.PENDING, op.operation_id)
        try:
            async with self.pool.acquire() as conn:
                saved = await self.store.save_receipt(conn, op.operation_id, grant.fence,
                                                     grant.owner_token, response.provider_id,
                                                     now=self._now())
        except asyncio.CancelledError:
            record_create_diagnostic("receipt_store", "cancelled")
            raise
        except Exception:
            record_create_diagnostic("receipt_store", "store_error")
            return BookingResult(BookingOutcome.PENDING, op.operation_id)
        if not saved:
            record_create_diagnostic("receipt_store", "not_saved")
            return BookingResult(BookingOutcome.PENDING, op.operation_id)
        record_create_diagnostic("receipt_store", "saved")
        verified = await self.mutations.read_exact(response.provider_id, request)
        if verified is None:
            record_create_diagnostic("readback", "unverified")
            return BookingResult(BookingOutcome.PENDING, op.operation_id)
        return await self._finalize(approval, context, op.operation_id, grant.fence,
                                    grant.owner_token, response.provider_id, request)

    async def _finalize(self, approval, context, operation_id, fence, owner_token,
                        provider_id, request) -> BookingResult:
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    row = await conn.fetchrow("""
                        SELECT state,fence,owner_token,receipt_provider_id,booking_id,
                               tenant_id,business_id,payload_hash
                        FROM public.booking_operations WHERE operation_id=$1 FOR UPDATE
                    """, operation_id)
                    if (row is None or row["state"] not in ("dispatched", "unresolved")
                            or row["fence"] != fence or row["owner_token"] != owner_token
                            or row["receipt_provider_id"] != provider_id or row["booking_id"] is not None
                            or row["tenant_id"] != context.tenant_id
                            or row["business_id"] != context.business_id):
                        return BookingResult(BookingOutcome.PENDING, operation_id)
                    from services.scheduling.operation_store import _intent
                    if row["payload_hash"] != _intent(approval)[1]:
                        return BookingResult(BookingOutcome.PENDING, operation_id)
                    booking_id = await insert_verified_booking(
                        conn, call_sid=context.call_sid, phone_number=request.customer_phone,
                        client_name=request.customer_name, client_email=request.customer_email,
                        accountant_name=approval.proposal.consultant_display,
                        appointment_time=request.start, language=context.language,
                        provider_appointment_id=provider_id, tenant_id=context.tenant_id,
                        business_id=context.business_id)
                    evidence = {"staff_id": request.staff_id, "service_id": request.service_id,
                                "start": request.start.isoformat(), "end": request.end.isoformat(),
                                "location": request.location, "customer_name": request.customer_name,
                                "customer_phone": request.customer_phone,
                                "customer_email": request.customer_email,
                                "pre_buffer_seconds": request.pre_buffer_seconds,
                                "post_buffer_seconds": request.post_buffer_seconds}
                    if not await self.store.finalize_verified(
                            conn, operation_id, fence, owner_token, provider_id,
                            booking_id, evidence, now=self._now()):
                        raise RuntimeError("finalization rejected")
                    return BookingResult(BookingOutcome.VERIFIED, operation_id, booking_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            return BookingResult(BookingOutcome.PENDING, operation_id)

    async def recover(self, approval: ApprovedProposal,
                      context: TrustedBookingContext) -> BookingResult:
        """Inspect a previously dispatched intent by its saved exact provider ID."""
        if type(context) is not TrustedBookingContext or not self._scope_matches(approval, context):
            return BookingResult(BookingOutcome.INVALID)
        try:
            request = self._request(approval, context)
            now = self._now()
        except (AttributeError, TypeError, ValueError, OverflowError):
            return BookingResult(BookingOutcome.INVALID)
        try:
            async with self.pool.acquire() as conn:
                admitted = await self.store.find_existing_intent(conn, approval)
        except asyncio.CancelledError:
            raise
        except Exception:
            return BookingResult(BookingOutcome.STORE_ERROR)
        if admitted.status is AdmissionStatus.STORE_ERROR:
            return BookingResult(BookingOutcome.STORE_ERROR)
        if admitted.status is not AdmissionStatus.EXISTING or admitted.operation is None:
            return BookingResult(BookingOutcome.INVALID)
        op = admitted.operation
        if op.state == "applied" and op.booking_id is not None:
            return BookingResult(BookingOutcome.VERIFIED, op.operation_id, op.booking_id)
        if op.state not in ("dispatched", "unresolved", "manual_review"):
            return BookingResult(BookingOutcome.PENDING, op.operation_id)
        try:
            async with self.pool.acquire() as conn:
                claim = await self.store.claim_reconciliation(conn, op.operation_id, now=now)
        except asyncio.CancelledError:
            raise
        except Exception:
            return BookingResult(BookingOutcome.PENDING, op.operation_id)
        if claim.status is not ReconcileStatus.GRANTED:
            return BookingResult(BookingOutcome.PENDING, op.operation_id)
        # The receipt is operation-owned evidence; the inspection fence may change.
        try:
            async with self.pool.acquire() as conn:
                current = await self.store.get(conn, op.operation_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            return BookingResult(BookingOutcome.PENDING, op.operation_id)
        if current is None or current.receipt_provider_id is None:
            return BookingResult(BookingOutcome.PENDING, op.operation_id)
        try:
            verified = await self.mutations.read_exact(current.receipt_provider_id, request)
            if verified is None:
                return BookingResult(BookingOutcome.PENDING, op.operation_id)
            return await self._finalize(approval, context, op.operation_id, claim.fence,
                                        claim.owner_token, current.receipt_provider_id, request)
        except asyncio.CancelledError:
            raise
        except Exception:
            return BookingResult(BookingOutcome.PENDING, op.operation_id)
