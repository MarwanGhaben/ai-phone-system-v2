"""Durable create-operation claims; no provider I/O or caller interpretation.

The trusted coordinator supplies an accepted approval and authoritative settlement
evidence. Each method uses the provided asyncpg connection. A caller may wrap
several methods and local writes in one outer transaction for atomic finalization.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
import json
from uuid import UUID, uuid4

from services.conversation.events import UtteranceIdentity
from services.scheduling.policy import AppointmentCandidate
from services.scheduling.proposals import ApprovedProposal, AppointmentProposal


def _utc(value: object) -> datetime | None:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


def _valid_text(value: object) -> bool:
    return type(value) is str and 0 < len(value) <= 255 and bool(value.strip())


def _us(value: timedelta) -> int:
    return (value.days * 86400 + value.seconds) * 1_000_000 + value.microseconds


class AdmissionStatus(Enum):
    CREATED = "created"
    EXISTING = "existing"
    PAYLOAD_CONFLICT = "payload_conflict"
    INTERVAL_CONFLICT = "interval_conflict"
    INVALID = "invalid"
    EXPIRED = "expired"
    STORE_ERROR = "store_error"


class DispatchStatus(Enum):
    GRANTED = "granted"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    STORE_ERROR = "store_error"


class OwnershipStatus(Enum):
    CURRENT = "current"
    STALE = "stale"
    INVALID = "invalid"
    STORE_ERROR = "store_error"


@dataclass(frozen=True, slots=True, repr=False)
class OwnershipResult:
    status: OwnershipStatus
    lease_until: datetime | None = None


class ReconcileStatus(Enum):
    GRANTED = "granted"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    STORE_ERROR = "store_error"


class SettlementKind(Enum):
    VERIFIED_APPLIED = "verified_applied"
    VERIFIED_NOT_APPLIED = "verified_not_applied"
    MANUAL_REVIEW = "manual_review"


class TransitionStatus(Enum):
    APPLIED = "applied"
    RELEASED = "released"
    MANUAL_REVIEW = "manual_review"
    CANCELLED = "cancelled"
    STALE_FENCE = "stale_fence"
    INVALID_STATE = "invalid_state"
    INVALID = "invalid"
    STORE_ERROR = "store_error"


class OperationReadError(Exception):
    """Fixed read-failure classification that carries no driver or row details."""


@dataclass(frozen=True, slots=True, repr=False)
class SettlementEvidence:
    kind: SettlementKind
    source: str
    observed_at: datetime
    provider_id: str | None = None


@dataclass(frozen=True, slots=True, repr=False)
class OperationRecord:
    operation_id: UUID
    state: str
    fence: int
    payload_hash: str
    snapshot_json: str
    starts_at: datetime
    ends_at: datetime
    admitted_at: datetime
    ownership_started_at: datetime | None
    lease_until: datetime | None
    receipt_provider_id: str | None = None
    booking_id: int | None = None


@dataclass(frozen=True, slots=True, repr=False)
class AdmissionResult:
    status: AdmissionStatus
    operation: OperationRecord | None = None

    def __bool__(self) -> bool:
        raise TypeError("inspect admission status explicitly")


@dataclass(frozen=True, slots=True, repr=False)
class DispatchResult:
    status: DispatchStatus
    operation_id: UUID | None = None
    fence: int | None = None
    owner_token: UUID | None = None
    create_permission: bool = False

    def __bool__(self) -> bool:
        raise TypeError("inspect dispatch status explicitly")


@dataclass(frozen=True, slots=True, repr=False)
class ReconcileResult:
    status: ReconcileStatus
    operation_id: UUID | None = None
    fence: int | None = None
    owner_token: UUID | None = None
    create_permission: bool = False

    def __bool__(self) -> bool:
        raise TypeError("inspect reconciliation status explicitly")


@dataclass(frozen=True, slots=True, repr=False)
class TransitionResult:
    status: TransitionStatus
    operation: OperationRecord | None = None

    def __bool__(self) -> bool:
        raise TypeError("inspect transition status explicitly")


def _record(row) -> OperationRecord:
    snapshot = row["payload_snapshot"]
    if type(snapshot) is not str:
        snapshot = json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False)
    return OperationRecord(row["operation_id"], row["state"], row["fence"],
                           row["payload_hash"], snapshot, row["starts_at"],
                           row["ends_at"], row["admitted_at"],
                           row["ownership_started_at"], row["lease_until"],
                           row.get("receipt_provider_id"), row.get("booking_id"))


_SELECT = """SELECT operation_id,state,fence,payload_hash,payload_snapshot,
                   starts_at,ends_at,admitted_at,ownership_started_at,lease_until,
                   receipt_provider_id,booking_id
            FROM public.booking_operations"""


def _intent(approval: object) -> tuple[dict, str] | None:
    if type(approval) is not ApprovedProposal or type(approval.proposal) is not AppointmentProposal:
        return None
    proposal = approval.proposal
    candidate = proposal.candidate
    identity = approval.identity
    if (type(candidate) is not AppointmentCandidate
            or type(identity) is not UtteranceIdentity
            or not _valid_text(identity.provider)
            or not _valid_text(identity.connection_epoch)
            or type(identity.commit_sequence) is not int or identity.commit_sequence <= 0
            or not _valid_text(proposal.proposal_id)
            or type(proposal.revision) is not int or proposal.revision <= 0
            or not all(_valid_text(value) for value in (
                candidate.scope.tenant_id, candidate.scope.business_id,
                candidate.scope.service_id, candidate.staff_id))
            or type(proposal.pre_buffer) is not timedelta
            or type(proposal.post_buffer) is not timedelta
            or not timedelta(0) <= proposal.pre_buffer <= timedelta(hours=2)
            or not timedelta(0) <= proposal.post_buffer <= timedelta(hours=2)):
        return None
    issued, expires, confirmed = (_utc(proposal.issued_at), _utc(proposal.expires_at),
                                   _utc(approval.approved_at))
    if (issued is None or expires is None or confirmed is None
            or not issued <= confirmed < expires
            or not timedelta(0) < expires - issued <= timedelta(minutes=2)
            or candidate.interval.start <= issued
            or not timedelta(0) < candidate.interval.end - candidate.interval.start <= timedelta(hours=4)):
        return None
    snapshot = {
        "proposal_id": proposal.proposal_id, "revision": proposal.revision,
        "proposal_fingerprint": proposal.fingerprint,
        "session": [proposal.session_scope.session_id, proposal.session_scope.caller_source_id],
        "scope": [candidate.scope.tenant_id, candidate.scope.business_id,
                  candidate.scope.service_id],
        "staff_id": candidate.staff_id,
        "start": candidate.interval.start.isoformat(),
        "end": candidate.interval.end.isoformat(),
        "duration_us": _us(candidate.interval.end - candidate.interval.start),
        "pre_buffer_us": _us(proposal.pre_buffer),
        "post_buffer_us": _us(proposal.post_buffer),
        "policy_version": candidate.policy_version,
        "source_query": [candidate.source_query.request_id,
                         candidate.source_query.window.start.isoformat(),
                         candidate.source_query.window.end.isoformat(),
                         list(candidate.source_query.staff_ids)],
        "source_observed_at": candidate.source_observed_at.isoformat(),
        "customer": [proposal.customer.name, proposal.customer.phone, proposal.customer.email],
        "consultant_display": proposal.consultant_display,
        "service_display": proposal.service_display,
        "location": proposal.location, "display_zone": proposal.display_zone,
        "issued_at": issued.isoformat(), "expires_at": expires.isoformat(),
        "confirmation": [identity.provider, identity.connection_epoch,
                         identity.commit_sequence, confirmed.isoformat()],
    }
    encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return snapshot, sha256(encoded.encode("utf-8")).hexdigest()


class OperationStore:
    """Database-owned state; instances hold no operation or replay registry."""

    def __repr__(self) -> str:
        return "OperationStore()"

    async def find_existing_intent(self, conn, approval: ApprovedProposal) -> AdmissionResult:
        """Read an exact immutable intent without ever acquiring an interval claim."""
        try:
            intent = _intent(approval)
        except (AttributeError, TypeError, ValueError, OverflowError):
            intent = None
        if intent is None:
            return AdmissionResult(AdmissionStatus.INVALID)
        proposal = approval.proposal
        candidate = proposal.candidate
        try:
            row = await conn.fetchrow(
                _SELECT + " WHERE tenant_id=$1 AND business_id=$2 AND action='create' "
                "AND proposal_id=$3 AND proposal_revision=$4",
                candidate.scope.tenant_id, candidate.scope.business_id,
                proposal.proposal_id, proposal.revision)
            if row is None:
                return AdmissionResult(AdmissionStatus.INVALID)
            if row["payload_hash"] != intent[1]:
                return AdmissionResult(AdmissionStatus.PAYLOAD_CONFLICT)
            return AdmissionResult(AdmissionStatus.EXISTING, _record(row))
        except Exception:
            return AdmissionResult(AdmissionStatus.STORE_ERROR)

    async def inspect_dispatch_owner(self, conn, operation_id: UUID, fence: int,
                                     owner_token: UUID, *, now: datetime) -> OwnershipResult:
        instant = _utc(now)
        if (not isinstance(operation_id, UUID) or type(fence) is not int or fence <= 0
                or not isinstance(owner_token, UUID) or instant is None):
            return OwnershipResult(OwnershipStatus.INVALID)
        try:
            row = await conn.fetchrow("""
                SELECT lease_until FROM public.booking_operations
                WHERE operation_id=$1 AND fence=$2 AND owner_token=$3
                  AND state='dispatched' AND lease_until > $4
                  AND ownership_started_at <= $4 AND updated_at <= $4
            """, operation_id, fence, owner_token, instant)
            return (OwnershipResult(OwnershipStatus.CURRENT, row["lease_until"])
                    if row is not None else OwnershipResult(OwnershipStatus.STALE))
        except Exception:
            return OwnershipResult(OwnershipStatus.STORE_ERROR)

    async def save_receipt(self, conn, operation_id: UUID, fence: int,
                           owner_token: UUID, provider_id: str,
                           *, now: datetime) -> bool:
        """Persist a create receipt independently of inspection ownership."""
        instant = _utc(now)
        if (not isinstance(operation_id, UUID) or type(fence) is not int or fence <= 0
                or not isinstance(owner_token, UUID) or not _valid_text(provider_id)
                or instant is None):
            return False
        try:
            row = await conn.fetchrow("""
                UPDATE public.booking_operations
                SET receipt_provider_id=$4,updated_at=$5
                WHERE operation_id=$1 AND fence=$2 AND owner_token=$3
                  AND state IN ('dispatched','unresolved')
                  AND receipt_provider_id IS NULL
                  AND ownership_started_at <= $5 AND updated_at <= $5
                RETURNING operation_id
            """, operation_id, fence, owner_token, provider_id, instant)
            return row is not None
        except Exception:
            return False

    async def finalize_verified(self, conn, operation_id: UUID, fence: int,
                                owner_token: UUID, provider_id: str,
                                booking_id: int, snapshot: dict,
                                *, now: datetime) -> bool:
        """Call inside the same transaction as booking and held-job insertion.

        The caller owns rollback. A stale fence or changed receipt cannot settle.
        """
        instant = _utc(now)
        if (not isinstance(operation_id, UUID) or type(fence) is not int or fence <= 0
                or not isinstance(owner_token, UUID) or not _valid_text(provider_id)
                or type(booking_id) is not int or booking_id <= 0
                or type(snapshot) is not dict or instant is None):
            return False
        row = await conn.fetchrow("""
            UPDATE public.booking_operations
            SET state='applied',provider_id=$4,booking_id=$5,
                verified_snapshot=$6::jsonb,settlement_source='exact_id_readback',
                settlement_observed_at=$7,updated_at=$7
            WHERE operation_id=$1 AND fence=$2 AND owner_token=$3
              AND state IN ('dispatched','unresolved')
              AND receipt_provider_id=$4 AND booking_id IS NULL
              AND ownership_started_at <= $7 AND updated_at <= $7
            RETURNING operation_id
        """, operation_id, fence, owner_token, provider_id, booking_id,
            json.dumps(snapshot, ensure_ascii=False, sort_keys=True, allow_nan=False), instant)
        return row is not None

    async def admit(self, conn, approval: ApprovedProposal, *, now: datetime) -> AdmissionResult:
        instant = _utc(now)
        try:
            intent = _intent(approval)
        except (AttributeError, TypeError, ValueError, OverflowError):
            intent = None
        if instant is None or intent is None:
            return AdmissionResult(AdmissionStatus.INVALID)
        proposal = approval.proposal
        candidate = proposal.candidate
        snapshot, digest = intent
        key = (candidate.scope.tenant_id, candidate.scope.business_id,
               "create", proposal.proposal_id, proposal.revision)
        admission_scope = json.dumps(
            ["booking-staff-admission-v1", candidate.scope.tenant_id,
             candidate.scope.business_id, candidate.staff_id],
            ensure_ascii=False, separators=(",", ":"))
        admission_lock = int.from_bytes(
            sha256(admission_scope.encode("utf-8")).digest()[:8], "big", signed=True)
        try:
            async with conn.transaction():
                # Serialize only admissions for this staff scope. A committed
                # conflicting claim can then be rejected without a speculative
                # GiST insertion waiting on its owner's receipt/finalization.
                await conn.execute("SELECT pg_advisory_xact_lock($1::bigint)", admission_lock)
                existing = await conn.fetchrow(_SELECT + " WHERE tenant_id=$1 AND business_id=$2 "
                    "AND action=$3 AND proposal_id=$4 AND proposal_revision=$5", *key)
                if existing is not None:
                    return AdmissionResult(
                        AdmissionStatus.EXISTING if existing["payload_hash"] == digest
                        else AdmissionStatus.PAYLOAD_CONFLICT,
                        _record(existing) if existing["payload_hash"] == digest else None)
                confirmed = _utc(approval.approved_at)
                issued = _utc(proposal.issued_at)
                if confirmed is None or issued is None or instant < confirmed or instant < issued:
                    return AdmissionResult(AdmissionStatus.INVALID)
                if instant >= proposal.expires_at:
                    return AdmissionResult(AdmissionStatus.EXPIRED)
                if candidate.interval.start <= instant:
                    return AdmissionResult(AdmissionStatus.INVALID)
                occupied = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM public.booking_operations
                        WHERE tenant_id=$1 AND business_id=$2 AND staff_id=$3
                          AND state <> 'released'
                          AND claim_span && tstzrange(
                              $4::timestamptz-$6::interval,
                              $5::timestamptz+$7::interval,'[)')
                    )
                """, candidate.scope.tenant_id, candidate.scope.business_id,
                    candidate.staff_id, candidate.interval.start, candidate.interval.end,
                    proposal.pre_buffer, proposal.post_buffer)
                if occupied:
                    return AdmissionResult(AdmissionStatus.INTERVAL_CONFLICT)
                row = await conn.fetchrow("""
                    INSERT INTO public.booking_operations
                      (operation_id,tenant_id,business_id,staff_id,service_id,action,
                       proposal_id,proposal_revision,payload_hash,payload_snapshot,
                       confirmation_identity,confirmed_at,issued_at,expires_at,
                       admitted_at,starts_at,ends_at,pre_buffer,post_buffer,claim_span,
                       created_at,updated_at)
                    VALUES ($1,$2,$3,$4,$5,'create',$6,$7,$8,$9::jsonb,$10::jsonb,
                            $11,$12,$13,$14,$15,$16,$17,$18,
                            tstzrange($15::timestamptz-$17::interval,
                                      $16::timestamptz+$18::interval,'[)'),$14,$14)
                    ON CONFLICT ON CONSTRAINT booking_operations_intent_key DO NOTHING
                    RETURNING operation_id,state,fence,payload_hash,payload_snapshot,
                              starts_at,ends_at,admitted_at,ownership_started_at,lease_until
                """, uuid4(), candidate.scope.tenant_id, candidate.scope.business_id,
                    candidate.staff_id, candidate.scope.service_id,
                    proposal.proposal_id, proposal.revision, digest,
                    json.dumps(snapshot, ensure_ascii=False, sort_keys=True),
                    json.dumps({"provider": approval.identity.provider,
                                "epoch": approval.identity.connection_epoch,
                                "sequence": approval.identity.commit_sequence}),
                    approval.approved_at, proposal.issued_at, proposal.expires_at,
                    instant, candidate.interval.start, candidate.interval.end,
                    proposal.pre_buffer, proposal.post_buffer)
                if row is not None:
                    return AdmissionResult(AdmissionStatus.CREATED, _record(row))
                existing = await conn.fetchrow(_SELECT + " WHERE tenant_id=$1 AND business_id=$2 "
                    "AND action=$3 AND proposal_id=$4 AND proposal_revision=$5", *key)
                if existing is None:
                    return AdmissionResult(AdmissionStatus.INTERVAL_CONFLICT)
                return AdmissionResult(
                    AdmissionStatus.EXISTING if existing["payload_hash"] == digest
                    else AdmissionStatus.PAYLOAD_CONFLICT,
                    _record(existing) if existing["payload_hash"] == digest else None)
        except Exception as exc:
            if getattr(exc, "sqlstate", None) == "23P01":
                return AdmissionResult(AdmissionStatus.INTERVAL_CONFLICT)
            return AdmissionResult(AdmissionStatus.STORE_ERROR)

    async def get(self, conn, operation_id: UUID) -> OperationRecord | None:
        if not isinstance(operation_id, UUID):
            return None
        try:
            row = await conn.fetchrow(_SELECT + " WHERE operation_id=$1", operation_id)
            return _record(row) if row is not None else None
        except Exception:
            raise OperationReadError("operation read failed") from None

    async def claim_dispatch(self, conn, operation_id: UUID, *, now: datetime,
                             lease: timedelta = timedelta(minutes=1)) -> DispatchResult:
        instant = _utc(now)
        if (not isinstance(operation_id, UUID) or instant is None or conn.is_in_transaction()
                or type(lease) is not timedelta
                or not timedelta(0) < lease <= timedelta(minutes=5)):
            return DispatchResult(DispatchStatus.INVALID)
        owner = uuid4()
        try:
            async with conn.transaction():
                row = await conn.fetchrow("""
                    UPDATE public.booking_operations
                    SET state='dispatched',fence=fence+1,dispatch_count=1,
                        owner_token=$2,lease_until=$3,ownership_started_at=$4,updated_at=$4
                    WHERE operation_id=$1 AND state='pending' AND expires_at > $4
                      AND confirmed_at <= $4 AND admitted_at <= $4
                    RETURNING fence
                """, operation_id, owner, instant + lease, instant)
            if row is None:
                return DispatchResult(DispatchStatus.UNAVAILABLE)
            return DispatchResult(DispatchStatus.GRANTED, operation_id,
                                  row["fence"], owner, True)
        except Exception:
            return DispatchResult(DispatchStatus.STORE_ERROR)

    async def claim_reconciliation(self, conn, operation_id: UUID, *, now: datetime,
                                   lease: timedelta = timedelta(minutes=1)) -> ReconcileResult:
        instant = _utc(now)
        if (not isinstance(operation_id, UUID) or instant is None or conn.is_in_transaction()
                or type(lease) is not timedelta
                or not timedelta(0) < lease <= timedelta(minutes=5)):
            return ReconcileResult(ReconcileStatus.INVALID)
        owner = uuid4()
        try:
            async with conn.transaction():
                row = await conn.fetchrow("""
                    UPDATE public.booking_operations
                    SET state='unresolved',fence=fence+1,owner_token=$2,
                        lease_until=$3,ownership_started_at=$4,
                        provider_id=NULL,settlement_source=NULL,
                        settlement_observed_at=NULL,updated_at=$4
                    WHERE operation_id=$1 AND state IN ('dispatched','unresolved','manual_review')
                      AND lease_until <= $4 AND ownership_started_at <= $4
                      AND confirmed_at <= $4 AND admitted_at <= $4
                      AND updated_at <= $4
                    RETURNING fence
                """, operation_id, owner, instant + lease, instant)
            if row is None:
                return ReconcileResult(ReconcileStatus.UNAVAILABLE)
            return ReconcileResult(ReconcileStatus.GRANTED, operation_id,
                                   row["fence"], owner, False)
        except Exception:
            return ReconcileResult(ReconcileStatus.STORE_ERROR)

    async def settle(self, conn, operation_id: UUID, fence: int, owner_token: UUID,
                     evidence: SettlementEvidence) -> TransitionResult:
        if (not isinstance(operation_id, UUID) or type(fence) is not int or fence <= 0
                or not isinstance(owner_token, UUID) or type(evidence) is not SettlementEvidence
                or type(evidence.kind) is not SettlementKind
                or not _valid_text(evidence.source) or _utc(evidence.observed_at) is None
                or (evidence.kind is SettlementKind.VERIFIED_APPLIED
                    and not _valid_text(evidence.provider_id))
                or (evidence.kind is not SettlementKind.VERIFIED_APPLIED
                    and evidence.provider_id is not None)):
            return TransitionResult(TransitionStatus.INVALID)
        state = {SettlementKind.VERIFIED_APPLIED: "applied",
                 SettlementKind.VERIFIED_NOT_APPLIED: "released",
                 SettlementKind.MANUAL_REVIEW: "manual_review"}[evidence.kind]
        try:
            async with conn.transaction():
                row = await conn.fetchrow("""
                    UPDATE public.booking_operations
                    SET state=$4,provider_id=$5,settlement_source=$6,
                        settlement_observed_at=$7,updated_at=$7
                    WHERE operation_id=$1 AND fence=$2 AND owner_token=$3
                      AND state IN ('dispatched','unresolved')
                      AND ownership_started_at <= $7
                      AND ($4 <> 'released' OR receipt_provider_id IS NULL)
                      AND ($4 <> 'applied' OR receipt_provider_id IS NULL
                           OR receipt_provider_id=$5)
                    RETURNING operation_id,state,fence,payload_hash,payload_snapshot,
                              starts_at,ends_at,admitted_at,ownership_started_at,lease_until
                """, operation_id, fence, owner_token, state, evidence.provider_id,
                    evidence.source, evidence.observed_at)
                if row is not None:
                    return TransitionResult({"applied": TransitionStatus.APPLIED,
                                             "released": TransitionStatus.RELEASED,
                                             "manual_review": TransitionStatus.MANUAL_REVIEW}[state],
                                            _record(row))
                actual = await conn.fetchrow("SELECT fence,owner_token FROM public.booking_operations "
                                             "WHERE operation_id=$1", operation_id)
                if actual is None or actual["fence"] != fence or actual["owner_token"] != owner_token:
                    return TransitionResult(TransitionStatus.STALE_FENCE)
                return TransitionResult(TransitionStatus.INVALID_STATE)
        except Exception:
            return TransitionResult(TransitionStatus.STORE_ERROR)

    async def cancel_before_dispatch(self, conn, operation_id: UUID, *, now: datetime) -> TransitionResult:
        instant = _utc(now)
        if not isinstance(operation_id, UUID) or instant is None:
            return TransitionResult(TransitionStatus.INVALID)
        try:
            async with conn.transaction():
                row = await conn.fetchrow("""
                    UPDATE public.booking_operations SET state='released',updated_at=$2
                    WHERE operation_id=$1 AND state='pending'
                      AND confirmed_at <= $2 AND admitted_at <= $2
                    RETURNING operation_id,state,fence,payload_hash,payload_snapshot,
                              starts_at,ends_at,admitted_at,ownership_started_at,lease_until
                """, operation_id, instant)
            return (TransitionResult(TransitionStatus.CANCELLED, _record(row)) if row is not None
                    else TransitionResult(TransitionStatus.INVALID_STATE))
        except Exception:
            return TransitionResult(TransitionStatus.STORE_ERROR)
