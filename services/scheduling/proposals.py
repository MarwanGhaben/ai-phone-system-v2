"""Per-session, pure proposal and caller-approval state.

The coordinator owns this object and supplies validated policy candidates, actual
playback completion events and admitted final caller utterances. A claim is only
evidence of approval; it grants no calendar or provider mutation authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
import json
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from services.conversation.events import UtteranceIdentity
from services.scheduling.policy import AppointmentCandidate
from services.scheduling import spoken_arabic


_LIFETIME = timedelta(minutes=2)
_NOTICE = timedelta(minutes=30)
_MAX_EVIDENCE_AGE = timedelta(minutes=5)
_MAX_RETAINED = 128


def _utc(value: object) -> datetime | None:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


def _text(value: object) -> bool:
    return type(value) is str and bool(value.strip())


def _micros(value: timedelta) -> int:
    return ((value.days * 86400 + value.seconds) * 1_000_000 + value.microseconds)


@dataclass(frozen=True, slots=True, repr=False)
class SessionScope:
    session_id: str
    caller_source_id: str

    def __post_init__(self) -> None:
        if not _text(self.session_id) or not _text(self.caller_source_id):
            raise ValueError("invalid session scope")


@dataclass(frozen=True, slots=True, repr=False)
class CallerProvenance:
    session_id: str
    caller_source_id: str

    def __post_init__(self) -> None:
        if not _text(self.session_id) or not _text(self.caller_source_id):
            raise ValueError("invalid caller provenance")


@dataclass(frozen=True, slots=True, repr=False)
class CustomerSnapshot:
    name: str
    phone: str
    email: str

    def __post_init__(self) -> None:
        if not _text(self.name) or not _text(self.phone) or type(self.email) is not str:
            raise ValueError("invalid customer snapshot")


class InputSource(Enum):
    CALLER_FINAL = "caller_final"
    AGENT_SPEECH = "agent_speech"
    PARTIAL = "partial"
    METADATA = "metadata"


class ProposalStatus(Enum):
    OFFERED = "offered"
    PRESENTATION_STARTED = "presentation_started"
    PRESENTED = "presented"
    INPUT_ADMITTED = "input_admitted"
    APPROVED = "approved"
    CLAIMED = "claimed"
    ALREADY_CLAIMED = "already_claimed"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    STALE = "stale"
    REPLAYED = "replayed"
    NOT_PRESENTED = "not_presented"
    INTERRUPTED = "interrupted"
    INVALID_INPUT = "invalid_input"
    CANCELLED = "cancelled"
    STALE_EVIDENCE = "stale_evidence"
    FUTURE_EVIDENCE = "future_evidence"
    EXPIRED = "expired"
    FOREIGN_SESSION = "foreign_session"
    CLOSED = "closed"
    OVERLOADED = "overloaded"


@dataclass(frozen=True, slots=True, repr=False)
class SessionKey:
    nonce: str


@dataclass(frozen=True, slots=True, repr=False)
class ProposalToken:
    nonce: str
    revision: int
    fingerprint: str
    session_nonce: str


@dataclass(frozen=True, slots=True, repr=False)
class PresentationToken:
    nonce: str
    proposal_token: ProposalToken


@dataclass(frozen=True, slots=True, repr=False)
class InputReceipt:
    nonce: str
    session_nonce: str
    identity: UtteranceIdentity
    source: InputSource
    provenance: CallerProvenance
    observed_at: datetime
    admitted_at: datetime


@dataclass(frozen=True, slots=True, repr=False)
class AppointmentProposal:
    proposal_id: str
    revision: int
    fingerprint: str
    session_scope: SessionScope
    candidate: AppointmentCandidate
    customer: CustomerSnapshot
    consultant_display: str
    service_display: str
    location: str
    display_zone: str
    pre_buffer: timedelta
    post_buffer: timedelta
    issued_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True, repr=False)
class ApprovedProposal:
    proposal: AppointmentProposal
    identity: UtteranceIdentity
    approved_at: datetime


@dataclass(frozen=True, slots=True, repr=False)
class ProposalDecision:
    status: ProposalStatus
    proposal: AppointmentProposal | None = None
    token: ProposalToken | None = None
    text: str | None = None
    presentation_token: PresentationToken | None = None
    receipt: InputReceipt | None = None
    approved: ApprovedProposal | None = None

    def __bool__(self) -> bool:
        raise TypeError("inspect proposal status explicitly")

    def __repr__(self) -> str:
        return f"ProposalDecision(status={self.status.name})"


def _fingerprint(scope: SessionScope, candidate: AppointmentCandidate,
                 customer: CustomerSnapshot, consultant: str, service: str,
                 location: str, zone: str, pre: timedelta, post: timedelta) -> str:
    query = candidate.source_query
    values = [
        scope.session_id, scope.caller_source_id,
        candidate.scope.tenant_id, candidate.scope.business_id,
        candidate.scope.service_id, candidate.staff_id,
        candidate.interval.start.isoformat(), candidate.interval.end.isoformat(),
        _micros(candidate.interval.end - candidate.interval.start),
        candidate.policy_version, query.request_id,
        query.window.start.isoformat(), query.window.end.isoformat(),
        list(query.staff_ids), candidate.source_observed_at.isoformat(),
        customer.name, customer.phone, customer.email,
        consultant, service, location, zone, _micros(pre), _micros(post),
    ]
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def _readback(proposal: AppointmentProposal, language: str) -> str:
    zone = ZoneInfo(proposal.display_zone)
    start = proposal.candidate.interval.start.astimezone(zone)
    end = proposal.candidate.interval.end.astimezone(zone)
    day = start.date().isoformat()
    clock = f"{start:%H:%M}–{end:%H:%M}"
    minutes = _micros(proposal.candidate.interval.end - proposal.candidate.interval.start) // 60_000_000
    if language == "en":
        return (
            f"{proposal.customer.name}, your {proposal.service_display} with "
            f"{proposal.consultant_display} is proposed for {day}, {clock} "
            f"({proposal.display_zone}), for {minutes} minutes, in person at "
            f"{proposal.location}. Contact: {proposal.customer.phone}, "
            f"{proposal.customer.email or 'no email'}. Do you confirm this exact appointment?"
        )
    return (
        f"{proposal.customer.name}، عندي لك {spoken_arabic.service(proposal.service_display)} مع "
        f"{spoken_arabic.consultant(proposal.consultant_display)} يوم {spoken_arabic.date(start)}، "
        f"من الساعة {spoken_arabic.clock(start)} إلى {spoken_arabic.clock(end)}، "
        f"{spoken_arabic.zone(proposal.display_zone)}. مدة الموعد {spoken_arabic.duration(minutes)}، "
        f"والحضور في المكتب، على عنوان {proposal.location}. رقم التواصل: {proposal.customer.phone}، "
        f"البريد الإلكتروني: {proposal.customer.email or 'غير مسجل'}. هل أحجز لك هذا الموعد؟"
    )


class ProposalState:
    """A synchronous owner-controlled session boundary; callers must serialize use."""

    def __init__(self, scope: SessionScope, *, max_receipts: int = 128,
                 max_presentations: int = 128, max_offers: int = 128) -> None:
        if type(scope) is not SessionScope or any(
            type(limit) is not int or not 0 < limit <= _MAX_RETAINED
            for limit in (max_receipts, max_presentations, max_offers)
        ):
            raise ValueError("invalid proposal state configuration")
        self._scope = scope
        self._nonce = uuid4().hex
        self._session_key = SessionKey(uuid4().hex)
        self._max_receipts = max_receipts
        self._max_presentations = max_presentations
        self._max_offers = max_offers
        self._revision = 0
        self._current: AppointmentProposal | None = None
        self._token: ProposalToken | None = None
        self._presentation: PresentationToken | None = None
        self._playback: tuple[str, str] | None = None
        self._rendered_text: str | None = None
        self._presentation_started: datetime | None = None
        self._presentation_completed: datetime | None = None
        self._approved: ApprovedProposal | None = None
        self._claimed = False
        self._marks: set[tuple[str, str]] = set()
        self._receipts: dict[str, InputReceipt] = {}
        self._identities: set[UtteranceIdentity] = set()
        self._used_receipts: set[str] = set()
        self._closed = False
        self._overloaded = False

    def __repr__(self) -> str:
        return "ProposalState()"

    @property
    def session_key(self) -> SessionKey:
        return self._session_key

    @property
    def current(self) -> AppointmentProposal | None:
        return self._current

    def _revoke(self) -> None:
        self._current = None
        self._token = None
        self._presentation = None
        self._playback = None
        self._rendered_text = None
        self._presentation_started = None
        self._presentation_completed = None
        self._approved = None
        self._claimed = False

    def _gate(self) -> ProposalDecision | None:
        if self._closed:
            return ProposalDecision(ProposalStatus.CLOSED)
        if self._overloaded:
            return ProposalDecision(ProposalStatus.OVERLOADED)
        return None

    def _expiry(self, now: object) -> ProposalDecision | None:
        instant = _utc(now)
        if instant is None:
            return ProposalDecision(ProposalStatus.INVALID_INPUT)
        if self._current is not None and instant >= self._current.expires_at:
            self._revoke()
            return ProposalDecision(ProposalStatus.EXPIRED)
        return None

    def offer(self, *, candidate: AppointmentCandidate, customer: CustomerSnapshot,
              consultant_display: str, service_display: str, location: str,
              display_zone: str, pre_buffer: timedelta, post_buffer: timedelta,
              now: datetime, maximum_evidence_age: timedelta) -> ProposalDecision:
        gate = self._gate()
        if gate is not None:
            return gate
        self._revoke()  # A correction revokes prior approval even if it cannot be offered.
        if self._revision >= self._max_offers:
            self._overloaded = True
            return ProposalDecision(ProposalStatus.OVERLOADED)
        instant = _utc(now)
        if (type(candidate) is not AppointmentCandidate
                or type(customer) is not CustomerSnapshot
                or any(not _text(value) for value in
                       (consultant_display, service_display, location, display_zone))
                or type(pre_buffer) is not timedelta or pre_buffer < timedelta(0)
                or type(post_buffer) is not timedelta or post_buffer < timedelta(0)
                or type(maximum_evidence_age) is not timedelta
                or not timedelta(0) < maximum_evidence_age <= _MAX_EVIDENCE_AGE
                or instant is None):
            return ProposalDecision(ProposalStatus.INVALID_INPUT)
        try:
            ZoneInfo(display_zone)
        except (ZoneInfoNotFoundError, OSError, ValueError):
            return ProposalDecision(ProposalStatus.INVALID_INPUT)
        observed = _utc(candidate.source_observed_at)
        if observed is None:
            return ProposalDecision(ProposalStatus.INVALID_INPUT)
        if observed > instant:
            return ProposalDecision(ProposalStatus.FUTURE_EVIDENCE)
        if instant - observed > maximum_evidence_age:
            return ProposalDecision(ProposalStatus.STALE_EVIDENCE)
        if candidate.interval.start < instant + _NOTICE:
            return ProposalDecision(ProposalStatus.INVALID_INPUT)
        self._revision += 1
        digest = _fingerprint(self._scope, candidate, customer, consultant_display,
                              service_display, location, display_zone,
                              pre_buffer, post_buffer)
        proposal = AppointmentProposal(
            uuid4().hex, self._revision, digest, self._scope, candidate, customer,
            consultant_display, service_display, location, display_zone,
            pre_buffer, post_buffer, instant, instant + _LIFETIME,
        )
        token = ProposalToken(uuid4().hex, self._revision, digest, self._nonce)
        self._current, self._token = proposal, token
        return ProposalDecision(ProposalStatus.OFFERED, proposal=proposal, token=token)

    def begin_presentation(self, token: ProposalToken, language: str,
                           playback_generation: str, mark: str,
                           at: datetime) -> ProposalDecision:
        gate = self._gate()
        if gate is not None:
            return gate
        if token is not self._token or self._current is None:
            return ProposalDecision(ProposalStatus.STALE)
        expiry = self._expiry(at)
        if expiry is not None:
            return expiry
        instant = _utc(at)
        if (instant is None or instant < self._current.issued_at
                or not _text(playback_generation) or not _text(mark)):
            return ProposalDecision(ProposalStatus.INVALID_INPUT)
        if language not in ("en", "ar") or type(language) is not str:
            return ProposalDecision(ProposalStatus.UNSUPPORTED_LANGUAGE)
        pair = (playback_generation, mark)
        if pair in self._marks or self._presentation is not None:
            return ProposalDecision(ProposalStatus.REPLAYED)
        if len(self._marks) >= self._max_presentations:
            self._overloaded = True
            self._revoke()
            return ProposalDecision(ProposalStatus.OVERLOADED)
        self._marks.add(pair)
        presentation = PresentationToken(uuid4().hex, token)
        rendered_text = _readback(self._current, language)
        self._presentation = presentation
        self._playback = pair
        self._rendered_text = rendered_text
        self._presentation_started = instant
        return ProposalDecision(ProposalStatus.PRESENTATION_STARTED,
                                text=rendered_text,
                                presentation_token=presentation)

    def complete_presentation(self, presentation_token: PresentationToken,
                              playback_generation: str, mark: str, success: bool,
                              completed_at: datetime) -> ProposalDecision:
        gate = self._gate()
        if gate is not None:
            return gate
        if presentation_token is not self._presentation or self._current is None:
            return ProposalDecision(ProposalStatus.STALE)
        if self._presentation_completed is not None:
            return ProposalDecision(ProposalStatus.REPLAYED)
        if self._playback != (playback_generation, mark):
            return ProposalDecision(ProposalStatus.STALE)
        expiry = self._expiry(completed_at)
        if expiry is not None:
            # A matching completion with unusable timing is terminal too: a
            # later callback must not revive uncertain presentation evidence.
            self._revoke()
            return expiry
        instant = _utc(completed_at)
        if (instant is None or self._presentation_started is None
                or instant < self._presentation_started or type(success) is not bool):
            self._revoke()
            return ProposalDecision(ProposalStatus.INVALID_INPUT)
        if success is False:
            self._revoke()
            return ProposalDecision(ProposalStatus.INTERRUPTED)
        self._presentation_completed = instant
        return ProposalDecision(ProposalStatus.PRESENTED)

    def admit_input(self, session_key: SessionKey, identity: UtteranceIdentity,
                    source: InputSource, provenance: CallerProvenance, *,
                    observed_at: datetime, admitted_at: datetime) -> ProposalDecision:
        if self._closed:
            return ProposalDecision(ProposalStatus.CLOSED)
        if session_key is not self._session_key:
            return ProposalDecision(ProposalStatus.FOREIGN_SESSION)
        if (type(identity) is not UtteranceIdentity
                or not _text(identity.provider) or not _text(identity.connection_epoch)
                or type(identity.commit_sequence) is not int
                or identity.commit_sequence < 1):
            return ProposalDecision(ProposalStatus.INVALID_INPUT)
        if identity in self._identities:
            return ProposalDecision(ProposalStatus.REPLAYED)
        gate = self._gate()
        if gate is not None:
            return gate
        observed, admitted = _utc(observed_at), _utc(admitted_at)
        if (type(source) is not InputSource or source is not InputSource.CALLER_FINAL
                or type(provenance) is not CallerProvenance
                or provenance.session_id != self._scope.session_id
                or provenance.caller_source_id != self._scope.caller_source_id
                or observed is None or admitted is None or observed > admitted):
            return ProposalDecision(ProposalStatus.INVALID_INPUT)
        if len(self._receipts) >= self._max_receipts:
            self._overloaded = True
            self._revoke()
            return ProposalDecision(ProposalStatus.OVERLOADED)
        item = InputReceipt(uuid4().hex, self._nonce, identity, source,
                            provenance, observed, admitted)
        self._receipts[item.nonce] = item
        self._identities.add(identity)
        return ProposalDecision(ProposalStatus.INPUT_ADMITTED, receipt=item)

    def confirm(self, token: ProposalToken, receipt: InputReceipt,
                literal_confirm: bool, now: datetime) -> ProposalDecision:
        gate = self._gate()
        if gate is not None:
            return gate
        if token is not self._token or self._current is None:
            return ProposalDecision(ProposalStatus.STALE)
        if (type(receipt) is not InputReceipt or receipt.session_nonce != self._nonce
                or self._receipts.get(receipt.nonce) is not receipt):
            return ProposalDecision(ProposalStatus.FOREIGN_SESSION)
        if receipt.nonce in self._used_receipts:
            return ProposalDecision(ProposalStatus.REPLAYED)
        if type(literal_confirm) is not bool:
            return ProposalDecision(ProposalStatus.INVALID_INPUT)
        expiry = self._expiry(now)
        if expiry is not None:
            return expiry
        instant = _utc(now)
        if instant is None or instant < receipt.admitted_at:
            return ProposalDecision(ProposalStatus.INVALID_INPUT)
        if literal_confirm is False:
            self._used_receipts.add(receipt.nonce)
            self._revoke()
            return ProposalDecision(ProposalStatus.CANCELLED)
        if self._approved is not None:
            return ProposalDecision(ProposalStatus.REPLAYED)
        if (self._presentation_completed is None
                or receipt.observed_at <= self._presentation_completed
                or receipt.observed_at >= self._current.expires_at):
            self._used_receipts.add(receipt.nonce)
            return ProposalDecision(ProposalStatus.NOT_PRESENTED)
        self._used_receipts.add(receipt.nonce)
        self._approved = ApprovedProposal(self._current, receipt.identity, instant)
        return ProposalDecision(ProposalStatus.APPROVED, approved=self._approved)

    def consume(self, token: ProposalToken, now: datetime) -> ProposalDecision:
        gate = self._gate()
        if gate is not None:
            return gate
        if token is not self._token or self._current is None:
            return ProposalDecision(ProposalStatus.STALE)
        expiry = self._expiry(now)
        if expiry is not None:
            return expiry
        if self._approved is None:
            return ProposalDecision(ProposalStatus.NOT_PRESENTED)
        if _utc(now) < self._approved.approved_at:
            return ProposalDecision(ProposalStatus.INVALID_INPUT)
        if self._claimed:
            return ProposalDecision(ProposalStatus.ALREADY_CLAIMED)
        self._claimed = True
        return ProposalDecision(ProposalStatus.CLAIMED, approved=self._approved)

    def close(self) -> ProposalDecision:
        self._closed = True
        self._revoke()
        self._receipts.clear()
        self._identities.clear()
        self._used_receipts.clear()
        self._marks.clear()
        return ProposalDecision(ProposalStatus.CLOSED)
