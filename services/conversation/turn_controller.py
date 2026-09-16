"""Synchronous per-call turn ownership and mutation-dispatch state.

One controller belongs to one logical call session and must be used on that call's
event loop. It performs no I/O and starts no tasks. Integration must admit every
validated final promptly, append each drained input to dialogue history, check a
generation token immediately before publishing output, and claim a mutation
immediately before calling its executor.

The controller does not establish caller consent, validate booking facts, authorize
a mutation, make operation history durable, or bound the upstream STT queue. A
granted claim only means that this in-memory session has reserved one possible
dispatch. Provider and durable-operation layers remain responsible for authority,
idempotency, readback and reconciliation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional
from uuid import uuid4

from services.conversation.events import UtteranceIdentity


class AdmissionReason(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE_IDENTITY = "duplicate_identity"
    INVALID_INPUT = "invalid_input"
    OVERLOADED = "overloaded"
    CLOSED = "closed"


class OutputReason(str, Enum):
    CURRENT = "current"
    NO_INPUT = "no_input"
    STALE_GENERATION = "stale_generation"
    FOREIGN_SESSION = "foreign_session"
    INVALID_TOKEN = "invalid_token"
    OVERLOADED = "overloaded"
    CLOSED = "closed"


class DispatchReason(str, Enum):
    GRANTED = "granted"
    NO_INPUT = "no_input"
    INVALID_TOKEN = "invalid_token"
    FOREIGN_SESSION = "foreign_session"
    STALE_GENERATION = "stale_generation"
    INVALID_OPERATION_ID = "invalid_operation_id"
    OPERATION_ID_REUSED = "operation_id_reused"
    GENERATION_ALREADY_CLAIMED = "generation_already_claimed"
    UNSETTLED_OPERATION = "unsettled_operation"
    OVERLOADED = "overloaded"
    CLOSED = "closed"


class OperationOutcome(str, Enum):
    PENDING = "pending"
    KNOWN_SUCCESS = "known_success"
    KNOWN_REJECTION = "known_rejection"
    UNKNOWN = "unknown"


class OutcomeReason(str, Enum):
    ACCEPTED = "accepted"
    IDEMPOTENT = "idempotent"
    OPERATION_NOT_FOUND = "operation_not_found"
    INVALID_OUTCOME = "invalid_outcome"
    CONFLICTING_RESULT = "conflicting_result"


class CloseReason(str, Enum):
    CLOSED = "closed"
    ALREADY_CLOSED = "already_closed"


@dataclass(frozen=True)
class FinalUtterance:
    identity: Optional[UtteranceIdentity] = field(repr=False)
    text: str = field(repr=False)
    language: Optional[str] = field(default=None, repr=False)
    is_final: bool = True


@dataclass(frozen=True)
class GenerationToken:
    session_identity: str = field(repr=False)
    generation: int


@dataclass(frozen=True)
class AdmissionDecision:
    reason: AdmissionReason
    token: Optional[GenerationToken] = None
    capacity: Optional[str] = None

    @property
    def accepted(self) -> bool:
        return self.reason is AdmissionReason.ACCEPTED


@dataclass(frozen=True)
class InputBatch:
    inputs: tuple[FinalUtterance, ...] = field(repr=False)
    token: GenerationToken


@dataclass(frozen=True)
class OutputDecision:
    reason: OutputReason

    @property
    def allowed(self) -> bool:
        return self.reason is OutputReason.CURRENT


@dataclass(frozen=True)
class DispatchDecision:
    reason: DispatchReason
    operation_id: Optional[str] = field(default=None, repr=False)
    capacity: Optional[str] = None

    @property
    def granted(self) -> bool:
        return self.reason is DispatchReason.GRANTED


@dataclass(frozen=True)
class OperationRecord:
    operation_id: str = field(repr=False)
    generation: int
    outcome: OperationOutcome
    needs_reconciliation: bool


@dataclass(frozen=True)
class OutcomeDecision:
    reason: OutcomeReason
    operation_id: Optional[str] = field(default=None, repr=False)

    @property
    def accepted(self) -> bool:
        return self.reason in (OutcomeReason.ACCEPTED, OutcomeReason.IDEMPOTENT)


@dataclass(frozen=True)
class CloseDecision:
    reason: CloseReason
    unsettled_operation: Optional[OperationRecord] = field(default=None, repr=False)


class TurnController:
    """Fail-closed generation and dispatch state for one logical call session.

    Transitions are synchronous and atomic with respect to one event-loop owner.
    The class is deliberately not thread-safe. Capacity exhaustion is permanent:
    integration must recover or hand off and close this controller rather than
    continue speaking or writing.
    """

    def __init__(
        self,
        *,
        pending_capacity: int = 32,
        identity_capacity: int = 256,
        operation_capacity: int = 256,
    ) -> None:
        capacities = (pending_capacity, identity_capacity, operation_capacity)
        if any(
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity < 1
            for capacity in capacities
        ):
            raise ValueError("capacities must be positive integers")

        self._pending_capacity = pending_capacity
        self._identity_capacity = identity_capacity
        self._operation_capacity = operation_capacity
        self._session_identity = uuid4().hex
        self._generation = 0
        self._closed = False
        self._overloaded = False
        self._overload_capacity: Optional[str] = None
        self._pending: deque[FinalUtterance] = deque()
        self._retained_identities: set[UtteranceIdentity] = set()
        self._operations: dict[str, OperationRecord] = {}
        self._claimed_generations: set[int] = set()
        self._unsettled_operation_id: Optional[str] = None

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def overloaded(self) -> bool:
        return self._overloaded

    @property
    def retained_identity_count(self) -> int:
        return len(self._retained_identities)

    def admit(self, utterance: FinalUtterance) -> AdmissionDecision:
        """Admit one validated final without classifying or changing its text."""
        if self._closed:
            return AdmissionDecision(AdmissionReason.CLOSED)
        if not self._valid_final(utterance):
            return AdmissionDecision(AdmissionReason.INVALID_INPUT)

        identity = utterance.identity
        if identity in self._retained_identities:
            return AdmissionDecision(AdmissionReason.DUPLICATE_IDENTITY)
        if self._overloaded:
            return AdmissionDecision(
                AdmissionReason.OVERLOADED,
                capacity=self._overload_capacity,
            )
        if len(self._pending) >= self._pending_capacity:
            self._latch_overload("pending_inputs")
            return AdmissionDecision(
                AdmissionReason.OVERLOADED,
                capacity="pending_inputs",
            )
        if len(self._retained_identities) >= self._identity_capacity:
            self._latch_overload("retained_identities")
            return AdmissionDecision(
                AdmissionReason.OVERLOADED,
                capacity="retained_identities",
            )

        self._retained_identities.add(identity)
        self._pending.append(utterance)
        self._generation += 1
        self._mark_unsettled_for_reconciliation()
        token = self._current_token()
        return AdmissionDecision(AdmissionReason.ACCEPTED, token=token)

    def take_pending(self) -> InputBatch:
        """Remove and return every admitted input in FIFO order."""
        inputs = tuple(self._pending)
        self._pending.clear()
        return InputBatch(inputs, self._current_token())

    def check_output(self, token: GenerationToken) -> OutputDecision:
        """Check only whether output belongs to the latest admitted input."""
        if self._closed:
            return OutputDecision(OutputReason.CLOSED)
        if self._overloaded:
            return OutputDecision(OutputReason.OVERLOADED)
        token_reason = self._token_problem(token)
        if token_reason is not None:
            return OutputDecision(token_reason)
        if self._generation == 0:
            return OutputDecision(OutputReason.NO_INPUT)
        return OutputDecision(OutputReason.CURRENT)

    def claim_dispatch(
        self,
        token: GenerationToken,
        operation_id: str,
    ) -> DispatchDecision:
        """Atomically reserve one possible mutation dispatch for this generation."""
        if self._closed:
            return DispatchDecision(DispatchReason.CLOSED)
        if self._overloaded:
            return DispatchDecision(
                DispatchReason.OVERLOADED,
                capacity=self._overload_capacity,
            )
        token_problem = self._dispatch_token_problem(token)
        if token_problem is not None:
            return DispatchDecision(token_problem)
        if self._generation == 0:
            return DispatchDecision(DispatchReason.NO_INPUT)
        if not isinstance(operation_id, str) or not operation_id:
            return DispatchDecision(DispatchReason.INVALID_OPERATION_ID)
        if operation_id in self._operations:
            return DispatchDecision(
                DispatchReason.OPERATION_ID_REUSED,
                operation_id=operation_id,
            )
        if self._generation in self._claimed_generations:
            return DispatchDecision(DispatchReason.GENERATION_ALREADY_CLAIMED)
        if self._unsettled_operation_id is not None:
            return DispatchDecision(DispatchReason.UNSETTLED_OPERATION)
        if len(self._operations) >= self._operation_capacity:
            self._latch_overload("operation_history")
            return DispatchDecision(
                DispatchReason.OVERLOADED,
                capacity="operation_history",
            )

        record = OperationRecord(
            operation_id=operation_id,
            generation=self._generation,
            outcome=OperationOutcome.PENDING,
            needs_reconciliation=False,
        )
        self._operations[operation_id] = record
        self._claimed_generations.add(self._generation)
        self._unsettled_operation_id = operation_id
        return DispatchDecision(
            DispatchReason.GRANTED,
            operation_id=operation_id,
        )

    def record_outcome(
        self,
        operation_id: str,
        outcome: OperationOutcome,
    ) -> OutcomeDecision:
        """Record or resolve the typed outcome for one previously granted claim."""
        if not isinstance(operation_id, str):
            return OutcomeDecision(OutcomeReason.OPERATION_NOT_FOUND)
        if not isinstance(outcome, OperationOutcome) or outcome is OperationOutcome.PENDING:
            return OutcomeDecision(
                OutcomeReason.INVALID_OUTCOME,
                operation_id=operation_id,
            )
        current = self._operations.get(operation_id)
        if current is None:
            return OutcomeDecision(
                OutcomeReason.OPERATION_NOT_FOUND,
                operation_id=operation_id,
            )
        if current.outcome is outcome:
            return OutcomeDecision(
                OutcomeReason.IDEMPOTENT,
                operation_id=operation_id,
            )
        if current.outcome in (
            OperationOutcome.KNOWN_SUCCESS,
            OperationOutcome.KNOWN_REJECTION,
        ):
            return OutcomeDecision(
                OutcomeReason.CONFLICTING_RESULT,
                operation_id=operation_id,
            )

        updated = replace(
            current,
            outcome=outcome,
            needs_reconciliation=(
                current.needs_reconciliation
                or outcome is OperationOutcome.UNKNOWN
            ),
        )
        self._operations[operation_id] = updated
        if outcome in (
            OperationOutcome.KNOWN_SUCCESS,
            OperationOutcome.KNOWN_REJECTION,
        ):
            if self._unsettled_operation_id == operation_id:
                self._unsettled_operation_id = None
        else:
            self._unsettled_operation_id = operation_id
        return OutcomeDecision(
            OutcomeReason.ACCEPTED,
            operation_id=operation_id,
        )

    def close(self) -> CloseDecision:
        """Permanently close the session while preserving reconciliation state."""
        if self._closed:
            return CloseDecision(
                CloseReason.ALREADY_CLOSED,
                unsettled_operation=self.unsettled_operation,
            )
        self._closed = True
        self._mark_unsettled_for_reconciliation()
        return CloseDecision(
            CloseReason.CLOSED,
            unsettled_operation=self.unsettled_operation,
        )

    @property
    def unsettled_operation(self) -> Optional[OperationRecord]:
        if self._unsettled_operation_id is None:
            return None
        return self._operations[self._unsettled_operation_id]

    @property
    def operation_history(self) -> tuple[OperationRecord, ...]:
        return tuple(self._operations.values())

    def _current_token(self) -> GenerationToken:
        return GenerationToken(self._session_identity, self._generation)

    @staticmethod
    def _valid_final(utterance: FinalUtterance) -> bool:
        identity = utterance.identity if isinstance(utterance, FinalUtterance) else None
        return (
            isinstance(utterance, FinalUtterance)
            and isinstance(identity, UtteranceIdentity)
            and isinstance(identity.provider, str)
            and bool(identity.provider)
            and isinstance(identity.connection_epoch, str)
            and bool(identity.connection_epoch)
            and isinstance(identity.commit_sequence, int)
            and not isinstance(identity.commit_sequence, bool)
            and identity.commit_sequence > 0
            and isinstance(utterance.text, str)
            and bool(utterance.text)
            and (utterance.language is None or isinstance(utterance.language, str))
            and utterance.is_final is True
        )

    def _token_problem(self, token: GenerationToken) -> Optional[OutputReason]:
        if not isinstance(token, GenerationToken):
            return OutputReason.INVALID_TOKEN
        if (
            not isinstance(token.session_identity, str)
            or not token.session_identity
            or isinstance(token.generation, bool)
            or not isinstance(token.generation, int)
            or token.generation < 0
        ):
            return OutputReason.INVALID_TOKEN
        if token.session_identity != self._session_identity:
            return OutputReason.FOREIGN_SESSION
        if token.generation != self._generation:
            return OutputReason.STALE_GENERATION
        return None

    def _dispatch_token_problem(
        self,
        token: GenerationToken,
    ) -> Optional[DispatchReason]:
        if not isinstance(token, GenerationToken):
            return DispatchReason.INVALID_TOKEN
        if (
            not isinstance(token.session_identity, str)
            or not token.session_identity
            or isinstance(token.generation, bool)
            or not isinstance(token.generation, int)
            or token.generation < 0
        ):
            return DispatchReason.INVALID_TOKEN
        if token.session_identity != self._session_identity:
            return DispatchReason.FOREIGN_SESSION
        if token.generation != self._generation:
            return DispatchReason.STALE_GENERATION
        return None

    def _latch_overload(self, capacity: str) -> None:
        self._overloaded = True
        self._overload_capacity = capacity
        self._mark_unsettled_for_reconciliation()

    def _mark_unsettled_for_reconciliation(self) -> None:
        operation_id = self._unsettled_operation_id
        if operation_id is None:
            return
        current = self._operations[operation_id]
        if not current.needs_reconciliation:
            self._operations[operation_id] = replace(
                current,
                needs_reconciliation=True,
            )

    def __repr__(self) -> str:
        return (
            "TurnController("
            f"generation={self._generation}, "
            f"pending_count={len(self._pending)}, "
            f"retained_identity_count={len(self._retained_identities)}, "
            f"operation_count={len(self._operations)}, "
            f"overloaded={self._overloaded}, "
            f"closed={self._closed})"
        )
