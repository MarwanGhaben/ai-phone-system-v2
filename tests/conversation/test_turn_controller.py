"""T031-A transition contracts for the isolated per-call turn-state core."""

from __future__ import annotations

import asyncio

import pytest

from services.conversation.events import UtteranceIdentity
from services.conversation.turn_controller import (
    AdmissionReason,
    CloseReason,
    DispatchReason,
    FinalUtterance,
    GenerationToken,
    OperationOutcome,
    OutcomeReason,
    OutputReason,
    TurnController,
)


def final(
    sequence: int,
    text: str = "yes",
    *,
    epoch: str = "epoch-a",
    language: str | None = "en",
    is_final: bool = True,
) -> FinalUtterance:
    return FinalUtterance(
        identity=UtteranceIdentity(
            provider="synthetic",
            connection_epoch=epoch,
            commit_sequence=sequence,
        ),
        text=text,
        language=language,
        is_final=is_final,
    )


class SyntheticExecutor:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def execute(self, claim) -> None:
        if claim.granted:
            self.writes.append(claim.operation_id)


class OutputRecorder:
    def __init__(self) -> None:
        self.outputs: list[str] = []

    def publish(self, controller: TurnController, token, output: str) -> bool:
        decision = controller.check_output(token)
        if decision.allowed:
            self.outputs.append(output)
        return decision.allowed


def test_distinct_repeated_text_is_fifo_and_duplicate_identity_is_rejected() -> None:
    controller = TurnController(pending_capacity=4, identity_capacity=4)
    inputs = (
        final(1, "نعم", language="ar"),
        final(2, "نعم", language="ar"),
        final(3, "yes"),
        final(4, "yes"),
    )

    decisions = [controller.admit(item) for item in inputs]
    duplicate = controller.admit(inputs[1])
    batch = controller.take_pending()

    assert all(decision.reason is AdmissionReason.ACCEPTED for decision in decisions)
    assert [decision.token.generation for decision in decisions] == [1, 2, 3, 4]
    assert duplicate.reason is AdmissionReason.DUPLICATE_IDENTITY
    assert duplicate.token is None
    assert batch.inputs == inputs
    assert batch.token.generation == 4
    assert controller.take_pending().inputs == ()


def test_missing_identity_empty_text_and_partial_input_are_rejected() -> None:
    controller = TurnController()
    malformed_identity = UtteranceIdentity(
        provider=["synthetic"],
        connection_epoch="epoch-a",
        commit_sequence=4,
    )
    cases = (
        FinalUtterance(identity=None, text="yes", language="en", is_final=True),
        FinalUtterance(
            identity=final(1).identity,
            text="",
            language="en",
            is_final=True,
        ),
        final(2, "yes", is_final=False),
        FinalUtterance(
            identity=final(3).identity,
            text="yes",
            language=object(),
            is_final=True,
        ),
        FinalUtterance(
            identity=malformed_identity,
            text="yes",
            language="en",
            is_final=True,
        ),
    )

    decisions = [controller.admit(item) for item in cases]

    assert all(decision.reason is AdmissionReason.INVALID_INPUT for decision in decisions)
    assert controller.generation == 0
    assert controller.take_pending().inputs == ()


@pytest.mark.asyncio
async def test_new_input_while_llm_waits_blocks_old_output_and_keeps_correction() -> None:
    controller = TurnController()
    recorder = OutputRecorder()
    controller.admit(final(1, "Hussam"))
    old_batch = controller.take_pending()
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_llm() -> str:
        started.set()
        await release.wait()
        return "stale reply"

    task = asyncio.create_task(fake_llm())
    await started.wait()
    correction = final(2, "Rami")
    controller.admit(correction)
    release.set()

    assert recorder.publish(controller, old_batch.token, await task) is False
    assert recorder.outputs == []
    assert controller.check_output(old_batch.token).reason is OutputReason.STALE_GENERATION
    assert controller.take_pending().inputs == (correction,)


def test_correction_immediately_before_dispatch_records_zero_writes() -> None:
    controller = TurnController()
    executor = SyntheticExecutor()
    controller.admit(final(1, "Monday"))
    stale_token = controller.take_pending().token
    correction = final(2, "Tuesday")
    controller.admit(correction)

    denied = controller.claim_dispatch(stale_token, "operation-before")
    executor.execute(denied)

    assert denied.reason is DispatchReason.STALE_GENERATION
    assert executor.writes == []
    assert controller.take_pending().inputs == (correction,)


@pytest.mark.asyncio
async def test_correction_after_claim_keeps_one_write_and_requires_reconciliation() -> None:
    controller = TurnController()
    executor = SyntheticExecutor()
    controller.admit(final(1, "Monday"))
    token = controller.take_pending().token
    claim = controller.claim_dispatch(token, "operation-after")
    executor.execute(claim)
    provider_started = asyncio.Event()
    provider_release = asyncio.Event()

    async def fake_provider() -> OperationOutcome:
        provider_started.set()
        await provider_release.wait()
        return OperationOutcome.KNOWN_SUCCESS

    task = asyncio.create_task(fake_provider())
    await provider_started.wait()
    correction = final(2, "Tuesday")
    controller.admit(correction)
    in_flight = controller.unsettled_operation
    provider_release.set()
    result = controller.record_outcome("operation-after", await task)

    assert claim.reason is DispatchReason.GRANTED
    assert executor.writes == ["operation-after"]
    assert in_flight is not None and in_flight.needs_reconciliation is True
    assert result.reason is OutcomeReason.ACCEPTED
    assert controller.operation_history[0].outcome is OperationOutcome.KNOWN_SUCCESS
    assert controller.operation_history[0].needs_reconciliation is True
    assert controller.take_pending().inputs == (correction,)


def test_unknown_outcome_blocks_write_until_authoritative_resolution() -> None:
    controller = TurnController()
    controller.admit(final(1, "book it"))
    first_token = controller.take_pending().token
    assert controller.claim_dispatch(first_token, "operation-one").granted
    unknown = controller.record_outcome("operation-one", OperationOutcome.UNKNOWN)
    unknown_record = controller.unsettled_operation
    controller.admit(final(2, "change it"))
    second_token = controller.take_pending().token

    blocked = controller.claim_dispatch(second_token, "operation-two")
    resolved = controller.record_outcome(
        "operation-one",
        OperationOutcome.KNOWN_SUCCESS,
    )
    next_claim = controller.claim_dispatch(second_token, "operation-two")

    assert unknown.reason is OutcomeReason.ACCEPTED
    assert unknown_record is not None and unknown_record.needs_reconciliation is True
    assert blocked.reason is DispatchReason.UNSETTLED_OPERATION
    assert resolved.reason is OutcomeReason.ACCEPTED
    assert next_claim.reason is DispatchReason.GRANTED
    assert controller.check_output(first_token).reason is OutputReason.STALE_GENERATION


def test_duplicate_claims_results_conflicts_and_operation_reuse_fail_safely() -> None:
    controller = TurnController()
    controller.admit(final(1))
    token = controller.take_pending().token
    assert controller.claim_dispatch(token, "operation-one").granted

    assert (
        controller.claim_dispatch(token, "operation-one").reason
        is DispatchReason.OPERATION_ID_REUSED
    )
    assert (
        controller.record_outcome("unrelated-operation", OperationOutcome.UNKNOWN).reason
        is OutcomeReason.OPERATION_NOT_FOUND
    )
    assert (
        controller.record_outcome("operation-one", OperationOutcome.UNKNOWN).reason
        is OutcomeReason.ACCEPTED
    )
    assert (
        controller.record_outcome("operation-one", OperationOutcome.UNKNOWN).reason
        is OutcomeReason.IDEMPOTENT
    )
    assert (
        controller.record_outcome(
            "operation-one",
            OperationOutcome.KNOWN_REJECTION,
        ).reason
        is OutcomeReason.ACCEPTED
    )
    assert (
        controller.record_outcome(
            "operation-one",
            OperationOutcome.KNOWN_REJECTION,
        ).reason
        is OutcomeReason.IDEMPOTENT
    )
    assert (
        controller.record_outcome(
            "operation-one",
            OperationOutcome.KNOWN_SUCCESS,
        ).reason
        is OutcomeReason.CONFLICTING_RESULT
    )
    assert (
        controller.claim_dispatch(token, "operation-two").reason
        is DispatchReason.GENERATION_ALREADY_CLAIMED
    )

    controller.admit(final(2))
    new_token = controller.take_pending().token
    assert (
        controller.claim_dispatch(new_token, "operation-one").reason
        is DispatchReason.OPERATION_ID_REUSED
    )


def test_call_isolation_overload_and_reused_call_identity_do_not_cross_sessions() -> None:
    first = TurnController(pending_capacity=1)
    second = TurnController(pending_capacity=2)
    shared_identity = final(1, "yes")
    first_token = first.admit(shared_identity).token
    second_token = second.admit(shared_identity).token

    assert first_token != second_token
    assert first.check_output(second_token).reason is OutputReason.FOREIGN_SESSION
    assert second.check_output(first_token).reason is OutputReason.FOREIGN_SESSION
    assert (
        first.claim_dispatch(second_token, "foreign-operation").reason
        is DispatchReason.FOREIGN_SESSION
    )

    forged_boolean_generation = GenerationToken(
        first_token.session_identity,
        True,
    )
    assert (
        first.check_output(forged_boolean_generation).reason
        is OutputReason.INVALID_TOKEN
    )
    assert (
        first.claim_dispatch(forged_boolean_generation, "forged-operation").reason
        is DispatchReason.INVALID_TOKEN
    )

    assert first.admit(final(2, "overflow")).reason is AdmissionReason.OVERLOADED
    assert first.overloaded is True
    assert second.check_output(second_token).allowed is True
    assert second.take_pending().inputs == (shared_identity,)

    first.close()
    replacement_for_same_call_sid = TurnController()
    assert (
        replacement_for_same_call_sid.check_output(first_token).reason
        is OutputReason.FOREIGN_SESSION
    )


def test_pending_capacity_overload_is_explicit_and_preserves_admitted_inputs() -> None:
    controller = TurnController(pending_capacity=2, identity_capacity=4)
    first = final(1)
    second = final(2)
    controller.admit(first)
    token = controller.admit(second).token

    exhausted = controller.admit(final(3))

    assert exhausted.reason is AdmissionReason.OVERLOADED
    assert exhausted.capacity == "pending_inputs"
    assert controller.overloaded is True
    assert controller.admit(second).reason is AdmissionReason.DUPLICATE_IDENTITY
    assert controller.admit(final(4)).reason is AdmissionReason.OVERLOADED
    assert controller.check_output(token).reason is OutputReason.OVERLOADED
    assert (
        controller.claim_dispatch(token, "blocked-write").reason
        is DispatchReason.OVERLOADED
    )
    assert controller.take_pending().inputs == (first, second)


def test_identity_capacity_overload_never_forgets_accepted_evidence() -> None:
    controller = TurnController(pending_capacity=4, identity_capacity=2)
    first = final(1)
    second = final(2)
    controller.admit(first)
    controller.admit(second)

    exhausted = controller.admit(final(3))

    assert exhausted.reason is AdmissionReason.OVERLOADED
    assert exhausted.capacity == "retained_identities"
    assert controller.retained_identity_count == 2
    assert controller.admit(first).reason is AdmissionReason.DUPLICATE_IDENTITY
    assert controller.take_pending().inputs == (first, second)


def test_operation_history_capacity_latches_overload_without_reusing_history() -> None:
    controller = TurnController(operation_capacity=1)
    controller.admit(final(1))
    first_token = controller.take_pending().token
    assert controller.claim_dispatch(first_token, "operation-one").granted
    controller.record_outcome("operation-one", OperationOutcome.KNOWN_SUCCESS)
    controller.admit(final(2))
    second_token = controller.take_pending().token

    exhausted = controller.claim_dispatch(second_token, "operation-two")

    assert exhausted.reason is DispatchReason.OVERLOADED
    assert exhausted.capacity == "operation_history"
    assert controller.overloaded is True
    assert len(controller.operation_history) == 1
    assert controller.operation_history[0].operation_id == "operation-one"


def test_close_before_claim_is_repeat_safe_and_never_reopens() -> None:
    controller = TurnController()
    controller.admit(final(1))
    token = controller.take_pending().token

    first_close = controller.close()
    second_close = controller.close()

    assert first_close.reason is CloseReason.CLOSED
    assert second_close.reason is CloseReason.ALREADY_CLOSED
    assert first_close.unsettled_operation is None
    assert controller.admit(final(2)).reason is AdmissionReason.CLOSED
    assert controller.check_output(token).reason is OutputReason.CLOSED
    assert (
        controller.claim_dispatch(token, "operation-after-close").reason
        is DispatchReason.CLOSED
    )


def test_close_after_claim_preserves_unknown_and_accepts_late_resolution() -> None:
    controller = TurnController()
    controller.admit(final(1))
    token = controller.take_pending().token
    controller.claim_dispatch(token, "operation-one")
    controller.record_outcome("operation-one", OperationOutcome.UNKNOWN)

    closed = controller.close()
    closed_again = controller.close()
    late = controller.record_outcome(
        "operation-one",
        OperationOutcome.KNOWN_SUCCESS,
    )

    assert closed.unsettled_operation is not None
    assert closed.unsettled_operation.outcome is OperationOutcome.UNKNOWN
    assert closed_again.unsettled_operation is not None
    assert late.reason is OutcomeReason.ACCEPTED
    assert controller.unsettled_operation is None
    assert controller.closed is True
    assert controller.check_output(token).reason is OutputReason.CLOSED
    assert (
        controller.claim_dispatch(token, "operation-two").reason
        is DispatchReason.CLOSED
    )


def test_close_with_pending_claim_keeps_reconciliation_evidence() -> None:
    controller = TurnController()
    controller.admit(final(1))
    token = controller.take_pending().token
    controller.claim_dispatch(token, "operation-pending")

    closed = controller.close()

    assert closed.unsettled_operation is not None
    assert closed.unsettled_operation.outcome is OperationOutcome.PENDING
    assert closed.unsettled_operation.needs_reconciliation is True
    assert (
        controller.record_outcome(
            "operation-pending",
            OperationOutcome.KNOWN_REJECTION,
        ).reason
        is OutcomeReason.ACCEPTED
    )
    assert controller.closed is True


def test_repr_and_errors_exclude_raw_input_and_operation_identity() -> None:
    text_sentinel = "synthetic-phone-6135550199"
    operation_sentinel = "synthetic-provider-secret-operation"
    language_sentinel = "synthetic-credential-language"
    controller = TurnController()
    item = final(1, text_sentinel, language=language_sentinel)
    admission = controller.admit(item)
    batch = controller.take_pending()
    claim = controller.claim_dispatch(batch.token, operation_sentinel)
    outcome = controller.record_outcome(operation_sentinel, OperationOutcome.UNKNOWN)
    close = controller.close()
    rendered = " ".join(
        repr(value)
        for value in (
            controller,
            item,
            admission,
            batch,
            batch.token,
            claim,
            outcome,
            controller.operation_history[0],
            close,
        )
    )

    assert text_sentinel not in rendered
    assert operation_sentinel not in rendered
    assert language_sentinel not in rendered
    with pytest.raises(ValueError) as caught:
        TurnController(pending_capacity=0)
    assert text_sentinel not in str(caught.value)
    assert operation_sentinel not in str(caught.value)
