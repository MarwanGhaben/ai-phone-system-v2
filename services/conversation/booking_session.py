"""Per-call ownership for the enabled verified booking phone route."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from loguru import logger

from services.conversation.events import UtteranceIdentity
from services.conversation.turn_controller import (FinalUtterance, OperationOutcome,
                                                   TurnController)
from services.scheduling.booking_service import (BookingOutcome, BookingService,
                                                 TrustedBookingContext)
from services.scheduling.proposals import (CallerProvenance, CustomerSnapshot,
                                           InputSource, ProposalState, ProposalStatus,
                                           SessionScope)
from services.scheduling.booking_check import EVIDENCE_AGE


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class BookingSession:
    """One bounded logical call session; all transitions run on its event loop."""

    def __init__(self, context, handler, pool, calendar, mutations,
                 owner_check=None, on_new_input=None, clock=None,
                 selection_loader=None):
        self.context, self.handler = context, handler
        self.pool, self.calendar, self.mutations = pool, calendar, mutations
        self.owner_check = owner_check or (lambda: True)
        self.on_new_input = on_new_input or (lambda: None)
        self.clock = clock or now_utc
        self.selection_loader = selection_loader
        self.scope = SessionScope(uuid4().hex, uuid4().hex)
        self.turn = TurnController(pending_capacity=16, identity_capacity=128)
        self.proposals = ProposalState(self.scope)
        self.receipts = {}
        self.event = asyncio.Event()
        self.worker = None
        self.closed = False
        self.offer_token = None
        self.presentation_token = None
        self.presentation_pair = None
        self.current_receipt = None
        self.current_text = ""
        self.active_token = None
        self.dispatch_token = None
        self.approval = None
        self.unresolved = False
        self.help_spoken = False

    def owns(self, context, handler) -> bool:
        return (not self.closed and self.context is context and self.handler is handler
                and self.owner_check())

    def admit(self, result) -> bool:
        """Called by the STT reader before any dialogue/provider await."""
        if self.closed or not self.owns(self.context, self.handler):
            return False
        identity = getattr(result, "utterance_id", None)
        observed = getattr(result, "received_at", None)
        instant = self.clock()
        if (type(identity) is not UtteranceIdentity or type(observed) is not datetime
                or observed.tzinfo is None or observed.utcoffset() is None
                or observed > instant or not getattr(result, "is_final", False)
                or type(getattr(result, "text", None)) is not str):
            self.unresolved = True
            self.event.set()
            return False
        decision = self.turn.admit(FinalUtterance(identity, result.text,
                                                  getattr(result, "language", None)))
        if not decision.accepted:
            if self.turn.overloaded:
                self.unresolved = True
                self.event.set()
            return False
        self.on_new_input()
        receipt = self.proposals.admit_input(
            self.proposals.session_key, identity, InputSource.CALLER_FINAL,
            CallerProvenance(self.scope.session_id, self.scope.caller_source_id),
            observed_at=observed, admitted_at=instant)
        if receipt.status is not ProposalStatus.INPUT_ADMITTED:
            self.unresolved = True
        else:
            self.receipts[identity] = receipt.receipt
            if self.proposals.current is not None:
                from services.llm.tool_protocol import literal_approval
                language = getattr(result, "language", None)
                if language not in ("en", "ar"):
                    language = self.context.language
                if literal_approval(result.text, language) is not True:
                    # A change, refusal or qualified answer revokes the old
                    # tuple before model classification or delayed lookup.
                    self.revoke_proposal()
        self.event.set()
        return receipt.status is ProposalStatus.INPUT_ADMITTED

    async def run(self, orchestrator, call_sid: str) -> None:
        try:
            while not self.closed:
                await self.event.wait()
                self.event.clear()
                if self.closed:
                    break
                batch = self.turn.take_pending()
                if not batch.inputs:
                    if self.unresolved or self.turn.overloaded:
                        await orchestrator._verified_help(call_sid, self)
                    continue
                # Every distinct identity is retained in dialogue order. Only
                # the newest generation may publish speech or dispatch.
                for item in batch.inputs:
                    self.context.add_user_message(item.text)
                latest = batch.inputs[-1]
                self.current_receipt = self.receipts.get(latest.identity)
                self.current_text = latest.text
                if self.current_receipt is None:
                    await orchestrator._verified_help(call_sid, self)
                    continue
                if self.turn.overloaded:
                    await orchestrator._verified_help(call_sid, self)
                    continue
                if self.unresolved:
                    await orchestrator._verified_help(call_sid, self)
                self.active_token = batch.token
                try:
                    await orchestrator._process_verified_turn(call_sid, self, latest, batch.token)
                finally:
                    self.active_token = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.unresolved = True
            frame = exc.__traceback__
            while frame is not None and frame.tb_next is not None:
                frame = frame.tb_next
            logger.warning(
                "BookingSession: verified worker failed "
                f"({type(exc).__name__}, line={frame.tb_lineno if frame else 0})")
            await orchestrator._verified_help(call_sid, self)

    def offer(self, candidate, customer: CustomerSnapshot, consultant: str,
              service: str, location: str, zone: str, pre, post):
        result = self.proposals.offer(
            candidate=candidate, customer=customer,
            consultant_display=consultant, service_display=service,
            location=location, display_zone=zone, pre_buffer=pre,
            post_buffer=post, now=self.clock(), maximum_evidence_age=EVIDENCE_AGE)
        self.offer_token = result.token if result.status is ProposalStatus.OFFERED else None
        return result

    def revoke_proposal(self) -> None:
        self.proposals.close()
        self.proposals = ProposalState(self.scope)
        self.offer_token = None
        self.presentation_token = None
        self.presentation_pair = None

    async def present(self, orchestrator, call_sid: str, language: str, token) -> bool:
        if (self.offer_token is None or not self.turn.check_output(token).allowed
                or self.unresolved or self.closed):
            return False
        presentation = None
        pair = None
        readback = None

        def begun(owner):
            nonlocal presentation, pair, readback
            if not self.turn.check_output(token).allowed or self.closed:
                return None
            # Twilio's mark name is private to wait_for_playback; that method
            # returns success only for this current PlaybackOwner's exact mark.
            # Bind both state tokens to its handler session and generation.
            pair = (owner.session_id, str(owner.generation))
            decision = self.proposals.begin_presentation(
                self.offer_token, language, *pair, at=self.clock())
            if decision.status is not ProposalStatus.PRESENTATION_STARTED:
                return None
            presentation = decision.presentation_token
            self.presentation_token, self.presentation_pair = presentation, pair
            readback = decision.text
            return decision.text

        success = False
        try:
            success = await orchestrator._speak_to_caller(
                call_sid, "", language, on_booking_playback=begun)
        finally:
            if presentation is not None and pair is not None:
                outcome = self.proposals.complete_presentation(
                    presentation, *pair,
                    success=bool(success and self.turn.check_output(token).allowed
                                 and self.owns(self.context, self.handler)),
                    completed_at=self.clock())
                success = outcome.status is ProposalStatus.PRESENTED
                if success and readback is not None:
                    self.context.add_assistant_message(readback)
        return success

    async def confirm(self, caller_text: str, language: str, token) -> BookingOutcome | None:
        from services.llm.tool_protocol import literal_approval
        if (self.closed or self.unresolved or self.current_receipt is None
                or self.offer_token is None or not self.turn.check_output(token).allowed):
            return None
        literal = literal_approval(caller_text, language)
        if literal is None:
            return None
        decision = self.proposals.confirm(self.offer_token, self.current_receipt,
                                          literal, self.clock())
        if decision.status is not ProposalStatus.APPROVED:
            return None
        claimed = self.proposals.consume(self.offer_token, self.clock())
        if claimed.status is not ProposalStatus.CLAIMED:
            return None
        operation_key = claimed.approved.proposal.proposal_id
        if not self.turn.claim_dispatch(token, operation_key).granted:
            return None
        self.dispatch_token, self.approval = token, claimed.approved
        trusted = TrustedBookingContext(
            self.mutations.tenant_id, self.mutations.business_id,
            self.scope.session_id, self.scope.caller_source_id,
            self.context.call_sid, language)
        service = BookingService(self.pool, self.calendar, self.mutations, self,
                                 clock=self.clock,
                                 **({"selection_loader": self.selection_loader}
                                    if self.selection_loader is not None else {}))
        try:
            result = await service.create(claimed.approved, trusted)
        except asyncio.CancelledError:
            self.turn.record_outcome(operation_key, OperationOutcome.UNKNOWN)
            self.unresolved = True
            raise
        except Exception:
            self.turn.record_outcome(operation_key, OperationOutcome.UNKNOWN)
            self.unresolved = True
            return BookingOutcome.PENDING
        if result.outcome is BookingOutcome.VERIFIED:
            self.turn.record_outcome(operation_key, OperationOutcome.KNOWN_SUCCESS)
        elif result.outcome in (BookingOutcome.INELIGIBLE, BookingOutcome.CONFLICT,
                                BookingOutcome.INVALID):
            self.turn.record_outcome(operation_key, OperationOutcome.KNOWN_REJECTION)
        else:
            self.turn.record_outcome(operation_key, OperationOutcome.UNKNOWN)
            self.unresolved = True
        if not self.turn.check_output(token).allowed:
            # A correction after possible dispatch cannot be treated as consent
            # for another operation, even if readback later completes.
            self.unresolved = True
        return result.outcome

    def current(self, context: TrustedBookingContext, approval) -> bool:
        return (self.owns(self.context, self.handler)
                and not self.unresolved
                and self.approval is approval
                and self.dispatch_token is not None
                and self.turn.check_output(self.dispatch_token).allowed
                and self.turn.unsettled_operation is not None
                and self.turn.unsettled_operation.operation_id == approval.proposal.proposal_id
                and context.session_id == self.scope.session_id
                and context.caller_source_id == self.scope.caller_source_id
                and context.call_sid == self.context.call_sid)

    async def valid(self, context: TrustedBookingContext, approval) -> bool:
        return self.current(context, approval)

    def close(self) -> None:
        self.closed = True
        self.turn.close()
        self.proposals.close()
        self.event.set()
