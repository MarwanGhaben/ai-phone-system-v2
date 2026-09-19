"""T015-A deterministic proposal, playback, caller input and claim transitions."""
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest
from zoneinfo import ZoneInfo

from services.conversation.events import UtteranceIdentity
from services.scheduling.models import (
    AvailabilityQuery, CalendarScope, TimeInterval,
)
from services.scheduling.policy import AppointmentCandidate
from services.scheduling.proposals import (
    CallerProvenance, CustomerSnapshot, InputSource, ProposalState,
    ProposalStatus, SessionScope,
)


UTC = timezone.utc
TORONTO = ZoneInfo("America/Toronto")
NOW = datetime(2030, 1, 7, 14, 0, tzinfo=UTC)
START = datetime(2030, 1, 8, 10, 0, tzinfo=TORONTO).astimezone(UTC)


def candidate(*, staff="staff-A", service="service-A", start=START,
              policy="approved-v1", observed=NOW - timedelta(seconds=10)):
    scope = CalendarScope("tenant-A", "business-A", service)
    window = TimeInterval(min(START, start) - timedelta(days=1),
                          max(START, start) + timedelta(days=3))
    query = AvailabilityQuery(scope, window, (staff,), "query-A")
    return AppointmentCandidate(scope, staff,
                                TimeInterval(start, start + timedelta(minutes=30)),
                                policy, query, observed)


def offer(state, *, now=NOW, chosen=None, customer=None, **changes):
    values = {
        "candidate": chosen or candidate(),
        "customer": customer or CustomerSnapshot("Marwan", "+14165550100", "m@example.invalid"),
        "consultant_display": "Hussam Saadaldin",
        "service_display": "30-min meeting",
        "location": "1901 Banff Ave, Ottawa, ON K1V 7W9",
        "display_zone": "America/Toronto",
        "pre_buffer": timedelta(0), "post_buffer": timedelta(0),
        "now": now, "maximum_evidence_age": timedelta(seconds=60),
    }
    values.update(changes)
    return state.offer(**values)


def session(name="A", **limits):
    return ProposalState(SessionScope(f"session-{name}", f"stream-{name}"), **limits)


def receipt(state, *, seq=1, at=NOW + timedelta(seconds=3), source=InputSource.CALLER_FINAL,
            provenance=None, key=None):
    return state.admit_input(
        key or state.session_key,
        UtteranceIdentity("stt", "epoch-A", seq), source,
        provenance or CallerProvenance("session-A", "stream-A"),
        observed_at=at, admitted_at=at,
    )


def presented(state, offered, *, language="en", at=NOW):
    started = state.begin_presentation(offered.token, language,
                                       "playback-A", "mark-A", at + timedelta(seconds=1))
    assert started.status is ProposalStatus.PRESENTATION_STARTED
    completed = state.complete_presentation(
        started.presentation_token, "playback-A", "mark-A", True,
        at + timedelta(seconds=2))
    assert completed.status is ProposalStatus.PRESENTED
    return started


def test_full_english_journey_claims_exact_tuple_once():
    state = session()
    offered = offer(state)
    assert offered.status is ProposalStatus.OFFERED
    assert offered.proposal.revision == 1
    assert offered.proposal.expires_at == NOW + timedelta(minutes=2)
    started = presented(state, offered)
    assert "Marwan" in started.text
    assert "Hussam Saadaldin" in started.text
    assert "2030-01-08" in started.text
    assert "America/Toronto" in started.text
    assert "1901 Banff Ave" in started.text
    assert "Do you confirm" in started.text
    admitted = receipt(state)
    assert admitted.status is ProposalStatus.INPUT_ADMITTED
    approved = state.confirm(offered.token, admitted.receipt, True,
                             NOW + timedelta(seconds=4))
    assert approved.status is ProposalStatus.APPROVED
    claimed = state.consume(offered.token, NOW + timedelta(seconds=5))
    assert claimed.status is ProposalStatus.CLAIMED
    assert claimed.approved.proposal is offered.proposal
    assert claimed.approved.proposal.candidate.interval.start == START
    assert state.consume(offered.token, NOW + timedelta(seconds=6)).status is ProposalStatus.ALREADY_CLAIMED


def test_claim_cannot_precede_approval_and_does_not_consume_it():
    state = session()
    offered = offer(state)
    presented(state, offered)
    admitted = receipt(state)
    approval_time = NOW + timedelta(seconds=4)
    assert state.confirm(offered.token, admitted.receipt, True,
                         approval_time).status is ProposalStatus.APPROVED
    assert state.consume(offered.token, NOW - timedelta(days=1)).status is ProposalStatus.INVALID_INPUT
    assert state.consume(offered.token, approval_time).status is ProposalStatus.CLAIMED


def test_confirmation_cannot_precede_receipt_admission():
    state = session()
    offered = offer(state)
    presented(state, offered)
    admitted = state.admit_input(
        state.session_key, UtteranceIdentity("stt", "epoch-A", 1),
        InputSource.CALLER_FINAL, CallerProvenance("session-A", "stream-A"),
        observed_at=NOW + timedelta(seconds=3),
        admitted_at=NOW + timedelta(seconds=10))
    assert admitted.status is ProposalStatus.INPUT_ADMITTED
    assert state.confirm(offered.token, admitted.receipt, True,
                         NOW + timedelta(seconds=4)).status is ProposalStatus.INVALID_INPUT
    assert state.confirm(offered.token, admitted.receipt, True,
                         NOW + timedelta(seconds=10)).status is ProposalStatus.APPROVED


def test_malformed_completion_clock_revokes_before_later_success():
    state = session()
    offered = offer(state)
    started = state.begin_presentation(offered.token, "en", "p", "m", NOW)
    assert state.complete_presentation(started.presentation_token, "p", "m", True,
        NOW.replace(tzinfo=None)).status is ProposalStatus.INVALID_INPUT
    assert state.current is None
    assert state.complete_presentation(started.presentation_token, "p", "m", True,
        NOW + timedelta(seconds=2)).status is ProposalStatus.STALE


def test_unhashable_identity_fields_are_rejected_without_poisoning_state():
    state = session()
    invalid = UtteranceIdentity(["private-canary"], "epoch-A", 1)
    result = state.admit_input(state.session_key, invalid, InputSource.CALLER_FINAL,
        CallerProvenance("session-A", "stream-A"), observed_at=NOW, admitted_at=NOW)
    assert result.status is ProposalStatus.INVALID_INPUT
    assert receipt(state).status is ProposalStatus.INPUT_ADMITTED


def test_arabic_exact_readback_and_unsupported_language():
    state = session()
    offered = offer(state, customer=CustomerSnapshot("مروان", "+14165550100", ""),
                    consultant_display="حسام", service_display="اجتماع 30 دقيقة")
    unsupported = state.begin_presentation(offered.token, "fr", "p1", "m1", NOW)
    assert unsupported.status is ProposalStatus.UNSUPPORTED_LANGUAGE
    assert unsupported.text is None
    started = state.begin_presentation(offered.token, "ar", "p2", "m2", NOW)
    assert started.status is ProposalStatus.PRESENTATION_STARTED
    for text in ("مروان", "حسام", "اجتماع 30 دقيقة", "2030-01-08",
                 "America/Toronto", "1901 Banff Ave"):
        assert text in started.text


@pytest.mark.parametrize("change", [
    {"customer": CustomerSnapshot("Other", "+14165550100", "m@example.invalid")},
    {"customer": CustomerSnapshot("Marwan", "+14165550101", "m@example.invalid")},
    {"customer": CustomerSnapshot("Marwan", "+14165550100", "other@example.invalid")},
    {"chosen": candidate(staff="staff-B")},
    {"chosen": candidate(service="service-B")},
    {"chosen": candidate(start=START + timedelta(minutes=30))},
    {"chosen": candidate(policy="approved-v2")},
    {"chosen": candidate(observed=NOW - timedelta(seconds=9))},
    {"consultant_display": "Other consultant"},
    {"service_display": "Other service"},
    {"location": "Other office"},
    {"display_zone": "UTC"},
    {"pre_buffer": timedelta(minutes=1)},
    {"post_buffer": timedelta(minutes=1)},
])
def test_every_tuple_change_invalidates_old_token_and_fingerprint(change):
    state = session()
    first = offer(state)
    second = offer(state, now=NOW + timedelta(seconds=1), **change)
    assert second.status is ProposalStatus.OFFERED
    assert second.proposal.revision == first.proposal.revision + 1
    assert second.proposal.fingerprint != first.proposal.fingerprint
    assert state.begin_presentation(first.token, "en", "old", "old", NOW + timedelta(seconds=2)).status is ProposalStatus.STALE


def test_lost_interrupted_late_or_repeated_playback_cannot_authorize():
    state = session()
    first = offer(state)
    started = state.begin_presentation(first.token, "en", "p1", "m1", NOW)
    early = receipt(state, at=NOW + timedelta(seconds=1))
    assert state.confirm(first.token, early.receipt, True,
                         NOW + timedelta(seconds=1)).status is ProposalStatus.NOT_PRESENTED
    assert state.complete_presentation(started.presentation_token, "p1", "m1", False,
                                       NOW + timedelta(seconds=2)).status is ProposalStatus.INTERRUPTED
    assert state.complete_presentation(started.presentation_token, "p1", "m1", True,
                                       NOW + timedelta(seconds=3)).status is ProposalStatus.STALE
    second = offer(state, now=NOW + timedelta(seconds=4))
    assert second.status is ProposalStatus.OFFERED
    assert state.begin_presentation(second.token, "en", "p1", "m1",
                                    NOW + timedelta(seconds=5)).status is ProposalStatus.REPLAYED
    started2 = presented(state, second, at=NOW + timedelta(seconds=5))
    assert state.complete_presentation(started2.presentation_token, "playback-A", "mark-A",
                                       True, NOW + timedelta(seconds=8)).status is ProposalStatus.REPLAYED


def test_literal_confirmation_receipt_authenticity_and_replay():
    state = session()
    offered = offer(state)
    presented(state, offered)
    first = receipt(state)
    for value in ("true", 1, [], None):
        assert state.confirm(offered.token, first.receipt, value,
                             NOW + timedelta(seconds=4)).status is ProposalStatus.INVALID_INPUT
    assert state.confirm(offered.token, first.receipt, True,
                         NOW + timedelta(seconds=4)).status is ProposalStatus.APPROVED
    assert state.confirm(offered.token, first.receipt, True,
                         NOW + timedelta(seconds=5)).status is ProposalStatus.REPLAYED
    assert receipt(state, seq=1).status is ProposalStatus.REPLAYED
    assert receipt(state, seq=2).status is ProposalStatus.INPUT_ADMITTED


def test_false_cancels_only_current_offer_and_old_token_cannot_erase_new_one():
    state = session()
    first = offer(state)
    second = offer(state, now=NOW + timedelta(seconds=1))
    admitted = receipt(state)
    assert state.confirm(first.token, admitted.receipt, False,
                         NOW + timedelta(seconds=4)).status is ProposalStatus.STALE
    assert state.current is second.proposal
    assert state.confirm(second.token, admitted.receipt, False,
                         NOW + timedelta(seconds=4)).status is ProposalStatus.CANCELLED
    assert state.current is None


def test_exact_expiry_and_stale_future_naive_clocks():
    state = session()
    assert offer(state, chosen=candidate(observed=NOW - timedelta(seconds=61))).status is ProposalStatus.STALE_EVIDENCE
    assert offer(state, chosen=candidate(observed=NOW + timedelta(seconds=1))).status is ProposalStatus.FUTURE_EVIDENCE
    assert offer(state, now=NOW.replace(tzinfo=None)).status is ProposalStatus.INVALID_INPUT
    offered = offer(state)
    assert state.begin_presentation(offered.token, "en", "p", "m",
                                    offered.proposal.expires_at).status is ProposalStatus.EXPIRED
    assert state.consume(offered.token, offered.proposal.expires_at).status is ProposalStatus.STALE
    renewed = offer(state)
    assert state.consume(renewed.token, renewed.proposal.expires_at).status is ProposalStatus.EXPIRED


def test_exact_notice_and_evidence_age_boundaries():
    near = NOW + timedelta(minutes=30)
    state = session()
    assert offer(state, chosen=candidate(start=near)).status is ProposalStatus.OFFERED
    assert offer(state, chosen=candidate(start=near - timedelta(microseconds=1))).status is ProposalStatus.INVALID_INPUT
    assert offer(state, chosen=candidate(observed=NOW - timedelta(seconds=60))).status is ProposalStatus.OFFERED
    assert offer(state, maximum_evidence_age=timedelta(minutes=6)).status is ProposalStatus.INVALID_INPUT


def test_mismatched_completion_and_untrusted_sources_do_not_present_or_approve():
    state = session()
    offered = offer(state)
    started = state.begin_presentation(offered.token, "en", "generation", "mark", NOW)
    assert state.complete_presentation(started.presentation_token, "generation", "wrong", True,
                                       NOW + timedelta(seconds=1)).status is ProposalStatus.STALE
    assert state.complete_presentation(started.presentation_token, "wrong", "mark", True,
                                       NOW + timedelta(seconds=1)).status is ProposalStatus.STALE
    assert receipt(state, source=InputSource.AGENT_SPEECH).status is ProposalStatus.INVALID_INPUT
    assert receipt(state, source=InputSource.PARTIAL).status is ProposalStatus.INVALID_INPUT
    assert receipt(state, source=InputSource.METADATA).status is ProposalStatus.INVALID_INPUT
    assert receipt(state, provenance=CallerProvenance("session-A", "foreign")).status is ProposalStatus.INVALID_INPUT
    assert receipt(state, key=session().session_key).status is ProposalStatus.FOREIGN_SESSION
    assert state.complete_presentation(started.presentation_token, "generation", "mark", True,
                                       NOW + timedelta(seconds=2)).status is ProposalStatus.PRESENTED
    admitted = receipt(state)
    assert state.confirm(offered.token, admitted.receipt, True, NOW + timedelta(seconds=4)).status is ProposalStatus.APPROVED


def test_malformed_completion_fails_closed_and_cannot_be_retried_as_success():
    state = session()
    offered = offer(state)
    started = state.begin_presentation(offered.token, "en", "p", "m", NOW)
    assert state.complete_presentation(started.presentation_token, "p", "m", 1,
                                       NOW + timedelta(seconds=1)).status is ProposalStatus.INVALID_INPUT
    assert state.current is None
    assert state.complete_presentation(started.presentation_token, "p", "m", True,
                                       NOW + timedelta(seconds=2)).status is ProposalStatus.STALE


def test_early_identity_cannot_be_reused_after_readback():
    state = session()
    offered = offer(state)
    early = receipt(state, at=NOW)
    presented(state, offered)
    assert state.confirm(offered.token, early.receipt, True,
                         NOW + timedelta(seconds=3)).status is ProposalStatus.NOT_PRESENTED
    assert state.confirm(offered.token, early.receipt, True,
                         NOW + timedelta(seconds=4)).status is ProposalStatus.REPLAYED
    assert receipt(state, seq=1, at=NOW + timedelta(seconds=4)).status is ProposalStatus.REPLAYED


def test_expiry_revokes_approval_and_late_old_completion_preserves_new_offer():
    state = session()
    first = offer(state)
    started = state.begin_presentation(first.token, "en", "first", "first", NOW)
    second = offer(state, now=NOW + timedelta(seconds=1))
    assert state.complete_presentation(started.presentation_token, "first", "first", True,
                                       NOW + timedelta(seconds=2)).status is ProposalStatus.STALE
    assert state.current is second.proposal
    presented(state, second, at=NOW + timedelta(seconds=1))
    admitted = receipt(state, at=NOW + timedelta(seconds=4))
    assert state.confirm(second.token, admitted.receipt, True,
                         NOW + timedelta(seconds=5)).status is ProposalStatus.APPROVED
    assert state.consume(second.token, second.proposal.expires_at).status is ProposalStatus.EXPIRED
    assert state.current is None


def test_presentation_and_offer_capacity_fail_closed():
    presentations = session(max_presentations=1)
    first = offer(presentations)
    presented(presentations, first)
    second = offer(presentations, now=NOW + timedelta(seconds=5))
    assert presentations.begin_presentation(second.token, "en", "other", "other",
                                            NOW + timedelta(seconds=6)).status is ProposalStatus.OVERLOADED
    assert presentations.current is None
    offers = session(max_offers=1)
    assert offer(offers).status is ProposalStatus.OFFERED
    assert offer(offers).status is ProposalStatus.OVERLOADED
    assert offers.current is None
    for invalid_limit in (True, 0, 129):
        with pytest.raises(ValueError, match="invalid proposal state configuration"):
            session(max_receipts=invalid_limit)


def test_scope_is_in_fingerprint_and_foreign_receipt_cannot_be_reconstructed():
    one, two = session("A"), session("B")
    first, second = offer(one), offer(two)
    assert first.proposal.fingerprint != second.proposal.fingerprint
    presented(one, first)
    admitted = receipt(one)
    cloned = replace(admitted.receipt)
    assert one.confirm(first.token, cloned, True,
                       NOW + timedelta(seconds=4)).status is ProposalStatus.FOREIGN_SESSION
    assert one.confirm(first.token, admitted.receipt, True,
                       NOW + timedelta(seconds=4)).status is ProposalStatus.APPROVED


def test_two_sessions_close_and_foreign_receipts():
    one, two = session("A"), session("B")
    a, b = offer(one), offer(two)
    presented(one, a)
    presented(two, b)
    foreign = receipt(one)
    assert two.confirm(b.token, foreign.receipt, True,
                       NOW + timedelta(seconds=4)).status is ProposalStatus.FOREIGN_SESSION
    assert one.close().status is ProposalStatus.CLOSED
    assert one.confirm(a.token, foreign.receipt, True,
                       NOW + timedelta(seconds=4)).status is ProposalStatus.CLOSED
    replacement = session("A")
    replaced = offer(replacement)
    assert replacement.confirm(replaced.token, foreign.receipt, True,
                               NOW + timedelta(seconds=4)).status is ProposalStatus.FOREIGN_SESSION


def test_capacity_overflow_latches_without_eviction():
    state = session(max_receipts=1)
    assert receipt(state, seq=1).status is ProposalStatus.INPUT_ADMITTED
    assert receipt(state, seq=2).status is ProposalStatus.OVERLOADED
    assert receipt(state, seq=1).status is ProposalStatus.REPLAYED
    assert offer(state).status is ProposalStatus.OVERLOADED


def test_canonical_serialization_and_immutability_and_safe_repr():
    state = session()
    first = offer(state, customer=CustomerSnapshot("a|b", "c", "d"))
    second = offer(state, now=NOW + timedelta(seconds=1),
                   customer=CustomerSnapshot("a", "b|c", "d"))
    assert first.proposal.fingerprint != second.proposal.fingerprint
    with pytest.raises(FrozenInstanceError):
        second.proposal.customer.name = "changed"
    for value in (state, second.proposal, second.token, second):
        rendered = repr(value)
        assert "a|b" not in rendered and "session-A" not in rendered
        assert "1416555" not in rendered
    with pytest.raises(TypeError):
        bool(second)
