"""Offline deterministic presentation/approval probe; no credentials or network."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from services.conversation.events import UtteranceIdentity
from services.scheduling.models import AvailabilityQuery, CalendarScope, TimeInterval
from services.scheduling.policy import AppointmentCandidate
from services.scheduling.proposals import (
    CallerProvenance, CustomerSnapshot, InputSource, ProposalState, ProposalStatus, SessionScope,
)
from services.scheduling import spoken_arabic


def main():
    now = datetime(2030, 1, 7, 14, tzinfo=timezone.utc)
    start = datetime(2030, 1, 8, 10, tzinfo=ZoneInfo("America/Toronto"))
    scope = CalendarScope("synthetic-tenant", "synthetic-business", "synthetic-service")
    candidate = AppointmentCandidate(scope, "synthetic-staff",
        TimeInterval(start, start + timedelta(minutes=30)), "synthetic-policy",
        AvailabilityQuery(scope, TimeInterval(now, now + timedelta(days=3)),
                          ("synthetic-staff",), "synthetic-query"), now)
    for language in ("ar", "en"):
        state = ProposalState(SessionScope("synthetic-session", "synthetic-stream"))
        offered = state.offer(candidate=candidate,
            customer=CustomerSnapshot("مروان", "+14165550100", "synthetic@example.invalid"),
            consultant_display="Hussam Saadaldin", service_display="appointment",
            location="1901 Banff Ave, Ottawa, ON K1V 7W9", display_zone="America/Toronto",
            pre_buffer=timedelta(0), post_buffer=timedelta(0), now=now,
            maximum_evidence_age=timedelta(seconds=60))
        assert offered.status is ProposalStatus.OFFERED
        started = state.begin_presentation(offered.token, language, "playback", "mark", now)
        assert started.status is ProposalStatus.PRESENTATION_STARTED
        if language == "ar":
            for text in ("حسام سعد الدين", "الثلاثاء", "الثامن من يناير", "العاشرة صباحاً",
                         "العاشرة والنصف صباحاً", "نصف ساعة", "بتوقيت تورونتو"):
                assert text in started.text
            for text in ("appointment", "2030-01-08", "10:00", "America/Toronto"):
                assert text not in started.text
        else:
            assert "2030-01-08" in started.text and "10:00" in started.text
        for text in ("1901 Banff Ave", "+14165550100", "synthetic@example.invalid"):
            assert text in started.text
        assert state.current is offered.proposal
        assert state.consume(offered.token, now).status is ProposalStatus.NOT_PRESENTED
        assert state.complete_presentation(started.presentation_token, "playback", "mark",
            True, now + timedelta(seconds=1)).status is ProposalStatus.PRESENTED
        receipt = state.admit_input(state.session_key, UtteranceIdentity("stt", "epoch", 1),
            InputSource.CALLER_FINAL, CallerProvenance("synthetic-session", "synthetic-stream"),
            observed_at=now + timedelta(seconds=2), admitted_at=now + timedelta(seconds=2))
        assert state.confirm(offered.token, receipt.receipt, True,
            now + timedelta(seconds=3)).status is ProposalStatus.APPROVED
        claimed = state.consume(offered.token, now + timedelta(seconds=4))
        assert claimed.status is ProposalStatus.CLAIMED
        assert claimed.approved.proposal is offered.proposal
        assert state.consume(offered.token, now + timedelta(seconds=5)).status is ProposalStatus.ALREADY_CLAIMED
    assert "العاشرة صباحاً" in spoken_arabic.slot(start)
    print("ARABIC_BOOKING_SPEECH_OFFLINE_PRESENTATION_CLAIM_OK")


if __name__ == "__main__":
    main()
