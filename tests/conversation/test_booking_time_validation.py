from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from services.conversation.orchestrator import (
    ConversationContext,
    ConversationOrchestrator,
    _parse_booking_datetime,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("date_time", [None, "", "not a real appointment time"])
async def test_unclear_booking_time_requests_repetition_and_clears_stale_slot(
    date_time: str | None,
) -> None:
    context = ConversationContext(call_sid="call-1", phone_number="+14165550100")
    context.pending_booking = {"appointment_time": "stale"}
    orchestrator = ConversationOrchestrator.__new__(ConversationOrchestrator)
    orchestrator._conversations = {context.call_sid: context}

    response = await orchestrator._check_booking(
        context.call_sid,
        {"date_time": date_time, "accountant_name": "Hussam"},
    )

    assert response.startswith("INVALID_DATE_TIME:")
    assert context.pending_booking is None


def test_valid_booking_time_preserves_the_requested_slot() -> None:
    appointment_time = _parse_booking_datetime("2099-06-18 14:30")

    assert appointment_time == datetime(
        2099, 6, 18, 14, 30, tzinfo=ZoneInfo("America/Toronto")
    )


def test_offset_booking_time_converts_to_toronto() -> None:
    appointment_time = _parse_booking_datetime("2099-06-18T14:30:00+00:00")

    assert appointment_time == datetime(
        2099, 6, 18, 10, 30, tzinfo=ZoneInfo("America/Toronto")
    )


def test_space_separated_booking_time_preserves_offset() -> None:
    appointment_time = _parse_booking_datetime("2099-06-18 14:30 +00:00")

    assert appointment_time == datetime(
        2099, 6, 18, 10, 30, tzinfo=ZoneInfo("America/Toronto")
    )
