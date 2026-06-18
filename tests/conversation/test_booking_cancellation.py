import pytest

from services.conversation.orchestrator import ConversationContext, ConversationOrchestrator


@pytest.mark.asyncio
async def test_cancellation_rejects_llm_id_without_lookup_context() -> None:
    call_sid = "call-1"
    context = ConversationContext(call_sid=call_sid, phone_number="+14165550100")
    orchestrator = ConversationOrchestrator.__new__(ConversationOrchestrator)
    orchestrator._conversations = {call_sid: context}

    async def speak(*args, **kwargs):
        return None

    orchestrator._speak_to_caller = speak

    response = await orchestrator._cancel_booking(
        call_sid,
        {"confirm_cancel": True, "appointment_id": "attacker-selected-id"},
    )

    assert response.startswith("CANCEL_ERROR: No appointment found")
