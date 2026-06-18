from types import SimpleNamespace

import pytest

from services.conversation import orchestrator as orchestrator_module
from services.conversation.orchestrator import ConversationContext, ConversationOrchestrator


class FakeResponse:
    status_code = 200
    text = ""


class FakeAsyncClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def post(self, url, **kwargs):
        return FakeResponse()


@pytest.mark.asyncio
async def test_transfer_has_no_fixed_five_second_delay(monkeypatch) -> None:
    call_sid = "call-1"
    context = ConversationContext(call_sid=call_sid, phone_number="+14165550100")
    orchestrator = ConversationOrchestrator.__new__(ConversationOrchestrator)
    orchestrator._conversations = {call_sid: context}
    observed_delays: list[float] = []

    async def speak(*args, **kwargs):
        return None

    async def record_sleep(delay: float):
        observed_delays.append(delay)

    settings = SimpleNamespace(
        transfer_phone_number="+14165550199",
        twilio_account_sid="ACtest",
        twilio_auth_token="token",
        twilio_phone_number="+14165550100",
    )
    orchestrator._speak_to_caller = speak
    monkeypatch.setattr(orchestrator_module, "get_settings", lambda: settings)
    monkeypatch.setattr(orchestrator_module.asyncio, "sleep", record_sleep)
    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    await orchestrator._handle_transfer(call_sid)

    assert 5 not in observed_delays
