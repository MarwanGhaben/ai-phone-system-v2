"""Verified phone requests through the installed SDK and an offline HTTP transport."""
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI

from services.llm.openai_service import OpenAILLM
from tests.conversation.test_safe_booking_flow import setup, final, take_pending


@pytest.mark.asyncio
@pytest.mark.parametrize("language,text,answer", [
    ("en", "Check Hussam's availability", "Which date would you prefer?"),
    ("ar", "أريد معرفة مواعيد حسام", "أي يوم تفضل؟"),
])
@pytest.mark.parametrize("with_tool", [False, True])
async def test_verified_request_reaches_http_with_single_tool_policy(language, text, answer, with_tool):
    requests = []

    def reply(request):
        requests.append(json.loads(request.content))
        message = {"role": "assistant", "content": answer}
        if with_tool:
            message.update(content=None, tool_calls=[{
                "id": "call_availability", "type": "function",
                "function": {"name": "check_appointment", "arguments": json.dumps({
                    "accountant_name": "Hussam Saadaldin",
                    "date_time": "2026-09-21 11:00"})}}])
        return httpx.Response(200, json={
            "id": "chatcmpl_test", "object": "chat.completion", "created": 1,
            "model": "gpt-4o", "choices": [{"index": 0, "message": message,
                "finish_reason": "tool_calls" if with_tool else "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    app, context, session = setup()
    context.language = language
    app._check_booking = AsyncMock(return_value="NO_ELIGIBLE_SLOT: no synthetic slot")
    app.llm = OpenAILLM(api_key="synthetic-key")
    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as http:
        async with AsyncOpenAI(api_key="synthetic-key", http_client=http, max_retries=0) as sdk:
            app.llm._client = sdk
            assert session.admit(final(1, text))
            batch = take_pending(session)
            context.add_user_message(text)
            try:
                result = await app._get_verified_response("call", session, text, batch.token)
                assert len(requests) == 1, f"SDK never reached HTTP; caller received: {result}"
                assert requests[0]["parallel_tool_calls"] is False
                assert requests[0]["messages"][-1]["content"] == text
                if with_tool:
                    app._check_booking.assert_awaited_once()
                    assert context.conversation_history[-1]["tool_call_id"] == "call_availability"
                else:
                    assert result == answer
                    app._check_booking.assert_not_awaited()
            finally:
                session.close()
