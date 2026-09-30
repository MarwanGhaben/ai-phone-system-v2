"""Luna compatibility through the installed SDK and a synthetic HTTP endpoint."""
import json

import httpx
import pytest
from openai import AsyncOpenAI

from services.llm.llm_base import LLMRequest, LLMRole, Message
from services.llm.openai_service import OpenAILLM


@pytest.mark.asyncio
@pytest.mark.parametrize('model', ['gpt-6-luna', 'gpt-4o'])
@pytest.mark.parametrize('mode', ['chat', 'stream', 'tools', 'single-tool'])
@pytest.mark.parametrize('text', ['Check Rami on Monday', 'شوف لي موعد مع رامي يوم الاثنين'])
async def test_wire_parameters_preserve_voice_budget_and_tool_contract(model, mode, text):
    bodies = []

    def reply(request):
        body = json.loads(request.content)
        bodies.append(body)
        message = {'role': 'assistant', 'content': text}
        finish = 'stop'
        if mode in ('tools', 'single-tool'):
            message.update(content=None, tool_calls=[{
                'id': 'call_slot', 'type': 'function',
                'function': {'name': 'check_appointment', 'arguments': '{"accountant_name":"Rami"}'},
            }])
            finish = 'tool_calls'
        completion = {'id': 'synthetic', 'object': 'chat.completion', 'created': 1,
                      'model': model, 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
                      'usage': {'prompt_tokens': 5, 'completion_tokens': 5, 'total_tokens': 10}}
        if mode == 'stream':
            events = []
            for delta, reason in [({'content': text}, None), ({}, 'stop')]:
                events.append('data: ' + json.dumps({'id': 'synthetic', 'object': 'chat.completion.chunk',
                    'created': 1, 'model': model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': reason}]}))
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                                  content=('\n\n'.join(events) + '\n\ndata: [DONE]\n\n').encode())
        return httpx.Response(200, json=completion)

    llm = OpenAILLM(api_key='synthetic', model=model)
    prompt = LLMRequest(messages=[Message(LLMRole.USER, text)], max_tokens=80, temperature=0.7,
                        tools=[{'name': 'check_appointment', 'parameters': {'type': 'object'}}],
                        metadata={'single_tool_call': mode == 'single-tool'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as http:
        async with AsyncOpenAI(api_key='synthetic', http_client=http, max_retries=0) as sdk:
            llm._client = sdk
            if mode == 'stream':
                chunks = [chunk async for chunk in llm.chat_stream(prompt)]
                assert ''.join(chunk.delta for chunk in chunks) == text
                assert chunks[-1].is_final
            elif mode == 'chat':
                assert (await llm.chat(prompt)).content == text
            else:
                response = await llm.chat_with_tools(prompt)
                assert response.tool_calls == [{'id': 'call_slot', 'name': 'check_appointment',
                                               'arguments': '{"accountant_name":"Rami"}'}]
    assert len(bodies) == 1
    body = bodies[0]
    assert body['messages'] == [{'role': 'user', 'content': text}]
    assert body['model'] == model and body['temperature'] == 0.7
    if model == 'gpt-6-luna':
        assert body.get('reasoning_effort') == 'none'
        assert body.get('max_completion_tokens') == 80
        assert 'max_tokens' not in body
    else:
        assert body['max_tokens'] == 80
        assert 'reasoning_effort' not in body and 'max_completion_tokens' not in body
    if mode == 'single-tool':
        assert body['parallel_tool_calls'] is False
