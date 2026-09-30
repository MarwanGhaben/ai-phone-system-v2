"""Synthetic Luna check. Offline by default; --live uses the server's existing key."""
import argparse
import asyncio
import json
import logging
import os
import re
import time

MODEL = 'gpt-6-luna'
TOOLS = [{
    'name': 'check_appointment',
    'description': 'Check a synthetic appointment; this test never books anything.',
    'parameters': {'type': 'object', 'properties': {
        'accountant_name': {'type': 'string', 'enum': ['Rami']},
        'date_time': {'type': 'string', 'enum': ['2030-01-08 10:00']},
    }, 'required': ['accountant_name', 'date_time'], 'additionalProperties': False},
}]


def language_matches(text, language):
    if not isinstance(text, str) or not text.strip():
        return False
    has_arabic = bool(re.search(r'[\u0621-\u064a]', text))
    return has_arabic if language == 'ar' else not has_arabic and bool(re.search('[A-Za-z]', text))


async def verify_client(llm):
    from services.llm.llm_base import LLMRequest, LLMRole, Message
    completed = 0
    for language, request_text in (
        ('en', 'Please check Rami on 2030-01-08 at 10:00.'),
        ('ar', 'ممكن تشوف لي موعد مع رامي يوم 2030-01-08 الساعة 10:00؟'),
    ):
        instruction = ('This is a synthetic receptionist test. Reply only in '
                       + ('Arabic' if language == 'ar' else 'English')
                       + '. Keep replies to one short sentence. Do not invent availability or book anything.')
        messages = [Message(LLMRole.SYSTEM, instruction), Message(LLMRole.USER, 'Say hello briefly.')]
        response = await llm.chat(LLMRequest(messages=messages, max_tokens=80))
        if response.finish_reason != 'stop' or not language_matches(response.content, language):
            raise ValueError('synthetic chat check failed')
        completed += 1

        messages = [Message(LLMRole.SYSTEM, instruction + ' Call check_appointment for the requested check.'),
                    Message(LLMRole.USER, request_text)]
        response = await llm.chat_with_tools(LLMRequest(messages=messages, max_tokens=150,
            tools=TOOLS, metadata={'single_tool_call': True}))
        if response.finish_reason != 'tool_calls' or len(response.tool_calls) != 1:
            raise ValueError('synthetic tool check failed')
        call = response.tool_calls[0]
        if (call['name'] != 'check_appointment' or not call['id']
                or json.loads(call['arguments']) != {'accountant_name': 'Rami', 'date_time': '2030-01-08 10:00'}):
            raise ValueError('synthetic arguments check failed')
        completed += 1

        messages.extend([
            Message(LLMRole.ASSISTANT, '', metadata={'tool_calls': [{
                'id': call['id'], 'type': 'function',
                'function': {'name': call['name'], 'arguments': call['arguments']},
            }]}),
            Message(LLMRole.TOOL, 'No appointments are available for Rami during the period checked. '
                    'Offer to check another consultant; no appointment has been booked.',
                    metadata={'tool_call_id': call['id']}),
        ])
        chunks = [chunk async for chunk in llm.chat_stream(LLMRequest(messages=messages, max_tokens=100))]
        if (not chunks or not chunks[-1].is_final or chunks[-1].finish_reason != 'stop'
                or not language_matches(''.join(chunk.delta for chunk in chunks), language)):
            raise ValueError('synthetic streaming check failed')
        completed += 1
    return completed


def offline_reply(request):
    """Exercise SDK serialization and streaming without contacting any endpoint."""
    import httpx
    body = json.loads(request.content)
    assert str(request.url) == 'https://api.openai.com/v1/chat/completions'
    assert body['model'] == MODEL and body['reasoning_effort'] == 'none'
    assert 'max_tokens' not in body and body['max_completion_tokens'] > 0
    language = 'ar' if 'only in Arabic' in body['messages'][0]['content'] else 'en'
    content = 'تحب أشوف لك موعد عند محاسب ثاني؟' if language == 'ar' else 'Would you like me to check another consultant?'
    message = {'role': 'assistant', 'content': content}
    finish = 'stop'
    if body.get('tools'):
        assert body['parallel_tool_calls'] is False
        message.update(content=None, tool_calls=[{'id': 'synthetic-call', 'type': 'function',
            'function': {'name': 'check_appointment', 'arguments': json.dumps({
                'accountant_name': 'Rami', 'date_time': '2030-01-08 10:00'})}}])
        finish = 'tool_calls'
    base = {'id': 'synthetic', 'created': 1, 'model': MODEL}
    if body['stream']:
        assert body['messages'][-1]['tool_call_id'] == 'synthetic-call'
        events = [json.dumps({**base, 'object': 'chat.completion.chunk', 'choices': [
            {'index': 0, 'delta': delta, 'finish_reason': reason}]})
            for delta, reason in [({'content': content}, None), ({}, 'stop')]]
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
            content=(''.join('data: ' + event + '\n\n' for event in events) + 'data: [DONE]\n\n').encode())
    return httpx.Response(200, json={**base, 'object': 'chat.completion', 'choices': [
        {'index': 0, 'message': message, 'finish_reason': finish}],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})


async def run(live):
    import httpx
    from openai import AsyncOpenAI
    from services.llm.openai_service import OpenAILLM
    if live:
        from config.settings import settings
        if settings.openai_model != MODEL:
            raise ValueError('candidate model setting differs')
        key = settings.openai_api_key
        transport = None
    else:
        key = 'synthetic'
        transport = httpx.MockTransport(offline_reply)
    async with httpx.AsyncClient(transport=transport, timeout=15, follow_redirects=False) as http:
        async with AsyncOpenAI(api_key=key, http_client=http, timeout=15, max_retries=0) as sdk:
            llm = OpenAILLM(api_key=key, model=MODEL)
            llm._client = sdk
            return await verify_client(llm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    from loguru import logger
    logger.remove()
    logging.disable(logging.CRITICAL)
    if not args.live:
        os.environ.update({'SECRET_KEY': 'synthetic', 'DATABASE_URL': 'postgresql://synthetic:synthetic@localhost/test',
            'TWILIO_ACCOUNT_SID': 'ACsynthetic', 'TWILIO_AUTH_TOKEN': 'synthetic',
            'TWILIO_PHONE_NUMBER': '+14165550100', 'DEEPGRAM_API_KEY': 'synthetic',
            'ELEVENLABS_API_KEY': 'synthetic', 'OPENAI_API_KEY': 'synthetic'})
    started = time.monotonic()
    try:
        completed = asyncio.run(asyncio.wait_for(run(args.live), 100))
    except Exception as error:
        code = getattr(error, 'status_code', None)
        category = ('api_rejected' if type(code) is int else 'timeout' if isinstance(error, TimeoutError)
                    else 'compatibility_check_failed')
        print(json.dumps({'status': 'LUNA_COMPATIBILITY_FAILED', 'category': category,
                          'http_status': code if type(code) is int else None}))
        return 1
    print(json.dumps({'status': 'LUNA_COMPATIBILITY_OK', 'mode': 'live' if args.live else 'offline',
                      'completed_requests': completed, 'elapsed_seconds': round(time.monotonic() - started, 2)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
