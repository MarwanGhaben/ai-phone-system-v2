"""Offline installed-image probe. Synthetic finals; no provider or database calls."""
import asyncio
from datetime import datetime, timezone
import logging
import os
import socket
from unittest.mock import AsyncMock, patch

os.environ.update({
    "SECRET_KEY": "synthetic", "DATABASE_URL": "postgresql://synthetic:synthetic@localhost/test",
    "TWILIO_ACCOUNT_SID": "ACsynthetic", "TWILIO_AUTH_TOKEN": "synthetic",
    "TWILIO_PHONE_NUMBER": "+14165550100", "DEEPGRAM_API_KEY": "synthetic",
    "ELEVENLABS_API_KEY": "synthetic", "OPENAI_API_KEY": "synthetic",
})
from loguru import logger
logger.remove()
logging.disable(logging.CRITICAL)
from services.conversation import orchestrator as module
from services.conversation import restored_turn_admission as admission
from services.conversation.events import UtteranceIdentity
from services.conversation.language_policy import explicit_language_request
from services.stt.stt_base import STTResult


async def journey(text, initial, expected, *, uncertain=False):
    agent = module.ConversationOrchestrator.__new__(module.ConversationOrchestrator)
    context = module.ConversationContext("synthetic", "+10000000000", language=initial)
    context.state = module.ConversationState.SPEAKING
    context.pending_booking = {"confirmed_available": True}
    final = STTResult(text, "en", None, True,
        utterance_id=UtteranceIdentity("elevenlabs_scribe", "synthetic", 1),
        received_at=datetime.fromtimestamp(100.5 if uncertain else 99, timezone.utc))
    class STT:
        language = initial
        async def get_transcript(self):
            yield final
            yield final  # Provider duplicate must not execute twice.
            context.state = module.ConversationState.ENDED
    stt = STT()
    agent._conversations = {"synthetic": context}
    agent._call_stt_instances = {"synthetic": stt}
    agent._call_state_locks = {"synthetic": asyncio.Lock()}
    agent._garbled_drop_count = {}
    agent._echo_guard_until = {}
    agent._barge_in_speech_start = {"synthetic": 100}
    agent._get_ai_response = AsyncMock(return_value="synthetic reply")
    agent._reconnect_stt_with_language = AsyncMock()
    async def speak(*args):
        context.state = module.ConversationState.LISTENING
        return True
    agent._speak_to_caller = AsyncMock(side_effect=speak)
    async def pause(_):
        context.state = module.ConversationState.LISTENING
    with patch.object(admission, "wall_time", return_value=101), patch.object(admission, "pause", pause):
        await agent._consume_stt_transcripts("synthetic")
    if uncertain:
        agent._get_ai_response.assert_not_awaited()
        assert context.pending_booking is None
        assert agent._speak_to_caller.await_count == 1
    else:
        agent._get_ai_response.assert_awaited_once_with("synthetic", text)
        assert context.language == expected


async def main():
    for text, initial, expected in (
        ("Please speak English", "ar", "en"),
        ("Arabic please", "en", "ar"),
        ("English please I want to book with Rami", "ar", "en"),
        ("احكي معي بالعربي بدي موعد مع رامي", "en", "ar"),
        ("No, Tuesday at four thirty, not Monday", "en", "en"),
    ):
        await journey(text, initial, expected)
    await journey("yes", "en", "en", uncertain=True)
    await journey("نعم", "ar", "ar", uncertain=True)
    for text in ('He said "English please"', "Don't speak English", "English please, Arabic please"):
        assert explicit_language_request(text) is None


def no_network(*args, **kwargs):
    raise AssertionError("offline probe attempted network access")


async def offline():
    with patch.object(socket.socket, "connect", no_network), patch.object(socket.socket, "connect_ex", no_network):
        await asyncio.wait_for(main(), 10)


asyncio.run(offline())
print("RESTORED_TURN_OFFLINE_OK")
