"""Bounded final-transcript admission for the restored serial phone consumer.

Receipt time is local adapter evidence, not proof of speaker identity. A final
already committed before the current playback can be deferred safely. A final
overlapping playback without that evidence requires clarification, never a
booking mutation. Existing acoustic/VAD and playback ownership stay intact.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime
import math
import time

from loguru import logger

from services.conversation.events import UtteranceIdentity
from services.conversation.language_policy import explicit_language_request

pause = asyncio.sleep
wall_time = time.time
MAX_WAIT_SECONDS = 30.0


@dataclass
class _AdmissionState:
    clarification_latched: bool = False


def _current(agent, call_sid, context, stt):
    return (
        agent._conversations.get(call_sid) is context
        and agent._call_stt_instances.get(call_sid) is stt
        and context.state.value != "ended"
    )


def _receipt(result, now):
    identity = getattr(result, "utterance_id", None)
    value = getattr(result, "received_at", None)
    if not isinstance(identity, UtteranceIdentity) or identity.provider != "elevenlabs_scribe":
        return None
    if not isinstance(value, datetime) or value.utcoffset() is None:
        return None
    try:
        stamp = value.timestamp()
    except (ValueError, OverflowError, OSError):
        return None
    return stamp if math.isfinite(stamp) and 0 <= now - stamp <= 120 else None


async def admit_final(agent, call_sid, context, stt, result):
    """Wait in the existing serial consumer; no new queue, task or provider call.

    Return true only when the same session is ready for this final. Unknown
    overlap invalidates a pending offer and prompts at most once until a fresh
    listening turn arrives. Cancellation propagates to the owning consumer.
    """
    if not _current(agent, call_sid, context, stt):
        return False
    state = getattr(context, "_restored_admission", None)
    if state is None:
        state = _AdmissionState()
        context._restored_admission = state
    now = wall_time()
    started = agent._barge_in_speech_start.get(call_sid, 0)
    guard = agent._echo_guard_until.get(call_sid, 0)
    receipt = _receipt(result, now)
    blocked = context.state.value == "speaking" or guard > now
    # Queued finals can be consumed after playback, once SPEAKING is already
    # cleared. Their receipt still tells us whether they overlapped that turn.
    overlaps_finished_playback = (
        receipt is not None and 0 < started <= receipt <= guard
    )
    uncertain = overlaps_finished_playback or (
        blocked and not (receipt is not None and 0 < receipt < started)
    )
    if uncertain:
        context.pending_booking = None
    deadline = asyncio.get_running_loop().time() + MAX_WAIT_SECONDS
    while _current(agent, call_sid, context, stt):
        if (context.state.value == "listening"
                and agent._echo_guard_until.get(call_sid, 0) <= wall_time()):
            break
        if asyncio.get_running_loop().time() >= deadline:
            context.pending_booking = None
            logger.warning("Caller admission stopped category=playback_wait_timeout")
            return False
        await pause(0.02)
    if not _current(agent, call_sid, context, stt):
        return False
    if uncertain:
        if not state.clarification_latched:
            state.clarification_latched = True
            language = explicit_language_request(result.text) or context.language
            language = "ar" if language == "ar" else "en"
            message = (
                "عذراً، تداخل الكلام. ممكن تعيد طلبك من فضلك؟"
                if language == "ar" else
                "Sorry, we spoke at the same time. Could you repeat your request?"
            )
            context.state = type(context.state).SPEAKING
            await agent._speak_to_caller(call_sid, message, language)
        logger.info("Caller admission deferred category=uncertain_overlap")
        return False
    state.clarification_latched = False
    return True
