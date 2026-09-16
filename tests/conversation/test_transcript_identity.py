"""T030-A dialogue-consumer contracts for per-call utterance identity."""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from services.conversation.events import UtteranceIdentity
from services.conversation.orchestrator import (
    ConversationContext,
    ConversationOrchestrator,
    ConversationState,
)
from services.stt.stt_base import STTResult


class FiniteSTT:
    def __init__(self, context: ConversationContext, results) -> None:
        self.context = context
        self.results = results

    async def get_transcript(self):
        for result in self.results:
            yield result
        self.context.state = ConversationState.ENDED


def identified(text: str, sequence: int, *, epoch: str = "epoch-a") -> STTResult:
    return STTResult(
        text=text,
        language="ar" if any("\u0600" <= char <= "\u06ff" for char in text) else "en",
        confidence=None,
        is_final=True,
        utterance_id=UtteranceIdentity("elevenlabs_scribe", epoch, sequence),
    )


def make_orchestrator(call_results: dict[str, list[STTResult]]) -> ConversationOrchestrator:
    orchestrator = ConversationOrchestrator.__new__(ConversationOrchestrator)
    orchestrator._conversations = {}
    orchestrator._call_stt_instances = {}
    orchestrator.process_transcript = mock.AsyncMock()
    for call_sid, results in call_results.items():
        context = ConversationContext(
            call_sid=call_sid,
            phone_number="+14165550100",
            language="en",
            state=ConversationState.LISTENING,
        )
        orchestrator._conversations[call_sid] = context
        orchestrator._call_stt_instances[call_sid] = FiniteSTT(context, results)
    return orchestrator


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["yes", "نعم"])
async def test_distinct_identical_answers_each_reach_dialogue(answer) -> None:
    orchestrator = make_orchestrator(
        {"call-a": [identified(answer, 1), identified(answer, 2)]}
    )

    await orchestrator._consume_stt_transcripts("call-a")

    assert orchestrator.process_transcript.await_count == 2
    assert [call.args[1] for call in orchestrator.process_transcript.await_args_list] == [
        answer,
        answer,
    ]


@pytest.mark.asyncio
async def test_duplicate_delivery_of_same_identity_reaches_dialogue_once() -> None:
    one = identified("yes", 1)
    orchestrator = make_orchestrator({"call-a": [one, one]})

    await orchestrator._consume_stt_transcripts("call-a")

    orchestrator.process_transcript.assert_awaited_once()


@pytest.mark.asyncio
async def test_unidentified_provider_results_are_forwarded_without_text_dedup() -> None:
    repeated = STTResult("yes", "en", 0.5, True)
    orchestrator = make_orchestrator({"call-a": [repeated, repeated]})

    await orchestrator._consume_stt_transcripts("call-a")

    assert orchestrator.process_transcript.await_count == 2


@pytest.mark.asyncio
async def test_same_identity_in_two_calls_is_scoped_independently() -> None:
    same = identified("yes", 1)
    orchestrator = make_orchestrator({"call-a": [same], "call-b": [same]})

    await asyncio.gather(
        orchestrator._consume_stt_transcripts("call-a"),
        orchestrator._consume_stt_transcripts("call-b"),
    )

    assert orchestrator.process_transcript.await_count == 2
    assert {call.args[0] for call in orchestrator.process_transcript.await_args_list} == {
        "call-a",
        "call-b",
    }


@pytest.mark.asyncio
async def test_partial_results_do_not_invoke_dialogue() -> None:
    partial = STTResult("ordinary partial", "en", None, False)
    orchestrator = make_orchestrator({"call-a": [partial]})

    await orchestrator._consume_stt_transcripts("call-a")

    orchestrator.process_transcript.assert_not_awaited()


@pytest.mark.asyncio
async def test_arabic_correction_is_a_distinct_consumer_boundary_final() -> None:
    correction = "لا، أقصد الساعة الحادية عشرة"
    orchestrator = make_orchestrator(
        {"call-a": [identified("نعم", 1), identified(correction, 2)]}
    )

    await orchestrator._consume_stt_transcripts("call-a")

    assert orchestrator.process_transcript.await_count == 2
    assert orchestrator.process_transcript.await_args_list[-1].args[1] == correction
