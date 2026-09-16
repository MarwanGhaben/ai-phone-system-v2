"""T030-A behavioral contracts for Scribe utterance identity and metadata."""

from __future__ import annotations

import asyncio
from collections import deque
import json
from math import inf, nan

import pytest
from loguru import logger
from unittest import mock

from services.conversation.events import MetadataDisposition
from services.conversation.orchestrator import (
    ConversationContext,
    ConversationOrchestrator,
    ConversationState,
)
from services.stt.elevenlabs_stt_service import ElevenLabsSTT
from services.stt.stt_base import STTResult


class ScriptedWebSocket:
    def __init__(self, events=()) -> None:
        self._events: asyncio.Queue = asyncio.Queue()
        self.recv_started = asyncio.Event()
        self.sent_messages: list[str] = []
        self.closed = False
        for event in events:
            self.feed(event)

    def feed(self, event) -> None:
        self._events.put_nowait(event if isinstance(event, str) else json.dumps(event))

    async def recv(self) -> str:
        self.recv_started.set()
        return await self._events.get()

    async def send(self, message: str) -> None:
        self.sent_messages.append(message)

    async def close(self) -> None:
        self.closed = True

    async def ping(self) -> None:
        return None


def commit(text: str, **metadata):
    return {"message_type": "committed_transcript", "text": text, **metadata}


def enrichment(text: str, **metadata):
    return {
        "message_type": "committed_transcript_with_timestamps",
        "text": text,
        **metadata,
    }


async def run_events(stt: ElevenLabsSTT, events) -> list[STTResult]:
    socket = ScriptedWebSocket([*events, {"message_type": "session_ended"}])
    stt._websocket = socket
    stt._transcript_queue = asyncio.Queue()
    stt._is_listening = True
    await stt._receive_loop()
    results = []
    while not stt._transcript_queue.empty():
        results.append(stt._transcript_queue.get_nowait())
    return results


class RecordingElevenLabsSTT(ElevenLabsSTT):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.yielded_results: list[STTResult] = []

    async def get_transcript(self):
        async for result in super().get_transcript():
            self.yielded_results.append(result)
            yield result


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["yes", "نعم"])
async def test_repeated_commits_flow_through_actual_adapter_and_consumer(answer, monkeypatch) -> None:
    stt = RecordingElevenLabsSTT(api_key="synthetic-key")
    socket = ScriptedWebSocket(
        [
            commit(answer),
            enrichment(answer),
            commit(answer),
            enrichment(answer),
            {"message_type": "session_ended"},
        ]
    )
    async def connect_socket(*args, **kwargs):
        return socket

    monkeypatch.setattr(
        "services.stt.elevenlabs_stt_service.websockets.connect", connect_socket
    )
    assert await stt.connect() is True
    await asyncio.wait_for(stt._receive_task, timeout=1)
    stt._is_listening = False

    orchestrator = ConversationOrchestrator.__new__(ConversationOrchestrator)
    context = ConversationContext(
        call_sid="call-a",
        phone_number="+14165550100",
        state=ConversationState.LISTENING,
    )
    orchestrator._conversations = {"call-a": context}
    orchestrator._call_stt_instances = {"call-a": stt}
    invocations = 0

    async def process(*args):
        nonlocal invocations
        invocations += 1
        if invocations == 2:
            context.state = ConversationState.ENDED

    orchestrator.process_transcript = mock.AsyncMock(side_effect=process)

    try:
        await asyncio.wait_for(orchestrator._consume_stt_transcripts("call-a"), timeout=1)
    finally:
        await stt.disconnect()

    assert orchestrator.process_transcript.await_count == 2
    assert [result.text for result in stt.yielded_results] == [answer, answer]
    assert stt.yielded_results[0].utterance_id != stt.yielded_results[1].utterance_id


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["yes", "نعم"])
async def test_identical_answers_from_distinct_commits_remain_distinct(answer) -> None:
    stt = ElevenLabsSTT(api_key="synthetic-key")

    results = await run_events(
        stt,
        [
            commit(answer),
            enrichment(answer, language_code="ar" if answer == "نعم" else "en"),
            commit(answer),
            enrichment(answer, language_code="ar" if answer == "نعم" else "en"),
        ],
    )

    assert [result.text for result in results] == [answer, answer]
    assert results[0].utterance_id != results[1].utterance_id
    assert all(result.utterance_id is not None for result in results)
    matched = [
        event for event in stt.metadata_events
        if event.disposition is MetadataDisposition.MATCHED
    ]
    assert [event.utterance_id for event in matched] == [results[0].utterance_id]
    assert stt.metadata_events[-1].utterance_id is None


@pytest.mark.asyncio
async def test_delayed_same_text_metadata_cannot_attach_to_a_later_commit() -> None:
    stt = ElevenLabsSTT(api_key="synthetic-key")

    results = await run_events(
        stt,
        [
            commit("yes"),
            enrichment("yes", language_code="en", confidence=0.8),
            commit("yes"),
            enrichment("yes", language_code="en", confidence=0.8),
        ],
    )

    assert [result.text for result in results] == ["yes", "yes"]
    assert results[0].utterance_id != results[1].utterance_id
    assert [event.disposition for event in stt.metadata_events] == [
        MetadataDisposition.MATCHED,
        MetadataDisposition.AMBIGUOUS,
    ]
    assert stt.metadata_events[0].utterance_id == results[0].utterance_id
    assert stt.metadata_events[1].utterance_id is None


@pytest.mark.asyncio
async def test_altered_repeated_metadata_cannot_attach_to_a_later_commit() -> None:
    stt = ElevenLabsSTT(api_key="synthetic-key")

    results = await run_events(
        stt,
        [
            commit("yes"),
            enrichment("yes", language_code="en", words=[{"start": 0, "end": 1}]),
            commit("yes"),
            enrichment("yes", language_code="ar", words=[{"start": 2, "end": 3}]),
        ],
    )

    assert len(results) == 2
    assert stt.metadata_events[0].utterance_id == results[0].utterance_id
    assert stt.metadata_events[1].utterance_id is None


@pytest.mark.asyncio
async def test_metadata_before_same_text_commit_prevents_later_association() -> None:
    stt = ElevenLabsSTT(api_key="synthetic-key")

    results = await run_events(
        stt,
        [
            enrichment("yes", language_code="en"),
            commit("yes"),
            enrichment("yes", language_code="ar"),
        ],
    )

    assert [result.text for result in results] == ["yes"]
    assert all(event.utterance_id is None for event in stt.metadata_events)
    assert not any(
        event.disposition is MetadataDisposition.MATCHED
        for event in stt.metadata_events
    )


@pytest.mark.asyncio
async def test_correlation_history_overflow_disables_matching_until_new_epoch() -> None:
    stt = ElevenLabsSTT(api_key="synthetic-key")
    socket = ScriptedWebSocket()
    stt._websocket = socket
    stt._transcript_queue = asyncio.Queue()
    stt._is_listening = True
    receive_task = asyncio.create_task(stt._receive_loop())
    await socket.recv_started.wait()
    overflow = [
        enrichment(f"unmatched-{index}")
        for index in range(stt.MAX_CORRELATION_TEXT_HISTORY + 1)
    ]
    for event in [
        *overflow,
        commit("after-overflow"),
        enrichment("after-overflow"),
        commit("processed-marker"),
    ]:
        socket.feed(event)

    results = [
        await asyncio.wait_for(stt._transcript_queue.get(), timeout=0.5)
        for _ in range(2)
    ]
    state = stt._current_receiver
    assert [result.text for result in results] == [
        "after-overflow",
        "processed-marker",
    ]
    assert stt.metadata_events[-1].utterance_id is None
    assert stt.metadata_events[-1].disposition is MetadataDisposition.AMBIGUOUS
    assert state.correlation_disabled is True
    assert len(state.correlation_text_history) == stt.MAX_CORRELATION_TEXT_HISTORY

    socket.feed({"message_type": "session_ended"})
    await receive_task
    assert state.correlation_text_history == {}

    fresh_results = await run_events(
        stt,
        [commit("fresh"), enrichment("fresh")],
    )
    assert stt.metadata_events[-1].disposition is MetadataDisposition.MATCHED
    assert stt.metadata_events[-1].utterance_id == fresh_results[0].utterance_id


@pytest.mark.asyncio
async def test_commit_is_delivered_without_waiting_for_optional_metadata() -> None:
    stt = ElevenLabsSTT(api_key="synthetic-key")
    socket = ScriptedWebSocket()
    stt._websocket = socket
    stt._transcript_queue = asyncio.Queue()
    stt._is_listening = True
    receiver = asyncio.create_task(stt._receive_loop())
    await socket.recv_started.wait()

    socket.feed(commit("prompt answer"))
    result = await asyncio.wait_for(stt._transcript_queue.get(), timeout=0.5)
    socket.feed({"message_type": "session_ended"})
    await receiver

    assert result.text == "prompt answer"
    assert result.is_final is True
    assert result.utterance_id is not None


@pytest.mark.asyncio
async def test_enrichment_is_classified_without_replaying_or_mutating_final() -> None:
    stt = ElevenLabsSTT(api_key="synthetic-key")

    results = await run_events(
        stt,
        [
            commit("one answer", confidence=0.42, language_code="en"),
            enrichment(
                "one answer",
                language_code="en",
                language_probability=0.8,
                confidence=0.7,
                words=[{"text": "private", "start": 0.1, "end": 0.4}],
            ),
        ],
    )

    assert len(results) == 1
    assert results[0].confidence == 0.42
    event = stt.metadata_events[-1]
    assert event.disposition is MetadataDisposition.MATCHED
    assert event.utterance_id == results[0].utterance_id
    assert event.language == "en"
    assert event.language_confidence == 0.8
    assert event.confidence == 0.7
    assert [(word.start_seconds, word.end_seconds) for word in event.word_timings] == [
        (0.1, 0.4)
    ]
    assert results[0].language == "en"
    assert "words" not in results[0].metadata


@pytest.mark.asyncio
async def test_ambiguous_repeated_text_metadata_never_attaches_or_replays() -> None:
    stt = ElevenLabsSTT(api_key="synthetic-key")

    results = await run_events(
        stt,
        [commit("yes"), commit("yes"), enrichment("yes"), enrichment("yes")],
    )

    assert len(results) == 2
    assert results[0].utterance_id != results[1].utterance_id
    ambiguous = [
        event for event in stt.metadata_events
        if event.disposition is MetadataDisposition.AMBIGUOUS
    ]
    assert len(ambiguous) == 2
    assert all(event.utterance_id is None for event in ambiguous)


@pytest.mark.asyncio
async def test_delayed_repeated_mismatched_and_reordered_metadata_stays_diagnostic() -> None:
    stt = ElevenLabsSTT(api_key="synthetic-key")

    results = await run_events(
        stt,
        [
            commit("first"),
            enrichment("mismatch"),
            enrichment("first"),
            enrichment("first"),
            commit("second"),
            commit("third"),
            enrichment("second"),
        ],
    )

    assert [result.text for result in results] == ["first", "second", "third"]
    dispositions = [event.disposition for event in stt.metadata_events]
    assert dispositions == [
        MetadataDisposition.UNMATCHED,
        MetadataDisposition.MATCHED,
        MetadataDisposition.REPEATED,
        MetadataDisposition.AMBIGUOUS,
    ]
    assert stt.metadata_events[-1].utterance_id is None


@pytest.mark.asyncio
async def test_invalid_enrichment_fields_are_retained_only_as_safe_unknowns() -> None:
    stt = ElevenLabsSTT(api_key="synthetic-key")

    results = await run_events(
        stt,
        [
            commit("answer"),
            enrichment(
                "answer",
                language_code=True,
                language_probability=nan,
                confidence=inf,
                words=[{"start": 0.5, "end": 0.1}],
            ),
        ],
    )

    assert len(results) == 1
    event = stt.metadata_events[-1]
    assert event.disposition is MetadataDisposition.MATCHED
    assert event.language is None
    assert event.language_confidence is None
    assert event.confidence is None
    assert event.word_timings == ()
    assert set(event.invalid_fields) == {
        "language_code",
        "language_probability",
        "confidence",
        "words",
    }


@pytest.mark.asyncio
async def test_metadata_partial_legacy_unknown_and_malformed_events_are_not_actionable() -> None:
    stt = ElevenLabsSTT(api_key="synthetic-key")

    results = await run_events(
        stt,
        [
            enrichment("metadata only", language_code="en"),
            {"message_type": "final_transcript", "text": "legacy final"},
            {"message_type": "partial_transcript", "text": "partial words"},
            {"message_type": "future_event", "text": "unknown payload"},
            ["malformed", "payload"],
        ],
    )

    assert results == []
    dispositions = {event.disposition for event in stt.metadata_events}
    assert MetadataDisposition.UNMATCHED in dispositions
    assert MetadataDisposition.UNSUPPORTED in dispositions
    assert MetadataDisposition.MALFORMED in dispositions


@pytest.mark.asyncio
async def test_new_connection_restarts_sequence_under_a_new_epoch() -> None:
    stt = ElevenLabsSTT(api_key="synthetic-key")

    first = await run_events(stt, [commit("same")])
    second = await run_events(stt, [commit("same")])

    assert first[0].utterance_id.commit_sequence == 1
    assert second[0].utterance_id.commit_sequence == 1
    assert first[0].utterance_id.connection_epoch != second[0].utterance_id.connection_epoch


@pytest.mark.asyncio
async def test_late_old_receiver_cannot_emit_under_the_new_epoch() -> None:
    stt = ElevenLabsSTT(api_key="synthetic-key")
    stt._transcript_queue = asyncio.Queue()
    stt._is_listening = True
    old_socket = ScriptedWebSocket()
    new_socket = ScriptedWebSocket()

    stt._websocket = old_socket
    old_task = asyncio.create_task(stt._receive_loop())
    await old_socket.recv_started.wait()
    stt._websocket = new_socket
    new_task = asyncio.create_task(stt._receive_loop())
    await new_socket.recv_started.wait()

    old_socket.feed(commit("late old answer"))
    old_socket.feed({"message_type": "session_ended"})
    new_socket.feed(commit("current answer"))
    new_socket.feed({"message_type": "session_ended"})
    try:
        await asyncio.wait_for(asyncio.gather(old_task, new_task), timeout=0.5)
    finally:
        for task in (old_task, new_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(old_task, new_task, return_exceptions=True)

    results = []
    while not stt._transcript_queue.empty():
        results.append(stt._transcript_queue.get_nowait())
    assert [result.text for result in results] == ["current answer"]
    assert results[0].utterance_id.commit_sequence == 1
    assert any(
        event.disposition is MetadataDisposition.OLD_EPOCH
        for event in stt.metadata_events
    )


@pytest.mark.asyncio
async def test_confidence_is_nullable_validated_and_backward_compatible() -> None:
    stt = ElevenLabsSTT(api_key="synthetic-key")
    values = [None, nan, inf, -inf, True, -0.1, 1.1, 0.0, 0.61, 1.0]
    events = [
        commit(f"answer-{index}", **({} if value is None else {"confidence": value}))
        for index, value in enumerate(values)
    ]

    results = await run_events(stt, events)

    assert [result.confidence for result in results] == [
        None, None, None, None, None, None, None, 0.0, 0.61, 1.0
    ]
    legacy = STTResult("legacy", "en", 0.5, True, ["alt"], {"source": "fake"})
    assert legacy.utterance_id is None
    assert legacy.alternatives == ["alt"]
    assert legacy.metadata == {"source": "fake"}


@pytest.mark.asyncio
async def test_oversized_and_invalid_commit_confidence_does_not_stop_reception() -> None:
    stt = ElevenLabsSTT(api_key="synthetic-key")
    invalid_values = [10**400, -(10**400), nan, inf, -inf, True]
    events = [
        commit(f"invalid-{index}", confidence=value)
        for index, value in enumerate(invalid_values)
    ]
    events.append(commit("subsequent-valid", confidence=0.5))

    results = await run_events(stt, events)

    assert [result.text for result in results] == [
        *(f"invalid-{index}" for index in range(len(invalid_values))),
        "subsequent-valid",
    ]
    assert [result.confidence for result in results[:-1]] == [None] * len(invalid_values)
    assert results[-1].confidence == 0.5


@pytest.mark.asyncio
async def test_invalid_enrichment_numbers_do_not_stop_reception_or_replay() -> None:
    stt = ElevenLabsSTT(api_key="synthetic-key")
    numeric_cases = [
        ("language_probability", {"language_probability": 10**400}),
        ("language_probability", {"language_probability": -(10**400)}),
        ("language_probability", {"language_probability": nan}),
        ("language_probability", {"language_probability": inf}),
        ("language_probability", {"language_probability": -inf}),
        ("language_probability", {"language_probability": True}),
        ("confidence", {"confidence": 10**400}),
        ("confidence", {"confidence": -(10**400)}),
        ("confidence", {"confidence": nan}),
        ("confidence", {"confidence": inf}),
        ("confidence", {"confidence": True}),
        ("words", {"words": [{"start": 10**400, "end": 10**400}]}),
        ("words", {"words": [{"start": -(10**400), "end": 0}]}),
        ("words", {"words": [{"start": 0, "end": 10**400}]}),
        ("words", {"words": [{"start": 0, "end": -(10**400)}]}),
        ("words", {"words": [{"start": nan, "end": 1}]}),
        ("words", {"words": [{"start": 0, "end": inf}]}),
        ("words", {"words": [{"start": -inf, "end": 1}]}),
        ("words", {"words": [{"start": True, "end": 1}]}),
        ("words", {"words": [{"start": 0, "end": False}]}),
    ]
    events = []
    for index, (_, payload) in enumerate(numeric_cases):
        text = f"metadata-{index}"
        events.extend([commit(text), enrichment(text, **payload)])
    events.append(commit("subsequent-valid"))

    results = await run_events(stt, events)

    assert [result.text for result in results] == [
        *(f"metadata-{index}" for index in range(len(numeric_cases))),
        "subsequent-valid",
    ]
    assert len(stt.metadata_events) == len(numeric_cases)
    assert all(event.utterance_id is not None for event in stt.metadata_events)
    assert [event.invalid_fields for event in stt.metadata_events] == [
        (field,) for field, _ in numeric_cases
    ]


@pytest.mark.asyncio
async def test_caches_are_bounded_teardown_releases_them_and_logs_exclude_text() -> None:
    secret = "synthetic-secret-transcript-937"
    captured: deque[str] = deque()
    sink = logger.add(lambda message: captured.append(str(message)), level="DEBUG")
    stt = ElevenLabsSTT(api_key="synthetic-key")
    try:
        commits = [commit(f"{secret}-{index}") for index in range(stt.MAX_PENDING_COMMITS + 5)]
        metadata = [enrichment(f"unmatched-{index}") for index in range(stt.MAX_METADATA_EVENTS + 5)]
        socket = ScriptedWebSocket()
        stt._websocket = socket
        stt._transcript_queue = asyncio.Queue()
        stt._is_listening = True
        receiver = asyncio.create_task(stt._receive_loop())
        await socket.recv_started.wait()
        for event in commits:
            socket.feed(event)
        results = [
            await asyncio.wait_for(stt._transcript_queue.get(), timeout=0.5)
            for _ in commits
        ]

        assert len(results) == len(commits)
        assert stt.pending_commit_count == stt.MAX_PENDING_COMMITS
        assert sum(
            event.disposition is MetadataDisposition.EVICTED
            for event in stt.metadata_events
        ) == 5

        for event in metadata:
            socket.feed(event)
        socket.feed({"message_type": "session_ended"})
        await receiver
        assert len(stt.metadata_events) <= stt.MAX_METADATA_EVENTS
        await stt.disconnect()
        assert stt.pending_commit_count == 0
        assert stt.metadata_events == ()
        assert secret not in "".join(captured)
    finally:
        logger.remove(sink)
