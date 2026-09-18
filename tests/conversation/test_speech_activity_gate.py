from __future__ import annotations

import asyncio
import math

import pytest

from services.conversation.speech_activity_gate import (
    FRAME_SAMPLES,
    MAX_CANDIDATE_BYTES,
    GateDecision,
    SpeechActivityGate,
    VadProbability,
)


class ControlledClock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeClassifier:
    def __init__(self, outputs, *, elapsed_ms: float = 0.1) -> None:
        self.outputs = iter(outputs)
        self.elapsed_ms = elapsed_ms
        self.frames = []
        self.reset_count = 0

    def create_state(self):
        return {"generation": 0}

    def reset_state(self, state) -> None:
        state["generation"] += 1
        self.reset_count += 1

    def classify(self, frame, _state):
        self.frames.append(tuple(frame))
        value = next(self.outputs)
        if isinstance(value, BaseException):
            raise value
        return VadProbability(float(value), self.elapsed_ms)


def make_gate(outputs, **kwargs):
    warnings = []
    classifier = FakeClassifier(outputs, elapsed_ms=kwargs.pop("elapsed_ms", 0.1))
    gate = SpeechActivityGate(
        classifier,
        probability_threshold=kwargs.pop("probability_threshold", 0.5),
        speech_duration_ms=kwargs.pop("speech_duration_ms", 160),
        max_input_gap_ms=kwargs.pop("max_input_gap_ms", 96),
        max_inference_ms=kwargs.pop("max_inference_ms", 20.0),
        warning=warnings.append,
        **kwargs,
    )
    return gate, classifier, warnings


@pytest.mark.asyncio
async def test_split_and_combined_callbacks_make_same_sample_duration_decision():
    payload = bytes(range(256)) * 5 + b"tail"
    combined, _, _ = make_gate([0.9] * 5)
    split, _, _ = make_gate([0.9] * 5)

    combined_result = await combined.analyze(payload)
    split_result = GateDecision.rejected()
    for start in range(0, len(payload), 73):
        result = await split.analyze(payload[start : start + 73])
        if result.triggered:
            split_result = result

    assert combined_result.triggered and split_result.triggered
    assert combined_result.audio == split_result.audio == payload
    assert combined.evidence_samples == split.evidence_samples == 5 * FRAME_SAMPLES


@pytest.mark.asyncio
async def test_rejected_frames_and_short_bursts_do_not_accumulate():
    gate, classifier, _ = make_gate([0.9] * 4 + [0.1] + [0.9] * 4)

    for _ in range(9):
        assert not (await gate.analyze(b"\x00" * FRAME_SAMPLES)).triggered

    assert gate.candidate_bytes == 4 * FRAME_SAMPLES
    assert gate.evidence_samples == 4 * FRAME_SAMPLES


@pytest.mark.asyncio
async def test_quiet_positive_audio_can_trigger_without_an_rms_floor():
    gate, _, _ = make_gate([0.99] * 5)
    quiet_mulaw = b"\xff" * (5 * FRAME_SAMPLES)

    decision = await gate.analyze(quiet_mulaw)

    assert decision.triggered
    assert decision.audio == quiet_mulaw


@pytest.mark.asyncio
async def test_real_input_gap_resets_partial_and_positive_evidence():
    clock = ControlledClock()
    gate, classifier, _ = make_gate([0.9] * 8, clock=clock)
    assert not (await gate.analyze(b"\x00" * (3 * FRAME_SAMPLES + 17))).triggered

    clock.advance(0.097)
    assert not (await gate.analyze(b"\x00" * (2 * FRAME_SAMPLES))).triggered
    assert classifier.reset_count == 1
    assert gate.evidence_samples == 2 * FRAME_SAMPLES

    assert (await gate.analyze(b"\x00" * (3 * FRAME_SAMPLES))).triggered


@pytest.mark.asyncio
async def test_candidate_and_callback_bounds_are_exact():
    gate, _, _ = make_gate([0.9] * 10, speech_duration_ms=320)
    result = await gate.analyze(b"\x00" * MAX_CANDIDATE_BYTES)

    assert result.triggered
    assert len(result.audio) == MAX_CANDIDATE_BYTES
    assert gate.max_candidate_bytes_observed == MAX_CANDIDATE_BYTES

    oversized, _, warnings = make_gate([0.9])
    assert not (await oversized.analyze(b"x" * 3201)).triggered
    assert oversized.retired
    assert warnings == ["input_oversized"]


@pytest.mark.asyncio
async def test_malformed_invalid_failure_and_overload_retire_without_fallback():
    malformed, _, malformed_warnings = make_gate([0.9])
    assert not (await malformed.analyze("not-bytes")).triggered
    assert malformed.retired
    assert malformed_warnings == ["input_malformed"]

    for value, elapsed, expected in [
        (math.nan, 0.1, "output_invalid"),
        (RuntimeError("private"), 0.1, "inference_failed"),
        (0.9, 20.1, "inference_overload"),
    ]:
        gate, _, warnings = make_gate([value], elapsed_ms=elapsed)
        first = await gate.analyze(b"\x00" * FRAME_SAMPLES)
        second = await gate.analyze(b"\x00" * FRAME_SAMPLES)
        assert not first.triggered and not second.triggered
        assert gate.retired and gate.candidate_bytes == 0
        assert warnings == [expected]


@pytest.mark.asyncio
async def test_retire_during_delayed_classification_discards_late_result():
    entered = asyncio.Event()
    release = asyncio.Event()

    class DelayedClassifier(FakeClassifier):
        async def classify(self, frame, state):
            entered.set()
            await release.wait()
            return super().classify(frame, state)

    warnings = []
    classifier = DelayedClassifier([0.99])
    gate = SpeechActivityGate(classifier, warning=warnings.append)
    task = asyncio.create_task(gate.analyze(b"\x00" * FRAME_SAMPLES))
    await entered.wait()
    gate.retire()
    release.set()

    assert not (await task).triggered
    assert gate.retired and gate.candidate_bytes == 0
    assert warnings == []


@pytest.mark.asyncio
async def test_gap_reset_failure_retires_gate_with_fixed_warning():
    clock = ControlledClock()
    gate, classifier, warnings = make_gate([0.9], clock=clock)
    await gate.analyze(b"\x00" * FRAME_SAMPLES)

    def fail_reset(_state):
        raise RuntimeError("private-reset-sentinel")

    classifier.reset_state = fail_reset
    clock.advance(0.2)
    assert not (await gate.analyze(b"\x00" * FRAME_SAMPLES)).triggered
    assert gate.retired and gate.candidate_bytes == 0
    assert warnings == ["state_reset_failed"]


@pytest.mark.asyncio
async def test_cancelled_classifier_permanently_retires_partial_evidence():
    entered, release = asyncio.Event(), asyncio.Event()

    class DelayedClassifier(FakeClassifier):
        async def classify(self, frame, state):
            entered.set()
            await release.wait()
            return super().classify(frame, state)

    gate = SpeechActivityGate(DelayedClassifier([0.9]))
    task = asyncio.create_task(gate.analyze(b"\x00" * FRAME_SAMPLES))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert gate.retired and gate.candidate_bytes == 0
        assert not (await gate.analyze(b"\x00" * FRAME_SAMPLES)).triggered
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
