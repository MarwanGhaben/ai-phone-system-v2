"""Bounded speech-evidence policy for caller barge-in audio."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import math
import time
from typing import Any, Callable, Optional


SAMPLE_RATE = 8000
FRAME_SAMPLES = 256
FRAME_MS = 32
MAX_CANDIDATE_MS = 320
MAX_CANDIDATE_BYTES = SAMPLE_RATE * MAX_CANDIDATE_MS // 1000
MAX_CALLBACK_BYTES = 3200


def _decode_mulaw(value: int) -> int:
    encoded = (~value) & 0xFF
    sign = encoded & 0x80
    exponent = (encoded >> 4) & 0x07
    mantissa = encoded & 0x0F
    sample = ((mantissa << 3) + 0x84) << exponent
    sample -= 0x84
    return -sample if sign else sample


_MULAW_TO_PCM16 = tuple(_decode_mulaw(value) for value in range(256))


@dataclass(frozen=True)
class VadProbability:
    probability: float
    elapsed_ms: float


@dataclass(frozen=True)
class GateDecision:
    triggered: bool
    audio: bytes = b""

    @classmethod
    def rejected(cls) -> "GateDecision":
        return cls(False, b"")

    @classmethod
    def accepted(cls, audio: bytes) -> "GateDecision":
        return cls(True, bytes(audio))


class SpeechActivityGate:
    """Reassemble mu-law callbacks and require sustained local-VAD evidence."""

    def __init__(
        self,
        classifier: Any,
        *,
        probability_threshold: float = 0.5,
        speech_duration_ms: int = 160,
        max_input_gap_ms: int = 96,
        max_inference_ms: float = 20.0,
        max_callback_bytes: int = MAX_CALLBACK_BYTES,
        clock: Callable[[], float] = time.monotonic,
        warning: Optional[Callable[[str], None]] = None,
    ) -> None:
        if not math.isfinite(probability_threshold) or not 0 <= probability_threshold <= 1:
            raise ValueError("probability threshold must be finite and in [0, 1]")
        if not FRAME_MS <= speech_duration_ms <= MAX_CANDIDATE_MS:
            raise ValueError("speech duration must be between 32 and 320 ms")
        if not FRAME_MS <= max_input_gap_ms <= 1000:
            raise ValueError("input gap must be between 32 and 1000 ms")
        if not math.isfinite(max_inference_ms) or not 0 < max_inference_ms <= 100:
            raise ValueError("inference budget must be finite and in (0, 100] ms")
        if not FRAME_SAMPLES <= max_callback_bytes <= MAX_CALLBACK_BYTES:
            raise ValueError("callback bound is outside the supported range")

        self._classifier = classifier
        self._state = classifier.create_state()
        self._threshold = float(probability_threshold)
        self._required_frames = math.ceil(speech_duration_ms / FRAME_MS)
        self._max_gap_seconds = max_input_gap_ms / 1000.0
        self._max_inference_ms = float(max_inference_ms)
        self._max_callback_bytes = max_callback_bytes
        self._clock = clock
        self._warning = warning
        self._partial = bytearray()
        self._candidate = bytearray()
        self._evidence_samples = 0
        self._accepted_evidence_samples = 0
        self._last_input_at: Optional[float] = None
        self._retired = False
        self._warning_emitted = False
        self._revision = 0
        self._analysis_lock = asyncio.Lock()
        self.max_candidate_bytes_observed = 0

    @property
    def retired(self) -> bool:
        return self._retired

    @property
    def evidence_samples(self) -> int:
        return self._accepted_evidence_samples or self._evidence_samples

    @property
    def candidate_bytes(self) -> int:
        return len(self._candidate)

    def reset_evidence(self) -> None:
        if self._retired:
            return
        self._revision += 1
        self._partial.clear()
        self._candidate.clear()
        self._evidence_samples = 0
        self._last_input_at = None
        self._classifier.reset_state(self._state)

    def retire(self) -> None:
        if self._retired:
            return
        self._revision += 1
        self._retired = True
        self._partial.clear()
        self._candidate.clear()
        self._evidence_samples = 0
        self._last_input_at = None
        self._state = None

    def _fail(self, category: str) -> GateDecision:
        self.retire()
        if not self._warning_emitted:
            self._warning_emitted = True
            if self._warning is not None:
                try:
                    self._warning(category)
                except Exception:
                    pass
        return GateDecision.rejected()

    def _reset_candidate(self) -> None:
        self._candidate.clear()
        self._evidence_samples = 0

    async def analyze(self, audio: bytes) -> GateDecision:
        if self._retired:
            return GateDecision.rejected()
        if not isinstance(audio, (bytes, bytearray, memoryview)):
            return self._fail("input_malformed")
        audio = bytes(audio)
        if len(audio) > self._max_callback_bytes:
            return self._fail("input_oversized")
        if not audio:
            return GateDecision.rejected()

        async with self._analysis_lock:
            if self._retired:
                return GateDecision.rejected()
            now = self._clock()
            if (
                self._last_input_at is not None
                and now - self._last_input_at > self._max_gap_seconds
            ):
                self._partial.clear()
                self._reset_candidate()
                try:
                    self._classifier.reset_state(self._state)
                except Exception:
                    return self._fail("state_reset_failed")
            self._last_input_at = now
            self._partial.extend(audio)

            while len(self._partial) >= FRAME_SAMPLES:
                frame_bytes = bytes(self._partial[:FRAME_SAMPLES])
                del self._partial[:FRAME_SAMPLES]
                pcm16 = tuple(_MULAW_TO_PCM16[value] for value in frame_bytes)
                revision = self._revision
                started = time.perf_counter()
                try:
                    result = self._classifier.classify(pcm16, self._state)
                    if inspect.isawaitable(result):
                        result = await result
                    measured_ms = (time.perf_counter() - started) * 1000.0
                except asyncio.CancelledError:
                    self.retire()
                    raise
                except Exception:
                    return self._fail("inference_failed")

                if self._retired or revision != self._revision:
                    return GateDecision.rejected()
                try:
                    probability = float(result.probability)
                    elapsed_ms = max(float(result.elapsed_ms), measured_ms)
                except (AttributeError, TypeError, ValueError, OverflowError):
                    return self._fail("output_invalid")
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    return self._fail("output_invalid")
                if not math.isfinite(elapsed_ms) or elapsed_ms > self._max_inference_ms:
                    return self._fail("inference_overload")

                if probability >= self._threshold:
                    self._candidate.extend(frame_bytes)
                    self._evidence_samples += FRAME_SAMPLES
                    self.max_candidate_bytes_observed = max(
                        self.max_candidate_bytes_observed, len(self._candidate)
                    )
                    if len(self._candidate) > MAX_CANDIDATE_BYTES:
                        return self._fail("candidate_overflow")
                    if self._evidence_samples >= self._required_frames * FRAME_SAMPLES:
                        accepted = bytes(self._candidate) + bytes(self._partial)
                        self._accepted_evidence_samples = self._evidence_samples
                        self._partial.clear()
                        self._candidate.clear()
                        self._evidence_samples = 0
                        self._revision += 1
                        self._retired = True
                        self._state = None
                        return GateDecision.accepted(accepted)
                else:
                    self._reset_candidate()

            return GateDecision.rejected()
