"""Pinned Silero VAD ONNX adapter with per-speaking-window recurrent state."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Sequence

from services.conversation.speech_activity_gate import FRAME_SAMPLES, VadProbability


SILERO_VAD_REVISION = "7e30209a3e901f9842f81b225f3e93d8199902b1"
SILERO_VAD_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
SILERO_VAD_SIZE = 2327524
SAMPLE_RATE = 8000
CONTEXT_SAMPLES = 32


class LocalVadError(RuntimeError):
    """Fixed public failure type; exception values must not enter call logs."""


@dataclass
class LocalVadState:
    recurrent: Any
    context: Any


class LocalVadModel:
    """An immutable CPU session shared by calls, with state supplied per gate."""

    def __init__(self, model_path: str | Path, *, max_inference_ms: float = 20.0) -> None:
        self.model_path = Path(model_path)
        self.max_inference_ms = float(max_inference_ms)
        if not math.isfinite(self.max_inference_ms) or not 0 < self.max_inference_ms <= 100:
            raise LocalVadError("invalid inference budget")
        try:
            stat = self.model_path.stat()
        except OSError as exc:
            raise LocalVadError("model unavailable") from exc
        if stat.st_size != SILERO_VAD_SIZE:
            raise LocalVadError("model integrity mismatch")
        digest = hashlib.sha256()
        try:
            with self.model_path.open("rb") as model_file:
                for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise LocalVadError("model unavailable") from exc
        if digest.hexdigest() != SILERO_VAD_SHA256:
            raise LocalVadError("model integrity mismatch")

        try:
            import numpy as np
            import onnxruntime as ort
        except Exception as exc:
            raise LocalVadError("runtime unavailable") from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        try:
            session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise LocalVadError("model load failed") from exc

        expected_inputs = {
            "input": ("tensor(float)", [None, None]),
            "state": ("tensor(float)", [2, None, 128]),
            "sr": ("tensor(int64)", []),
        }
        expected_outputs = {
            "output": ("tensor(float)", [None, 1]),
            "stateN": ("tensor(float)", [None, None, None]),
        }
        if self._metadata(session.get_inputs()) != expected_inputs:
            raise LocalVadError("unexpected model input contract")
        if self._metadata(session.get_outputs()) != expected_outputs:
            raise LocalVadError("unexpected model output contract")
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise LocalVadError("unexpected inference provider")

        self._np = np
        self._session = session
        self._sample_rate = np.asarray(SAMPLE_RATE, dtype=np.int64)

    @staticmethod
    def _metadata(items) -> dict:
        return {item.name: (item.type, list(item.shape)) for item in items}

    def create_state(self) -> LocalVadState:
        return LocalVadState(
            recurrent=self._np.zeros((2, 1, 128), dtype=self._np.float32),
            context=self._np.zeros((1, CONTEXT_SAMPLES), dtype=self._np.float32),
        )

    def reset_state(self, state: LocalVadState) -> None:
        state.recurrent.fill(0)
        state.context.fill(0)

    def classify(self, frame: Sequence[int], state: LocalVadState) -> VadProbability:
        if len(frame) != FRAME_SAMPLES:
            raise LocalVadError("invalid frame size")
        try:
            samples = self._np.asarray(frame, dtype=self._np.float32).reshape(1, FRAME_SAMPLES)
            samples = samples / self._np.float32(32768.0)
            model_input = self._np.concatenate((state.context, samples), axis=1)
            started = time.perf_counter()
            output, next_state = self._session.run(
                ["output", "stateN"],
                {"input": model_input, "state": state.recurrent, "sr": self._sample_rate},
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            raise LocalVadError("inference failed") from exc

        if output.shape != (1, 1) or next_state.shape != (2, 1, 128):
            raise LocalVadError("invalid output shape")
        probability = float(output[0, 0])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise LocalVadError("invalid probability")
        if not bool(self._np.isfinite(next_state).all()):
            raise LocalVadError("invalid recurrent state")
        state.recurrent = next_state
        state.context = samples[:, -CONTEXT_SAMPLES:].copy()
        return VadProbability(probability, elapsed_ms)
