"""Bounded, provider-independent observations of barge-in decisions."""

from __future__ import annotations

from collections import deque
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import json
import time
from typing import Callable, Deque, Optional
from uuid import uuid4


MAX_ACTIVE_CALLS = 64
MAX_SPEAKING_WINDOWS = 16
MAX_COUNTER = 2**31 - 1
DIAGNOSTIC_FAILURE = "observer_failure"

DETECTOR_CATEGORIES = frozenset(
    {
        "no_audio",
        "grace_suppressed",
        "decode_failed",
        "below_threshold",
        "above_threshold",
        "gate_fired",
    }
)
ROUTE_CATEGORIES = frozenset({"forwarded", "dropped"})
ACTION_NAMES = frozenset({"interrupt", "reset"})
ACTION_OUTCOMES = frozenset({"succeeded", "not_owned", "failed"})
WINDOW_OUTCOMES = frozenset(
    {"active", "completed", "interrupted", "replaced", "failed", "teardown"}
)

_UNBOUND = object()
_BOUND_STATE: ContextVar[object] = ContextVar(
    "barge_in_diagnostic_state", default=_UNBOUND
)
_BOUND_WINDOW: ContextVar[object] = ContextVar(
    "barge_in_diagnostic_window", default=_UNBOUND
)


def bind_diagnostic_state(
    state: Optional["CallDiagnostics"],
    window: Optional["SpeakingWindow"] = None,
) -> tuple[Token, Token]:
    """Bind callback ownership across an awaited detector invocation."""
    return _BOUND_STATE.set(state), _BOUND_WINDOW.set(window)


def reset_diagnostic_state(tokens: tuple[Token, Token]) -> None:
    state_token, window_token = tokens
    _BOUND_WINDOW.reset(window_token)
    _BOUND_STATE.reset(state_token)


def bound_diagnostic_state() -> tuple[bool, Optional["CallDiagnostics"]]:
    state = _BOUND_STATE.get()
    if state is _UNBOUND:
        return False, None
    return True, state  # type: ignore[return-value]


def bound_diagnostic_window() -> tuple[bool, Optional["SpeakingWindow"]]:
    window = _BOUND_WINDOW.get()
    if window is _UNBOUND:
        return False, None
    return True, window  # type: ignore[return-value]


def _increment(value: int, amount: int = 1) -> int:
    return min(MAX_COUNTER, value + max(0, amount))


@dataclass(eq=False)
class SpeakingWindow:
    sequence: int
    started_at: float
    ended_at: Optional[float] = None
    outcome: str = "active"
    received_callbacks: int = 0
    received_bytes: int = 0
    received_min_bytes: Optional[int] = None
    received_max_bytes: int = 0
    analyzed_callbacks: int = 0
    analyzed_bytes: int = 0
    no_audio_count: int = 0
    grace_suppressed_count: int = 0
    decode_failed_count: int = 0
    below_threshold_count: int = 0
    above_threshold_count: int = 0
    gate_trigger_count: int = 0
    forwarded_callbacks: int = 0
    forwarded_bytes: int = 0
    dropped_callbacks: int = 0
    dropped_bytes: int = 0
    max_rms: int = 0
    max_consecutive: int = 0
    dispatch_gap_count: int = 0
    dispatch_gap_sum_seconds: float = 0.0
    dispatch_gap_min_seconds: Optional[float] = None
    dispatch_gap_max_seconds: float = 0.0
    last_dispatch_at: Optional[float] = None
    interrupt_succeeded_count: int = 0
    interrupt_not_owned_count: int = 0
    interrupt_failed_count: int = 0
    reset_succeeded_count: int = 0
    reset_not_owned_count: int = 0
    reset_failed_count: int = 0


@dataclass(eq=False)
class CallDiagnostics:
    diagnostic_id: str
    started_at: float
    windows: Deque[SpeakingWindow] = field(
        default_factory=lambda: deque(maxlen=MAX_SPEAKING_WINDOWS)
    )
    current_window: Optional[SpeakingWindow] = None
    windows_started: int = 0
    window_overflow_count: int = 0
    observer_failure_count: int = 0
    failure_category: Optional[str] = None
    received_callbacks: int = 0
    received_bytes: int = 0
    received_min_bytes: Optional[int] = None
    received_max_bytes: int = 0
    analyzed_bytes: int = 0
    max_rms: int = 0
    max_consecutive: int = 0
    trigger_count: int = 0
    forwarded_callbacks: int = 0
    forwarded_bytes: int = 0
    dropped_callbacks: int = 0
    dropped_bytes: int = 0
    speaking_seconds: float = 0.0
    finished: bool = False


class BargeInDiagnostics:
    """Collect fixed numeric aggregates without retaining identifiers or audio."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        emit: Optional[Callable[[str], None]] = None,
        max_active_calls: int = MAX_ACTIVE_CALLS,
    ) -> None:
        self._clock = clock
        self._emit = emit or (lambda _summary: None)
        self._max_active_calls = max(0, min(MAX_ACTIVE_CALLS, max_active_calls))
        self._active: dict[str, CallDiagnostics] = {}
        self.capacity_skip_count = 0

    @property
    def active_count(self) -> int:
        return len(self._active)

    def start_call(self) -> Optional[CallDiagnostics]:
        if len(self._active) >= self._max_active_calls:
            self.capacity_skip_count = _increment(self.capacity_skip_count)
            return None
        state = CallDiagnostics(diagnostic_id=uuid4().hex, started_at=self._clock())
        self._active[state.diagnostic_id] = state
        return state

    def mark_failed(self, state: Optional[CallDiagnostics]) -> None:
        if not self._is_active(state):
            return
        self._record_failure(state)

    def start_speaking(
        self, state: Optional[CallDiagnostics]
    ) -> Optional[SpeakingWindow]:
        if not self._is_active(state):
            return None
        now = self._clock()
        if state.current_window is not None:
            self.end_speaking(state, state.current_window, "replaced")
        state.windows_started = _increment(state.windows_started)
        if len(state.windows) == MAX_SPEAKING_WINDOWS:
            state.window_overflow_count = _increment(state.window_overflow_count)
        window = SpeakingWindow(sequence=state.windows_started, started_at=now)
        state.windows.append(window)
        state.current_window = window
        return window

    def ensure_speaking(
        self, state: Optional[CallDiagnostics]
    ) -> Optional[SpeakingWindow]:
        if not self._is_active(state):
            return None
        return state.current_window or self.start_speaking(state)

    def end_speaking(
        self,
        state: Optional[CallDiagnostics],
        window: Optional[SpeakingWindow],
        outcome: str,
    ) -> None:
        if (
            not self._is_active(state)
            or self._owned_window(state, window) is None
            or window.ended_at is not None
        ):
            return
        if outcome not in WINDOW_OUTCOMES:
            outcome = "failed"
        if window.outcome != "interrupted":
            window.outcome = outcome
        window.ended_at = self._clock()
        state.speaking_seconds += max(0.0, window.ended_at - window.started_at)
        if state.current_window is window:
            state.current_window = None

    def observe_dispatch(
        self, state: Optional[CallDiagnostics], byte_count: int
    ) -> Optional[SpeakingWindow]:
        window = self.ensure_speaking(state)
        if window is None:
            return None
        now = self._clock()
        window.received_callbacks = _increment(window.received_callbacks)
        window.received_bytes = _increment(window.received_bytes, byte_count)
        window.received_max_bytes = max(window.received_max_bytes, byte_count)
        window.received_min_bytes = (
            byte_count
            if window.received_min_bytes is None
            else min(window.received_min_bytes, byte_count)
        )
        state.received_callbacks = _increment(state.received_callbacks)
        state.received_bytes = _increment(state.received_bytes, byte_count)
        state.received_max_bytes = max(state.received_max_bytes, byte_count)
        state.received_min_bytes = (
            byte_count
            if state.received_min_bytes is None
            else min(state.received_min_bytes, byte_count)
        )
        if window.last_dispatch_at is not None:
            gap = max(0.0, now - window.last_dispatch_at)
            window.dispatch_gap_count = _increment(window.dispatch_gap_count)
            window.dispatch_gap_sum_seconds += gap
            window.dispatch_gap_max_seconds = max(window.dispatch_gap_max_seconds, gap)
            if window.dispatch_gap_min_seconds is None:
                window.dispatch_gap_min_seconds = gap
            else:
                window.dispatch_gap_min_seconds = min(
                    window.dispatch_gap_min_seconds, gap
                )
        window.last_dispatch_at = now
        return window

    def observe_detector(
        self,
        state: Optional[CallDiagnostics],
        category: str,
        *,
        analyzed_bytes: int = 0,
        rms: Optional[int] = None,
        consecutive: Optional[int] = None,
        window: Optional[SpeakingWindow] = None,
    ) -> None:
        window = self._owned_window(state, window)
        if window is None or category not in DETECTOR_CATEGORIES:
            return
        if analyzed_bytes:
            window.analyzed_callbacks = _increment(window.analyzed_callbacks)
            window.analyzed_bytes = _increment(window.analyzed_bytes, analyzed_bytes)
            state.analyzed_bytes = _increment(state.analyzed_bytes, analyzed_bytes)
        field_name = {
            "no_audio": "no_audio_count",
            "grace_suppressed": "grace_suppressed_count",
            "decode_failed": "decode_failed_count",
            "below_threshold": "below_threshold_count",
            "above_threshold": "above_threshold_count",
            "gate_fired": "gate_trigger_count",
        }[category]
        setattr(window, field_name, _increment(getattr(window, field_name)))
        if rms is not None:
            window.max_rms = max(window.max_rms, max(0, int(rms)))
            state.max_rms = max(state.max_rms, max(0, int(rms)))
        if consecutive is not None:
            window.max_consecutive = max(
                window.max_consecutive, max(0, int(consecutive))
            )
            state.max_consecutive = max(
                state.max_consecutive, max(0, int(consecutive))
            )
        if category == "gate_fired":
            state.trigger_count = _increment(state.trigger_count)

    def observe_route(
        self,
        state: Optional[CallDiagnostics],
        category: str,
        byte_count: int,
        *,
        window: Optional[SpeakingWindow] = None,
    ) -> None:
        window = self._owned_window(state, window)
        if window is None or category not in ROUTE_CATEGORIES:
            return
        callback_field = f"{category}_callbacks"
        byte_field = f"{category}_bytes"
        setattr(window, callback_field, _increment(getattr(window, callback_field)))
        setattr(window, byte_field, _increment(getattr(window, byte_field), byte_count))
        setattr(state, callback_field, _increment(getattr(state, callback_field)))
        setattr(state, byte_field, _increment(getattr(state, byte_field), byte_count))

    def observe_action(
        self,
        state: Optional[CallDiagnostics],
        action: str,
        outcome: str,
        *,
        window: Optional[SpeakingWindow] = None,
    ) -> None:
        window = self._owned_window(state, window)
        if (
            window is None
            or action not in ACTION_NAMES
            or outcome not in ACTION_OUTCOMES
        ):
            return
        field_name = f"{action}_{outcome}_count"
        setattr(window, field_name, _increment(getattr(window, field_name)))
        if action == "interrupt" and outcome == "succeeded":
            window.outcome = "interrupted"

    def snapshot(self, state: CallDiagnostics) -> dict:
        now = self._clock()
        return {
            "schema": 1,
            "event": "barge_in_diagnostics_summary",
            "diagnostic_id": state.diagnostic_id,
            "elapsed_seconds": round(max(0.0, now - state.started_at), 6),
            "windows_started": state.windows_started,
            "windows_retained": len(state.windows),
            "window_overflow_count": state.window_overflow_count,
            "observer_failure_count": state.observer_failure_count,
            "failure_category": state.failure_category,
            "totals": {
                "received_callbacks": state.received_callbacks,
                "received_bytes": state.received_bytes,
                "received_min_bytes": state.received_min_bytes,
                "received_max_bytes": state.received_max_bytes,
                "analyzed_bytes": state.analyzed_bytes,
                "max_rms": state.max_rms,
                "max_consecutive": state.max_consecutive,
                "trigger_count": state.trigger_count,
                "forwarded_callbacks": state.forwarded_callbacks,
                "forwarded_bytes": state.forwarded_bytes,
                "dropped_callbacks": state.dropped_callbacks,
                "dropped_bytes": state.dropped_bytes,
                "speaking_seconds": round(state.speaking_seconds, 6),
            },
            "windows": [self._window_snapshot(window, now) for window in state.windows],
        }

    def finish_call(self, state: Optional[CallDiagnostics]) -> Optional[dict]:
        if not self._is_active(state):
            return None
        summary = None
        payload = None
        try:
            if state.current_window is not None:
                self.end_speaking(state, state.current_window, "teardown")
            summary = self.snapshot(state)
            payload = json.dumps(summary, separators=(",", ":"), sort_keys=True)
        except Exception:
            self._record_failure(state)
        finally:
            state.current_window = None
            state.finished = True
            self._active.pop(state.diagnostic_id, None)

        if payload is None:
            return None
        try:
            self._emit(payload)
        except Exception:
            self._record_failure(state)
        return summary

    def _is_active(self, state: Optional[CallDiagnostics]) -> bool:
        return (
            state is not None
            and not state.finished
            and self._active.get(state.diagnostic_id) is state
        )

    def _owned_window(
        self,
        state: Optional[CallDiagnostics],
        window: Optional[SpeakingWindow],
    ) -> Optional[SpeakingWindow]:
        if not self._is_active(state):
            return None
        # None means acquisition failed or no window was owned. Never adopt
        # whichever invocation happens to be current when a delayed event lands.
        candidate = window
        if candidate is None:
            return None
        if not any(retained is candidate for retained in state.windows):
            return None
        return candidate

    @staticmethod
    def _record_failure(state: CallDiagnostics) -> None:
        state.observer_failure_count = _increment(state.observer_failure_count)
        state.failure_category = DIAGNOSTIC_FAILURE

    @staticmethod
    def _window_snapshot(window: SpeakingWindow, now: float) -> dict:
        end = window.ended_at if window.ended_at is not None else now
        mean_gap = (
            window.dispatch_gap_sum_seconds / window.dispatch_gap_count
            if window.dispatch_gap_count
            else None
        )
        return {
            "sequence": window.sequence,
            "elapsed_seconds": round(max(0.0, end - window.started_at), 6),
            "outcome": window.outcome,
            "received_callbacks": window.received_callbacks,
            "received_bytes": window.received_bytes,
            "received_min_bytes": window.received_min_bytes,
            "received_max_bytes": window.received_max_bytes,
            "analyzed_callbacks": window.analyzed_callbacks,
            "analyzed_bytes": window.analyzed_bytes,
            "no_audio_count": window.no_audio_count,
            "grace_suppressed_count": window.grace_suppressed_count,
            "decode_failed_count": window.decode_failed_count,
            "below_threshold_count": window.below_threshold_count,
            "above_threshold_count": window.above_threshold_count,
            "gate_trigger_count": window.gate_trigger_count,
            "forwarded_callbacks": window.forwarded_callbacks,
            "forwarded_bytes": window.forwarded_bytes,
            "dropped_callbacks": window.dropped_callbacks,
            "dropped_bytes": window.dropped_bytes,
            "max_rms": window.max_rms,
            "max_consecutive": window.max_consecutive,
            "dispatch_gap_count": window.dispatch_gap_count,
            "dispatch_gap_min_seconds": _rounded(window.dispatch_gap_min_seconds),
            "dispatch_gap_mean_seconds": _rounded(mean_gap),
            "dispatch_gap_max_seconds": round(window.dispatch_gap_max_seconds, 6),
            "interrupt_succeeded_count": window.interrupt_succeeded_count,
            "interrupt_not_owned_count": window.interrupt_not_owned_count,
            "interrupt_failed_count": window.interrupt_failed_count,
            "reset_succeeded_count": window.reset_succeeded_count,
            "reset_not_owned_count": window.reset_not_owned_count,
            "reset_failed_count": window.reset_failed_count,
        }


def _rounded(value: Optional[float]) -> Optional[float]:
    return round(value, 6) if value is not None else None
