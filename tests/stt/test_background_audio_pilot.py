"""Offline contracts for the paired background-audio pilot."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pilot = _load("background_audio_pilot", ROOT / "scripts" / "evaluate-stt-background-audio.py")
probe = _load("provider_compatibility_probe", ROOT / "scripts" / "check-stt-provider-compatibility.py")


class FakeTransport:
    def __init__(self) -> None:
        self.aborted = False

    def abort(self) -> None:
        self.aborted = True


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.value

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds
        await asyncio.sleep(0)


class FakeSocket:
    def __init__(
        self,
        *events: object,
        error_after_sends: int | None = None,
        events_after_sends: dict[int, list[object]] | None = None,
    ) -> None:
        self.events: asyncio.Queue[object] = asyncio.Queue()
        for event in events:
            self.feed(event)
        self.error_after_sends = error_after_sends
        self.events_after_sends = events_after_sends or {}
        self.sent: list[str] = []
        self.closed = False
        self.transport = FakeTransport()

    def feed(self, event: object) -> None:
        self.events.put_nowait(event)

    async def recv(self) -> str:
        event = await self.events.get()
        if isinstance(event, BaseException):
            raise event
        return event if isinstance(event, str) else json.dumps(event)

    async def send(self, message: str) -> None:
        self.sent.append(message)
        for event in self.events_after_sends.get(len(self.sent), []):
            self.feed(event)
        if self.error_after_sends == len(self.sent):
            self.feed({"message_type": "invalid_request", "error": "PRIVATE_BODY"})
        await asyncio.sleep(0)

    async def close(self) -> None:
        self.closed = True


class CancellationResistantSocket(FakeSocket):
    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def recv(self) -> str:
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.release.wait()
        raise RuntimeError("PRIVATE_LATE_ERROR")


class LateConnector:
    def __init__(self, socket: FakeSocket) -> None:
        self.socket = socket
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def __call__(self, *_args, **_kwargs):
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.release.wait()
        return self.socket


def _session_started(enabled: bool = False) -> dict:
    return {
        "message_type": "session_started",
        "config": {"filter_background_audio": enabled},
    }


def _case(audio: bytes | None = None, language: str = "en") -> object:
    audio = audio if audio is not None else bytes((index % 251 for index in range(8001)))
    return pilot.ValidatedCase(
        case_id="synthetic-en-normal",
        audio_path=Path("outside.raw"),
        audio=audio,
        sha256=hashlib.sha256(audio).hexdigest(),
        duration_seconds=len(audio) / 8000,
        language_hint=language,
        foreground_expected_wording="Please move the synthetic appointment to three.",
        expected_entities=("three",),
        background_expected_wording=None,
        foreground_absent=False,
        source_kind="synthetic_test_fixture",
        permission_status=pilot.APPROVED_AUDIO_STATUS,
    )


def _manifest(path: Path, audio_path: Path, audio: bytes, **overrides) -> dict:
    item = {
        "case_id": "synthetic-en-normal",
        "audio_path": audio_path.name,
        "sha256": hashlib.sha256(audio).hexdigest(),
        "encoding": "mulaw",
        "sample_rate_hz": 8000,
        "channels": 1,
        "duration_seconds": len(audio) / 8000,
        "language_hint": "en",
        "foreground_expected_wording": "Please move the synthetic appointment to three.",
        "expected_entities": ["three"],
        "background_expected_wording": None,
        "foreground_absent": False,
        "source_kind": "consented_owner_recording",
        "permission_status": pilot.APPROVED_AUDIO_STATUS,
    }
    item.update(overrides)
    return {
        "schema": 1,
        "test_audio_status": pilot.APPROVED_AUDIO_STATUS,
        "cases": [item],
    }


def _audio_messages(socket: FakeSocket) -> list[bytes]:
    decoded = []
    for raw in socket.sent:
        event = json.loads(raw)
        assert event == {
            "message_type": "input_audio_chunk",
            "audio_base_64": event["audio_base_64"],
        }
        decoded.append(base64.b64decode(event["audio_base_64"], validate=True))
    return decoded


def test_query_matches_accepted_compatibility_diagnostic() -> None:
    for language in ("", "en", "ar"):
        for filtering in (False, True):
            assert pilot.build_connection_url(language, filtering) == probe.build_probe_url(
                language, filtering
            )


def test_manifest_permission_hash_format_and_exclusive_report_gate(tmp_path) -> None:
    audio = bytes([0x7F]) * 8000
    audio_path = tmp_path / "approved.raw"
    audio_path.write_bytes(audio)
    manifest_path = tmp_path / "manifest.json"
    manifest = _manifest(manifest_path, audio_path, audio)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    case = pilot.validate_case(manifest_path, "synthetic-en-normal")
    assert case.audio is not audio
    assert case.audio == audio
    assert case.sha256 == hashlib.sha256(audio).hexdigest()

    report = tmp_path / "result.json"
    pilot.reserve_report(report, case, 42).close()
    with pytest.raises(pilot.PilotError, match="report_exists"):
        pilot.reserve_report(report, case, 42)

    manifest["test_audio_status"] = "permission_pending"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(pilot.PilotError, match="audio_not_approved"):
        pilot.validate_case(manifest_path, "synthetic-en-normal")


def test_manifest_rejects_hash_format_and_duration_inconsistencies(tmp_path) -> None:
    audio = bytes([0x7F]) * 8000
    audio_path = tmp_path / "approved.raw"
    audio_path.write_bytes(audio)
    manifest_path = tmp_path / "manifest.json"

    for overrides, category in (
        ({"sha256": "0" * 64}, "audio_hash_mismatch"),
        ({"encoding": "pcm_s16le"}, "audio_format_invalid"),
        ({"sample_rate_hz": 16000}, "audio_format_invalid"),
        ({"channels": 2}, "audio_format_invalid"),
        ({"duration_seconds": 2.0}, "audio_duration_invalid"),
    ):
        manifest_path.write_text(
            json.dumps(_manifest(manifest_path, audio_path, audio, **overrides)),
            encoding="utf-8",
        )
        with pytest.raises(pilot.PilotError, match=category):
            pilot.validate_case(manifest_path, "synthetic-en-normal")


@pytest.mark.asyncio
async def test_byte_preservation_final_short_chunk_silence_and_pacing(monkeypatch) -> None:
    monkeypatch.setattr(pilot, "OBSERVATION_SECONDS", 0.01)
    case = _case()
    socket = FakeSocket(_session_started(False))
    clock = FakeClock()

    async def connect(*_args, **_kwargs):
        return socket

    result = await pilot.run_arm(
        pilot.Arm("baseline", False), case, "secret", connect, clock=clock
    )
    chunks = _audio_messages(socket)
    audio_chunks = (len(case.audio) + 159) // 160

    assert result["status"] == "complete"
    assert b"".join(chunks[:audio_chunks]) == case.audio
    assert len(chunks[audio_chunks - 1]) == 1
    assert b"".join(chunks[audio_chunks:]) == bytes([0xFF]) * 16000
    assert result["audio_bytes_sent"] == len(case.audio)
    assert result["silence_bytes_sent"] == 16000
    assert result["first_audio_send_seconds"] == 0.0
    assert result["last_chunk_send_seconds"] is not None
    assert result["latency_scoring_valid"] is True
    assert all(delay >= 0 for delay in clock.sleeps)
    assert socket.closed is True


@pytest.mark.asyncio
async def test_repeated_finals_and_delayed_language_metadata_remain_separate(monkeypatch) -> None:
    monkeypatch.setattr(pilot, "OBSERVATION_SECONDS", 0.01)
    socket = FakeSocket(
        _session_started(True),
        events_after_sends={
            1: [
                {"message_type": "committed_transcript", "text": "yes"},
                {"message_type": "committed_transcript", "text": "yes"},
            ],
            2: [
                {
                    "message_type": "committed_transcript_with_timestamps",
                    "text": "yes",
                    "language_code": "en",
                    "words": [{"text": "PRIVATE", "start": 0, "end": 1}],
                }
            ],
        },
    )

    async def connect(*_args, **_kwargs):
        return socket

    result = await pilot.run_arm(
        pilot.Arm("candidate", True), _case(), "secret", connect, clock=FakeClock()
    )

    assert [event["sequence"] for event in result["finals"]] == [1, 2]
    assert [event["text"] for event in result["finals"]] == ["yes", "yes"]
    assert result["language_metadata"][0]["sequence"] == 1
    assert result["language_metadata"][0]["detected_language"] == "en"
    assert result["language_metadata"][0]["received_seconds"] is not None
    assert "text" not in result["language_metadata"][0]


@pytest.mark.asyncio
async def test_pair_sends_identical_clip_and_silence_in_seeded_order(monkeypatch) -> None:
    monkeypatch.setattr(pilot, "OBSERVATION_SECONDS", 0.01)
    retained = [FakeSocket(_session_started(False)), FakeSocket(_session_started(True))]
    sockets = list(retained)

    async def connect(*_args, **_kwargs):
        return sockets.pop(0)

    report = await pilot.run_pair(_case(), "secret", 7, connect, clock_factory=FakeClock)

    assert report["arm_order"] == [arm.label for arm in pilot.ordered_arms(7)]
    assert _audio_messages(retained[0]) == _audio_messages(retained[1])
    assert all(arm["status"] == "complete" for arm in report["arms"])


@pytest.mark.asyncio
async def test_early_provider_error_stops_audio_and_is_not_zero_final_success(monkeypatch) -> None:
    monkeypatch.setattr(pilot, "OBSERVATION_SECONDS", 0.01)
    socket = FakeSocket(_session_started(True), error_after_sends=3)

    async def connect(*_args, **_kwargs):
        return socket

    result = await pilot.run_arm(
        pilot.Arm("candidate", True), _case(), "secret", connect, clock=FakeClock()
    )

    assert result["status"] == "incomplete"
    assert result["error_category"] == "invalid_request"
    assert result["zero_final_result"] is False
    assert 0 < len(socket.sent) < 151
    assert result["audio_bytes_sent"] < len(_case().audio)
    assert result["finals"] == []


@pytest.mark.asyncio
async def test_handshake_timeout_differs_from_true_empty_result(monkeypatch) -> None:
    monkeypatch.setattr(pilot, "HANDSHAKE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(pilot, "OBSERVATION_SECONDS", 0.01)
    accepted = FakeSocket(_session_started(False))
    stalled = FakeSocket()
    sockets = iter((accepted, stalled))

    async def connect(*_args, **_kwargs):
        return next(sockets)

    empty = await pilot.run_arm(
        pilot.Arm("baseline", False), _case(), "secret", connect, clock=FakeClock()
    )
    timed_out = await pilot.run_arm(
        pilot.Arm("candidate", True), _case(), "secret", connect, clock=FakeClock()
    )

    assert empty["status"] == "complete"
    assert empty["finals"] == []
    assert empty["zero_final_result"] is True
    assert empty["error_category"] is None
    assert timed_out["status"] == "incomplete"
    assert timed_out["zero_final_result"] is False
    assert timed_out["error_category"] == "handshake_timeout"


@pytest.mark.asyncio
async def test_late_connector_socket_is_closed_and_not_reused(monkeypatch) -> None:
    monkeypatch.setattr(pilot, "HANDSHAKE_TIMEOUT_SECONDS", 0.01)
    socket = FakeSocket()
    connector = LateConnector(socket)
    owner = pilot.TaskOwner()

    result = await pilot.open_socket(connector, "wss://synthetic.invalid", "secret", owner)
    assert result is None
    await asyncio.wait_for(connector.cancelled.wait(), timeout=0.1)
    connector.release.set()
    await pilot.drain_tasks(owner)

    assert socket.closed is True
    assert not owner.tasks


@pytest.mark.asyncio
async def test_cancellation_resistant_receiver_is_owned_and_consumed() -> None:
    socket = CancellationResistantSocket()
    state = pilot.ReceiverState()
    owner = pilot.TaskOwner()
    task = asyncio.create_task(pilot.receive_events(socket, state, FakeClock()))
    await asyncio.sleep(0)

    await pilot._stop_task(task, owner)
    assert socket.cancelled.is_set()
    assert owner.tasks
    socket.release.set()
    await pilot.drain_tasks(owner)

    assert not owner.tasks


@pytest.mark.asyncio
async def test_second_arm_failure_preserves_completed_first(monkeypatch) -> None:
    monkeypatch.setattr(pilot, "OBSERVATION_SECONDS", 0.01)
    sockets = iter(
        (
            FakeSocket(_session_started(False), {"message_type": "committed_transcript", "text": "first"}),
            FakeSocket(_session_started(True), {"message_type": "quota_exceeded", "error": "PRIVATE"}),
        )
    )

    async def connect(*_args, **_kwargs):
        return next(sockets)

    report = await pilot.run_pair(
        _case(), "secret", 1, connect, clock_factory=FakeClock
    )

    assert len(report["arms"]) == 2
    assert report["arms"][0]["status"] == "complete"
    assert report["arms"][0]["finals"][0]["text"] == "first"
    assert report["arms"][1]["status"] == "incomplete"
    assert report["arms"][1]["error_category"] == "quota_exceeded"
    assert report["status"] == "incomplete"


def test_stdout_redacts_transcript_and_report_is_external(monkeypatch, capsys, tmp_path) -> None:
    audio = bytes([0x55]) * 8000
    audio_path = tmp_path / "approved.raw"
    audio_path.write_bytes(audio)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest(manifest_path, audio_path, audio)), encoding="utf-8"
    )
    report_path = tmp_path / "report.json"

    async def result(*_args, **_kwargs):
        case = pilot.validate_case(manifest_path, "synthetic-en-normal")
        report = pilot._report_base(case, 7, pilot.ordered_arms(7))
        arm = pilot._empty_arm_result(pilot.Arm("baseline", False), case)
        arm["status"] = "complete"
        arm["finals"] = [{"sequence": 1, "text": "PRIVATE_TRANSCRIPT", "received_seconds": 1.0}]
        report["arms"] = [arm, dict(arm, label="candidate", filtering=True)]
        report["status"] = "complete"
        return report

    monkeypatch.setattr(pilot, "_load_configuration", lambda: "PRIVATE_API_KEY")
    monkeypatch.setattr(pilot, "run_pair", result)

    exit_code = pilot.main(
        [
            "--allow-provider-contact",
            "--manifest",
            str(manifest_path),
            "--case",
            "synthetic-en-normal",
            "--report",
            str(report_path),
            "--seed",
            "7",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "PRIVATE_TRANSCRIPT" not in stdout
    assert "PRIVATE_API_KEY" not in stdout
    assert json.loads(stdout)["report_written"] is True
    assert "PRIVATE_TRANSCRIPT" in report_path.read_text(encoding="utf-8")
    assert report_path.resolve().is_relative_to(tmp_path.resolve())


@pytest.mark.asyncio
async def test_cleanup_failure_invalidates_silence_and_latency(monkeypatch):
    class BrokenClose(FakeSocket):
        async def close(self):
            raise RuntimeError("synthetic private close failure")

    socket = BrokenClose(_session_started())
    async def connect(*_args, **_kwargs):
        return socket

    monkeypatch.setattr(pilot, "OBSERVATION_SECONDS", 0.001)
    result = await pilot.run_arm(
        pilot.Arm("baseline", False), _case(), "secret", connect, clock=FakeClock()
    )
    assert result["status"] == "incomplete"
    assert result["zero_final_result"] is False
    assert result["latency_scoring_valid"] is False


@pytest.mark.asyncio
async def test_provider_failure_during_pacing_sleep_sends_no_next_chunk():
    state = pilot.ReceiverState()
    class ErrorClock(FakeClock):
        async def sleep(self, seconds):
            await super().sleep(seconds)
            state.terminal.set()

    socket = FakeSocket()
    owner = pilot.TaskOwner()
    result = await pilot.stream_audio(socket, _case(), state, owner, ErrorClock())
    await pilot.drain_tasks(owner)
    assert not result["complete"]
    assert len(socket.sent) == 1


@pytest.mark.asyncio
async def test_cancel_open_keeps_late_socket_disposal_owned():
    started, release = asyncio.Event(), asyncio.Event()
    socket = FakeSocket()
    children = []
    async def connect(*_args, **_kwargs):
        children.append(asyncio.current_task())
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return socket

    owner = pilot.TaskOwner()
    opening = asyncio.create_task(pilot.open_socket(connect, "synthetic", "secret", owner))
    await asyncio.wait_for(started.wait(), 0.5)
    opening.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await opening
        release.set()
        await asyncio.gather(*children, return_exceptions=True)
        await pilot.drain_tasks(owner)
        assert socket.closed, "cancelled open must dispose its late socket"
    finally:
        release.set()
        await asyncio.gather(*children, return_exceptions=True)
        await socket.close()
        await pilot.drain_tasks(owner)


def test_nonfinite_manifest_duration_is_rejected(tmp_path):
    audio = bytes([0xff]) * 8000
    path = tmp_path / "audio.raw"
    path.write_bytes(audio)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(manifest, path, audio, duration_seconds=float('nan'))))
    with pytest.raises(pilot.PilotError, match="audio_duration_invalid"):
        pilot.validate_case(manifest, "synthetic-en-normal")


@pytest.mark.asyncio
async def test_cancelled_handshake_removes_both_waiters():
    state = pilot.ReceiverState()
    receiver_release = asyncio.Event()
    receiver = asyncio.create_task(receiver_release.wait())
    before = set(asyncio.all_tasks())
    handshake = asyncio.create_task(pilot._await_handshake(state, receiver, 10))
    await asyncio.sleep(0)
    waiters = set(asyncio.all_tasks()) - before - {handshake}
    handshake.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await handshake
        assert waiters and all(task.done() for task in waiters)
    finally:
        receiver_release.set()
        await receiver
        for task in waiters:
            task.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)


def test_reserved_report_does_not_overwrite_replaced_path(tmp_path):
    report = tmp_path / "report.json"
    reservation = pilot.reserve_report(report, _case(), 1)
    try:
        try:
            report.rename(tmp_path / "original.json")
        except PermissionError:
            # Windows can prevent replacement while the exclusive descriptor is
            # held; in that case the replacement attack itself is blocked.
            assert sys.platform == "win32"
            return
        report.write_text("unrelated existing data", encoding="utf-8")
        with pytest.raises(pilot.PilotError, match="report_write_failed"):
            pilot.write_report(reservation if reservation is not None else report, {"status": "complete"})
        assert report.read_text() == "unrelated existing data"
    finally:
        if reservation is not None:
            reservation.close()


@pytest.mark.asyncio
async def test_blocked_send_does_not_catch_up_and_invalidates_latency():
    clock = FakeClock()
    sent_at = []
    class SlowFirstSend(FakeSocket):
        async def send(self, message):
            sent_at.append(clock.monotonic())
            if len(sent_at) == 1:
                clock.value += 0.12
            await super().send(message)

    owner = pilot.TaskOwner()
    result = await pilot.stream_audio(
        SlowFirstSend(), _case(), pilot.ReceiverState(), owner, clock
    )
    await pilot.drain_tasks(owner)
    assert result["complete"]
    assert not result["latency_scoring_valid"]
    assert result["max_schedule_deviation_seconds"] >= 0.12
    assert all(b - a >= 0.0199 for a, b in zip(sent_at, sent_at[1:]))


@pytest.mark.asyncio
async def test_cancelled_close_retains_and_aborts_socket():
    started, release = asyncio.Event(), asyncio.Event()
    class LateClose(FakeSocket):
        async def close(self):
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            self.closed = True

    socket = LateClose()
    owner = pilot.TaskOwner()
    close = asyncio.create_task(pilot.close_socket(socket, owner))
    await asyncio.wait_for(started.wait(), 0.5)
    close.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await close
        assert socket.transport.aborted and owner.tasks
    finally:
        release.set()
        await pilot.drain_tasks(owner)
    assert socket.closed and not owner.tasks


@pytest.mark.asyncio
async def test_timer_oversleep_does_not_accumulate_across_clip():
    class OversleepClock(FakeClock):
        async def sleep(self, seconds):
            await super().sleep(seconds + 0.011)

    clock = OversleepClock()
    sent_at = []
    class Sink(FakeSocket):
        async def send(self, message):
            sent_at.append(clock.monotonic())
            await super().send(message)

    result = await pilot.stream_audio(Sink(), _case(), pilot.ReceiverState(), pilot.TaskOwner(), clock)
    assert result['complete'] and result['latency_scoring_valid']
    assert result['max_schedule_deviation_seconds'] < 0.02
    assert abs(sent_at[-1] - sent_at[0] - (len(sent_at)-1)*0.02) < 0.02
    assert min(b-a for a,b in zip(sent_at,sent_at[1:])) >= 0.005 - 1e-8


@pytest.mark.asyncio
async def test_small_send_cost_does_not_accumulate_across_clip():
    clock = FakeClock()
    class Sink(FakeSocket):
        async def send(self, message):
            clock.value += 0.003
            await super().send(message)
    result = await pilot.stream_audio(Sink(), _case(), pilot.ReceiverState(), pilot.TaskOwner(), clock)
    assert result['complete'] and result['latency_scoring_valid']
    assert result['max_schedule_deviation_seconds'] < 0.004


@pytest.mark.asyncio
async def test_long_sleep_stall_preserves_spacing_and_invalidates_timing():
    class StallClock(FakeClock):
        stalled = False
        async def sleep(self, seconds):
            await super().sleep(seconds)
            if not self.stalled:
                self.stalled = True
                self.value += 0.12
    clock = StallClock()
    sent_at = []
    class Sink(FakeSocket):
        async def send(self, message):
            sent_at.append(clock.monotonic())
            await super().send(message)
    result = await pilot.stream_audio(Sink(), _case(), pilot.ReceiverState(), pilot.TaskOwner(), clock)
    assert result['complete'] and not result['latency_scoring_valid']
    assert result['max_schedule_deviation_seconds'] >= 0.12 - 1e-8
    assert all(b-a >= 0.0199 for a,b in zip(sent_at,sent_at[1:]))


@pytest.mark.asyncio
async def test_windows_timer_wait_is_off_loop_and_bounded(monkeypatch):
    calls = []
    async def run_in_thread(function, seconds):
        calls.append((function, seconds))
        await asyncio.sleep(0)
    monkeypatch.setattr(pilot.sys, 'platform', 'win32')
    monkeypatch.setattr(pilot.asyncio, 'to_thread', run_in_thread)
    await pilot.RealClock.sleep(0.25)
    assert calls == [(pilot.time.sleep, 0.02)]


@pytest.mark.asyncio
async def test_early_timer_wakeup_does_not_dispatch_before_deadline():
    class EarlyClock(FakeClock):
        early = True
        async def sleep(self, seconds):
            if self.early:
                self.early = False
                await super().sleep(seconds / 2)
            else:
                await super().sleep(seconds)
    clock = EarlyClock()
    sent_at = []
    class Sink(FakeSocket):
        async def send(self, message):
            sent_at.append(clock.monotonic())
            await super().send(message)
    result = await pilot.stream_audio(Sink(), _case(), pilot.ReceiverState(), pilot.TaskOwner(), clock)
    assert result['latency_scoring_valid']
    assert all(t-sent_at[0] >= index*0.02-1e-8 for index,t in enumerate(sent_at))


def test_stdout_exposes_invalid_timing_without_transcript():
    report = {'case_id': 'synthetic', 'status': 'complete', 'arms': [
        {'label': 'candidate', 'status': 'complete', 'finals': [{'text': 'PRIVATE_TEXT'}],
         'latency_scoring_valid': False}]}
    summary = pilot.stdout_summary(report, True)
    assert summary['arms'][0]['latency_scoring_valid'] is False
    assert summary['arms'][0]['final_count'] == 1
    assert 'PRIVATE_TEXT' not in json.dumps(summary)
