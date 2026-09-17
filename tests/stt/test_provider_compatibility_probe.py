"""Offline tests for the standalone provider compatibility probe."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
from urllib.parse import parse_qs, urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check-stt-provider-compatibility.py"
SPEC = importlib.util.spec_from_file_location("stt_provider_probe", SCRIPT)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class SyntheticRemoteClose(Exception):
    pass


class FakeSocket:
    def __init__(self, *events: object) -> None:
        self.events: asyncio.Queue[object] = asyncio.Queue()
        for event in events:
            self.events.put_nowait(event)
        self.sent: list[str] = []
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        event = await self.events.get()
        if isinstance(event, BaseException):
            raise event
        return event if isinstance(event, str) else json.dumps(event)

    async def close(self) -> None:
        self.closed = True


class StalledSocket(FakeSocket):
    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.recv_cancelled = asyncio.Event()

    async def recv(self) -> str:
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.recv_cancelled.set()
            await self.release.wait()
        raise SyntheticRemoteClose()


def _websockets(connect):
    module = types.ModuleType("websockets")
    module.connect = connect
    exceptions = types.ModuleType("websockets.exceptions")
    exceptions.ConnectionClosed = SyntheticRemoteClose
    module.exceptions = exceptions
    return module


def _session_started(**extra):
    return {"message_type": "session_started", "config": {}, **extra}


def _query(url: str) -> dict[str, list[str]]:
    return parse_qs(urlsplit(url).query, keep_blank_values=True)


def test_probe_queries_match_actual_local_adapter_constructor() -> None:
    from services.stt.elevenlabs_stt_service import ElevenLabsSTT

    for arm in probe.arms():
        adapter = ElevenLabsSTT(
            api_key="synthetic-secret",
            language=arm.language,
            model=probe.REVIEWED_MODEL,
            sample_rate=8000,
        )
        # This diagnostic is published before the new adapter. Exercise the
        # deployed baseline constructor, then require exactly one candidate
        # parameter addition; no dependency on unpublished T033-A code.
        expected = adapter._connection_url()
        if arm.filtering:
            expected += "&filter_background_audio=true"
        assert probe.build_probe_url(arm.language, arm.filtering) == expected


def test_default_and_help_are_offline_and_do_not_load_settings(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        probe,
        "_load_configuration",
        lambda: (_ for _ in ()).throw(AssertionError("settings loaded")),
    )

    assert probe.main([]) == 0
    output = capsys.readouterr().out
    assert "no provider contacted" in output.lower()
    assert "ELEVENLABS" not in output

    with pytest.raises(SystemExit) as error:
        probe._parser().parse_args(["--help"])
    assert error.value.code == 0
    help_output = capsys.readouterr().out
    assert "--allow-provider-contact" in help_output
    assert "no secret" not in help_output.lower()


def test_incompatible_local_configuration_is_explicit_and_unattempted(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        probe,
        "_load_configuration",
        lambda: (_ for _ in ()).throw(probe.LocalConfigurationError()),
    )

    assert probe.main(["--allow-provider-contact"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "local_configuration"
    assert len(result["arms"]) == 6
    assert all(arm["outcome"] == "unattempted" for arm in result["arms"])


@pytest.mark.asyncio
async def test_six_sequential_arms_use_actual_reviewed_queries_and_no_audio(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []
    sockets: list[FakeSocket] = []

    async def connect(url: str, **kwargs):
        assert not sockets or sockets[-1].closed
        calls.append((url, kwargs))
        socket = FakeSocket(_session_started())
        sockets.append(socket)
        return socket

    monkeypatch.setitem(sys.modules, "websockets", _websockets(connect))
    monkeypatch.setattr(probe, "_load_configuration", lambda: ("synthetic-secret", probe.REVIEWED_MODEL))

    output = await probe.run_probe()

    assert output["aggregate_success"] is True
    assert [arm["label"] for arm in output["arms"]] == [
        "auto-baseline",
        "auto-filtered",
        "en-baseline",
        "en-filtered",
        "ar-baseline",
        "ar-filtered",
    ]
    assert len(calls) == 6
    assert all(socket.sent == [] for socket in sockets)
    assert all(socket.closed for socket in sockets)
    for index, (url, kwargs) in enumerate(calls):
        query = _query(url)
        assert query["model_id"] == ["scribe_v2_realtime"]
        assert query["audio_format"] == ["ulaw_8000"]
        assert query["sample_rate"] == ["8000"]
        assert query["commit_strategy"] == ["vad"]
        assert query["vad_silence_threshold_secs"] == ["0.3"]
        assert query["include_language_detection"] == ["true"]
        assert "include_timestamps" not in query
        assert kwargs["extra_headers"] == {"xi-api-key": "synthetic-secret"}
        assert kwargs["max_size"] == probe.MAX_MESSAGE_BYTES
        assert "xi-api-key" not in url
        if index % 2:
            assert query["filter_background_audio"] == ["true"]
        else:
            assert "filter_background_audio" not in query
        if index < 2:
            assert "language_code" not in query
        elif index < 4:
            assert query["language_code"] == ["en"]
        else:
            assert query["language_code"] == ["ar"]
    assert all(arm["filtering_echo"] == "unknown" for arm in output["arms"])


@pytest.mark.asyncio
async def test_acceptance_is_distinct_from_open_and_echo_is_honest(monkeypatch) -> None:
    sockets = [FakeSocket(_session_started()), FakeSocket()]  # second never sends acceptance

    async def connect(*_args, **_kwargs):
        return sockets.pop(0)

    monkeypatch.setattr(probe, "ARM_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setitem(sys.modules, "websockets", _websockets(connect))
    accepted = await probe.run_arm(probe.Arm("auto-baseline", "", False), "secret")
    open_only = await probe.run_arm(probe.Arm("en-baseline", "en", False), "secret")

    assert accepted["outcome"] == "accepted"
    assert accepted["filtering_echo"] == "unknown"
    assert open_only["outcome"] == "timeout"
    assert open_only["outcome"] != "accepted"


@pytest.mark.asyncio
async def test_named_error_is_redacted_and_not_retried(monkeypatch) -> None:
    calls: list[str] = []

    async def connect(url: str, **_kwargs):
        calls.append(url)
        return FakeSocket(
            {
                "message_type": "invalid_request",
                "error": "PRIVATE_PROVIDER_BODY",
                "url": "wss://private.invalid",
            }
        )

    monkeypatch.setitem(sys.modules, "websockets", _websockets(connect))
    monkeypatch.setattr(probe, "_load_configuration", lambda: ("SECRET_VALUE", probe.REVIEWED_MODEL))
    output = json.dumps(await probe.run_probe())
    assert "PRIVATE_PROVIDER_BODY" not in output
    assert "private.invalid" not in output
    assert "SECRET_VALUE" not in output
    assert len(calls) == 6
    result = json.loads(output)
    assert all(arm["outcome"] == "provider_error" for arm in result["arms"])
    assert all(arm["provider_error"] == "invalid_request" for arm in result["arms"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (SyntheticRemoteClose(), "remote_closed"),
        ("not-json", "malformed_event"),
        ("x" * (64 * 1024 + 1), "message_too_large"),
    ],
    ids=["remote-close", "malformed", "oversized"],
)
async def test_close_malformed_and_oversized_events_stop_arm(monkeypatch, event, expected) -> None:
    async def connect(*_args, **_kwargs):
        return FakeSocket(event)

    monkeypatch.setitem(sys.modules, "websockets", _websockets(connect))
    result = await probe.run_arm(probe.Arm("auto-baseline", "", False), "secret")
    assert result["outcome"] == expected


@pytest.mark.asyncio
async def test_unknown_flood_is_bounded_and_does_not_print_event_name(monkeypatch) -> None:
    async def connect(*_args, **_kwargs):
        return FakeSocket(*({"message_type": "private_unknown_event"} for _ in range(20)))

    monkeypatch.setitem(sys.modules, "websockets", _websockets(connect))
    result = await probe.run_arm(probe.Arm("auto-baseline", "", False), "secret")
    assert result["outcome"] == "unknown_event_limit"
    assert "private_unknown_event" not in json.dumps(result)


@pytest.mark.asyncio
async def test_timeout_retains_cancelled_recv_child_until_released(monkeypatch) -> None:
    socket = StalledSocket()

    async def connect(*_args, **_kwargs):
        return socket

    monkeypatch.setattr(probe, "ARM_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setitem(sys.modules, "websockets", _websockets(connect))
    owner = probe.CleanupOwner(set())
    started = time.monotonic()
    result = await probe.run_arm(
        probe.Arm("auto-baseline", "", False), "secret", owner=owner
    )
    assert time.monotonic() - started < 0.5
    assert result["outcome"] == "timeout"
    await asyncio.wait_for(socket.recv_cancelled.wait(), 0.5)
    assert owner.tasks
    socket.release.set()
    await probe._drain_cleanup(owner)
    assert not owner.tasks


@pytest.mark.asyncio
async def test_timed_out_connector_eventually_returning_socket_is_closed(monkeypatch):
    socket = FakeSocket(_session_started())
    release, cancelled = asyncio.Event(), asyncio.Event()

    async def connect(*_args, **_kwargs):
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return socket

    monkeypatch.setattr(probe, "ARM_TIMEOUT_SECONDS", 0.01)
    owner = probe.CleanupOwner(set())
    try:
        result = await probe.run_arm(probe.arms()[0], "secret", connect, owner=owner)
        assert result["outcome"] == "timeout"
        await asyncio.wait_for(cancelled.wait(), 0.5)
    finally:
        release.set()
        await probe._drain_cleanup(owner)
    assert socket.closed, "a late socket must be disposed, not just its task result consumed"
    assert not owner.tasks


@pytest.mark.asyncio
async def test_overall_deadline_preserves_completed_and_marks_only_remaining(monkeypatch):
    sockets = []

    async def connect(*_args, **_kwargs):
        socket = FakeSocket(_session_started()) if not sockets else FakeSocket()
        sockets.append(socket)
        return socket

    monkeypatch.setitem(sys.modules, "websockets", _websockets(connect))
    monkeypatch.setattr(probe, "_load_configuration", lambda: ("secret", probe.REVIEWED_MODEL))
    monkeypatch.setattr(probe, "TOTAL_TIMEOUT_SECONDS", 0.03)
    monkeypatch.setattr(probe, "CLOSE_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(probe, "ARM_TIMEOUT_SECONDS", 0.05)
    result = await asyncio.wait_for(probe.run_probe(), 0.5)
    assert result["status"] == "overall_timeout"
    assert result["arms"][0]["outcome"] == "accepted"
    assert result["arms"][1]["outcome"] == "timeout"
    assert all(arm["outcome"] == "unattempted" for arm in result["arms"][2:])
    assert len(sockets) == 2 and all(socket.closed for socket in sockets)


@pytest.mark.asyncio
async def test_synchronous_close_failure_is_redacted_and_not_success(monkeypatch):
    class BrokenCloseSocket(FakeSocket):
        def close(self):
            raise RuntimeError("PRIVATE_CLOSE_FAILURE")

    async def connect(*_args, **_kwargs):
        return BrokenCloseSocket(_session_started())

    result = await probe.run_arm(probe.arms()[0], "secret", connect)
    assert result["outcome"] == "cleanup_failed"
    assert "PRIVATE_CLOSE_FAILURE" not in json.dumps(result)


def test_main_unexpected_local_failure_returns_fixed_json(monkeypatch, capsys):
    async def broken_probe():
        raise RuntimeError("PRIVATE_UNEXPECTED_FAILURE")

    monkeypatch.setattr(probe, "run_probe", broken_probe)
    assert probe.main(["--allow-provider-contact"]) == 1
    captured = capsys.readouterr()
    assert "PRIVATE_UNEXPECTED_FAILURE" not in captured.out + captured.err
    assert json.loads(captured.out)["status"] == "probe_failed"


@pytest.mark.asyncio
async def test_close_deadline_aborts_socket_and_releases_child(monkeypatch):
    release = asyncio.Event()

    class Transport:
        aborted = False

        def abort(self):
            self.aborted = True
            release.set()

    class SlowClose(FakeSocket):
        transport = Transport()

        async def close(self):
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            self.closed = True

    socket = SlowClose(_session_started())

    async def connect(*_args, **_kwargs):
        return socket

    owner = probe.CleanupOwner(set())
    monkeypatch.setattr(probe, "CLOSE_TIMEOUT_SECONDS", 0.01)
    try:
        result = await asyncio.wait_for(
            probe.run_arm(probe.arms()[0], "secret", connect, owner=owner), 0.5
        )
        assert result["outcome"] == "cleanup_failed"
        assert socket.transport.aborted
    finally:
        release.set()
        await probe._drain_cleanup(owner)
    assert socket.closed and not owner.tasks
