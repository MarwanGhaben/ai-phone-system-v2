"""Reply-length repair applied to the exact deployed restored orchestrator."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHANGED_LINES = {2809: (80, 220), 2857: (60, 160), 2915: (60, 160),
                 2955: (80, 220), 3048: (50, 120), 3089: (80, 220)}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reply():
    return _load("restored_reply_builder", "scripts/build_restored_reply_repair.py")


@pytest.fixture(scope="module")
def baseline():
    availability = _load("restored_availability_builder_reply", "scripts/deploy-restored-availability.py")
    turn = _load("restored_turn_builder_reply", "scripts/build_restored_turn_repair.py")
    original = subprocess.check_output(
        ["git", "-c", "safe.directory=*", "show",
         availability.RESTORE + ":services/conversation/orchestrator.py"], cwd=ROOT)
    return turn.patch_orchestrator(availability.patch_orchestrator(original))


def _digest(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def test_deployed_baseline_reconstructs_to_the_pinned_hash(reply, baseline):
    assert _digest(baseline) == reply.BASELINE_SHA
    assert len(baseline.decode().splitlines()) == 3854


def test_only_the_six_reply_cap_lines_change(reply, baseline):
    patched = reply.patch_orchestrator(baseline)
    old = baseline.decode().split("\n")
    new = patched.decode().split("\n")
    assert len(old) == len(new)
    changed = {i + 1: (a, b) for i, (a, b) in enumerate(zip(old, new)) if a != b}
    assert sorted(changed) == sorted(CHANGED_LINES)
    for number, (before, after) in CHANGED_LINES.items():
        assert changed[number][0].strip() == f"max_tokens={before},"
        assert changed[number][1].strip() == f"max_tokens={after},"
        assert changed[number][0].replace(str(before), str(after)) == changed[number][1]


def test_result_compiles_and_has_no_old_reply_caps(reply, baseline):
    patched = reply.patch_orchestrator(baseline)
    compile(patched, "patched", "exec")
    text = patched.decode()
    start = text.index("    async def _get_ai_response(")
    end = text.index("    async def _speak_to_caller(")
    body = text[start:end]
    for old in ("max_tokens=80,", "max_tokens=60,", "max_tokens=50,"):
        assert old not in body
    assert body.count("max_tokens=220,") == 3
    assert body.count("max_tokens=160,") == 2
    assert body.count("max_tokens=120,") == 1


def test_the_separate_greeting_call_is_untouched(reply, baseline):
    patched = reply.patch_orchestrator(baseline).decode().split("\n")
    assert patched[707].strip() == "max_tokens=60,"
    assert baseline.decode().split("\n")[707] == patched[707]


def test_crlf_input_gives_the_same_result(reply, baseline):
    assert reply.patch_orchestrator(baseline.replace(b"\n", b"\r\n")) == reply.patch_orchestrator(baseline)


@pytest.mark.parametrize("mutate", [
    lambda b: b + b"# unreviewed\n",
    lambda b: b.replace(b"max_tokens=80,", b"max_tokens=81,", 1),
    lambda b: b"unexpected production source",
    lambda b: b"",
])
def test_modified_input_is_refused(reply, baseline, mutate):
    with pytest.raises(RuntimeError):
        reply.patch_orchestrator(mutate(baseline))


def test_anchor_drift_with_matching_hash_is_refused(reply, baseline, monkeypatch):
    # Even a source that somehow passes the hash gate cannot be half-patched.
    changed = baseline.replace(b"max_tokens=50,", b"max_tokens=51,")
    monkeypatch.setattr(reply, "BASELINE_SHA", _digest(changed))
    with pytest.raises(RuntimeError, match="token caps"):
        reply.patch_orchestrator(changed)
