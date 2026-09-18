# T033-G implementation result

Lead review completed afterward: see `T033-G-review.md` for direct recovery
corrections, independent Python 3.11/Linux tests, local private Arabic replay and
resource measurements. The implementation report below remains historical; its
tool availability and private-audio limitations describe the implementer's host.

T033-G is implemented locally on `codex/phase-4-arabic-voice-quality` at unchanged
HEAD `4c135d883554496a250e135368e7e6eade90cc7a`. Nothing was committed, pushed,
deployed or sent to a live provider.

## Implementation

- Added a bounded, callback-split-independent speech policy with five-frame
  (160 ms) starting evidence, 320 ms onset retention, monotonic input-gap reset,
  fixed-category fail-closed behavior and exact current-callback tail retention.
- Added a checksum-verified Silero VAD v6.2.1 ONNX adapter using a shared CPU
  session and per-window context/recurrent state.
- Bound enabled gate state to the exact conversation context and playback owner.
  Accepted audio is sent after owned clear and STT reset; audio arriving during
  reset is bounded and drained in order. Replacement calls and late results fail
  ownership checks.
- Preserved the legacy path when disabled and added a default-false Compose flag
  plus bounded settings. ONNX Runtime is pinned exactly; no startup download is
  used.
- Kept existing diagnostics schemas unchanged and documented their enabled-mode
  limitation.

## Failing-first evidence

Before the production modules existed:

`C:/Python314/python.exe -m pytest -q tests/conversation/test_speech_activity_gate.py`

failed during collection with
`ModuleNotFoundError: services.conversation.speech_activity_gate`.

## Validation

Development dependencies were installed only in `.repo-review/t033g-venv`.
The required `C:/Python311/python.exe` and Docker CLI are unavailable on this host,
so equivalent offline runs used the isolated Python 3.14 environment.

- New policy, callback ownership and real-model tests: **22 passed**.
- Existing STT, TTS, telephony and conversation regression set: **341 passed**.
- Conversation-only set with isolated dummy configuration: **175 passed**.
- Syntax compilation of all changed Python modules/tests: passed.
- Real model: generated silence, tones, mixture and seeded noise rejected; pinned
  public upstream speech fixture detected; corrupt and absent models rejected.
- Protected preservation: all 65 normalized protected file hashes unchanged;
  all four booking method AST segment hashes unchanged.
- Text whitespace equivalent check passed. The unchanged upstream MIT license is
  intentionally byte-identical and has no terminal newline.
- The Git executable is unavailable, so `git diff --check` could not run; the
  equivalent trailing-whitespace scan over every changed text file passed.

One preceding combined run reported **339 passed, 1 failed** when the existing
`test_receiver_and_close_delays_remain_tracked_after_disconnect` exceeded its
0.2-second test deadline. The unchanged test passed immediately in isolation and
the complete combined command then passed all 341 tests.

The two historical release-manifest assertions fail exactly as the assignment
predicts because they pin earlier runtime hashes:

- voice release manifest: expected old `orchestrator.py` hash;
- diagnostics release manifest: expected old `config/settings.py` hash.

Their assertions and manifests were not changed.

The unfiltered full-suite command cannot be completed in this host environment.
Its current collection stops first because the interrupted development-only
dependency installation did not leave `twilio` installed. Python 3.11 is absent,
and the available Python 3.14 also cannot import the repository's pinned
SQLAlchemy 2.0.25 (`TypingOnly` assertion). The scoped runtime regression command
does not import those unavailable/incompatible paths and passed.

## Model and measurements

- Revision: `7e30209a3e901f9842f81b225f3e93d8199902b1`
- Model SHA-256:
  `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`
- License: MIT, packaged unchanged
- Runtime: `onnxruntime==1.30.0`, CPU only, one intra-op and one inter-op thread
- Warmed inference wall time: 0.508 ms mean, 0.616 ms p95, 93.419 ms maximum
  scheduler outlier
- Measured session RSS delta: 18,776,064 bytes (17.9 MiB); approximately 71.6 MiB
  across four independent workers if fully resident and unshared
- Per-call numeric state: 1,152 bytes, plus bounded audio buffers and object
  overhead

## Remaining limitations

- The owner's exact private music and approved private Arabic normal/quiet files
  were unavailable, so neither was rerun locally.
- Nearby speech and vocal music can still be classified as speech. This is voice
  activity detection, not speaker identity or transcript validation.
- Linux/Python 3.11 CPU and RSS measurements remain for the release rehearsal.
- The feature stays disabled by default. A separate reviewed release assignment
  is required before any owner reproduction or production enablement.

## Cleanup

The task-specific virtual environment, official-source download copy, measurement
scripts, preservation helper/baseline and fixture scratch file were removed after
validation. No task Python process remains. Docker is unavailable on this host, so
no container, image, volume or network was created.
