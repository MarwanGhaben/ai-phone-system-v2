# T033-H speech-gate release implementation result

Implemented locally on `codex/phase-4-arabic-voice-quality` at unchanged HEAD
`4c135d883554496a250e135368e7e6eade90cc7a`. Nothing was committed, staged,
pushed, deployed or sent to the server/provider. Detector code, thresholds,
dependencies, old release files, migrations, manifests, task checkboxes and all
51 frozen assets remain unchanged.

## Four-file implementation

- `scripts/deploy-speech-aware-barge-in.py` is an import-safe, root/40-hex/lock
  protected app-only release. It validates the exact deployed source/image,
  checkout/nginx hotfix, private override, complete mount/environment semantics,
  Redis/database/schema 0004, diagnostics-on/filter-off settings and public TLS.
- `tests/integration/test_speech_gate_release.py` supplies failing-first release,
  integrity, settings, package, resource, cutover/recovery, cleanup/redaction and
  opt-in Docker rehearsal contracts.
- `docs/operations/T033-H-speech-gate-rollout.md` retains an intentionally unfilled
  publication placeholder and defines the bounded owner rollout and phone check.
- This result records implementation and validation without marking T033-H done.

The release stages and verifies the canonical manifest plus all 51 assets. Text
groups receive only CRLF-to-LF normalization; all three `model_assets` entries are
hashed and staged as exact raw bytes. The candidate derives from a verified local
tag of the exact running image and copies only `config/settings.py`, the
orchestrator, the local VAD/gate modules and the three model assets. Eight protected
runtime hashes remain exact.

Before selecting wheels, a network-disabled one-shot probe checks the actual image
is CPython 3.12/Linux/x86_64. The release then downloads exact official package-host
wheels and rejects any filename or digest difference:

| Package | Wheel SHA-256 |
| --- | --- |
| onnxruntime 1.30.0 | `fa688e7891a6aa206636fe7372e27ee75fd17713289f6b4fc7b190e0a7de9328` |
| flatbuffers 25.12.19 | `7634f50c427838bb021c2d66a3d1168e9d199b0607e6329399f04846d42e20b4` |
| protobuf 7.36.1 | `97198b77e369a0abd8e262b8f6c7266c55ddb796a3a12c76d7b8881188ed83aa` |

The candidate build is network-disabled and installs those wheels with
`--no-index --no-deps --only-binary=:all:`. Complete distribution inventories must
equal the baseline plus exactly those three versions; NumPy must remain 1.26.4 and
every other distribution must remain unchanged. No host/global pip install occurs
in the release.

The candidate test container is network-disabled/read-only and mounts only frozen
synthetic tests, scripts, Compose support and model assets. It runs all four speech
test directories plus `tests/evaluation/test_local_vad_audio.py`. A separate
one-shot effective-mount probe verifies every runtime/model hash, exact settings
intersection/additions, exact package inventory and successful `LocalVadModel`
construction without application lifespan, workers or provider clients. A bounded
500-frame warm probe records numeric load RSS and mean/p95/max time, rejects p95
above the 20 ms budget, and requires available memory for four measured model
deltas plus a 384 MiB coexistence reserve before calls stop.

The candidate override adds exactly the six manifest values and replaces only the
nested read-only `/app/config/settings.py` overlay. Cutover uses app-only
`--no-deps --no-build --pull never` replacement. Unit failure injection at stop,
override installation, candidate replacement/readiness and TLS restores the exact
old override/image state; failed recovery closes ingress and never prints success.

## Failing-first and validation evidence

The new release-contract test was created first. Its initial standard-library run
failed during collection with `FileNotFoundError` for the not-yet-created
`scripts/deploy-speech-aware-barge-in.py`, establishing the missing release package.

After implementation:

```text
Bundled Python 3.12.14:
python -B -m unittest tests.integration.test_speech_gate_release
Ran 25 tests in 1.703s
OK (skipped=1)

Python 3.12.14 disposable venv:
python -B -m pytest -q tests/integration/test_speech_gate_release.py
24 passed, 1 skipped in 2.95s

python -B -m pytest -q tests/conversation/test_speech_activity_gate.py \
  tests/conversation/test_speech_aware_barge_in.py \
  tests/evaluation/test_local_vad_audio.py
28 passed, 1 warning in 12.64s
```

The broader frozen speech/model command executed 350 cases: 349 passed and one
failed. The failure is the unchanged
`test_connection_and_close_timeouts_are_bounded_and_logs_are_safe`; its first
synthetic connection exceeded the test's 10 ms Windows deadline and returned
false. An isolated retry reproduced the same setup timeout and pending fake-socket
cleanup. T033-H does not change that frozen STT runtime/test. This is reported as
an environment-specific pre-existing test-harness failure, not hidden as a green
suite and not repaired by weakening a frozen guard.

The full working-tree run completed with **600 passed, 53 skipped and 8 failed**.
Six failures are the documented historical manifest guards: four in
`test_barge_in_diagnostics_release.py` and two in `test_voice_release.py`. The
other two are frozen Windows timing assumptions: the 10 ms STT connection setup
above and `test_overall_deadline_preserves_completed_and_marks_only_remaining`,
whose 30 ms provider-probe deadline completed before its expected timeout on this
host. No T033-H release test failed. The extra skip relative to the recorded
Python 3.11 baseline is environment-specific; Docker remains unavailable.

Independent worktree audit:

```text
canonical manifest SHA-256:
4d1aeb6f59b5291c88094493457edcd11806d49cddd56a4f1b7e129c489390a4
asset entries: 51
hash mismatches: []
```

Both new Python files compile under Python 3.14. The opt-in rehearsal remains
gated by `T033_H_RUN_DOCKER_TESTS=1`; this host has no Docker CLI/engine or usable
WSL distribution, so the real Linux derived-image/replacement rehearsal was not
run and is not claimed. Its harness reconstructs the deployed 4c135d8 source over
an existing local dependency image, uses the real pinned wheels/model, runs the
candidate tests and measurements, verifies package preservation, replacement and
exact old-image rollback, then checks that named containers/images were removed.

## Scope and remaining gates

No real server capacity, package state, private mounts, nginx, TLS, four-worker
load or phone behavior has been tested by this local implementation. Actual music
suppression and intended-speaker identity remain unverified. The release runbook
therefore contains no usable commit and no deployment occurred. Independent lead
review, Linux Docker rehearsal and later publication are required before the owner
receives a pinned command.
# Lead follow-up

The implementation evidence below is historical. Independent lead review fixed
the unnecessary optional SymPy requirement and the Windows Docker driver's Linux
memory read. The real Linux candidate and exact image/settings rollback now pass:
26 release tests, including the opt-in rehearsal. See
[T033-H-review.md](T033-H-review.md) for full-suite limitations and release scope.
