# T033-G independent lead review

## Decision

Accepted locally as a default-off speech-aware beta candidate after the corrections
below. This does not establish rejection of the owner's unrecorded music, nearby
speaker identity, echo cancellation or complete voice acceptance. No publication,
server change or provider contact occurred. HEAD remains
`4c135d883554496a250e135368e7e6eade90cc7a` on phase4.

## Findings and direct corrections

The 22 delivered focused tests passed with Python 3.11 once the exact runtime was
installed in an isolated review venv. Three additional lead probes failed against
the delivered code:

1. STT reset failure after playback was removed left context SPEAKING with no
   active playback. Later audio was discarded forever by the gate path.
2. Cancelling the callback during reset left the same stranded state.
3. A classifier-state reset exception after an input gap escaped unsanitized and
   did not retire the gate.

Command: `C:/Python311/python.exe -B -m pytest -q .repo-review/test_t033g_lead_probes.py`
Before correction: 3 failed. After correction: 3 passed.

Lead added synchronous, exact-context/STT/handler/state-lock/active-playback guarded
recovery to LISTENING when an incomplete interruption flow releases ownership.
This does not fabricate a connected STT transport or revive a replacement call;
it prevents stale SPEAKING state from suppressing all subsequent input. Existing
transport failure handling still applies. Input-gap reset failure now retires the
gate with a fixed category. Cancellation during awaited classification retires
partial evidence before propagating cancellation.

Six permanent regressions cover clear/reset/forward failures, cancelled reset,
input-gap reset failure and cancelled classification. Real-model negative tests
now assert the classifier remained healthy: detector failure/overload can no
longer masquerade as successful acoustic rejection. Lead also added the explicit
`models/vad/* -text` Git attribute to preserve exact model/provenance/license bytes at publication.

## Validation

- Initial normal Python 3.11 focused run: 20 passed, 2 failed because ONNX Runtime
  was absent. No runtime defect inferred from that setup failure.
- Isolated review venv: Python 3.11, ONNX Runtime 1.30.0, existing NumPy 1.26.4.
  Delivered focused suite then passed 22/22.
- Delivered speech regression suite: 341 passed, two known warnings.
- Corrected focused suite plus original three lead probes: 31 passed.
- Corrected full suite: **579 passed, 51 skipped, 6 failed**, two known warnings.
  All six failures belong to historical release assets pinned to older hashes:
  four in `test_barge_in_diagnostics_release.py` and two in `test_voice_release.py`.
  Their guards remain unchanged. Do not describe the entire suite as green or
  repeat the implementer's claim of only two failing assertions.
- Actual local Linux container: Python 3.12.14 (the existing image lineage),
  ONNX Runtime 1.30.0 and NumPy 1.26.4. Corrected focused/model suite: **28 passed**.
  Container had network disabled and synthetic settings; no app lifespan or
  credentials were used. This is local Linux compatibility, not a server rehearsal.
- `git diff --check` passed, with only normal CRLF notices.

The task-specific review environment is `.repo-review/t033g-review-venv` and is
retained for subsequent lead release tests. Global Python packages were unchanged.
The temporary Linux validation image/container are removed after measurement;
no live provider or server was accessed. Build cache may remain in Docker.

## Real private Arabic evidence

Existing owner-approved raw mu-law files were read locally, with SHA-256 matched
against their conversion inventory. No upload, new recording or transcript read
was required. Default 0.5/160 ms/20 ms settings, 160-byte callbacks:

| Clip | First trigger position in clip | Retained onset bytes | Failures |
| --- | --- | --- | --- |
| Normal Arabic | 2.22 s | 1376 | none |
| Quiet Arabic correction | 1.48 s | 1344 | none |
| Distant Arabic voice | 5.80 s | 1344 | none |
| Mixed roleplay | 1.64 s | 1344 | none |

These positions include original silence and are NOT measured speech-onset
latencies. The distant and mixed results show the intended-speaker limitation.
The exact high-volume music used in the phone reproduction was not recorded.

A separate actual-callback probe with the REAL model rejected a generated 440 Hz
tone with zero clears/resets/forwarded bytes, then accepted quiet Arabic with one
clear, one reset and 1344 onset bytes forwarded. This exercises integration rather
than only classifier predictions. All 256 mu-law decoding entries matched Python
3.11 audioop's independent reference.

## Local Linux resource measurements

500 warmed 32 ms frames, one process, CPU only:

- mean 0.233 ms, p95 0.424 ms, maximum 0.888 ms;
- model/runtime initialization RSS increase 58,298,368 bytes (~55.6 MiB);
- four independent worker deltas would total roughly 222.4 MiB before call state
  and existing application load. This is a conservative planning estimate, not
  measured DigitalOcean capacity or a four-worker load test.

Use the Linux measurement for rollout capacity planning, not only the smaller
Windows implementer measurement. Rehearsal must check the actual image and available
memory before cutover. Model digest and public fixture remain unchanged.

## Next

User-mediated SOL/high T033-H packages the frozen corrected candidate, exact ONNX
runtime/model, default-off fallback and protected settings overlay for the current
4c135d8 server. No new voice behavior in that assignment. Owner phone testing comes
after independent release review and a separately published pinned command.
