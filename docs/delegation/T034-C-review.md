# T034-C independent review and release acceptance

## Post-publication preflight correction

Owner execution of2d6d23f stopped at preflight in
`/opt/ai-phone-voice-release.gpq1nhql`, before any cutover marker. The lead found
a deterministic packaging defect: SOURCE_SCOPE still equalled all8 packaged
runtime paths, while the actual Git diff from deployed2f7e15e to2d6d23f contains
only orchestrator.py. verify_source therefore rejects the intended release before
capacity/settings/build/cutover. This was the lead's packaging error; the prior
mock mirrored RUNTIME_PATHS and hid it, and image rehearsal did not exercise this
host Git-history check. No private server logs were requested or read.

Corrected SOURCE_SCOPE to exactly orchestrator.py, retaining all8 asset hash checks
and rejecting changes to other voice files. Added a real temporary Git-history
regression: baseline checkout plus nginx hotfix, prior8-file voice release,
one-file candidate, then extra STT change. It accepts only the one-file candidate.
Both the corrected mock fixture and real-history test failed before the fix and
passed afterward. Focused release suite21 passed/1 Docker skip. Runtime, tests
inside the candidate image, manifest and provider configuration are unchanged;
repeating image rehearsal is unnecessary for this source-scope-only correction.
The stopped release directory is retained. All normal preflight guards still run
on the next owner attempt; no live server success is inferred from local tests.
Final normal full suite:392 passed,50 optional skips,2 existing warnings.

Baseline: phase4 HEAD `2f7e15e1b38f051615fc63c11f7bf6a66d827ad9`.
Owner/server still runs that voice release until a separate manual rollout.

## Standards review

The submitted interruption test simulated speech completion and reset flags,
leaving the real audio callback/interruption/playback boundary unproven. The lead
added a synthetic-audio regression using the real RMS decoder/gate, audio callback,
speech cancellation, Twilio playback and serial transcript consumer. It verifies
that interrupted Hussam dispatch is aborted and the queued Rami correction reaches
exactly one lookup. Provider boundaries remain synthetic. All tasks/streams and
handlers are closed by finally blocks in this added test.

SOL correctly reported that its environment could not execute behavioral tests.
The lead supplied independent execution; no missing-tool result is counted as a pass.

## Spec review and direct lead corrections

1. Initial focused run:12 failed/33 passed. The test factory defaulted to None,
   causing the tool stub to emit JSON null for normal scenarios. Corrected the
   factory to use its existing explicit default sentinel. Focused45 then passed.
2. Reproduced stale pending authorization: a new availability request awaited
   acknowledgement before `_check_booking` could clear the old proposal. Failed
   or interrupted acknowledgement retained the obsolete proposal. Clear it in
   the dispatch branch before the new await; all four booking methods stay intact.
3. Reproduced stale fallback EOF: call end/replacement after the last streamed
   chunk, before EOF, bypassed chunk-only guards. Added post-loop session check
   before appending response history/returning final text.
4. Established the availability session snapshot before acknowledgement, so an
   exception at that new boundary can also fence fallback after teardown. Added
   a synthetic exception regression (defensive boundary test, not a live incident).

All three added behavioral probes failed before the respective fixes and passed
afterward. The EOF test initially used an incorrect test-only history attribute;
that setup error was corrected before its behavioral red result was recorded.

Final focused acknowledgement/playback/booking guards:49 passed, two existing
optional ffmpeg/audioop warnings. All four booking method AST source hashes equal
HEAD and the values recorded in T034-C-result.md. Python3.11 syntax and whitespace
checks pass. No STT/TTS/Twilio/provider settings, SQL or notification code changed.

## Release packaging

Reused the accepted app-only release procedure, pinned to the actual previous
voice source/image from the owner console. Added a new T034-C manifest with26
assets, including the new regression module. Historical T034-B manifest remains
unchanged (SHA256 710960d4778877451287e9f816d88884d32e85bd2af45931a6ce7dc4dfded66a).
The release procedure retains exact settings/private mount preservation, schema0004
read-only gates, ingress control, candidate testing and old-image recovery.

Initial full run had two expected old-manifest failures; new manifest resolves
the intentionally changed orchestrator fingerprint. Updated the asset-count test
from25 to26. Final normal full suite:391 passed,50 optional skips,2 old warnings.

Explicit local Docker run: real candidate test/image checks and synthetic Compose
replacement/rollback passed. That invocation also ran the old already-loaded25
count assertion, resulting in20 passed/1 failed; the corrected offline release
suite is verified separately. No server/provider contact. The local rehearsal uses
a locally available synthetic base, not the server's private image/configuration;
the exact server baseline is checked by the owner-run release preflight.

Corrected offline release suite:20 passed/1 Docker skip. Independent cleanup
inventory found no containers, only default bridge/host/none networks, and no
temporary T034 image tags. Full-suite optional skips are not live-provider evidence.
Clean HEAD Git archive plus only the seven selected release files:391 passed,
50 optional skips,2 existing warnings. Temporary clean checkout removed. This
checks that untracked planning/tooling/fixtures are not hidden release dependencies.

## Limits

Telephone pronunciation/timing remain owner smoke checks after deployment.
Acknowledgement is synchronous and generic; it does not announce unverified names
or dates. Concurrent correction intake during a running lookup remains open.
Background speech filtering is NOT part of this release. Booking/public-launch
P0s remain in the existing roadmap. Do not rerun the old 2f7e15e rollout command.
