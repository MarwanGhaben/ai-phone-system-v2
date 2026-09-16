# T034-B lead acceptance — 2026-09-16

Accepted for the first bounded voice beta rollout after independent review and
local corrections. Server execution and telephone quality are separate gates.

## Independent evidence

- Initial full suite:361 passed,50 optional skips,2 existing local warnings.
- The local Docker engine was stopped; restarted the existing Desktop engine.
  No server service was accessed or changed.
- Actual derived Linux/Python3.12 image exposed three timing assumptions in the
  accepted TTS tests: first_chunk means queued, not consumed. Cancelling at that
  point correctly could yield zero delivered chunks. Windows had happened to
  consume first. Tests now wait on an explicit consumer event before cancellation,
  retain every expected-audio assertion, wait for all tested stream closures,
  and release gates/join tasks in finally blocks. No runtime TTS change was made.
- A new failing-first readiness regression proved HTTP-ready could return while
  Docker health was still starting. The release wait now requires both within its
  existing bounded loop, avoiding a false post-start rejection and rollback.
- Synthetic harness now retains stdout failure evidence as well as stderr, and
  cleanup inventory commands must succeed before absence counts as verified.
- Runbook command template now runs in a fail-fast bash/pipefail block.
- Final explicit Docker release suite:21 passed in35.98s. Actual candidate image
  hashes/imports and163 speech/booking tests passed. Synthetic Compose replaced
  old with candidate then restored old, preserving mount/environment contracts.
- Final normal full suite:362 passed,50 optional skips,2 existing warnings in12.16s.
- Clean baseline Git archive plus ONLY selected release files:362 passed,
  50 optional skips,2 existing warnings in13.96s. This verifies untracked planning,
  local tooling and other workspace files are not hidden release dependencies.
- Independent Docker inventory after rehearsal: no containers; only bridge/host/
  none networks; no temporary T034 image tags. Existing unrelated images retained.
- `git diff --check` passed. Eight runtime and17 test manifest entries match.

The local warnings remain optional ffmpeg discovery and audioop deprecation.
The Linux image reports dateutil/audioop deprecations. These are not new failures.

## Manifest amendment by the lead

All eight accepted runtime hashes remain unchanged. Only the test hash for
`tests/tts/test_call_isolation.py` was updated to reflect deterministic consumer
and closure synchronization. This supersedes the original implementer manifest
preservation claim; it is not an unreported production edit.
Final manifest SHA-256:
`710960d4778877451287e9f816d88884d32e85bd2af45931a6ce7dc4dfded66a`.

## Deployment boundary

The server still runs notifications source3e1a749/schema0004 until the owner runs
the new reviewed commit. The release script uses that exact server image as parent,
preserves its settings/private mounts and performs read-only0004 checks; no
migration/restore or changed notification policy. Local rehearsal used an existing
local candidate base, synthetic data and no provider access. Actual private mounts,
public TLS and live DB checks are server preflight/cutover gates, not claimed as
already exercised locally. Real Arabic accuracy and interruption acceptance await
supervised calls; concurrent turn intake remains unintegrated.
