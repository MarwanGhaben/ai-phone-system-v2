# T049-A lead review: accepted locally after corrections

## Current decision — 2026-09-13

T049-A is accepted locally on `codex/phase-3-safe-booking`, unchanged HEAD
`424ee6ada7244bf7b8f89b2e3fbb50255b77a1de`. SOL's follow-up resolved the original
five findings below. Lead independently reviewed the actual files and corrected
one further confirmation-recovery issue. All work remains unstaged/uncommitted;
server remains 424ee6a with schema 0003. No live SMS or provider calls occurred.

The earlier review below is retained as **historical failing-first evidence**;
its unaccepted/production-untouched statements describe that earlier review only.

### Additional lead correction

A held job is dispatchable by the worker, so being held alone does not prove that
every new confirmation is blocked. The reproducible defect was recovery after an
outage: a due confirmation with a future exponential retry retained that delay
after an exact positive observation, potentially expiring before dispatch.
The real-DB probe expected `(pending, eligible)` but observed `(held, ineligible)`.

Lead changed `notification_outbox.release_current_jobs` to reset valid matching
jobs after fresh presence, retaining original due times and scope/version fences.
Shared validity windows are 15 minutes for confirmations and five for reminders;
past appointments and stale jobs are not revived. The worker also suppresses a
confirmation after appointment start. A permanent real-DB regression verifies
fresh-presence recovery through one accepted synthetic SMS, repeated-run no-send,
expired confirmation/reminder suppression and past-appointment suppression.
No SQL bytes changed in this lead correction.

### Independent final evidence

- Corrected SOL baseline, real PostgreSQL + notification/startup/dashboard suites:
  **82 passed, 1 optional Compose skip, 231.11s**. This run preceded the additional
  lead correction; the affected tests were rerun below after that correction.
- Final normal full suite: `C:/Python311/python.exe -m pytest -q --tb=short`:
  **183 passed, 47 opt-in skips, 2 existing warnings, 7.90s**. Warnings remain
  audioop deprecation and missing local ffmpeg; no speech acceptance is claimed.
- Final real 0004 regression subset with `T005_RUN_DOCKER_TESTS=1`:
  `pytest -q tests/integration/test_migrations.py -k 0004 --tb=short`:
  **12 passed, 43 deselected, 81.81s**.
- Independent real-DB review probes: **4 passed, 24.003s** (confirmation recovery,
  11-job fairness, inventory failure resetting evidence, shared provider pause).
- Explicit Docker image exclusion and synthetic Compose/startup suites with
  `T004_RUN_DOCKER_TESTS=1` and `T005_RUN_COMPOSE_TESTS=1`: **21 passed, 16.13s**.
- Actual four-worker candidate image on an internal local Docker network reached
  `/ready` and `/health` after prepare through 0004. Enabled reconciliation with
  synthetic missing credentials recorded `check_failed:configuration` and created
  zero SMS jobs; intentional 0004 ledger drift returned 503. This is startup and
  failure-handling evidence, not successful live delivery.
  Local image: `sha256:dbaab7a24155d5f5c3a6af2a8daaaba83ec6f9562069a039aa6163e651fce615`,
  tag `ai-phone-t049a-candidate:20260913`.
- Image rehearsal cleanup confirmed `t005c-image-d57d84de309b`; probe cleanup
  confirmed `t005-migrations-7cb11c2b6321450ba91bf43bde9fc917`. Final Docker listing
  had no containers and only default bridge/host/none networks.
- `git diff --check` passed; Git emitted line-ending conversion notices, not
  whitespace errors. Bootstrap/0001/0002/0003 hashes remain unchanged. Accepted
  unapplied 0004 is `a1c66c7705adbd49918966c7d41d8bbbc5e80d3d5d0534fe641fd15579731a9e`.

Next is [T049-B rollout packaging](T049-B-notification-release.md), including an
explicit paused-worker recovery mode. Simply turning the feature off after 0004
would reactivate the legacy JSON reminders and must not be used as recovery.
This is scoped local acceptance only: live removal/SMS smoke, direct Outlook
propagation, carrier delivery receipts and parent T018/T025/T048/T049 remain open.

## Historical original review

Reviewed locally 2026-09-13 on phase3, HEAD424ee6ada7244bf7b8f89b2e3fbb50255b77a1de.
SOL's implementation is present in the shared checkout. It is not accepted or
deployed. Lead changed only review/probe/planning files; production code is untouched.
Applied Clean Code Guard and Test Guard, prioritizing runtime behavior over style.

## Independent evidence

- Normal full suite: `C:/Python311/python.exe -m pytest -q`:
  **181 passed, 40 skipped, 2 existing warnings, 11.94s**. This includes the
  Arabic/Toronto tests unavailable to SOL. Warnings: audioop and local ffmpeg.
- Local Docker PostgreSQL enabled with `T005_RUN_DOCKER_TESTS=1`, running
  test_migrations.py, test_automatic_notifications.py, test_database_startup.py
  and tests/dashboard/test_dashboard_metrics.py:
  **71 passed, 3 failed, 1 optional Compose skip, 187.77s**.
  Two failures are a production SQL error; one is the drift test transaction error.
  Initial sandboxed Docker access was denied before container creation; permitted
  local Docker execution then ran the real suite. No server/provider access.
- Independent real-DB reproductions in `.repo-review/t049a_review_probes.py`:
  **3 tests, 3 failures, 19.351s**, confirming findings 2-4 below.
  Uses existing disposable PostgreSQL harness, real store/poller/worker and fake
  Graph/SMS boundaries only. The initial backoff fixture lacked observed provider
  identity, causing an unrelated failure; corrected fixture includes identity and
  asserts no internal error. Only the corrected run supports finding4.
- Test-harness cleanup completed. Latest probe container explicitly removed:
  `t005-migrations-2437134e604540e6981a91fc00b56405`.
- Tracked `git diff --check` passed. No image/release acceptance is claimed;
  additional image work is deferred until these blockers are corrected.

## Findings

### 1. P1: accepted SMS cannot be persisted

`services/sms/notification_worker.py:264-271` binds `$2` as the varchar state
assignment and also compares it to the untyped 'accepted' literal in a CASE.
Real asyncpg raises `AmbiguousParameterError: inconsistent types deduced for
parameter $2 (text versus character varying)` AFTER the SMS submission. The
accepted message ID is lost locally; the dispatch eventually becomes unknown.

Reproduced by both existing real tests:
`test_0004_dispatch_acceptance_and_uncertain_intent_are_not_reposted` and
`test_0004_caller_cancel_serializes_with_inflight_reminder`.
Use an explicit consistent SQL parameter type. Preserve intent-before-send and
unknown-no-repost behavior; do not convert a failed write into accepted locally.

### 2. P1: held removal notices starve later valid messages

`services/sms/notification_worker.py:15-19,77-98` always selects the same first
10 due jobs ordered by original due_at. Held removal notices do not expire or
advance selection priority. Ten persistent per-appointment failures block every
later confirmation/reminder/removal indefinitely.

Real probe: 11 cancellation notices, first10 return network failures, eleventh
returns404 eligible for its recorded cancellation. After three real worker ticks:
**30 reads, 0 SMS, eleventh still pending**. Add durable bounded next-attempt
scheduling/fair selection, separate due time from retry time, and preserve recovery
semantics. Increasing LIMIT does not fix starvation.

### 3. P1: inventory failures do not invalidate consecutive absence evidence

`services/scheduling/provider_observations.py:243-274` handles inventory pause
and finalization but never records a failed/contradictory inventory into the
reconciliation evidence. `removal_reconciliation.py:130-164` therefore leaves
missing_count=2 and last_result=unavailable after an authorization failure.
The next404 may reuse that sequence despite the brief requiring an error to break it.

Real probe observed `(2,'unavailable')`, expected `(0,'check_failed')` after a
qualifying404 followed by inventory authorization failure. Found-in-list and
malformed/paging/time-budget failures need equivalent conservative reset behavior.
Worker reads also currently ingest only `present` (`notification_worker.py:243`),
so contradictory error/changed results from that reader are not shared evidence.
Persist appropriately fenced evidence from all relevant reads, without counting
concurrent worker reads as independent spaced missing checks.

### 4. P1: shared Microsoft pause is bypassed by in-flight batches

The observer checks next_request_at once at tick entry (provider_observations.py:
187-193); the new SMS worker may change it mid-batch. Later observation requests
and inventory pages do not recheck it. Workers also admit their own requests
independently from the observation lock.

Real probe imposed an hour-long shared pause while the first observation was in
flight. The poller made **2 reads instead of1**, with no internal error. New request
admission must honor the shared pause across both readers, including OAuth and
inventory continuation, with a consistent lock order. Requests already in flight
cannot be unsent; do not promise otherwise.

### 5. Test oracle repair: drift call lacks transaction

`tests/integration/test_migrations.py:1699-1701` calls runtime compatibility
outside a transaction; that contract uses LOCK TABLE. It raises
`NoActiveSQLTransactionError` instead of the intended schema incompatibility.
Wrap this assertion in the same read-only transaction contract as real readiness.
Do not weaken production readiness or change it to accept the dropped CHECK.

## Acceptance status

Migration and most integration coverage ran successfully, but notification dispatch
and policy tests above block acceptance. Existing tests do not provide one complete
real-DB poller -> inferred removal -> worker -> accepted SMS scenario. Add that
positive flow plus the negative cases; a missing-module failing-first check is
not evidence for those invariants. Keep parent tasks open and feature disabled.

Next handoff: `T049-A-followup.md`, SOL/high. Preserve immutable historical SQL;
0004 has never been applied on the server and may change for the bounded fix if
its checksum/contracts/tests are updated together. Do not deploy0004 yet.
