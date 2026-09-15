# T049-A focused correction brief

Use owner's GPT SOL/high in the existing CLI. Queue fairness, shared provider
admission and transactional SMS result handling need careful code changes and
real database acceptance. Read `T049-A-review.md` and the original T049-A brief.
Required branch `codex/phase-3-safe-booking`, HEAD
`424ee6ada7244bf7b8f89b2e3fbb50255b77a1de`. Preserve all implementation and
untracked work. No commits, pushes, installation, deployment, server access,
real Graph/SMS calls, task checkbox edits, or further delegation.

Correct the four production findings and one test error in this single pass.
Do not rewrite unrelated booking/dialogue/provider mutation behavior.

## Scope

notification_worker.py, notification_outbox.py, removal_reconciliation.py,
provider_observations.py, booking_readback.py; a small shared provider-admission
module if cohesion requires it; unapplied0004 SQL/runner/contract only if required;
notification/migration/startup tests and relevant new module image sentinel;
automatic-notifications.md and T049-A-result.md. Other scope needs lead review.
Do not edit `.repo-review/t049a_review_probes.py` to make it pass. Promote its
behavior into permanent tests with the original expectations; its local harness
is a review aid, not production API. Earlier0001-0003/bootstrap bytes are immutable.

## Corrections and required evidence

1. Fix accepted-result SQL parameter typing. Existing real PostgreSQL worker
   acceptance and cancellation-serialization tests must pass. Keep message ID,
   accepted_at and accepted state atomic. Unknown submissions stay non-retryable.
   No success fallback for failed DB persistence; no change to SMS carrier contract.

2. Add persistent retry eligibility/fair ordering for held jobs so ten failing
   notices cannot permanently block the eleventh healthy appointment. Maintain
   original due_at for lateness rules; do not slide reminder due times forward.
   Avoid busy-polling each held row every tick. Apply bounded per-job/provider
   backoff; preserve attempts versus actual POST attempts clearly. Do not auto
   repost unknown/expired dispatch intents or retry terminal recipient failures.
   Include real multi-tick and restart tests with >10 jobs, while other workers
   contend. Confirm the healthy later job progresses and failed jobs remain held.

3. Inventory error/contradiction must reset the missing sequence under the same
   booking/scope/version fence. Persist fixed reason for auth, malformed/incomplete
   inventory, found-ID or timeout; never treat list failure as absence. Boundaries
   include an outer tick timeout during inventory. A found ID is not proof its
   time matches: hold until exact readback rather than restoring old reminders.
   Integrate relevant negative/changed worker reads into shared evidence, not
   just successes. Stale read completion must not overwrite a newer outcome;
   worker concurrency cannot manufacture two spaced checks. Test404 -> inventory
   error/found-ID ->404 requires a new sequence, and fresh present/changed resets.

4. All provider reads in this feature must use shared request admission/backoff:
   observe, OAuth/token refresh, business check, collection and continuation pages,
   across notification and observation workers. Recheck before each new request;
   once a response supplies Retry-After, publish it before granting later requests.
   A request already admitted/in flight may finish; a later request must wait.
   Persist pauses so worker replacement honors them; retain401 refresh after pause.
   Use one consistent lock order. Do NOT simply take the observer's existing global
   lock inside the SMS worker while holding its booking lock: the observer already
   takes global then booking and that creates a deadlock. No SQL transaction across
   provider HTTP. Document the chosen ownership/admission order and test contention,
   cancellation/cleanup, hour-long Retry-After, and a pause imposed mid-inventory.
   The observer's deadline must still protect other calls and DB pool resources.

5. Repair only the drift test transaction setup at test_migrations.py:1699-1701.
   With the required transaction, the removed CHECK must still cause the expected
   SchemaCompatibilityError. Do not loosen production compatibility checks.

Add a full synthetic HTTP/real-PostgreSQL positive integration: live enrollment,
two spaced404s, correct-business complete inventory absence, automatic transition,
per-booking reminder suppression, fresh404 before notice and one accepted Telnyx
message ID persisted. Re-run to prove no second POST. Negative controls must
exercise the same real path, not mock internal reducer/enqueue/authorization calls.
Use another appointment on the same phone to prove isolation. Keep existing
rollback/unknown-send/cancellation race regressions.

## Validation and return

Run available offline tests plus Python3.11 grammar checks. If Docker/pytest are
available, run the commands from the review with T005_RUN_DOCKER_TESTS=1 against
the existing synthetic harness. Do not install packages or access the server to
compensate for missing tools. Lead will run real tests independently.

Return exact changed paths, failing-first evidence, final commands/results/skips,
historical SQL hashes and new0004 hash if changed. Explain lock order, fair retry
selection, missing-evidence ordering and acceptance persistence. State remaining
limits truthfully. Keep T049-A open pending independent review. No rollout work.
