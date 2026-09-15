# T049-B: package automatic notifications for manual rollout

Implement this bounded assignment. Recommended implementer: the owner's GPT SOL
with high reasoning in the separate CLI. This task involves database cutover,
private mounted settings and recovery without duplicate SMS. It fits the current
SOL routing lane; it is not a claim that SOL is best for every task. Lead will
independently review/test the actual files. If a prerequisite is inconsistent or
a broader runtime defect appears, return the exact blocker; do not expand scope,
switch models or rerun the whole audit. Report unavailable tools once.

## Starting state and scope

Branch `codex/phase-3-safe-booking`; required unchanged HEAD:
`424ee6ada7244bf7b8f89b2e3fbb50255b77a1de`.
Accepted T049-A files are present but uncommitted in this checkout. Z: and the UNC
share may refer to this same workspace. Preserve every unrelated/untracked file.
Read `specs/001-production-readiness/plan.md`, its linked spec/tasks, T049-A brief,
follow-up, current lead review and `docs/operations/automatic-notifications.md`.
The lead's new confirmation recovery correction is accepted; do not overwrite it.
Compare `docs/delegation/T049-A-accepted-hashes.json` before/after your work.

Create only:

- `scripts/deploy-automatic-notifications.py`
- `tests/integration/test_notification_release.py`
- `docs/operations/T049-B-notification-rollout.md`
- `docs/delegation/T049-B-result.md`

Modify only, for the narrow paused-worker setting described below:

- `config/settings.py`
- `api/main.py`
- `docker-compose.yml`
- `tests/integration/test_database_startup.py`

All other T049-A runtime/tests/SQL assets are frozen. No dependency installation,
Git staging/commit/push, task checkbox edits, server access, live provider call,
message sending or deployment. Do not invoke an existing rollout script.

## One small runtime prerequisite: safe pause

Add boolean `automatic_notification_workers_paused`, environment alias
`AUTOMATIC_NOTIFICATION_WORKERS_PAUSED`, default false. It is valid only with
`automatic_notifications_enabled=true`; retain the existing observation-enabled
requirement. Add the corresponding Compose mapping with default false.

In paused mode, lifespan still checks the complete 0004 schema and serves the
app, but starts neither ObservationPoller nor NotificationWorker. It must also
keep the legacy JSON reminder scheduler off. Leave feature mode enabled, so new
bookings still atomically create durable notification jobs and the orchestrator
does not use legacy immediate SMS. Do not suppress/delete/requeue existing jobs.
Log one fixed paused-mode message without secrets. Existing disabled/default and
enabled-unpaused lifecycle behavior stays intact. Pausing takes effect on restart;
it cannot recall messages already submitted. Test startup/shutdown and invalid
flag combinations, including absence of all three background send/poll paths.

This mode is a **degraded recovery option**, not rollback to old application
behavior: it uses the accepted 0004-capable runtime with workers paused. If that
runtime cannot become ready, keep ingress stopped and report recovery required.
Do not patch source text dynamically, monkeypatch worker methods, or reactivate
legacy messaging as a shortcut. Do not claim it recovers every application bug.

## Authoritative deployed baseline

- Server root `/opt/ai-phone-system-v2`; checkout intentionally remains phase2
  `1480eeb0e36596e26d0dedde77f08beee91aec5b` with the nginx hotfix.
- Running source label `424ee6ada7244bf7b8f89b2e3fbb50255b77a1de`.
- Running immutable image
  `sha256:2ac002d96da4541ed08e0f221bb559bc4c70d9762b3fab5735c81d92171459ea`.
- Schema history 0001/0002/0003; canonical bookings and observation data exist.
  No 0004 or notification enrollment/outbox exists on the server yet.
- Observation settings enabled=true, interval=60, freshness=180.
- Root-only private Compose override preserves actual environment, image and
  mounts. Parent config directory is mounted read-only to `/app/config`.
  `/opt/ai-phone-observation-release.apge12qe/candidate-settings.py` supplies the
  **existing nested read-only `/app/config/settings.py` mount**. A new release
  must replace that one file mount source, not append a duplicate destination.
  Preserve the parent config mount and every unrelated mount/option.
- `/opt/ai-phone-release.02W6q2/runtime.env` remains required at `/app/.env`.
  All prior release directories and protected backups must remain.
- nginx SHA-256
  `59ac9bdaa9a86c1022a573c9709faddf7e93a53015da8acfcc4330b3e288f3ab`;
  certificate-validated public health `https://aiagent.ghaben.ca:8443/health`.
  Preserve TLS files, renewal timer and configuration.
- Small server: 2 GiB RAM, no swap last reported. Verify capacity within release;
  no owner preflight loop, destructive pruning or deletion of rollback assets.
  Owner accepted on-server-only backup risk; do not ask again.

Read the current `deploy-provider-observations.py`, its tests/runbook and earlier
persistence release for understood mechanisms. Keep the prior mount-order fix,
authenticated final-TCP PostgreSQL readiness and settings-file overlay lessons.
Old source/image/0002 assumptions and empty-observation assertions do not apply.
Do not implement by chains of text replacements of executable historical source.

## Immutable source and build contract

Accept exactly one full 40-hex source commit argument; lead will later publish
the accepted application and release files together. Never invent that commit.
Read whitelisted Git blobs at that commit, not working-tree files or branch tip.
Verify baseline ancestry and scoped changes, including templates and Compose.
No full-repository COPY or dependency installation. Derive the candidate from
the exact verified running image with network disabled and no new pulls.

Candidate runtime copies (19 paths):

```
api/main.py
config/settings.py
services/database.py
migrations/runner.py
migrations/schema_contract.py
migrations/0004_appointment_notifications.sql
services/calendar/booking_readback.py
services/calendar/provider_admission.py
services/scheduling/booking_records.py
services/scheduling/provider_observations.py
services/scheduling/removal_reconciliation.py
services/sms/notification_text.py
services/sms/notification_outbox.py
services/sms/notification_worker.py
services/sms/telnyx_submission.py
services/sms/telnyx_sms_service.py
services/conversation/orchestrator.py
services/dashboard/dashboard_routes.py
templates/dashboard.html
```

The runtime image is the same for active and paused recovery, with separately
verified protected overrides. Label it with source commit; record immutable ID.
Base Compose stays unchanged on the server. Preserve all previous settings;
only three new settings are introduced: enabled=true, interval=60, paused=false
for active mode, or paused=true for rehearsal/recovery. Existing observation
settings remain unchanged. Verify actual settings inside the mounted candidate.
Allow only the exact intended image/environment/settings-file-source changes;
compare mount contents order-independently with duplicates preserved. Never just
ignore all mounts/settings differences. Preserve dollar signs/multiline values.
Keep schema-capable candidate image pinned for future explicit migrate service use.

SQL SHA-256 values must match accepted bytes and ledger constants:

- bootstrap: `1c72bbe0a120860902c565e5573c05ad1cd640891a42bb35cca2c6e0876f31e8`
- 0001: `53c861ac91cae5ecaf9f794b15563c88ee58bc0dafa5ae64fa81c89d907aa3a9`
- 0002: `62addab700e047dff2a13973c881a874bfe6c7f88967d593f189987192782be6`
- 0003: `b3abaa09c340f89c32778b0b17b5c2a584845983b13eb95dbb9c366f3d6c5830`
- 0004: `a1c66c7705adbd49918966c7d41d8bbbc5e80d3d5d0534fe641fd15579731a9e`

No SQL edits, ledger repair, down-migration or historical timestamp conversion.
Git SQL blobs retain LF bytes. Working-copy Python CRLF vs Git LF differences
alone are not drift; distinguish published-source and working-copy checks.

## Release sequence and failure behavior

1. Root/umask077, deployment lock, unique root-owned non-symlink release directory
   and protected logs. Verify baseline app label/image, health, exact 0003 history
   and contract, target database correspondence, config/override/mounts/TLS and
   capacity. Check required Graph/SMS configuration presence/format locally without
   printing values or contacting providers. Stop on missing prerequisites with
   fixed categories. No source/environment dump in public output.
2. Build bounded candidate, write protected active and paused overrides and
   settings-file overlay. Pure settings/readiness probes must not start workers
   or contact Graph/Telnyx before cutover. Snapshot previous state for recovery.
3. Fresh protected pg_dump, hash and archive index verification. Restore using
   current PostgreSQL image into resource-capped tmpfs on an internal isolated
   network, no live mounts, synthetic credentials. Wait for final TCP PostgreSQL
   with an authenticated query. Verify old-image 0003 compatibility; apply 0004
   with candidate --prepare, repeat prepare, verify exact history and contracts.
   Existing 17 business tables AND existing provider observation/control records
   must be unchanged by migration. New notification tables start empty; never
   enroll restored rows or send SMS. Exercise actual paused candidate lifespan and
   readiness on the restored copy with outbound network blocked and synthetic
   provider configuration. Always remove and verify exact temporary resources,
   including partial create/start/timeouts. Keep protected backups/logs.
4. Recheck complete current-state fingerprints immediately before stopping writers.
   Owner runbook says finish calls/pause beta testing. Mark cutover-started, stop
   ingress/app gracefully and verify stopped, final protected backup, apply 0004,
   verify schema and preserved rows. Never automatically restore over live data.
5. Activate paused candidate first, verify image/settings/0004 readiness and
   nginx/certificate-validated HTTPS. Then stop ingress/app for the explicit
   activation restart and use active override with paused=false. Verify settings,
   /ready, health, nginx and HTTPS again. Feature activation is the authorized point
   for real background provider reads and SMS. Do not make an extra test SMS or
   live Graph call in the deployment script. Provider rejection/outage is not the
   same as local deployment failure; expose worker state through normal dashboard.
6. Before migration success, old-image recovery is allowed only if exact 0003
   history/contract is independently verified. Once 0004 is committed, recover
   with the verified paused candidate; never disable the feature or restore the
   old JSON sender. After any uncertain/failed cutover, stop app/ingress and verify
   stopped before recovery, preserve committed jobs/accepted/unknown states and
   report the original error even when degraded recovery succeeds. A previously
   sent SMS cannot be undone. If schema is ambiguous or paused candidate fails
   readiness, leave ingress stopped and report a safe recovery-needed marker.
   Do not blindly rerun any release with cutover-started marker.

Success markers should identify 0004 migration, worker activation, verified
readiness/HTTPS, source commit/image and release directory. Degraded recovery
must print a distinct SMS-paused marker, never the success marker. Preserve
checkout, database/redis/certbot services, backup directories and legacy JSON file.

## Acceptance tests and return evidence

Use boundary fakes for release failure decisions and a real opt-in local synthetic
Docker test for actual release methods. Cover:

- Preflight refusal leaves live resources untouched; pinned source/copy scope.
- Existing parent+nested settings mounts, reversed mount list, missing/duplicate
  destination refusal, environment dollar signs/multiline values, active/paused
  settings comparisons; no probes against stale mounted settings.
- Nonempty 0003 aware/naive bookings and observations preserved through restored
  0004 migration/repeat-run; new tables empty before activation.
- Paused lifecycle: app ready; neither new worker/poller nor legacy scheduler starts;
  enabled booking path still durably enqueues, and no direct confirmation SMS.
- Failures before stop, migration rollback to0003, paused startup, active activation
  and health checks. Correct old/degraded/stop-ingress recovery decision by actual
  schema; no automatic database restore or accepted/unknown job requeue.
- Recovery after synthetic accepted and unknown SMS records preserves those states
  and performs zero carrier submissions. Default/active lifecycle still works.
- Cleanup on partial resources, timeouts and recovery failures. Original error
  retained; secret sentinels never appear in public output.

Local lead has Python3.11/pytest/asyncpg and working Docker CLI. Accepted local
T049-A image tag `ai-phone-t049a-candidate:20260913` has ID
`sha256:dbaab7a24155d5f5c3a6af2a8daaaba83ec6f9562069a039aa6163e651fce615`.
It does not yet include this pause setting. Local IDs differ from server IDs;
inject test-only baselines into the test harness, not production bypass flags.
Existing `.repo-review/t049a_image_rehearsal.py` proves T049-A startup, not this
new release procedure. Preserve its accepted files; no need to rerun unrelated
historical suites repeatedly. If tools are unavailable, report which tests are
unexecuted and leave real acceptance to lead rather than fabricate success.

Return exact changed files, focused failing-first/final tests, frozen-file hashes,
cleanup evidence and unresolved questions. Runbook includes an eventual short
pinned fetch/show invocation with a clearly non-runnable commit placeholder until
lead publishes it. One eventual owner smoke batch: make a fresh test booking,
verify enrollment/confirmation acceptance, cancel that same booking through
Bookings, verify automatic removal/notice and no second notice on later polls.
Do not reuse unenrolled missing row18 as a positive removal-notification test.
Distinguish SMS accepted from received. No owner/server/GitHub action in this task.
