# T049-A: Automatic removal reconciliation and appointment-specific SMS

Implement this bounded runtime slice locally. Use the owner's GPT SOL, high
reasoning, separate GPT CLI. Transactional state, concurrent workers and ambiguous
SMS outcomes justify SOL here. Lead reviews the actual changes and runs real
PostgreSQL tests before packaging a separate manual release. Do not delegate
further. Escalate contract gaps to the lead rather than inventing provider behavior.

Required branch: `codex/phase-3-safe-booking`.
Required HEAD: `424ee6ada7244bf7b8f89b2e3fbb50255b77a1de`.
Workspace: `F:/Hossam-AI-Project/ai-voice-platform-v2`; the owner's mapped share is
equivalent. Verify HEAD and branch, preserve all existing untracked work. No reset,
staging, commit, push, installs, server access, real provider calls, or checkboxes.

## Outcome and evidence boundary

The owner explicitly wants automatic customer notification when a consultant
removes an appointment, without a routine dashboard approval. The deployed 0003
poller correctly shows an existing appointment and a missing appointment, but
does not change booking status or reminders. This task connects those behaviors.

Important: the Bookings API does not provide a cancellation-reason guarantee
through appointment GET 404. Repeated missing responses plus a complete inventory
are a conservative **application removal policy**, not Microsoft proof of who
cancelled or why. Name the resulting state/reason `removed_externally` /
`inferred_provider_removal`, not `verified_cancellation`. Customer wording says
the original appointment is no longer scheduled and invites them to contact the
office; do not claim Hussam cancelled it or that no replacement appointment exists.

No routine human click is required for eligible removals. Provider outages,
unproven identity, ambiguous collection results and ambiguous SMS submission
remain exceptions, visible on the dashboard, without false cancellation messages.
Direct Outlook-copy deletion remains unverified; this slice handles removal
observable through the configured Microsoft Bookings appointment identity.

## Read before editing

- AGENTS.md; specs/001-production-readiness/{spec,plan,tasks,data-model}.md.
- services/scheduling/{booking_records,provider_observations}.py.
- services/calendar/booking_readback.py; ms_bookings_service.py cancellation flow.
- services/sms/{reminder_scheduler,telnyx_sms_service}.py.
- orchestrator.py: persist_booking_record caller, _send_booking_sms, _cancel_booking.
- api/main.py lifecycle; config/settings.py; docker-compose.yml.
- migrations/{runner,schema_contract}.py and four immutable SQL assets.
- dashboard_routes.py bookings/deletion routes, templates/dashboard.html.
- existing observation/persistence/startup/migration/image tests.
- docs/operations/2026-09-13-provider-observations-deployment.md.

Official contracts checked by lead on 2026-09-13:

- https://learn.microsoft.com/en-us/graph/api/bookingbusiness-list-appointments?view=graph-rest-1.0
  GET the configured business's appointments collection. Do not invent `$filter`
  support; a date-window collection cannot establish global identity absence.
- https://learn.microsoft.com/en-us/graph/api/bookingappointment-cancel?view=graph-rest-1.0
  A successful explicit cancellation action returns 204. This task does not call it.
- https://developers.telnyx.com/docs/messaging/messages/send-message
  Submission acceptance is not delivery. Do not invent exactly-once delivery or
  an undocumented SMS idempotency header.

## Bounded scope

New migration `0004_appointment_notifications.sql`, runner/schema contract,
new cohesive modules under services/scheduling and services/sms for reconciliation,
outbox/storage and worker; existing observation/readback modules as necessary;
booking_records.py and only notification/local-finalization portions of
orchestrator.py; reminder_scheduler.py, telnyx_sms_service.py, api/main.py lifecycle,
settings/Compose flags; bookings dashboard endpoint/template; focused tests and
existing migration/startup/image sentinels. Add operations/automatic-notifications.md
and delegation/T049-A-result.md. No release script changes, dependency upgrades,
FAQ/STT/TTS/LLM prompt edits, broader booking validation rewrite or provider writes.

Preserve bootstrap and 0001/0002/0003 bytes and pinned checksums. Extend the
explicit migration runner and read-only readiness contract, including fresh and
upgraded databases and rollback. No startup DDL or historical timezone inference.

## Required runtime behavior

### 1. Durable identity and enrollment

Use local numeric booking ID AND a snapshot of provider scope (tenant/business),
provider ID and canonical start for each managed appointment. Record the scope
without credentials. Never match or cancel notifications by phone alone.
Provider-ID duplication across local rows is ambiguous: hold, do not notify twice.

Add durable per-booking reconciliation state, not just last-attempt observation:
last matching present observation, first missing time, consecutive missing count,
snapshot version, disposition and fixed reason, transition time. All timestamps
are aware UTC. Reset evidence when scope/ID/time changes or a present/changed/error
result interrupts the missing sequence. Configuration scope changes cannot make
all old appointments appear cancelled.

Enrollment requires a successful exact-ID matching-time read in the current
scope AFTER this feature is enabled. Do not seed prior presence from a local row,
the old last-attempt table, a screenshot or a successful create flag. Existing
already-missing bookings (including the old cancelled test) remain held/review,
with no retroactive customer messages. Existing future appointments can enroll
after a new positive read; initial confirmation SMS must not be replayed.

### 2. Automatic removal policy

For an enrolled future booking whose exact local snapshot remains confirmed:

- First appointment GET 404: persist missing evidence and hold its pending
  reminder. Do not change local booking to cancelled or enqueue cancellation SMS.
- Require at least two consecutive exact-ID 404 checks, at least 60 seconds apart
  and at most 10 minutes apart. A successful response/error/changed snapshot
  breaks the sequence. Multiple workers do not count as independent checks.
- On the qualifying check, verify configured business identity using a successful
  exact-business read and completely traverse that business's appointments list.
  Only missing from a valid, fully traversed list qualifies. A matching ID anywhere
  (including a later page or another date) vetoes removal. No narrow date filter.
- Reject malformed pages/IDs, repeated or unsafe nextLinks, cross-business paths,
  redirects, auth failures, timeouts, truncation or budgets exceeded. Set bounded
  budgets (maximum 20 pages/2000 entries/30 seconds); exhaustion means inconclusive,
  never absent. Respect existing shared Retry-After/backoff across all new reads.
  Do not log/store unrelated appointment bodies or customer data from inventory.
- Recheck the local snapshot under lock before applying. In one transaction:
  record the inferred-removal evidence/reason, set bookings.status to
  `removed_externally`, suppress that booking's pending confirmation/reminder jobs,
  and enqueue one removal notice with a unique durable event key. No hard delete.
- A later reappearance must not automatically reverse the disposition or send
  new confirmation messages. Show a discrepancy. For pending removal SMS, a fresh
  exact-ID read that finds the appointment must suppress the notice; failed reads
  delay it. Re-check missing evidence before dispatch as described below.

The count/time/list criteria are lead-selected operational defaults, not a claim
that Microsoft documents cancellation semantics this way. Document that distinction.

### 3. Appointment notification storage and dispatch

Replace the active JSON reminder sender with a PostgreSQL outbox when the new
feature flag is enabled. Preserve the legacy file; do not infer booking IDs from
phone/name/time or bulk-import ambiguous reminders. Never run both senders at once.
Default the new feature off until a reviewed migration/release enables it. When
off, legacy behavior remains explicit; no claim that old reminders are protected.

For NEW bookings with the feature on, persist the booking and initial confirmation
and 24-hour reminder jobs atomically in the same transaction. Use the returned
booking ID, not a phone key. Prevent the old fire-and-forget confirmation/reminder
path from also running. Existing enrolled bookings may receive only a future-due
reminder; no overdue reminders and no replayed confirmations. Migration itself
does not send or enqueue customer messages.

Jobs need unique event keys, booking/scope/version identity, kind, due UTC instant,
state, attempts, fixed error classification, provider message ID and timestamps.
Retain the minimal recipient/language/content snapshot needed for a stable notice;
do not expose raw content, phone, provider IDs or errors in diagnostic logs.
States distinguish pending, held, dispatching, accepted, failed, unknown, suppressed.
Accepted is not delivered. Do not claim delivery without validated provider evidence.

Use database claims and per-booking serialization shared with cancellation/local
finalization. Commit dispatch intent BEFORE external SMS submission. Do not keep
a SQL transaction open over network calls. A crashed/expired dispatch claim becomes
unknown, not automatically pending: blindly reclaiming can send a duplicate.
Fence stale workers and serialize competing finalization/dispatch. Database
deduplication prevents duplicate job creation, not carrier-level exactly-once SMS.

Before every reminder or delayed confirmation, obtain fresh exact-ID provider
readback in the current scope; require matching canonical start and a still-current
local confirmed snapshot. Missing/changed/failed/stale results hold the job, never
send old details. Do not revive overdue reminders after an outage. A removal notice
requires current removal state and a fresh exact-ID 404 after the complete-inventory
removal decision; scope/errors/reappearance veto dispatch. Provider failures use
shared backoff, not a new retry loop bypassing the observation control.

There is an unavoidable interval between external checks and carrier submission.
A reminder already submitted before a subsequent cancellation cannot be recalled.
Document this limit; test that locally known cancellation before dispatch prevents
the send and cancellation waits/serializes correctly with an in-flight dispatch.

Telnyx boundary: a valid successful response with message ID records accepted;
missing ID, timeout, uncertain transport failure, malformed success, or failure
to persist after possible submission is unknown and is not blindly retried.
Retry only documented definitely-not-accepted failures, with bounded attempts and
backoff. Provider rejection/invalid phone/missing SMS config is a visible state.
Do not use the old bool return as proof of safe retry. Preserve any legacy wrapper
needed by unrelated callers, but the new worker must use a structured result.

### 4. Existing cancellation and deletion coexistence

Do not alter calendar ownership/selection/provider mutation logic in this slice.
When the existing caller-cancellation path reports success, route its local status,
specific reminder suppression and SMS job through the same transaction/event
identity instead of phone-wide cancellation and direct cancellation SMS. If the
provider ID cannot map uniquely to a canonical local booking, do not fabricate
an association; report the local notification outcome as unresolved.
This does not certify the earlier provider mutation implementation as production safe.

Ensure external-removal and caller-cancellation races cannot each queue a separate
cancellation notice for the same appointment lifecycle. Future reschedule/rebook
must use a new version; never permanently dedupe on phone or swallow another booking.

Existing dashboard hard-delete/clear-all routes must not silently erase jobs or
audit evidence or delete a booking during dispatch. Block deletion of managed
bookings with a fixed conflict response rather than adding a new destructive workflow.

### 5. Customer and dashboard behavior

Removal notice templates in English and clear standard Arabic, deterministic,
no LLM. Include consultant and original appointment date/time in America/Toronto
(DST-aware). Meaning: 'Your appointment with {consultant} on {date/time} is no
longer scheduled. Please contact Flexible Accounting to arrange another time.'
No invented cancellation reason, replacement booking, phone number or delivery
promise. Use persisted language `ar` for Arabic; handle any known existing alias
only with source evidence. Unknown language uses the documented English fallback.

Dashboard must distinguish 'Removed from Microsoft (automatic reconciliation)',
'Reminders on hold', and notice queued/accepted/failed/unknown. Keep original time
and historical record. Counts must exclude removed appointments from upcoming.
Show fixed failure labels, not provider payloads. No administrator approval button.
Last observed evidence and last notification attempt are separate timestamps.

Start worker only after readiness; validate that observation is enabled whenever
the new notification feature is enabled. Stop/cancel/await both workers before
closing DB/HTTP clients. Disabled mode performs no new notification dispatch.

## Independent acceptance tests (synthetic only)

1. First missing result: reminder held; no removal/no SMS. Error, throttle,
   permissions change or malformed result never becomes cancellation.
2. Prior live presence + two spaced 404s + complete correct-scope inventory absence:
   one removed transition, one notice, pending jobs suppressed atomically.
3. No prior presence, wrong business, duplicate local provider ID, later-page
   match, malformed/looped/unsafe pagination or budget exhaustion: no removal SMS.
4. 404 -> present/error -> 404 resets evidence; snapshot/scope changes fence old
   results. Restart preserves evidence; simultaneous workers cannot advance count.
5. Two appointments for one phone: removing A does not touch B's reminders or jobs.
6. Rollback after any local finalization step leaves no partial status/job change.
   Repeated ticks, duplicate success callbacks and concurrent caller cancellation
   yield one notice lifecycle. Deletion cannot erase active work.
7. Reminder due during missing/provider outage is held; recovery after due/appointment
   expiration does not send stale reminders. Positive current future booking sends
   one eligible reminder. Reappearing appointment suppresses an unsent removal notice.
8. Worker crash before/after dispatch intent, timeout after potential acceptance,
   accepted response with DB failure: no automatic duplicate POST on restart.
   Missing/invalid response message ID is not accepted/delivered.
9. Real PostgreSQL: fresh/upgraded/repeat-run migration, constraints, unique event
   keys, concurrency/serialization, rollback, drift rejection, immutable SQL bytes.
10. Real lifespan/route tests: feature disabled, missing observation dependency,
    cleanup on partial startup, managed deletion conflict and accurate counts/states.
11. English/Arabic templates, Toronto summer/winter boundaries, aware scheduling,
    invalid recipient and unconfigured provider. Redaction tests use sentinel secrets.
12. New-booking integration asserts atomic jobs and no legacy duplicate path; legacy
    JSON is preserved/unread by active new sender. Existing test bookings cause no
    historical confirmation or already-missing cancellation message at enablement.

Use unittest-compatible offline tests plus real asyncpg/PostgreSQL tests under
the repository's explicit Docker opt-in. Run focused/full pytest if installed;
report missing tooling honestly. Never change expected behavior to fit code or
replace production modules with stubs that invalidate the oracle. Lead runs real
database acceptance independently. No live phone/Graph/SMS tests by implementer.

## Return evidence

Return changed paths/diff, failing-first results, final commands/results/skips,
immutable SQL hashes, worker/transaction race rationale and any unresolved issues.
Report exact behavior left out, including delivery receipts and direct Outlook
propagation. No rollout script in this assignment. Lead will package 0004 and a
schema-compatible fallback after acceptance, preserving the current settings-file
mount and TLS. Do not replay the 0003 deployment command.
