# T005-D: Save new bookings with an unambiguous appointment time

## Assignment and starting state

Recommended implementer: Marwan's GPT SOL, high reasoning, in his separate GPT CLI.
Reason: this change joins migration history, datetime semantics, database writes
and caller-facing failure handling. Lead independently reviews and runs real
PostgreSQL tests. If unavailable or scope conflicts arise, return to the lead;
do not substitute a broad rewrite or ask another model to guess the design.

Implement this assignment now, locally only. Branch: `codex/phase-3-safe-booking`.
Required HEAD: `4d87c3a3633680350552fe1fd9a1f10498adce08`.
The Windows repository is `F:/Hossam-AI-Project/ai-voice-platform-v2`; the existing
Z: mapping is acceptable. Verify branch/HEAD without resetting. Preserve all
existing untracked files and unrelated changes. Do not stage, commit, push, merge,
deploy, install dependencies, download images, contact real providers, or change
task checkboxes. This handoff supersedes historical next-action notes only for
this bounded slice. Parent T005/T017/T018 remain open.

Read:
- `specs/001-production-readiness/{spec,plan,tasks,data-model}.md` and interfaces.
- `docs/operations/database-contract-map.md`, `migrations.md`, `database-startup.md`.
- `docs/operations/2026-09-11-booking-persistence-proof.md`.
- `migrations/runner.py`, `schema_contract.py`, immutable SQL revisions/bootstrap.
- `_confirm_booking` in `services/conversation/orchestrator.py`.
- `services/dashboard/dashboard_routes.py` booking list/count consumers.
- Existing migration, startup, image-content and booking-safety tests.

## Defect and scope

The lead reproduced the real orchestrator with real asyncpg and isolated
PostgreSQL 16: aware appointment datetime rejected by the legacy `timestamp`
column, zero local booking rows, yet BOOKING_SUCCESS and a confirmation SMS task.
The calendar was mocked successful; this proves the code defect, not the exact
history of the owner's deleted September appointment.

This slice fixes that deterministic failure and stops normal success handling on
a failed local save. It prepares records for synchronization. It does NOT implement
Outlook subscriptions, durable pre-dispatch operations, idempotency, conflict
locking, a notification outbox, or automatic recovery after process/DB failure.
Those remain release blockers in their existing tasks; do not claim atomicity
between Microsoft and PostgreSQL or full booking safety.

## Required behavior

1. Add immutable `migrations/0002_bookings_aware_time.sql`: one nullable
   `public.bookings.appointment_time_utc TIMESTAMPTZ` column, no default and no
   historical backfill. Preserve original naive values, rows, IDs and both legacy
   schema families. Unknown historical times remain NULL in the new column.
   Never alter previously accepted bootstrap or 0001 SQL bytes/checksums.
2. Extend the existing checksummed runner/history contract to support 0002. Keep
   status read-only, migration locking/timeouts, transactional rollback and drift
   rejection. Recognize old valid 0001 histories (with/without bootstrap provenance)
   as predecessors and current histories as complete. Empty --prepare applies the
   existing bootstrap plus revisions to reach current schema. New application
   readiness requires 0002 and the new column. Never stamp a missing predecessor,
   hide extra history rows with a fixed LIMIT, or fabricate bootstrap provenance.
   Preserve old --apply admin-only behavior as documented; integrate 0002 through
   full-schema --prepare and report pending/current status accurately. If these
   CLI contracts cannot coexist, report the exact conflict before changing them.
3. Before provider dispatch, reject a missing/naive/invalid pending datetime.
   For NEW successful bookings, persist the exact aware instant in the new column;
   dual-write an explicitly derived America/Toronto wall time to the old column
   solely for legacy compatibility. Convert to Toronto before removing tzinfo
   for that old field. Never remove tzinfo from the canonical value or infer the
   timezone of any existing row. Both writes belong to the same INSERT.
4. Use a small persistence helper in `services/scheduling/booking_records.py`
   (create package initializer if required). Require a nonempty provider appointment
   ID; use INSERT RETURNING id and require an actual inserted record. Do not silently
   accept ON CONFLICT DO NOTHING as proof of persistence. Do not invent a uniqueness
   constraint over historical data or use call_sid as a universal booking identity.
5. `_confirm_booking` may run the current normal success/SMS path only after the
   new row is saved. If provider success is followed by database failure, missing
   provider ID, or no inserted row: clear pending authorization, do not set completion,
   do not dispatch success SMS/reminders, return BOOKING_OUTCOME_UNKNOWN with neutral
   guidance that an appointment may exist and staff must check it. No second create,
   automatic provider cancellation, or caller advice to book again. Log a fixed
   classification and safe exception type, not raw SQL args, exception text or PII.
   Preserve cancellation propagation; do not claim this solves crash reconciliation.
6. In the authenticated booking-list endpoint, use the canonical value for new
   rows and render it in America/Toronto with an explicit offset, independent of
   PostgreSQL session timezone. Retain the old display for legacy rows but expose
   `time_verified: false` when canonical time is absent. Exclude unresolved legacy
   timestamps from an asserted verified-upcoming count and expose a separate
   unresolved-time count. Do not guess their zone. No frontend redesign or changes
   to unrelated metric queries in this slice.

## Allowed files

- New migration above; runner/schema contract changes strictly needed for evolution.
- New booking_records helper; orchestrator imports and `_confirm_booking` only.
- Booking-list endpoint in dashboard_routes only.
- Migration/startup tests where revision expectations genuinely change; new
  `tests/integration/test_booking_persistence.py`; focused dashboard behavior tests.
- Image-content sentinels only to require the new migration in packaged images.
- Migration operations documentation and `docs/delegation/T005-D-result.md`.

Do not edit accepted fixtures, immutable SQL, provider adapters, availability,
prompts outside result wording, TLS, deployment scripts, Compose, dependencies,
live database data or the historical proof script. Do not widen readiness to
accept arbitrary schemas just to make the new column pass.

## Acceptance: tests first, using actual state

- Reproduce the missing-row bug before fixing it using real asyncpg, the real
  orchestrator and isolated PostgreSQL. Mock provider/TTS/SMS boundaries only.
  The regression expects ONE stored row, exact provider ID and exact UTC instant;
  it must fail against the starting application, not merely inspect SQL strings.
- January Toronto 11:00 -> 16:00Z; September Toronto 11:00 -> 15:00Z. Equivalent
  UTC inputs retain those instants. Old compatibility field remains 11:00.
  Naive inputs cause zero provider calls. Test under UTC and non-UTC DB sessions.
- Provider-success/local-save-failure suppresses BOOKING_SUCCESS and SMS, clears
  pending, and never retries provider creation. Missing ID and no inserted row
  also fail safely. Capture output to verify synthetic private sentinels are absent.
- Empty preparation, old complete 0001 upgrade, repeated preparation, read-only
  status/readiness, missing/unknown/checksum-drift history, wrong-type preexisting
  column, migration rollback and concurrent runner cases use real PostgreSQL.
  Preserve legacy row values and immutable file hashes. No real customer rows.
- Dashboard shows correct new-row time/offset and upcoming count across DB timezones;
  a legacy naive row stays unresolved and is not silently counted as verified.
- Existing booking guards, migrations and startup regressions still pass.

Use the existing opt-in isolated Docker harness. Never use configured live DSNs.
Run available tests and return precise missing-tool evidence if blocked; the lead
has Python 3.11/asyncpg/pytest and a working local Docker engine for independent
execution. Do not spend time installing a second environment.

## Return evidence

Changed files and actual diff; failing-first and final commands/results with skips
separate; immutable SQL hashes; preserved historical-data evidence; exact cleanup
result; branch/HEAD; remaining limitations. No checkbox completion or deployment.

Next, separately: T017 durable dispatch/finalization and T018 provider readback,
then external-calendar reconciliation using verified Bookings/mailbox identity.
Outlook deletion alone is not yet evidence that the authoritative Bookings record
was cancelled. No live deletion mapping is assumed by this assignment.
