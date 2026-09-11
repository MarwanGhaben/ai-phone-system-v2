# Database preparation and application readiness

Locally accepted 2026-09-11 after [lead fixes and actual-image rehearsal](../delegation/T005-C-review.md).
Server preflight passed on 2026-09-11. Current release preparation is described in
[the first beta release procedure](phase-2-first-release.md). No server deployment
has occurred yet; the historical implementation evidence remains below.

Compose now leaves a new PostgreSQL volume empty and runs a short-lived `migrate`
service before the application. The service uses the application image, receives
only `MIGRATION_DATABASE_URL`, and runs:

```text
python -m migrations.runner --prepare
```

PostgreSQL must first be healthy. The application starts only when preparation exits
successfully, and nginx waits for the application's `/ready` health check. Existing
named database volumes are preserved; this ordering does not remove or recreate
services or volumes.

The current preparation target includes checksummed revisions `0001` and `0002`.
Revision `0002` adds nullable `bookings.appointment_time_utc TIMESTAMPTZ` without
backfilling or changing the legacy naive column. Runtime readiness now requires the
new column and its valid history row; an otherwise valid 0001 database remains a
recognized preparation predecessor but is not application-ready until upgraded.

Each Uvicorn worker opens its normal application pool and performs a bounded,
read-only compatibility check before starting reminders or prewarming TTS. Startup
aborts and closes the pool if the exact 17-table column/key contract or recognized
migration history is missing or drifted. No worker, lifespan handler, or readiness
request runs DDL or writes migration history.

`/health` remains lightweight process liveness. `/ready` returns 200 only after
lifespan startup and a current bounded database check; shutdown, database outage,
history drift, or structural incompatibility returns a generic 503 body. Both probe
paths bypass request rate limiting.

The runtime contract checks table kinds, exact columns/base types/varchar lengths/
nullability, primary/unique/foreign keys, foreign-key delete actions, inheritance,
partitions and row-level security. It intentionally does not claim equivalence of
defaults, sequence ownership, secondary indexes, grants, or trigger bodies. The
known live caller-trigger repair, partial-family adoption, timestamp conversion,
administrator bootstrap and full schema-semantic convergence remain later work.

For local acceptance, render Compose only with synthetic environment values and do
not start the real project stack. Database tests use the existing opt-in disposable
PostgreSQL fixture described in [migrations.md](migrations.md). No DigitalOcean
action is part of this startup integration.
