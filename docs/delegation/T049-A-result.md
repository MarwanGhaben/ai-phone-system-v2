# T049-A local implementation result

Branch `codex/phase-3-safe-booking` and HEAD
`424ee6ada7244bf7b8f89b2e3fbb50255b77a1de` were read from the local
`.git/HEAD` and branch ref after editing. No commit, push, deployment,
installation, server access or live provider call was made. Git is unavailable
in this CLI, so a Git diff/stat could not be generated here. Existing unrelated
work was preserved.

## Changes

- `migrations/0004_appointment_notifications.sql`,
  `migrations/runner.py`, `migrations/schema_contract.py`: additive
  reconciliation/outbox/contract tables, explicit revision order, checksum and
  read-only readiness including rollback and drift checks.
- `services/calendar/booking_readback.py`,
  `services/scheduling/{provider_observations,removal_reconciliation,booking_records}.py`:
  exact-business complete inventory, durable enrollment/missing evidence,
  appointment-scoped atomic finalization, and transactional jobs for new bookings.
- `services/sms/{notification_text,notification_outbox,notification_worker,telnyx_submission,telnyx_sms_service}.py`:
  deterministic English/Arabic notices, booking/version event keys, durable
  claims and structured submission outcomes.
- `services/conversation/orchestrator.py`, `api/main.py`,
  `config/settings.py`, `docker-compose.yml`, `services/database.py`:
  enabled-mode booking/cancellation routing, exclusive worker lifecycle,
  disabled-by-default configuration and 0004 readiness requirement.
- `services/dashboard/dashboard_routes.py`, `templates/dashboard.html`:
  fixed reconciliation/notification states, separate observation and attempt
  times, upcoming counts excluding removed bookings and managed-delete conflict.
- `tests/integration/{test_automatic_notifications,test_migrations,test_database_startup,test_image_contents}.py`,
  `tests/dashboard/test_dashboard_metrics.py`: synthetic policy, HTTP boundary,
  migration, worker, lifecycle, image and dashboard regressions.
- `docs/operations/automatic-notifications.md`: behavior and operational
  limits. This result file records the handoff.

## Evidence

Failing first: the initial four new notification tests failed with
`ModuleNotFoundError` for the not-yet-created notification modules. After
implementation, `python -B -m unittest discover -s tests/integration -p
test_automatic_notifications.py -q` ran 10 tests: 8 passed, 2 skipped because
this Python CLI has no Toronto timezone data. The opt-in real PostgreSQL
suite is present in `test_migrations.py`. Its offline run
(`python -B -m unittest discover -s tests/integration -p
test_migrations.py -q`) ran 48 tests: 13 passed, 35 opt-in PostgreSQL tests
skipped. Python 3.11-mode AST parsing passed for 21 changed Python files.

Pinned historical SQL SHA-256 values are unchanged:

| Asset | SHA-256 |
| --- | --- |
| bootstrap | `1c72bbe0a120860902c565e5573c05ad1cd640891a42bb35cca2c6e0876f31e8` |
| 0001 | `53c861ac91cae5ecaf9f794b15563c88ee58bc0dafa5ae64fa81c89d907aa3a9` |
| 0002 | `62addab700e047dff2a13973c881a874bfe6c7f88967d593f189987192782be6` |
| 0003 | `b3abaa09c340f89c32778b0b17b5c2a584845983b13eb95dbb9c366f3d6c5830` |
| new 0004 after focused correction | `a1c66c7705adbd49918966c7d41d8bbbc5e80d3d5d0534fe641fd15579731a9e` |

The worker holds a per-booking session advisory lock across provider read and
SMS submission; local finalization obtains the matching transaction advisory
lock. It commits a claim token before POST and records a possible lost result
as `unknown`, without automatic repost. The policy cannot guarantee carrier
exactly-once submission or retract a reminder already sent.

## Lead review still required

This CLI has no `pytest`, `asyncpg`, `fastapi`, `tzdata` or Docker command.
The lead must run the opt-in PostgreSQL cases, full pytest suite, startup and
dashboard route tests, Toronto summer/winter template tests and image check in
the reviewed local environment. No real database or image pass is claimed.
The application policy does not prove Microsoft cancellation cause, direct
Outlook-copy deletion propagation, SMS delivery, or safety of the existing
provider cancellation mutation. No rollout script or production change is
included.

## Focused correction after lead review

The lead's independent local PostgreSQL run found three failing tests and
separate reproductions for accepted-result typing, held-job starvation,
inventory-error evidence and shared Microsoft pause admission. The follow-up
changes are limited to `services/sms/{notification_worker,notification_outbox}.py`,
`services/scheduling/{removal_reconciliation,provider_observations}.py`,
`services/calendar/{booking_readback,provider_admission}.py`, unapplied 0004 SQL
and its schema contract, focused migration/notification/image tests, and these
two documentation files. The historical 0001-0003/bootstrap hashes above are
still unchanged. No server migration or provider call occurred.

The accepted-result update now types the state parameter consistently. A
durable `next_attempt_at` and bounded `retry_count` separate held eligibility
from the unchanged reminder `due_at`; `attempts` advances only for a committed
possible SMS POST. The observer and worker use one session advisory lock for
provider request admission, with persistent pause checks before OAuth, exact
reads, business reads and every inventory page. The admission lock is released
before observer booking transactions; the worker retains its booking session
lock while it takes admission, without a SQL transaction over HTTP. Inventory
failure or found-ID clears the qualifying missing sequence under the booking
snapshot fence. Read-start timestamps prevent older worker observations from
overwriting newer evidence, and worker 404s never add a spaced missing count.

The added opt-in PostgreSQL tests include the complete synthetic HTTP/poller/
inventory/removal/outbox/accepted-SMS flow, negative inventory results,
11-job fairness across ticks/workers, and a pause imposed during a batch and
mid-inventory. The existing drift oracle now holds the required read-only
transaction. The lead must rerun these real-DB tests independently; this CLI
has only Python 3.14 and no local Docker/pytest/asyncpg installation.

Final focused checks in this CLI: `python -B -m unittest discover -s
tests/integration -p test_automatic_notifications.py -q` ran 12 tests (10
passed, 2 optional skips); the same command for `test_migrations.py` ran 54
(13 passed, 41 opt-in skips); `test_provider_observations.py` ran 19 (18
passed, 1 opt-in skip). Python 3.11 grammar parsing passed for the 10 focused
Python files. A broader `unittest discover -s tests/integration -q` run reached
110 tests but had seven import errors because this CLI lacks `pytest`,
`asyncpg` and `loguru`; it is not a full-suite pass. The lead's real PostgreSQL
failing-first evidence is in `T049-A-review.md` (two accepted-result failures,
one drift-oracle failure and three independent behavioral probe failures).
The focused follow-up also includes a cancellation/lock-cleanup regression for
an admission waiter. No new real-PostgreSQL result is claimed here.

Separate lead-review question outside the listed corrections: the existing
`release_current_jobs` predicate requires `due_at >= CURRENT_TIMESTAMP` even
for a new confirmation created due immediately. A positive read occurring
later appears unable to release that held confirmation. This follow-up leaves
that pre-existing behavior unchanged under the bounded correction brief.
