# T049-B implementation result

T049-B is implemented locally on `codex/phase-3-safe-booking` at unchanged HEAD
`424ee6ada7244bf7b8f89b2e3fbb50255b77a1de`. Nothing was committed, pushed,
deployed or sent to Graph/Telnyx. The accepted T049-A hash manifest remains
unchanged and every frozen entry still matches it.

## Runtime prerequisite

`AUTOMATIC_NOTIFICATION_WORKERS_PAUSED` is a false-by-default boolean and is valid
only while automatic notifications are enabled. Paused mode still runs the full
database compatibility gate and leaves automatic feature mode enabled, so new
booking persistence continues to create durable jobs and the orchestrator cannot
fall back to immediate/legacy messaging. Startup constructs neither the provider
poller nor notification worker and does not start the legacy JSON scheduler. It
logs one fixed degraded-mode message. Existing disabled and active behavior is
unchanged.

## Release package

The owner-run script verifies the deployed 424ee6a/image/schema-0003 baseline and
the existing nested settings mount, then reads an exact 19-file whitelist from one
future reviewed 40-hex commit. It derives a single 0004-capable image from the
running image with no network or pulls. Protected active and paused overrides
replace exactly one `/app/config/settings.py` source, retain every other mount and
environment value, and pin migrate to the candidate.

The restored-copy rehearsal checks authenticated PostgreSQL readiness, nonempty
aware/naive bookings and provider observations, repeat-safe 0004 migration,
unchanged fingerprints for 17 business tables and the two existing observation
tables, empty reconciliation/outbox data, actual paused lifespan readiness, and
one synthetic durable booking with two held jobs and no legacy SMS-log change,
plus cleanup of every named temporary resource. Cutover performs a final backup after
writer stop, starts paused first, then makes a separate activation restart. After
0004, recovery can use only the verified candidate with workers paused; it never
reactivates legacy messaging or changes accepted/unknown outbox states.

## Local validation

The focused release suite currently contains 20 standard-library boundary tests.
It covers copy scope, no-network build, settings/mount comparison including reverse
order and duplicates, secret preservation, read-only probes, settings-only startup,
rehearsal migration/repeat/readiness and cleanup, pre-stop refusal, paused-first
activation, 0003 and 0004 recovery choices, ambiguous schema, private output, and
absence of live restore/requeue/provider calls. The pytest-based startup tests add
paused readiness, invalid flag combinations and Compose mapping coverage.
One additional `T049_RUN_DOCKER_TESTS=1` case uses the accepted local T049-A image
as a test-only parent, creates a nonempty synthetic 0003 archive, and executes the
actual candidate build and restored-copy rehearsal methods through paused readiness
with explicit container/network/image cleanup.

This CLI has Python 3.14 but no pytest, asyncpg, pydantic, Docker or Git executable.
The lead must run the pytest/startup tests and real synthetic Docker release methods
in the accepted local environment. No real database, candidate image, Compose or
cleanup pass is claimed here. Final command results and accepted-hash comparison
are recorded in the task return.

Available final checks: the release suite ran 21 tests with 20 passed and the one
explicit Docker rehearsal skipped; the accepted automatic-notification suite ran
12 with 10 passed and 2 optional skips; migration tests ran 55 with 13 passed and
42 PostgreSQL skips; provider-observation tests ran 19 with 18 passed and 1 skip.
Python 3.11 grammar parsing passed for all five changed Python files. Direct startup
test collection remains blocked at import because `pytest` is unavailable here.

Deployment and the supervised booking/cancellation smoke remain separate owner
steps after lead acceptance and publication. Carrier acceptance is not delivery;
this work does not add delivery receipts or prove direct Outlook propagation.
