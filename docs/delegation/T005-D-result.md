# T005-D implementation result

## Starting state and scope

- Branch: `codex/phase-3-safe-booking`.
- Required starting HEAD verified from the local branch ref:
  `4d87c3a3633680350552fe1fd9a1f10498adce08`.
- No reset, staging, commit, push, merge, deployment, provider contact, live DSN,
  server access, dependency installation, or task-checkbox change was performed.
- Existing unrelated files and work were left in place.

## Failing-first evidence

`tests/integration/test_booking_persistence.py` was added before production files.
Against the starting implementation:

```text
python -B tests/integration/test_booking_persistence.py -v
exit: 1
FAILED (errors=3, skipped=1)
```

All three runnable groups failed because the required
`services/scheduling/booking_records.py` boundary did not exist. The opt-in real
PostgreSQL test was separately skipped because `T005_RUN_DOCKER_TESTS` was not set.
The existing lead proof remains the real failing-state reproduction: one simulated
provider create, zero local rows, BOOKING_SUCCESS, and one SMS task.

## Implemented changes

- Added immutable revision `0002_bookings_aware_time.sql`, which adds nullable
  `bookings.appointment_time_utc TIMESTAMPTZ` without a default or backfill.
- Extended the checksummed manifest, ledger predecessor/order checks, preparation,
  read-only status, and runtime readiness to revision 0002. Full-schema status is
  `pending 0002` or `up-to-date 0002`; narrow 0001 apply behavior remains intact.
- Added a booking-record persistence helper that requires an aware datetime and a
  nonempty provider ID, writes the exact UTC instant plus an explicitly derived
  Toronto naive compatibility value in one `INSERT ... RETURNING id`, and rejects
  a missing returned row.
- `_confirm_booking` rejects invalid/naive times before hold speech or provider
  dispatch. Provider success reaches normal success/SMS handling only after local
  persistence. Local failure, missing provider ID, or missing inserted row clears
  authorization and returns sanitized `BOOKING_OUTCOME_UNKNOWN` without a retry.
- The authenticated booking list renders verified canonical times in Toronto with
  an explicit offset, labels legacy rows `time_verified: false`, excludes them from
  verified upcoming counts, and reports `unresolved_time` separately.
- Updated migration, image-content, persistence, failure-path, dashboard, timezone,
  rollback, drift, predecessor, status, readiness, and concurrency expectations.

## Immutable SQL hashes

```text
migrations/bootstrap_schema.sql
1c72bbe0a120860902c565e5573c05ad1cd640891a42bb35cca2c6e0876f31e8

migrations/0001_admin_users_updated_at.sql
53c861ac91cae5ecaf9f794b15563c88ee58bc0dafa5ae64fa81c89d907aa3a9

migrations/0002_bookings_aware_time.sql
62addab700e047dff2a13973c881a874bfe6c7f88967d593f189987192782be6
```

The accepted bootstrap and 0001 hashes remain unchanged. The real migration tests
assert that old rows and sequence values survive and receive only a NULL canonical
column, but those PostgreSQL assertions could not execute in this environment.

## Test results

Available focused persistence/dashboard checks:

```text
python -B tests/integration/test_booking_persistence.py -v
Ran 8 tests in 0.353s
OK (skipped=2 real PostgreSQL tests)
```

Available migration checks:

```text
python -B tests/integration/test_migrations.py -v
Ran 35 tests in 0.205s
OK (12 passed, 23 real PostgreSQL tests skipped)
```

Static validation:

```text
Python 3.11 syntax and trailing whitespace: PASS
```

The prescribed pytest group could not start:

```text
C:\Python314\python.exe: No module named pytest
exit: 1
```

Only Python 3.14 is available. Python 3.11, pytest, asyncpg, Docker, and Git are
unavailable, so no PostgreSQL container was created and there was no test resource
to clean up. Real PostgreSQL migration/persistence/dashboard tests, the existing
booking-guard/startup/image suites, the full suite, `git diff --check`, and Git
status remain for lead acceptance. Skips are not reported as passes.

## Remaining limitations

This slice does not make the external provider write atomic with PostgreSQL and
does not add durable dispatch operations, crash reconciliation, idempotency,
conflict locking, provider readback, Outlook subscriptions, or a notification
outbox. T017, T018, and the existing synchronization tasks remain required.
