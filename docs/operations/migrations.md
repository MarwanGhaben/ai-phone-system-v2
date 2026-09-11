# Database migrations

## T005-D booking timestamp revision

The current full-schema revision is `0002`. Its immutable SQL adds nullable
`public.bookings.appointment_time_utc TIMESTAMPTZ` without a default or historical
backfill. Existing `appointment_time` wall-clock values and every existing row/ID
remain unchanged; a NULL canonical value explicitly means that the historical
instant is unresolved.

The checksummed manifest is ordered `0001`, then `0002`. Valid current histories
contain both revisions, optionally preceded by the already recognized
`bootstrap-v1` provenance row. A 0001-only history, with or without bootstrap
provenance, is a valid predecessor for `--prepare`. Version 0002 without 0001,
unknown rows, checksum drift, wrong column shape, or reversed application times
fail compatibility checks. Ledger validation reads every row rather than hiding
extra history behind a fixed limit.

`--prepare` now returns `prepared bootstrap-v1 0001 0002`, `applied 0002`, or
`up-to-date 0002` as appropriate. Read-only `--status` reports `pending 0002` or
`up-to-date 0002` for a complete application schema. The deliberately narrow
administrator predecessor retains its `pending 0001`/`up-to-date 0001` status and
`--apply` behavior; `--apply` does not apply 0002 or create a full schema.

Application readiness requires the exact new nullable column and a valid 0002
history row. All migration DDL, postconditions, and ledger writes remain in the
existing bounded transaction and advisory-lock contract. Revision 0002 has no down
migration and does not infer or convert any historical timezone.

The previous application rejects the new column/history at readiness. A plain
old-image rollback after applying 0002 therefore cannot recover service. The lead
rehearsed a fallback that retains the previous application but uses the current
schema checker; see `docs/delegation/T005-D-review.md`. Live release packaging
must prepare and verify that fallback before migration. Never remove the new
column after new writes as an improvised rollback. Do not use the T020-A app-only
deployment script for this schema-changing release.

## T005-A/T005-C foundation history

T005-A accepted locally after independent review and real PostgreSQL tests on
2026-09-10; see [lead evidence](../delegation/T005-A-review.md).
This does not complete parent T002, T005 or T006 or authorize deployment.

## Scope and review

The work was performed in the shared checkout
`\\192.168.0.125\ai-voice-platform-v2` (mapped as `Z:\` in this session).
SOL used that mapping; the lead verified the same files in the active local
`F:/Hossam-AI-Project/ai-voice-platform-v2` checkout. The required branch is
`codex/phase-2-database-foundation`;
HEAD remains `1affd83320e157343691a44934fac84f7aaa2490`.

Git was unavailable in SOL's environment. A read-only parser of the existing Git v2
index compared all 91 indexed file contents against their stored blob hashes
(allowing CRLF normalization); no differences were found before implementation.
This is not a claim that `git status` or a staged diff was executed.
Hashes were also recorded for 198 existing source/planning/fixture files,
excluding runtime/cache/nested-checkout directories. Those files are preserved.
No staging, commit, push, deployment, application/bootstrap change, provider call
or task-checkbox update is part of this slice.

All five outputs are new files; none replaced pre-existing task work:

| Added file | Change |
|---|---|
| `migrations/__init__.py` | Side-effect-free package marker. |
| `migrations/runner.py` | Explicit status/apply CLI, predecessor and ledger validation, bounded connection/locking, atomic repair and history. |
| `migrations/0001_admin_users_updated_at.sql` | Two additive ALTER statements; no row update/backfill. |
| `tests/integration/test_migrations.py` | Offline error/configuration/cleanup tests and opt-in real PostgreSQL acceptance scenarios. |
| `docs/operations/migrations.md` | Contract, local usage and verification evidence. |

Lead review and independent local database acceptance are now complete for this
slice. The lead started local Docker and downloaded its official PostgreSQL 16
test image with permission; SOL made no installation or download.

## Runner contract

Invoke `python -m migrations.runner` with exactly one of `--status` or
`--apply`. Only the explicit `MIGRATION_DATABASE_URL` environment variable
supplies its destination. A complete PostgreSQL URI must include hostname,
username, password (explicit empty is allowed) and database; omitted port means
5432. Percent-encode reserved characters in credentials. Destination and identity
are passed explicitly to asyncpg to prevent fallback to PGHOST, PGUSER,
PGDATABASE, PGPASSWORD or a password file. No application settings or dotenv
are imported, and there is no DSN/SQL/file-selection CLI option.

The original T005-A manifest contained only version `0001` and its reviewed SQL file.
The checksum is SHA-256 of exact file bytes, including line endings. Do not edit
an applied revision. A missing/undecodable file or differing recorded checksum
fails; directory discovery is never used.

Status uses a repeatable-read, read-only transaction. It reads catalog metadata
and (when present) the ledger. It never creates a ledger, takes an advisory lock,
runs a sequence function or writes a row. Results are `pending 0001` or
`up-to-date 0001`; invalid states fail rather than being presented as pending.

Apply uses one dedicated connection and one read-committed transaction. A stable
signed 64-bit key derived from
`ai-voice-platform-v2:public:schema_migrations` obtains a nonblocking
transaction advisory lock. A competing runner reports `busy` or sees the
committed revision on its next run. Apply also holds an ACCESS EXCLUSIVE lock on
the target and any existing ledger, then rechecks their metadata before DDL.
Preflight, ledger creation, repair, postcondition and history insert all occur
inside the same transaction.

Bounds: connection 10 seconds; table locks 3 seconds; server statements 10
seconds; asyncpg commands 15 seconds; idle transaction 15 seconds; graceful
connection close 5 seconds, with synchronous termination on close failure.
Status uses ordinary read locks only. Apply's short table lock blocks concurrent
writers during the transaction; this local implementation is not a deployed
writer-drain or operational recovery plan.

## Recognized predecessor and ledger

The target must be a permanent ordinary `public.admin_users` table without
inheritance/partitioning, row-level security, rules, user triggers, dropped column
slots, identity or generated columns. Its eight observed columns must match:

| Column | Type | Nullability |
|---|---|---|
| id | integer | NOT NULL |
| username | varchar(100) | NOT NULL |
| email | varchar(255) | NOT NULL |
| password_hash | varchar(255) | NOT NULL |
| is_active, is_superuser | boolean | nullable |
| created_at, last_login | timestamp without time zone, unrestricted precision | nullable |

The validated constraints are one primary key on id and unique keys on username
and email, all immediate/nondeferrable and validated, with valid ready plain
unique indexes. Additional constraints fail closed; names are not hardcoded.
Additional target indexes are untouched, not semantically audited.
Defaults of those eight columns are neither validated nor changed.

An optional ninth column, `updated_at`, must be nullable timestamp without time
zone with unrestricted precision. Its default must be absent, builtin `now()`
or `CURRENT_TIMESTAMP`. Deparsing with a pg_catalog-only search path distinguishes
an application-defined `public.now()`; the expression is never executed or
printed. Compatible existing values and defaults are preserved exactly.

The ledger must be an ordinary permanent `public.schema_migrations` table:
`version text PRIMARY KEY`, `checksum text NOT NULL`,
`applied_at timestamptz NOT NULL`, no defaults, extra columns/indexes/constraints,
user triggers, RLS, rules, inheritance, generated/identity or dropped columns.
It can be empty or contain the single matching version/checksum and a finite
applied instant. Unknown versions, malformed shape and checksum drift fail
before mutation. Recorded migrations still require the target postcondition on
both status and apply.

For an eight-column target, the SQL first adds nullable `updated_at` with no
default, then sets CURRENT_TIMESTAMP as the default for future inserts. Existing
rows retain NULL. A compatible ninth column is adopted without executing either
ALTER statement. History records the apply/adoption instant as an aware timestamp.
Subsequent applies preserve history time, all data and schema.

This validates only the narrow repair contract. It does not prove unrelated table,
default, sequence binding, function, role/grant or historical timezone semantics.
No booking-family merge, sequence repair, caller trigger, administrator bootstrap,
historical timestamp conversion, UTC shadow field or full fresh-install support
is included. Legacy naive authentication semantics remain until the later aware
time cutover.

## Errors and retry

Fixed safe classifications include `configuration`, `connection`,
`incompatible schema`, `unknown version`, `checksum drift`, `busy` and
`migration failed`. Failures return nonzero and never echo the URL, driver
message, raw SQL, customer values, unknown version contents or traceback locals.
Argument errors also avoid echoing supplied arguments.

After a failed apply, rerun status before retrying. Transaction failures roll
back the repair and ledger together; a lost commit response can leave an
unknown client outcome, so status must determine whether it committed.
No automatic down migration or destructive column removal is supplied.

## Local test commands

Use the repository root and an already provisioned Python 3.11 environment with
the existing requirements. The lead verified `C:/Python311/python.exe`, pytest
and asyncpg 0.29.0 on the local Windows host; SOL's separate environment lacked them.

**Local PowerShell**, once the existing approved development tools are available:

```powershell
Set-Location 'F:/Hossam-AI-Project/ai-voice-platform-v2'
$priorT005 = [Environment]::GetEnvironmentVariable('T005_RUN_DOCKER_TESTS','Process')
$priorT004 = [Environment]::GetEnvironmentVariable('T004_RUN_DOCKER_TESTS','Process')
try {
    [Environment]::SetEnvironmentVariable('T005_RUN_DOCKER_TESTS',$null,'Process')
    [Environment]::SetEnvironmentVariable('T004_RUN_DOCKER_TESTS',$null,'Process')
    & 'C:/Python311/python.exe' -B -m pytest tests/integration/test_migrations.py -q
    if ($LASTEXITCODE -ne 0) { throw 'Focused offline tests failed' }
    & 'C:/Python311/python.exe' -B -m pytest tests -q
    if ($LASTEXITCODE -ne 0) { throw 'Full offline tests failed' }
    $env:T005_RUN_DOCKER_TESTS = '1'
    & 'C:/Python311/python.exe' -B -m pytest tests/integration/test_migrations.py -q -s
    if ($LASTEXITCODE -ne 0) { throw 'Real PostgreSQL tests failed' }
} finally {
    [Environment]::SetEnvironmentVariable('T005_RUN_DOCKER_TESTS',$priorT005,'Process')
    [Environment]::SetEnvironmentVariable('T004_RUN_DOCKER_TESTS',$priorT004,'Process')
}
```

The real suite requires an existing official PostgreSQL 16 image and the explicit
local `desktop-linux` Docker context. It rejects TCP/SSH daemon endpoints,
requires Linux, uses an existing immutable image ID with pull disabled and never
reads an application/server DSN. It creates a unique container/database with
192 MiB memory/no additional swap, half a CPU, 128 PIDs, 32 MiB shared memory,
128 MiB tmpfs data and a random loopback-only port. Shared buffers/connections
are reduced for this disposable fixture. No host files or existing volumes mount.

Each test starts from synthetic tables, two invented accounts/sessions and 16
invented bookings. Database tests reproduce both actual password UPDATE strings
without importing the authentication module. They check old-row NULLs, new-insert
defaults, complete preserved values and sequence states, idempotence/adoption,
read-only status, rejected shapes/ledger contents, post-apply drift, two-runner
contention, bounded table-lock failure and injected failure after DDL before
history insertion followed by retry.

Candidate names are registered before create dispatch. Cleanup attempts removal
of every exact candidate and verifies absence even after setup failures/timeouts.
Any uncertain resource is named; setup errors retain their original failure and
a teardown error is separately reported by the test framework. Successful real
runs print `T005 cleanup confirmed: t005-migrations-...`.

No DigitalOcean or GitHub action is needed or authorized by these instructions.
Operational backup/role/rehearsal/writer-drain gates remain in
[the catalog comparison](2026-09-10-catalog-comparison.md).

## T005-C preparation mode

Accepted locally after lead corrections, 33 real migration/app tests, 16 real
Docker/startup tests and actual Python 3.12 image rehearsal. See
[T005-C evidence](../delegation/T005-C-review.md). Server rollout is separate;
Marwan has waived the off-server backup copy requirement for the first rollout.

Application startup now uses the explicit one-shot command:

```powershell
$env:MIGRATION_DATABASE_URL = 'postgresql://USER:PERCENT_ENCODED_PASSWORD@HOST:5432/DATABASE'
python -m migrations.runner --prepare
```

`--prepare` is the fresh-install and full-family startup gate. It accepts only an
entirely empty public application schema, the reviewed complete 17-table family
with an optional missing `admin_users.updated_at`, or an already prepared family.
It rejects partial eight/nine-table declarations, unexpected empty-path objects,
malformed or drifted history, and incompatible columns or keys before committing.

On an empty schema, the bootstrap asset, validation, ledger creation and both
`bootstrap-v1` and `0001` records commit in one transaction. An existing complete
family receives only the `0001` repair/adoption record; no bootstrap provenance is
invented. Valid steady-state history is either `0001` alone or `bootstrap-v1`
followed by `0001`. The narrower `--apply` and read-only `--status` commands keep
their T005-A predecessor behavior; `--apply` does not create a fresh schema.

All modes use only `MIGRATION_DATABASE_URL`. They never import application
settings, and diagnostics remain fixed safe classifications.

## Historical evidence from SOL's implementation environment

- Tests were written before the runner. The first stdlib run produced nine
  expected import errors because `migrations` did not exist; six initial database
  test methods skipped. This is failing-first offline evidence, not a reproduced
  PostgreSQL failure.
- Latest `python -B tests/integration/test_migrations.py -v` on
  `C:/Python314/python.exe`: **10 passed, 7 PostgreSQL test methods skipped**,
  0.324 seconds. Parameterized subcases run within the database methods.
  Passing offline tests prove only their mocked/configuration/cleanup boundaries.
- Explicit `T005_RUN_DOCKER_TESTS=1` invocation was attempted and the variable
  restored in finally. It stopped in database class setup with
  `ModuleNotFoundError: asyncpg`. At that point the nine then-existing offline
  tests passed. No database claim, container creation or resource cleanup was
  executed; there are no created test resources from this session.
- `python -B -m pytest tests -q` was attempted: `No module named pytest`.
  The existing full offline suite has not run in this environment; prior session
  results are not substituted.
- Python 3.11 grammar parsing and Python 3.14 in-memory compilation passed for
  both new Python modules and the test file. These are static checks, not a
  Python 3.11 runtime or PostgreSQL SQL execution.
- Python 3.11, Git and Docker were absent from PATH and the brief's known local
  locations. Python 3.14 is available; pytest and asyncpg are absent. No
  dependencies, tools or images were installed or downloaded.

SOL's then-remaining acceptance blockers: run on the approved Python 3.11 development
environment with its existing requirements, local desktop-linux engine and
already available PostgreSQL 16 image; run the real focused and full offline
suites, verify cleanup and obtain lead review. No schema/application acceptance
or parent task completion was claimed by SOL. Those local test/review blockers
were subsequently resolved in the linked lead review; server deployment remains separate.

Catalog/driver details were checked against the primary references:
[PostgreSQL 16 attributes](https://www.postgresql.org/docs/16/catalog-pg-attribute.html),
[constraints](https://www.postgresql.org/docs/16/catalog-pg-constraint.html),
[indexes](https://www.postgresql.org/docs/16/catalog-pg-index.html),
[locking](https://www.postgresql.org/docs/16/explicit-locking.html) and the
[asyncpg API](https://magicstack.github.io/asyncpg/current/api/index.html).
