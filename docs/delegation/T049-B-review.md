# T049-B lead review

**Accepted locally after the corrections below.** Manual server rollout and
live message receipt remain pending; no task-wide production sign-off is claimed.

Reviewed locally on 2026-09-14 from HEAD
`424ee6ada7244bf7b8f89b2e3fbb50255b77a1de`, branch
`codex/phase-3-safe-booking`. Server still runs that source with schema 0003.
This record distinguishes local acceptance/publication from owner-run deployment.

## Corrections found during actual execution

SOL's original offline checks did not exercise PostgreSQL or FastAPI startup.
The first lead run returned **3 failed, 31 passed, 1 optional skip**:

- Two test expectations omitted the extra database check performed by `/ready`
  and the unconditional pool cleanup on rejected startup. Lead corrected the
  assertions without weakening the lifecycle behavior.
- The synthetic seed had a fresh-install bootstrap ledger entry, while the
  release intentionally requires the deployed legacy-adoption history. The
  isolated fixture now represents that history. No production ledger is changed.

After correcting that fixture, actual pg_dump/pg_restore exposed a real existing
readiness defect. PostgreSQL 16 reparses some CHECK expressions into equivalent
forms: varchar-literal array casts become casts on each element; the leading
missing-count BETWEEN expansion becomes a flat AND conjunction. The persisted
baseline keeps its original spelling, so exact string equality rejects a valid
restored database. This affected both 0003 observations and 0004 notifications.

Lead made a narrow exception to the frozen T049-A files in
`migrations/schema_contract.py` and `tests/integration/test_migrations.py`:
only those two experimentally observed forms are canonicalized. Values, types,
bounds, column names, operators, key sets and remaining expression text stay
exact. There is no general SQL equivalence heuristic, catalog-baseline rewrite,
SQL migration edit or ledger repair. New tests reject altered values, casts,
bounds and Boolean rules. Two real backup/restore cycles retain unchanged stored
baselines and pass readiness/repeat migration; altered observation/outbox CHECKs
still fail. The restored predecessor probe uses the corrected candidate contract;
the original live image is independently verified during preflight.

The original accepted-hashes manifest is retained as historical evidence, not
rewritten to conceal these corrections. All other frozen T049-A files match.
Historical bootstrap/0001/0002/0003 and accepted unapplied 0004 hashes are unchanged.

## Verification

- Final real migration/restore subset: **14 passed, 43 deselected, 85.69s**.
  Includes all 0004 lifecycle tests, two backup/restore round trips, unchanged
  baseline checks, repeat preparation and genuine constraint-drift rejection.
- Actual Docker Compose active/paused settings probes passed with the existing
  parent config mount and nested settings-file overlay. Dollar signs and multiline
  synthetic settings were preserved; the previous sources were unchanged. Local
  test resources were removed. This exercised the actual release override methods.
- Release rehearsal exercises nonempty restored 0003 data, repeat 0004 migration,
  paused four-worker startup, synthetic durable booking creation and paused restart
  preserving accepted/unknown outbox rows. Final explicit Docker release,
  startup, image-content and synthetic Compose suites: **45 passed, 93.92s**.
- Final full suite: **207 passed, 49 optional skips, 2 existing warnings, 5.99s**.
  Warnings are audioop deprecation and missing local ffmpeg, not new failures.
- Python 3.11 syntax checks passed. Source whitespace checks passed; Git's CRLF
  conversion notices are distinguished from errors. Published-archive scope/SQL
  verification is performed before pushing the concrete release.

Only synthetic local resources are used. No Graph/Telnyx call, server command or
customer message is performed by lead validation. Internal networks prevent
outbound provider traffic. Live cancellation/SMS receipt remains an owner smoke
test after the separately published manual deployment command.

## Recovery boundary

Before 0004 commits, verified unchanged 0003 can recover to the previous image.
After commit, recovery uses the candidate with feature mode enabled but both
workers paused; legacy JSON messaging stays off and accepted/unknown jobs are not
requeued. This is degraded recovery, not a claim that every candidate bug can be
recovered. If it cannot become ready, ingress remains stopped for investigation.
