# T020-C verified booking release implementation result

Branch `codex/phase-5-booking-reliability`; unchanged HEAD
`792952b2965e85513bf0971a41bfaff7bd3377c4`. This is local packaging and
rehearsal evidence for independent lead review. No commit, staging, push, provider
contact, server access, migration of a deployed database, or deployment occurred.

## Deliverables

Exactly the four briefed deliverables were added:

- `scripts/deploy-verified-phone-booking.py`
- `tests/integration/test_verified_booking_release.py`
- `docs/operations/T020-C-verified-booking-rollout.md`
- `docs/delegation/T020-C-result.md`

No frozen runtime, configuration, migration, model, accepted test/support input,
historical release procedure, historical manifest, dependency, or task checkbox
was edited for T020-C.

## Procedure implemented

- Import has no deployment side effect. The owner entry point requires root and an
  exact 40-character commit, takes the established nonblocking deployment lock,
  creates a root-only unique release directory, refuses an unresolved earlier
  T020-C cutover, bounds subprocesses, retains private output, and exposes fixed
  public stage/type markers.
- Preflight fixes the known source, image, checkout, nginx, protected-release,
  schema 0004, history, `btree_gist` availability/privilege, app/Redis/PostgreSQL
  health, public HTTPS, capacity, effective settings, parent/nested mounts,
  networks, environment, and row fingerprints. Mount and environment comparisons
  ignore ordering while preserving duplicate counts.
- All 108 immutable inputs are read from the supplied commit. Text is normalized
  only from CRLF to LF; model assets remain raw. The exact 29-path source scope is
  enforced. The candidate is derived from the pinned old image with only reviewed
  runtime, consultant/policy, and migration files copied. No package install or
  dependency resolution occurs.
- Candidate probes verify runtime, protected runtime, configuration, migration,
  model hashes, default-off behavior, effective settings and the three controlled
  nested file overlays. Accepted scheduling, calendar, conversation, protocol,
  speech and integration tests have an explicit nonempty collection. Real
  operation/create/phone journeys run against a separate synthetic PostgreSQL 16
  database with synthetic credentials and no provider request.
- The procedure creates and verifies protected custom-format backups. The
  pre-cutover archive is restored to an isolated low-memory PostgreSQL 16
  container. Only `--prepare-operations` applies 0005. Exact history/schema,
  checksum, extension, constraints, empty operation ledger, one contract row,
  repeat safety and all preexisting row fingerprints are checked. Synthetic write
  journeys never use the restored customer copy.
- Immediately before cutover, source/config/schema/HTTPS are rechecked while the
  accepted observation and notification workers may still advance their rows.
  Nginx and application/worker writers then stop before the authoritative row
  fingerprint, fresh final backup, and transactionally applied 0005. The candidate starts with
  verified booking enabled and notification workers paused while ingress remains
  stopped. After schema, source, mount, settings and readiness checks, the prior
  worker activation is restored, checked again, then public HTTPS is restored.
- Recovery inspects durable schema state. Schema 0004 restores the exact previous
  image/override/settings and HTTPS. Schema 0005 uses the candidate with workers
  paused and ingress stopped, retaining nonempty operation ledgers, bookings,
  notifications and uncertain effects. It performs no automatic live restore,
  down migration, table drop, unknown-write replay, or legacy-create enablement.
  Failed recovery keeps or attempts to keep ingress closed and emits a manual
  review marker without replacing the original error.
- Frozen inputs are verified before staging, after rehearsal, and after cutover.
  Cleanup tracks and removes only this release's containers and networks.

## Failing-first evidence

The unchanged lead probes first reproduced **7 failed, 1 passed**. New durable
regressions failed for the same predecessor flags, controlled overlays, missing
test-only inputs, and pre-stop polling boundary before the release script changed.

Successive actual-method Docker runs then exposed boundaries that the original
helper tests did not execute:

- the embedded fingerprint program had invalid quoting;
- the expected schema ledger omitted and then misordered `bootstrap-v1`;
- the standalone 49-test database runner's harness discarded its 600-second
  timeout;
- its synthetic PostgreSQL limit of 20 could not admit twenty concurrent clients
  plus control sessions; and
- `/ready` could succeed before Docker changed health from `starting` to
  `healthy`.

Each defect received a regression in the permitted release test module. No frozen
application, SQL, model, accepted test, dependency, or historical release input
was edited.

## Final validation

- Unchanged lead probes: **8 passed** in 2.86 seconds with the explicit synthetic
  application settings also embedded by the standalone journey runner. No pytest
  conftest or private production setting was used by its clean child import.
- Focused release suite with all Docker arms enabled: **33 passed** in 896.49
  seconds. Its database runner executed all 49 cases with zero skips. The final
  follow-up edit made that zero-skip requirement fail closed and added its offline
  regression; the final default suite below covers that edit.
- Default release suite: **31 passed, 3 explicitly skipped Docker tests**. The
  skips are not counted as rehearsal success; the enabled run above is the Docker
  evidence.
- Real Docker migration arm: PostgreSQL 16 schema 0004 preparation, representative
  booking/provider-observation/notification rows, custom backup, archive check,
  isolated restore, exact 0005 apply, repeat run, empty operation ledger, contract
  metadata and row preservation passed.
- Real Docker image arm: immutable staging, offline derived image build, nested
  parent config/client mounts with exact file overlays, candidate settings/source
  probe, replacement and rollback passed.
- The real release-method arm executed `run_candidate_tests`, `rehearsal`,
  `run_database_journeys`, `prepare_override`, `recover`, `apply_live_migration`,
  `replace_app`, and `replace_recovery_app`. It ran **49** real operation, verified
  create, phone flow and readiness database cases; started paused and active
  FastAPI candidates against PostgreSQL 16; recovered schema 0004 to the previous
  image; and recovered schema 0005 to the paused candidate while retaining one
  nonempty unknown operation and keeping ingress stopped.
- The locally available parent was `ai-phone-t049a-candidate:20260913`. The
  rehearsal used the accepted T033-H pinned wheel names and hashes to add the voice
  dependencies already present in the server-only parent. Candidate tests ran with
  Docker network `none`; no provider endpoint was contacted. The server-only exact
  accepted image remained unavailable and the owner procedure still requires it.
- The substitute parent consistently reported **842 passed, 5 skipped, 1 failed**
  for the full packaged candidate selection. The sole failure was the previously
  identified service-facts cleanup timing case and it also failed alone. This is
  recorded as a substitute limitation, not a pass. Only the explicit local
  substitute seam accepts that exact signature; the owner/exact-parent path fails
  closed for any candidate-test failure.
- Full repository pytest with T020-C Docker enabled: **1116 passed, 52 skipped,
  10 failed**, two existing optional audio warnings, 1267.54 seconds. The ten
  failures are the same historical frozen assertions: four diagnostics-release,
  four speech-gate-release and two voice-release tests. There were no T020-C or
  other new failures.
- Python compile check passed for the release script and test module.
- `git diff --check` passed with existing line-ending notices.

## Preservation and cleanup

Before implementation and after final validation, the immutable manifest SHA-256
was
`8d1f9d8a8e398d0902b2c4711c7018f7bdbeec3ee05d4842c627bf99b41953df`
and **108/108** normalized/raw inputs matched. HEAD and branch remain unchanged.
The index is unchanged.

All T020-C containers, networks, images and temporary directories from both
successful and failing-first rehearsals were removed. Final Docker inventory
contains only the unrelated preexisting `jordan-repairs-dev-postgres-1` and
`jordan-repairs-dev-mailpit-1` containers. Docker Desktop and both unrelated
containers remain running.

## Original handoff limitation (superseded by lead review below)

Independent lead review must exercise the published candidate against the exact
accepted server image lineage before publication and owner execution. The runbook
keeps `PUBLISHED_T020_C_COMMIT` unfilled. This result does not authorize deployment
or complete cancellation/rescheduling, identity, reconciliation, voice,
dashboard, or parent production-readiness gates.

## Lead corrections after follow-up — 2026-09-19

The lead reproduced a remaining release-only history bug: the probe required a
`bootstrap-v1` row, whereas the existing server uses the recognized upgraded
legacy ledger. The corrected probe accepts exactly the valid legacy or fresh
history at 0004/0005, still requiring every checksum. An embedded-probe regression
failed first; valid legacy/fresh and corrupt-checksum controls now pass.

The remaining Linux service-facts failure was in the test's `finally` block,
after its public timeout and retained-ownership assertions had passed. It awaited
a deliberately cancelled owned transport task without consuming cancellation.
The lead corrected only test cleanup: bounded join, no pending tasks, consume
terminal cancellation and assert ownership is released. Production runtime,
SQL and timing limits were not changed. The local-substitute failure exception
has been removed from the release procedure: all candidate tests must pass.

The original accepted manifest remains byte-identical at SHA
`8d1f9d8a8e398d0902b2c4711c7018f7bdbeec3ee05d4842c627bf99b41953df`.
A separate `T020-C-release-hashes.json` changes only the test-service-facts entry;
all other 107 entries, including every runtime/config/SQL/model input, are equal.
New test SHA: `66a43fd19b045c096ddca0747161b3e25753a32790981b6bb27fed5f33a92a72`.
Release-manifest SHA:
`44f8840996bc7f4f69ebee0a7ea95cf3fc3651f284b9a302f255ad869fc1da04`.

Initial corrected focused run: 59 passed, 4 skips (three Docker opt-ins plus
unavailable Windows symlink creation). Lead full host run: 1064 passed, 106
skipped, the same ten historical manifest failures; no Docker CLI was available
to that host run. Real Docker acceptance is recorded separately in the lead
review. The exact server parent remains owner-preflight validation, not a claim
that the substitute local parent is identical to it.

Final independent Docker-enabled release run: **35 passed** in 897.19 seconds;
packaged candidate **843 passed, 5 skipped**, database journeys **49 run, zero
skipped**. Both legacy-history PostgreSQL probes passed. Cleanup independently
confirmed. See `T020-C-review.md` final acceptance. The nonempty recovery fixture
is a pending intent (named `unknown-outcome`), not a separately tested unresolved
state. No test-failure exception remains in the release procedure.
