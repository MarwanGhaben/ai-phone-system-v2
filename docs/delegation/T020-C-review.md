# T020-C independent lead review — 2026-09-19

**Accepted locally after follow-up and lead corrections.** The final acceptance
below supersedes the initial rejection. The accepted booking runtime is preserved;
the lead changed release packaging and one test-cleanup block only.

Branch remains `codex/phase-5-booking-reliability`, HEAD
`792952b2965e85513bf0971a41bfaff7bd3377c4`. The server is unchanged at that
release and schema 0004. No provider/server contact, publication or deployment
occurred during this review.

## Independent validation

- Full pytest with `T020_C_RUN_DOCKER_TESTS=1`: **1107 passed, 52 skipped,
  10 failed**, two existing optional audio warnings, 456.94 seconds. All ten
  failures are the already classified historical release assertions: four
  diagnostics, four speech-gate and two voice-release tests. Both submitted
  Docker arms passed. There were no other failures in this full run.
- Default release-only run: **21 passed, 3 skipped**. Two skips are opt-in Docker;
  one is unavailable Windows symlink creation in the non-elevated run.
- Independent release-boundary probes: **7 failed, 1 passed** against the
  submitted procedure. File: `.repo-review/test_t020c_lead.py`; captured output:
  `.repo-review/T020-C-lead-probes.txt`. Preserve these probes unchanged for
  follow-up; they are synthetic and do not contact a provider or server.
- Additional local Linux derived-image probe executed the actual
  `run_candidate_tests` method, with external networking disabled:
  **833 passed, 5 skipped, 10 failed**. Seven failures are missing packaged
  repository metadata (Compose, Dockerfile, T007-A result report); two expose
  missing ONNX Runtime in the substitute local parent, and one is an unchanged
  timing-sensitive service-facts cancellation test. The latter three are not
  claimed production runtime regressions. The exact server parent was unavailable.
  Probe: `.repo-review/t020c_candidate_probe.py`; output:
  `.repo-review/T020-C-candidate-method.txt`. Its containers/images were removed.
- The manifest test verifies all **108/108** frozen inputs, source-scope paths
  and manifest SHA
  `8d1f9d8a8e398d0902b2c4711c7018f7bdbeec3ee05d4842c627bf99b41953df`.
- `git diff --check` passed, with existing CRLF notices. No accepted application,
  SQL, model or test input was edited by the lead.
- Full-run disposable containers, networks and T020-C tagged images were removed.
  The two unrelated Jordan development containers and Docker Desktop were left
  running.

## Blocking findings

### 1. The known predecessor is rejected by configuration comparisons

`prepare_override` rejects any existing `AUTOMATIC_NOTIFICATION_WORKERS_PAUSED`
key. The accepted automatic-notification release writes that key as `false`, and
the subsequent voice releases preserve it. `normalized_compose` also rejects the
key on the baseline side. This is an expected input, not unexpected drift.

There are two additional independent comparison defects:

- Candidate Compose normalization removes its exact settings overlay, but the
  baseline normalization retains the old overlay that preflight requires. Even
  with the pause flag omitted, the normalized configurations differ.
- Candidate container normalization removes the pause environment key, but the
  baseline container normalization retains it. The otherwise unchanged effective
  container contracts therefore differ.

The first three independent probes reproduce these separately. The unrelated-env
positive control remains passing: unrelated changes must continue to be rejected.
Do not fix this by dropping all environment/mount verification or accepting an
unvalidated image. Both absent/false verified-phone baselines must follow the
approved baseline contract; true is not an accepted predecessor.

### 2. Candidate tests lack required repository inputs

`run_candidate_tests` mounts only the selected test tree and two scripts. It
selects `test_image_contents.py` and `test_database_startup.py`, which read
`/app/.dockerignore`, `/app/Dockerfile` and `/app/docker-compose.yml`. Those files
are deliberately excluded from the parent runtime image by `.dockerignore`.
The new image build does not supply them, and the method does not mount them.
The actual image run also exposed missing `docs/delegation/T007-A-result.md`,
which preserved contract tests read for their frozen hashes. Audit all selected
test inputs rather than fixing only the first missing path.

An independent captured-invocation probe proves the missing Dockerfile/Compose
test mounts. Run the **actual method** inside the disposable candidate; do not
claim a host pytest run proves the packaged method works. Obtain test-only inputs
from pinned source and preserve the existing tests, deployment image scope and
all 108 frozen assets.

### 3. The standalone database journey program lacks application settings

`run_database_journeys` invokes a generated unittest program directly, without
pytest/conftest. Its env file supplies only `MIGRATION_DATABASE_URL`, and the
`one_shot` call omits the already available synthetic application settings.
Importing the phone journey imports the actual orchestrator/settings and fails
before the tests execute. A clean child-process import independently returned
`ValidationError`; no environment values or private error bodies were printed.

Supply isolated synthetic settings explicitly, with database routing restricted
to the separately owned synthetic database. Do not borrow production settings or
run destructive test setup on the restored customer copy. Exercise this exact
generated program and prove the required journeys actually ran.

### 4. Normal active polling is mistaken for pre-cutover database drift

`preflight` snapshots every preserved row. `recheck_before_stop` demands equality
after the potentially long build/test/rehearsal, while the old app and its
observation/notification workers are still active. Those workers legitimately
update observation `checked_at` and reconciliation/control data without a new
phone call. The independent probe holds the image, settings, files, schema,
health and booking rows constant, advances only an observation hash, and gets
`live business rows changed before cutover`.

Keep all pre-cutover configuration/schema/identity safeguards. Capture the
authoritative preservation snapshot and final backup after verified writer stop;
compare against that snapshot across migration/replacement. The implementation
already has that later preservation boundary. Tests must distinguish ordinary
pre-stop activity from forbidden changes after the preservation boundary.

### 5. Current Docker arms miss the failing release boundaries

The migration Docker arm uses a separate helper. The derived-image Docker arm
constructs Compose by hand and runs `python -c '...sleep...'`, then probes files
and settings. It does not execute `prepare_override`, `run_candidate_tests`,
`run_database_journeys`, real FastAPI readiness/worker startup, or the actual
phase-aware recovery path against PostgreSQL. Mocked cutover tests verify call
ordering, not that those commands can run with the accepted mount/environment
shape. This explains why both supplied Docker arms pass while the release fails
the independent boundaries above.

Extend the local rehearsal to use the actual release methods and real readiness,
with synthetic credentials and isolated resources. Cover pre-0005 recovery and
post-0005 paused recovery retaining nonempty/unknown operation evidence. A
substitute local parent remains acceptable as explicitly limited evidence when
the server-only image is unavailable; do not weaken the owner's exact-parent
preflight or represent the substitute as the production image.

## Initial decision and next action (historical)

One consolidated owner-mediated SOL/high correction:
`docs/delegation/T020-C-followup.md`. Only the original four release deliverables
may change. Keep T020-C unchecked, the publication placeholder unfilled, the
accepted manifest and every accepted runtime input unchanged. No new owner calls
or server inventory are needed. Review the corrected actual release paths before
publication and the owner deployment command.

## Follow-up lead review — 2026-09-19

The original rejection above is historical. SOL corrected the predecessor
configuration, test-only inputs, standalone database settings and preservation
boundary, and added the actual-method Docker rehearsal. The lead then found and
corrected two bounded issues directly rather than requesting another handoff:

1. The schema probe had been changed to require the fresh `bootstrap-v1` ledger
   entry. A failing-first embedded-probe regression reproduced refusal of the
   known upgraded predecessor. It now accepts only the exact recognized legacy
   or fresh ledger, at 0004 or 0005, with all checksums enforced. Independent real
   PostgreSQL legacy-0004 and legacy-0005 probes both pass.
2. The remaining Linux service-facts test passed its public timeout and retained
   cleanup assertions, then failed while joining a deliberately cancelled task
   in `finally`. Test-only cleanup now performs a bounded join, consumes terminal
   cancellation, rejects unfinished cleanup and checks released ownership. No
   runtime behavior, timeout or SQL changed. The substitute-specific test-failure
   exception was removed from the release procedure; failed tests block release.

The original accepted manifest is unchanged. The separately versioned
`T020-C-release-hashes.json` differs in exactly one test hash; the other 107
entries, including all production/model/configuration/SQL inputs, are identical.
Release-manifest SHA:
`44f8840996bc7f4f69ebee0a7ea95cf3fc3651f284b9a302f255ad869fc1da04`.
The staged Git objects independently match all 108 release hashes, four separately
pinned test inputs and the exact 29-path production source scope.

Corrected focused host checks: 59 passed, four skips (three Docker opt-ins and
one Windows symlink permission). Full host run: 1064 passed, 106 skipped,
the same ten historical manifest assertions; Docker was not on that run's PATH.
The real Docker release/recovery acceptance is recorded below when complete;
host skips alone are not release acceptance.

## Final lead acceptance

The corrected Docker-enabled release suite passed **35/35 tests** in 897.19
seconds. Its actual packaged candidate run passed **843 tests, five optional
skips**, with no tolerated failure. The generated database runner executed
**49 real PostgreSQL journeys with zero skips**. Actual release methods exercised
restored-copy migration, repeat safety, real FastAPI readiness, paused/active
replacement, pre-0005 recovery and post-0005 paused recovery with a nonempty
operation ledger. The retained fixture is a pending intent; its name is
`unknown-outcome`, but it is not an `unresolved` state row. Do not misdescribe that
fixture as a separate unresolved-state recovery test.

Both independent upgraded-history database probes passed. All release/test-only
hashes and production source scope match staged Git objects. `git diff --check`
passes. Independent post-run inventory confirms zero T020-C rehearsal containers,
networks or tagged images; the unrelated development containers/Desktop remain.

The release package is accepted for publication and the owner-run guarded beta
rollout. The exact server-only parent is still verified by the owner procedure,
not claimed as locally tested. Keep the ten historical release-manifest failures
classified, and leave cancellation/rescheduling, identity, broader reconciliation,
voice evaluation and parent production gates open. Server/schema0004 unchanged
until the owner returns successful rollout markers.
