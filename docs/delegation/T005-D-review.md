# T005-D lead acceptance, 2026-09-11

Accepted locally after source review, corrections and real integration execution.
This is new-booking persistence acceptance, not external-calendar synchronization
or completion of parent T005/T017/T018. Server deployment remains separate.

## Corrections after SOL

- Replaced the new fixed-offset pytz substitute and fake application modules with
  actual application imports and actual timezone rules. Failures now originate at
  the database/provider boundary rather than a mocked persistence helper. The
  original SOL test file is retained locally under `.repo-review` for comparison.
- Reused existing America/Toronto timezone configuration. Normalize confirmed
  aware inputs before passing them to the legacy calendar adapter and formatting
  speech/SMS; equivalent UTC inputs must still say 11 AM, not 3 or 4 PM.
- Clear pending authorization on interrupted local persistence and clear stale
  completion flags/summary on local-save failure. Retrying confirmation after the
  failure cannot dispatch another create.
- Validate the 0002 SQL checksum during read-only status as well as preparation.
- Seed actual legacy booking data in the already-0001 upgrade test; the old test
  compared empty booking lists. Preserve original naive values and provider IDs.

The explicit keyword parameters on the persistence boundary correspond to one
SQL row; retained rather than introducing a new domain model in this interim slice.
Catch-all save handling is intentional at the external-success/local-failure
boundary: it produces an unknown outcome and suppresses normal success/SMS.
No claim of durable recovery or provider/database atomicity is made.

## Final validation

Full pytest with `T005_RUN_DOCKER_TESTS=1`, `T004_RUN_DOCKER_TESTS=1`, and
`T005_RUN_COMPOSE_TESTS=1`, using local Python 3.11 and desktop-linux:

```text
128 passed, 2 warnings in 107.88 seconds; no skips
```

Includes actual PostgreSQL 16 migrations, rollback/concurrency/drift, new booking
rows, winter/summer and equivalent UTC instants, UTC/Asia-Kolkata DB sessions,
dashboard provenance/counts, existing confirmation guards, app startup, image
content positive/negative checks and synthetic Compose failure ordering.
Warnings are the existing audioop deprecation and local ffmpeg absence.

Real Python 3.12 Docker application rehearsal also passed:

- Candidate built from existing phase-3 local test image with only seven approved
  runtime files copied into a fresh build context. No repository/.env mount.
- Candidate migration preparation, actual four-worker app startup, /ready and
  /health succeeded on an isolated synthetic database/internal network.
- Corrupting the 0002 ledger checksum made /ready return 503.
- Compatibility rollback image retained the previous app and changed only its
  schema checker to the current contract. Against a candidate-prepared database,
  actual app startup/readiness/health and drift rejection all passed.
- Rehearsal temporary containers and networks removed with verified absence:
  prefixes `t005c-image-019a425ad877` and `t005c-image-d321b1c39d30`.
- Local candidate/rollback image tags retained for inspection:
  `ai-phone-t005d-candidate:20260911`,
  `ai-phone-t005d-rollback-compatible:20260911`.

The first build attempt used a raw image ID in FROM, which BuildKit treated as a
registry name and rejected; no image was downloaded. Re-ran using the verified
existing local base tag. A lead test-fixture omission was corrected before the
final all-gates run. No failed attempt is included in the passing count.

`git diff --check` passed. Immutable SQL SHA-256 values:

```text
bootstrap_schema.sql: 1c72bbe0a120860902c565e5573c05ad1cd640891a42bb35cca2c6e0876f31e8
0001_admin_users_updated_at.sql: 53c861ac91cae5ecaf9f794b15563c88ee58bc0dafa5ae64fa81c89d907aa3a9
0002_bookings_aware_time.sql: 62addab700e047dff2a13973c881a874bfe6c7f88967d593f189987192782be6
```

## Deployment boundary

Do not rerun `deploy-booking-guards.py`: it only replaces the orchestrator and
assumes the pre-phase-3 image. This revision also needs a database migration.
Plain rollback to the old image after 0002 is not sufficient: its strict checker
rejects the new column/history. The compatibility rollback approach above was
rehearsed locally, but a server-specific deployment script still needs packaging.

Next: verify running server image/health and existing migration ledger, then
package an exact-commit candidate and compatible fallback from that running image.
Preserve private override, runtime.env mount and TLS configuration. Rehearse a
protected fresh backup in an isolated database before live migration; stop writers
for final backup/migration/cutover. Do not drop the new column to roll back after
new writes. The owner previously accepted keeping backups only on the server.
