# Phase 2: first beta database release

The candidate contains accepted T004 build exclusions and T005-A/B/C database
preparation, structural compatibility and application readiness. It does not fix
Arabic comprehension or booking correctness, and does not enable public launch.

## Server preflight received

Marwan returned successful preflight on 2026-09-11: clean main at 1affd833,
Compose v5.0.2, five healthy/running existing containers, 4.0G free disk and
888 MiB available memory. The application role is a PostgreSQL superuser and
can access all 17 public tables. This permits the scoped migration; least-privilege
separation remains later security work. Existing backup SHA-256 still matches
996edee63a77f6ddba88257a853b54746982112f0e57669991b4e9ccae691eb4.
The owner explicitly accepts keeping the recovery copy only on this server.

## Candidate preparation

Run `scripts/prepare-database-release.sh` only on the verified DigitalOcean server,
as root, with the full reviewed commit present in its repository. The lead supplies
an exact fetch/invocation after publishing that commit. Do not substitute a moving
branch reference for the commit argument.

The script leaves the running app and live database schema unchanged. It:

1. Checks the expected baseline/clean tracked checkout and running app health.
2. Saves a rollback image tag and root-only recovery metadata under a newly created
   `/opt/ai-phone-release.*` directory. The inspection/environment files contain
   secrets: keep them private and never paste or commit them.
3. Exports only committed files into an isolated build context, builds a candidate
   tagged for this run, and records its immutable image ID. Build output stays in
   a protected log. The existing unpinned dependency ranges mean this is a new
   build, not automatically the previously tested local image.
4. Takes a fresh compressed backup and restores it into a resource-capped
   PostgreSQL container with no external network, published ports or live mounts.
5. Runs the actual candidate migration and read-only runtime compatibility check
   against that restored copy, then removes the rehearsal containers/tmpfs data.

The 3 GiB disk and 600 MiB available-memory checks are conservative entry guards,
not guarantees against exhaustion. No images, build cache or production volumes
are pruned. On a build or restore failure, the script stops and preserves protected
logs for diagnosis. Return the output markers, not raw logs or recovery files.
Rehearsal uses a synthetic database owner and omits original grants; it validates
the restored schema, not full role/ACL recovery. Actual application-role privileges
were separately inspected on the live container.

## Cutover after successful rehearsal

The lead will use the reported release directory and immutable candidate ID to
prepare the manual cutover: stop new test calls, drain active calls/writers, take
the final pre-change snapshot, run the same migration image against the existing
database, replace only the app, and verify readiness and proxy routing. Preserve
the old image/configuration for rollback; an additive migration is not reversed
by blindly restoring an older database over newer records.

Do not run the old `scripts/deploy.sh`, `docker compose down`, a stack-wide rebuild,
or a volume deletion. The preparation command is not permission to start cutover
before its result has been reviewed. No extra coding-model assignment is needed.
