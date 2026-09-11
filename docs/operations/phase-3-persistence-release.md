# T005-D migration-aware beta rollout

Application source: `075d8cf6fc29c4cbdf69b5c82baa130d5c0c540d`.
Owner returned the expected healthy running image
`sha256:8c0b299785bed1703cb897205591e63121256772f138021e687f6b71e80d7264`
and 0001 checksum
`53c861ac91cae5ecaf9f794b15563c88ee58bc0dafa5ae64fa81c89d907aa3a9`.
Server directory: `/opt/ai-phone-system-v2`. Existing checkout remains 1480eeb;
running application is the image selected by the protected Compose override.

## Stopped rehearsal and correction (2026-09-11)

The owner's first attempt at script eee5706 stopped after `rehearsal.dump`.
Read-only marker inspection confirmed no cutover or deployment; the previous
b1c4dc4 app remained healthy. Sanitized log categories locate the restore failure
at a missing database. Source inspection identified a startup race: socket-only
`pg_isready` can accept the image's temporary initialization server before the
requested database exists. The [official image entrypoint](https://github.com/docker-library/postgres/blob/master/docker-entrypoint.sh)
starts that socket-only server before creating POSTGRES_DB and later replaces it
with the final server.

The corrected gate requires a successful, password-authenticated TCP query
returning `current_database() = rehearsal` before TCP restore, with bounded
waiting. Live migration, application source, resource limits, backup protection
and fallback behavior are unchanged. No server memory change is justified by the
earlier diagnostic: its broad OOMKilled match also matched false values. The
classifier now distinguishes true OOM flags and excludes normal network-not-found
cleanup evidence.

Validation: 18 focused release/diagnostic tests passed, including a delayed/missing
database gate and diagnostic false-positive regression. The actual release
rehearsal passed locally against isolated PostgreSQL 16 with synthetic legacy
rows: authenticated readiness, restore, old contract, migration, historical-row
preservation, candidate/fallback contracts and cleanup. Server retry remains
pending; keep the failed release's protected files.

## Owner execution

End test calls before running the pinned script supplied by the lead. Use the
DigitalOcean browser console as root. Fetch `codex/phase-3-safe-booking`, then
pipe `scripts/deploy-booking-persistence.py` from the exact release commit to
`python3 - RELEASE_COMMIT`. The script loads its shared release helper from that
same exact commit. It always packages application source 075d8cf.

Do not rerun the older app-only deployment command. No local PowerShell, GitHub
website change, server checkout switch, broad cleanup or Docker volume deletion
is required. This is an existing beta rollout, not public-launch acceptance.

## What executes

1. Acquire the existing deployment lock; verify checkout, scoped source changes,
   image/health, nginx fingerprint, protected override, settings, mounts, networks,
   database identity and old read-only schema compatibility.
2. Build an allowlisted candidate from the exact current image. Build a fallback
   retaining the previous app with only the current schema checker. No image pull,
   dependency installation or whole-repository build context. The seven runtime
   paths must exactly match the reviewed source difference.
3. Validate effective settings and mounts for both app choices. Pin the Compose
   migrate service to the candidate image so future starts use the current runner.
   Retain existing environment values, secret mounts, dollar escaping and TLS.
4. Create a protected custom-format database dump and verify its archive index.
   Restore into an isolated temporary PostgreSQL container on an internal network.
   Check old contract, apply 0002, verify both candidate/fallback contracts, compare
   a fingerprint of all historical booking row values and require their new
   canonical column to remain NULL. Remove temporary containers/network.
5. Recheck configuration/schema; gracefully stop nginx and app, create another
   protected dump after writers stop, apply 0002, replace app, verify readiness,
   settings/image, atomically install the private override, restart nginx and
   check public HTTPS health. Database/Redis/certbot services are not replaced.

Backups, settings snapshots, escaped overrides, image IDs and private logs remain
under a root-only `/opt/ai-phone-persistence-release.*` directory. Keep it and the
earlier runtime.env directory. Do not paste private files or dumps. The owner has
accepted backups remaining on-server for this rollout.

## Recovery

Every migration/check container is removed even if the Docker client times out.
After a cutover error, stop ingress/app and inspect actual schema compatibility:
use the compatible fallback on 0002, or the original image if schema remains 0001.
Do not infer migration failure from a timed-out client. Unknown schema or failed
recovery stays an explicit failure; no automatic database restore/drop-column.
Successful recovery prints `RECOVERY_READY_HTTPS_OK` and its image ID, then the
original deployment attempt still exits nonzero. Fallback restores prior beta
behavior; it does not include the new booking-save fix.

Success prints:

```text
RESTORED_BACKUP_MIGRATION_AND_FALLBACK_OK
LIVE_DATABASE_0002_OK
BOOKING_PERSISTENCE_DEPLOYED_READY_HTTPS_OK
DEPLOYED_COMMIT=075d8cf6fc29c4cbdf69b5c82baa130d5c0c540d
DEPLOYED_IMAGE=sha256:...
```

If `RELEASE_STOPPED_TYPE=...` appears, return the public markers and release
directory to the lead; do not repeatedly rerun or expose private logs.

## Validation evidence

- Application acceptance: 128 tests passed with all database/Docker opt-ins;
  actual candidate and compatible fallback image startup/drift checks passed.
- Release harness: 15 focused tests passed, covering original release behavior
  and failure injection for backup, pre-commit migration, ambiguous post-commit
  timeout, app replacement and HTTPS; unknown-schema fail-closed; configuration
  drift; atomic override preservation; no live database restore on rollback.
- Full normal suite after packaging: 109 passed, 27 opt-in skips, two existing
  warnings. Application/database acceptance was already executed with those
  opt-ins; packaging does not change the reviewed application source.
- Actual new release restore/migrate/check/cleanup methods executed locally with
  a synthetic PostgreSQL 16 dump, candidate and fallback images. The isolated
  database/network were removed and harness cleanup verified. No live providers.

This is packaging evidence. Deployment and a new phone/Outlook/dashboard smoke
test still need owner-returned results. Automatic external calendar sync remains
unimplemented and is not implied by this release.
