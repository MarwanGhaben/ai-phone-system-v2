# T018-C provider-observation rollout (manual owner step)

This procedure passed local lead acceptance. It has not been executed on
DigitalOcean. Use the exact published 40-character commit supplied by the lead
for `<REVIEWED_RELEASE_COMMIT>` below; do not substitute a branch tip. The source
commit cannot contain its own hash, so this document keeps that parameter explicit.

The server checkout remains at `1480eeb0e36596e26d0dedde77f08beee91aec5b`.
The running app is source `075d8cf6fc29c4cbdf69b5c82baa130d5c0c540d`,
immutable image `sha256:f65fd1b449e2e107d2137158fc05bdbcc376c301b74c97e4a56c57a73b576a21`.
The script verifies those independently. It also verifies the accepted 0001/0002
ledger and nginx hotfix before any cutover. A drift refusal is a stop condition,
not a reason to edit the server manually.

Before the owner runs this, finish calls and pause beta testing. The release
script needs root, existing Docker/Compose and `curl`, at least 1 GiB free disk
and 512 MiB available memory. It creates root-protected backups and a disposable
isolated PostgreSQL rehearsal, then stops ingress and app for the final backup,
0003 migration and app-only replacement. It does not change the base Compose
file, checkout, TLS files, database/Redis/certbot containers, old image or
private runtime.env mount. The existing on-server-only backup risk was accepted.

**DigitalOcean browser console: use the lead's published exact commit:**

```sh
cd /opt/ai-phone-system-v2
REVIEWED_COMMIT=<REVIEWED_RELEASE_COMMIT>
git fetch --no-tags origin "$REVIEWED_COMMIT"
set -o pipefail; git show "$REVIEWED_COMMIT":scripts/deploy-provider-observations.py | sudo python3 - "$REVIEWED_COMMIT"
```

The command reads the script and every staged runtime asset from the same exact
commit. Do not copy a long Python paste or run the old persistence release
script. The lead will supply the pinned SHA and final short command after local
acceptance; the placeholder here is intentionally not runnable.

The settings probe uses `compose run --no-deps --pull never` with an existing
immutable image. `--no-build` is deliberately absent because `compose run` does
not support that option; app replacement uses `compose up --no-build`.

Expected safe markers progress through `PREFLIGHT_0002_AND_HTTPS_OK`,
`PROTECTED_BACKUP_VERIFIED=rehearsal.dump`,
`RESTORED_0002_TO_0003_AND_FALLBACK_OK`, `REHEARSAL_RESOURCES_REMOVED`,
`PRE_CUTOVER_FINGERPRINTS_OK`, `CUTOVER_STARTED_STOPPING_BETA_TRAFFIC`,
`PROTECTED_BACKUP_VERIFIED=final-after-writer-stop.dump`,
`LIVE_0003_AND_BUSINESS_ROWS_OK`, then
`PROVIDER_OBSERVATIONS_DEPLOYED_READY_HTTPS_OK`. Record the printed
`DEPLOYED_COMMIT`, immutable `DEPLOYED_IMAGE`, and `RELEASE_DIRECTORY`.
Normal HTTPS health proves app/ingress health only; it does not prove a Microsoft
readback result or actual Outlook-copy propagation.

On failure, return only the fixed public markers and the stage/type marker to
the lead. Do **not** paste `private.log`, overrides, environment, settings,
database dumps or raw provider responses. If cutover started, never blindly
rerun this script. `RECOVERY_READY_HTTPS_OK` means the compatible old app or
fallback returned; the original release still failed and needs review.
`RECOVERY_REQUIRES_REVIEW_INGRESS_LEFT_STOPPED` or
`RECOVERY_UNVERIFIED_INGRESS_MUST_REMAIN_STOPPED` means ingress must stay stopped
for a schema-aware manual decision. Never downgrade 0003 or automatically
restore an old backup over live data.

After a successful deployment, perform one authenticated dashboard observation
check without deleting or changing appointments. Booking row 18 should be
flagged **unavailable** if Microsoft still returns 404. Booking row 19 should
show a **freshly present** observation if it still exists in Microsoft.
Report the displayed state/timestamp and any safe marker, without names,
provider payloads or secrets. A different result is evidence to investigate,
not an instruction to edit the provider. Observation polling is explicitly
enabled at interval 60 seconds with 180-second freshness on the new app only;
remote failure should leave local readiness intact.

For review, the script preserves the first and final protected archives,
SHA-256 files, rendered private overrides and logs in its release directory.
Its rehearsal uses a synthetic isolated network, capped PostgreSQL 16 tmpfs
container and authenticated TCP readiness; it checks nonempty aware/naive rows,
all 17 preexisting business-table row fingerprints, repeat-safe 0003 and both
candidate/fallback schema compatibility. It verifies removal of the named
rehearsal container and network even after partial creation.
