# T049-B automatic-notification rollout

This is a manual beta rollout procedure for migration 0004 and the accepted
T049-A automatic-notification runtime. It has not been executed on DigitalOcean.
The lead must replace `<REVIEWED_RELEASE_COMMIT>` with the accepted, published
40-character commit. The placeholder is intentionally not runnable.

The script requires the known server baseline before it changes anything:

- checkout `1480eeb0e36596e26d0dedde77f08beee91aec5b`;
- running source `424ee6ada7244bf7b8f89b2e3fbb50255b77a1de` and image
  `sha256:2ac002d96da4541ed08e0f221bb559bc4c70d9762b3fab5735c81d92171459ea`;
- exact 0001/0002/0003 history with canonical booking and provider-observation
  rows, and no 0004 notification tables;
- the protected Compose override, parent `/app/config` mount, and exactly one
  nested read-only settings-file mount from the accepted observation release;
- the existing observation settings, required Graph/Telnyx configuration,
  nginx hotfix, database/Redis health, and certificate-validated HTTPS health;
- at least 1 GiB free disk and 512 MiB available memory.

The script reads its 19 application assets and Compose scope from the exact
commit, derives one candidate from the running immutable image with build
networking disabled, and replaces the prior nested settings source with the new
release's `candidate-settings.py`. It preserves the parent config mount, private
runtime.env mount, TLS files, environment values, dollar signs, multiline values,
networks and service options. Keep the successful release directory because the
running application uses its settings-file mount.

## Owner execution after lead acceptance

Finish test calls and pause beta testing. In the **DigitalOcean browser console**,
as root, run only the lead-supplied commit after it has been published:

```sh
cd /opt/ai-phone-system-v2
REVIEWED_COMMIT=<REVIEWED_RELEASE_COMMIT>
git fetch --no-tags origin "$REVIEWED_COMMIT"
set -o pipefail; git show "$REVIEWED_COMMIT":scripts/deploy-automatic-notifications.py | sudo python3 - "$REVIEWED_COMMIT"
```

Do not substitute a branch name, reuse an older deployment script, edit the
server checkout, prune images, delete release directories, or rerun an attempt
whose release directory contains `cutover-started` without lead review. Do not
paste protected logs, dumps, overrides, environment files, settings snapshots,
provider responses, message bodies, or customer identifiers.

## What the script does

1. It obtains the root-only deployment lock and performs read-only baseline,
   configuration, schema, capacity, mount, nginx and HTTPS checks.
2. It builds the bounded candidate without pulls or dependency installation and
   renders protected active and paused overrides. Settings-only probes do not
   start the application, workers, Graph reads or SMS submissions.
3. It makes and verifies a protected backup, restores it into a resource-capped
   PostgreSQL 16 container on an internal temporary network, and waits for an
   authenticated TCP query to the intended database. It verifies exact 0003,
   applies 0004 twice, and checks that all 17 business tables plus provider
   observation/control rows retain identical values. Reconciliation and outbox
   rows must be empty. The actual candidate then reaches readiness with automatic
   notifications enabled and all notification workers paused. A synthetic booking
   on that restored copy must atomically create held confirmation/reminder jobs
   without changing legacy SMS logs. Temporary app, database and network resources
   are always removed and checked.

   The corrected candidate contract validates the restored 0003 structure because
   PostgreSQL can serialize equivalent CHECK expressions differently after restore.
   It recognizes only the two tested equivalent forms and never rewrites recorded
   baselines. The original live image/0003 contract is verified before cutover.
4. Immediately before cutover it repeats the complete live fingerprints. It marks
   cutover started, gracefully stops nginx and app, verifies both stopped, makes
   the final protected backup, applies 0004, and repeats contract and row checks.
   It never restores a backup over the live database automatically.
5. It first starts the 0004-capable candidate with automatic notifications enabled
   and workers paused, then checks readiness and public HTTPS. It performs a second
   explicit stop/restart with the active override. That restart sets paused=false
   and is the point at which normal Graph observation and notification processing
   may begin. The script sends no test SMS and makes no direct provider probe.

Expected success markers include:

```text
PREFLIGHT_0003_AND_HTTPS_OK
PROTECTED_BACKUP_VERIFIED=rehearsal.dump
RESTORED_0003_TO_0004_AND_PAUSED_READY_OK
REHEARSAL_RESOURCES_REMOVED
PRE_CUTOVER_FINGERPRINTS_OK
CUTOVER_STARTED_STOPPING_BETA_TRAFFIC
PROTECTED_BACKUP_VERIFIED=final-after-writer-stop.dump
LIVE_0004_AND_PRESERVED_ROWS_OK
PAUSED_CANDIDATE_READY_HTTPS_OK
AUTOMATIC_NOTIFICATIONS_0004_WORKERS_ACTIVE_READY_HTTPS_OK
DEPLOYED_COMMIT=<40-hex commit>
DEPLOYED_IMAGE=sha256:...
```

Record those public markers and `RELEASE_DIRECTORY`. HTTPS/readiness proves local
application health; it does not prove Graph success, carrier acceptance, customer
receipt, or direct Outlook propagation.

## Recovery behavior

Before 0004 commits, recovery may restore the verified old image and exact prior
override only when the database still passes the exact 0003 contract. Once 0004
is committed, the old image and feature-disabled mode are incompatible recovery
choices: disabling automatic notifications would reactivate the legacy JSON
scheduler. Recovery instead uses the same 0004-capable candidate with
`AUTOMATIC_NOTIFICATIONS_ENABLED=true` and
`AUTOMATIC_NOTIFICATION_WORKERS_PAUSED=true`. It does not delete, suppress,
requeue or resend pending, accepted or unknown jobs.

`RECOVERY_SMS_PAUSED_READY_HTTPS_OK` means the original rollout failed but the
degraded candidate is ready with both new workers and the legacy scheduler off.
It is not a deployment-success marker. `RECOVERY_PRE_0004_READY_HTTPS_OK` means
the database remained exact 0003 and the old application recovered.
`RECOVERY_REQUIRES_REVIEW_INGRESS_LEFT_STOPPED` or
`RECOVERY_UNVERIFIED_INGRESS_MUST_REMAIN_STOPPED` means nginx and app must remain
stopped for schema-aware review. A submitted SMS cannot be recalled, and the
paused mode takes effect only on application restart.

## One post-rollout smoke batch

After the success marker, perform one supervised batch using a newly created test
booking. Verify that the same booking enrolls after a fresh Microsoft read and its
confirmation records a provider message ID as **accepted**. Accepted means the
carrier accepted the submission; separately record whether the test phone received
it. Cancel that same appointment through Microsoft Bookings, then verify two spaced
404 observations plus complete inventory absence lead to its local automatic
removal, its reminder suppression and one removal notice. Let later polls run and
verify no second removal notice. Do not use historical unenrolled row 18 as the
positive test and do not change any unrelated provider appointment.
