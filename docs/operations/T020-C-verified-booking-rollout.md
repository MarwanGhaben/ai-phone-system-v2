# T020-C verified phone booking and schema 0005 rollout

Status: c60e774's owner rollout stopped before cutover during the synthetic
20-request journey. A targeted owner replay confirmed PostgreSQL deadlocks during
admission and provider-receipt persistence. The next candidate adds staff-scoped
admission serialization and a committed-claim check; it does not retry provider
creation, change SQL/checksums, or relax the database exclusion constraint.
Do not rerun c60e774. Await the newly reviewed and published correction SHA.
The corrected packaged candidate passed all 51 real PostgreSQL journeys with
zero skips. The real receipt-writer regression failed before the correction and
passed afterward; the current release offline suite passes 31 checks.

Historical acceptance before that server finding included all 35 Docker-enabled release
tests, 843 packaged candidate passes and 49 real database journeys. Use the exact
published commit supplied by the lead for the owner-controlled server run.
`PUBLISHED_T020_C_COMMIT` is deliberately unfilled. Use only the exact reviewed
40-character commit supplied after publication.

## Release boundary

The release derives a candidate from source
`792952b2965e85513bf0971a41bfaff7bd3377c4` and image
`sha256:14589fc6411ef54f64e0a519c41729698b113680ebc02fdbd94d53b5a0fa4369`.
It validates all 108 frozen inputs from
`T020-C-contention-hashes.json`, whose SHA-256 is
`c3f89ece10d06f9edf14ea191fbad9f2b12c1eaa7cc8533b0ec528607a46de4a`.
Both prior manifests are preserved. Relative to the published release manifest,
only the operation-store module and its PostgreSQL regression test file change;
the other 106 assets, including all SQL, remain identical.
Text inputs use CRLF-to-LF normalization and the three VAD model assets use raw
bytes. The candidate copies only the declared runtime, consultant/policy, and
migration assets. It installs no package and uses no floating dependency.

Candidate tests receive separately pinned, read-only copies of `.dockerignore`,
the accepted T007-A/T013-A result evidence, and the dashboard template. Dockerfile
and Compose come from the frozen 108 inputs. These files exist only in test
containers and do not expand the deployed application image or source scope.

The procedure preserves the active observation and automatic-notification
features, diagnostics, speech-aware gate, background-filter setting, model bytes,
ports, networks, credentials, private parent mounts, nested settings overlay,
nginx configuration, and every protected release directory. It adds schema 0005
with checksum
`cf9c9e2cdc53569a59833235efa98633b5f7b233f98549a8b2f1b16614866100`
and enables `VERIFIED_PHONE_BOOKING_ENABLED=true`. Notification workers are paused
while the candidate is verified with phone ingress stopped, then returned to their
previous active state before HTTPS ingress is restored.

The pre-cutover rehearsal creates a protected custom-format backup, restores it
into an isolated low-memory PostgreSQL 16 container, applies only
`python -m migrations.runner --prepare-operations`, verifies repeat safety and all
preexisting row fingerprints, and checks that the operation ledger contains zero
business operations plus one contract row. Accepted operation, create, phone,
readiness, scheduling, protocol, and speech tests use synthetic credentials and a
separate synthetic database. No live provider booking or SMS probe is performed.

Observation and notification workers may legitimately update their rows while the
old application remains active during build and rehearsal. The procedure therefore
rechecks source, image, settings, mounts, schema, nginx and HTTPS before stopping
writers, but does not compare the early row fingerprint across that live interval.
After nginx and the application have verifiably stopped, it takes the final backup
and authoritative row fingerprint. That fingerprint must remain exact through
0005 migration and both candidate replacements.

## Owner handoff: three steps after publication

1. Finish active beta calls and keep the deployment window free of new calls.
2. In the DigitalOcean browser console, run the command below only after the lead
   replaces `PUBLISHED_T020_C_COMMIT` with the reviewed full commit:

   ```bash
   bash <<'BASH'
   set -euo pipefail
   cd /opt/ai-phone-system-v2
   release='PUBLISHED_T020_C_COMMIT'
   test "$release" != 'PUBLISHED_T020_C_COMMIT' && test "${#release}" -eq 40
   git fetch --no-tags origin codex/phase-5-booking-reliability
   git show "$release:scripts/deploy-verified-phone-booking.py" | sudo python3 - "$release"
   BASH
   ```

3. Return the protected release path and fixed public markers, including the
   printed commit and image. Do not return `private.log`, backups, settings,
   environment output, database fingerprints, customer data, phone numbers,
   transcripts, provider output, CallSid, or stream IDs.

Do not run the placeholder command, a branch name, an abbreviated SHA, or the
mutable checkout copy. Do not rerun after any failure. Keep the new protected
release directory and all older release directories.

## Successful completion

The terminal success evidence is:

```text
VERIFIED_PHONE_BOOKING_DEPLOYED_READY_HTTPS_OK
DEPLOYED_COMMIT=<reviewed 40-character commit>
DEPLOYED_IMAGE=sha256:<candidate image ID>
PROTECTED_RELEASE_DIRECTORY=/opt/ai-phone-verified-booking-release.<unique suffix>
```

Earlier fixed markers record the 0004 baseline, 108 input hashes, derived image,
candidate test collection, backup verification, restore/migration rehearsal,
stopped ingress, paused candidate, schema/readiness checks, worker activation, and
final HTTPS restoration. Process exit alone is not release evidence.

## Failure and recovery states

The procedure records the original error in the root-only release evidence and
prints only fixed public markers.

- **Before schema 0005 commits:** the exact previous override, image, settings,
  schema 0004 readiness, and public HTTPS are restored. The terminal recovery
  marker is `RECOVERY_PRE_0005_PREVIOUS_APP_READY_HTTPS_OK`.
- **After schema 0005 commits:** the old schema-0004 image is not started. The
  schema-0005-aware candidate is installed with notification workers paused and
  booking ingress stopped. Existing operations, bookings, notifications, provider
  observations, and uncertain effects are retained. The terminal markers are
  `RECOVERY_0005_CANDIDATE_PAUSED_INGRESS_STOPPED` and
  `RECOVERY_MANUAL_REVIEW_REQUIRED`.
- **Unverified recovery:** ingress is forced closed where possible. Return
  `RECOVERY_FAILED_INGRESS_CLOSED` or
  `RECOVERY_FAILED_INGRESS_CLOSURE_UNVERIFIED`, followed by
  `RECOVERY_UNVERIFIED_MANUAL_REVIEW_REQUIRED`.

In either failure state, stop. Do not restore a live database automatically, run a
down migration, drop operation tables, enable the legacy create path, replay an
unknown provider write, or retry the deployment. The lead will inspect the
protected evidence and provide the next bounded action.

## Scope limits

This is a supervised create-side beta. It does not complete owned cancellation or
rescheduling, caller identity verification, unknown-ID reconciliation, all
handoff/overload work, voice and speaker evaluation, dashboard redesign, or the
remaining production-readiness parent gates. No post-release booking call or SMS
test is part of the deployment procedure.
