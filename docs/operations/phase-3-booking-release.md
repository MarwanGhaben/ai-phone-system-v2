# Phase 3: bounded booking-guard release

This is an owner-run app-only update for the inspected beta server, not general
production readiness or a fix for all Arabic/scheduling defects.

## Verified starting state

Owner console reported checkout 1480eeb on codex/phase-2-database-foundation;
the only tracked modification is nginx/nginx.conf, SHA-256
59ac9bdaa9a86c1022a573c9709faddf7e93a53015da8acfcc4330b3e288f3ab.
The running app image is
sha256:01cee33a5a13aea37919568f6d00e0f2c12a8e98bd6eb93ab063b73f51711605.
The private Compose override exists. Disk free: 8.4 GiB. Preserve the phase-2
release directory and its runtime.env mount, the TLS hotfix, certificates and timer.

## Procedure

The lead supplies a full published commit on codex/phase-3-safe-booking. Finish
test calls before running `scripts/deploy-booking-guards.py COMMIT` as root in the
DigitalOcean browser console. Do not run the old database cutover or generic deploy
script. No additional database migration is included in this change.

The script:

1. Locks concurrent executions and verifies checkout, image, private override,
   nginx fingerprint, free disk and the app-only source scope.
2. Stores protected recovery files in a new /opt/ai-phone-booking-release.* directory
   and tags the previous image. Logs/settings may contain credentials; do not paste
   or publish them. It does not take or restore a database snapshot: no schema
   change is being applied and existing protected backups remain in place.
3. Builds a layer on a verified local tag of the exact deployed image, copying only
   services/conversation/orchestrator.py. It does not reinstall dependencies or use
   a broad build context. Source differences in other runtime/config/schema files
   cause rejection. The image revision label identifies the reviewed Git commit.
4. Runs the two focused booking test files in the candidate with network disabled,
   synthetic test settings, resource limits and no live configuration mounts.
   Removes the exact temporary test container and verifies removal.
5. Copies the existing override and changes only services.app.image. Existing
   escaped environment strings, command, runtime.env and other mounts are preserved.
   It compares rendered mounts/networks and effective settings in a configuration-
   only probe before downtime. The migrate service remains on its previous image
   and is never executed by this app-only release.
6. Gracefully stops nginx and the app, replaces only the app with --no-deps, checks
   database readiness and effective settings, updates the private image pin,
   restarts nginx and validates public HTTPS /health without bypassing TLS checks.
7. Attempts to restore the previous app image and exact override on cutover failure,
   verifies readiness/settings/HTTPS, and reports rollback success. If rollback
   cannot complete, the common stop marker requires operator review; do not rerun
   blindly. Existing database records are never rewound.

This image release does not switch the server source checkout or rewrite nginx's
single-file mount. The server checkout remains phase 2; the app's actual phase-3
commit/image are recorded in the protected release directory's `deployed` file
and the image revision label. Use those for deployment identity. A later full
checkout alignment must preserve the live TLS mount and private configuration.

Expected success markers:

```text
CANDIDATE_IMAGE_BOOKING_TESTS_PASSED
EFFECTIVE_SETTINGS_AND_MOUNTS_PRESERVED
BOOKING_GUARDS_DEPLOYED_READY_HTTPS_OK
DEPLOYED_COMMIT=...
DEPLOYED_IMAGE=...
```

If stopped, return only the console markers and release directory; keep protected
files for diagnosis. After success, retry the dashboard and one short inbound call.
Actual appointment tests require deliberate tester choices; this deployment does
not itself create a calendar event.

## Validation evidence

- Seven subprocess-boundary scenarios pass: successful pin/preservation, replacement
  failure, running-settings drift, HTTPS failure, override drift, probe-settings
  drift, and mount drift. These simulate Docker/Git execution, not a live cutover.
- Real local Docker derived-image build and network-disabled tests passed: 18 tests,
  one existing Python 3.12 dateutil deprecation warning, 2.90 seconds. Test container
  removal verified. This used the local phase-2 validation image, not the server's
  image; the script repeats the image tests against the actual server base.
- Actual Docker verification caught raw sha256 image-ID interpretation in FROM and
  parent conftest discovery. The corrected procedure uses a verified local image
  tag and passes the test root directory so synthetic settings load.
- No live server execution is claimed by this document.
