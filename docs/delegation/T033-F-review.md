# T033-F independent release review — 2026-09-17

Accepted locally after lead corrections and an actual isolated Docker rehearsal.
The incoming summary listed only its result document, but all six required new
release/readout/test/runbook/result files were present and inspected.

## Findings corrected by the lead

1. The preflight required an explicit false background-filter setting from the
   deployed 2ba16c1 Settings model, which predates that field. A new regression
   failed first. Missing new fields are now treated as disabled only on the pinned
   old-version path; explicit enabled values still fail, and candidate settings
   still require both exact new values plus all unchanged old settings.
2. The opt-in local rehearsal used a phase-3 dependency image as though it already
   contained phase-4 voice files. Its real execution failed at the missing
   `services/conversation/events.py`. The test now builds a synthetic baseline
   source layer from the actual deployed 2ba16c1 Git files over the existing local
   dependency image. The candidate still copies only the four allowed modules.
   Rollback now uses the actual old Settings source and proves the new fields are
   absent, rather than using the candidate Settings file as an old-version fake.
3. Clean Git staging exposed a raw CRLF manifest hash mismatch. A new test failed
   against the exact LF-normalized manifest Git publishes. The script now pins
   and hashes that declared normalization, with no content or entry changes.
   Original local raw SHA remains `f2e9581422ffebaf364019cd3c42ae6d2a3e55581bbcba3c42f28f34c626ef7b`;
   canonical/published SHA is `c606c818fcdcb9bd0fc64ff06087d8e8a6409e80b25819d00f6a569e805da18c`.

A lead harness attempt initially used a raw image ID in Dockerfile FROM, which
BuildKit interpreted as a registry reference and refused. It was corrected to a
verified local seed tag; no image was downloaded. This was a harness correction,
not a deployment-script defect. The real server script already uses a pinned tag.

## Validation

- Initial focused suites:33 passed/one opt-in skip.
- Corrected offline release/readout suites:34 passed/one opt-in skip.
- Final offline suites after the Git-normalization regression:35 passed/one skip.
- Full working-tree suite:554 passed/51 skipped/two old voice-release manifest
  failures, with existing audioop/ffmpeg warnings. The old release guards remain
  unchanged; the new T033-F package validates all36 frozen assets independently.
- Clean staged Git export:555 passed/51 skipped/the same two old voice-release
  manifest failures. No additional failures or missing test-support files.
- Explicit `T033_F_RUN_DOCKER_TESTS=1`, desktop-linux: **one real rehearsal passed**
  in 50.71 seconds. It ran candidate image source/import checks, all four speech
  test directories, parent-plus-nested settings mounts, diagnostics-on/filter-off,
  replacement and exact previous-image/settings/contract restoration.
- All36 accepted normalized asset hashes and the manifest's original SHA-256
  remain unchanged. `git diff --check` passed with existing line-ending notices.

The Docker engine was initially stopped; the installed background engine was
started locally. No packages were installed. Rehearsal flags were restored.
Independent final Docker inventories contained no containers, only the three
default networks, and exactly the nine original image tags. All rehearsal
containers, networks, baseline/seed/parent/candidate tags were removed.

The rehearsal used synthetic credentials and no public ports or voice/SMS/provider
traffic. It does not prove the DigitalOcean private mounts, nginx/TLS or real call
quality; the owner-run procedure validates server fingerprints before cutover.
No database migration, backup restore or server operation was performed locally.

## Release boundary

Source defaults remain diagnostics off and filtering off. The reviewed rollout
enables numeric diagnostics only on the beta server, retains filter false, and
preserves schema0004 and automatic-notification settings. It retains exact
previous-image/override recovery and provides a bounded numeric-only log reader.

Owner sequence after publication: finish beta calls; execute the pinned release;
after success make quiet and nearby-speech non-booking calls; return the reader's
JSON and observed behavior. No new recording-file pilot, threshold change or claim
that diagnostics solve unwanted interruptions. Parent noise/turn tasks remain open.
