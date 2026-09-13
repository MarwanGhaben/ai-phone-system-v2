# T018-C lead acceptance, 2026-09-12

Accepted for the separate manual beta rollout after two direct release fixes.
Server remains source075d8cf/imagef65fd1b4 with schema0002; no server or Microsoft
access occurred during this review. Accepted T018-B application files were not
changed by the release assignment or these corrections.

## Corrections

- Removed unsupported `--no-build` from `docker compose run`. Actual installed
  Compose help confirms the flag is absent; the prior mocked assertion required
  it and would have let the owner hit a pre-cutover deployment failure. Existing
  immutable images and `--pull never` remain; app replacement retains the valid
  `compose up --no-build` flag.
- If recovery fails after restarting nginx, attempt to stop both ingress and app
  again. A failed first stop does not skip the second. Keep the original cutover
  failure and an explicit unverified-recovery marker; never claim stop succeeded
  without verification. Added the failure regression.
- Report sanitized failure type even for errors before release-directory setup
  (argument, lock, prior incomplete cutover), instead of silent process exit.

## Evidence

- Initial SOL release suite: 16 passed locally. Corrected suite: 17 passed.
- Full normal pytest: 158 passed, 35 optional skips, two existing warnings,
  6.28 seconds. Prior accepted T018-B evidence includes real PG 75 passed and
  explicit Docker/Compose18 passed; application files have not changed since.
- Actual new release `rehearsal` method ran against a compressed synthetic
  PostgreSQL16 backup containing aware and naive bookings. It verified predecessor
  history, applied0003, verified repeat safety, candidate/fallback compatibility,
  no backfill and identical row fingerprints for all17 business tables.
- Actual `prepare_override` and `settings_probe` methods ran with local Compose
  and real candidate/fallback images, preserving dollar signs, multiline settings
  and read-only mounts while accepting only the three intentional new settings.
  On Windows, the proof adapts the bind source spelling to Docker's translation;
  it does not change the server preservation code.
- Temporary seed/rehearsal/settings containers and networks were removed and
  cleanup confirmed. Existing validation images and synthetic proof artifacts
  remain local. Proof source: `.repo-review/t018c_release_rehearsal.py`.
- The first local proof attempt had a fixture-only missing `applied_at` when
  seeding0002 history; fixed the synthetic seed and reran. No runtime/SQL change.

Recovery decision failures are covered at the deployment I/O boundaries; no live
DigitalOcean cutover was exercised locally. Actual candidate and compatible
fallback app startup/readiness and0003 drift rejection were separately proven
in T018-B's real image rehearsal. Remote outage is allowed to degrade observation
freshness while the phone app remains ready.

## Owner outcome and limits

The eventual script enables polling every60seconds with180-second freshness,
preserves private overrides/runtime.env/TLS and historical bookings, and checks
a restored backup before stopping test traffic. Row18 may become unavailable in
Microsoft while its original booking is retained. Row19 may remain freshly
present. Neither result is preclaimed. This does not infer cancellation from404,
prove direct Outlook-copy propagation or suppress legacy reminder records.
No new owner inventory or another SOL implementation round is needed.
