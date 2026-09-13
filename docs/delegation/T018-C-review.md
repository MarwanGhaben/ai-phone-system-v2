# T018-C lead acceptance, 2026-09-12

## Pre-cutover mount-order correction, 2026-09-13

Owner diagnostic for release `.oomp1k6t` confirmed cutover_started=false,
deployed_marker=false and every health/configuration check true except
mount_list_order_matches. mount_contents_match=true proves the same mount
dictionaries were returned in a different order. The pre-cutover guard still
compared raw lists, although replacement verification already ignored ordering.

Added a failing-first regression against the actual pre-cutover method: the old
code raised `running app changed before cutover` for reversed identical mounts.
Both pre-cutover and replacement now use the same sorted full-dictionary
fingerprint. Every field and duplicate count remains significant; no mount is
dropped or compared by path alone. The regression rejects changed source,
destination, read/write, mode, propagation or type, and missing/duplicate mounts.
No server configuration, runtime application, database or dependency changes.
Focused release/diagnostic21 passed. The last server attempt's settings and
restored0003/fallback rehearsal already passed; it never stopped writers.
The next owner action is a fresh guarded attempt using the corrected pinned
commit, not a manual edit of the private override or a skip of pre-cutover checks.

## Settings-mount incident and correction (latest evidence)

The first server attempt stopped at settings, before cutover, in
`/opt/ai-phone-observation-release._jjs4r2g`. Owner diagnostic confirms no cutover
marker, matching Compose values/mounts/networks, and the original app still healthy.
Its settings probe returned the old model without observation fields.

The server's existing `/app/config` bind mount hides the new image settings.py.
The initial local proof lacked that directory mount and missed this packaging
condition. Lead reproduced the exact behavior with the old config directory
mounted onto the real candidate image: observation fields disappear even though
the new environment variables are present. A pinned candidate-only settings.py
file bind restores the new settings without modifying the shared config folder.
The corrected preservation checks permit only this precise read-only source/target
in addition to the three new environment values; all previous mounts/settings
remain required. Fallback gets no additional file mount.

The initial diagnostic's import-error category was a false positive from literal
`except ImportError:` source logged during Git blob reads. Corrected it to match
exception lines at the start of a line; added a regression. No Pydantic dependency
change was needed. Application source and migration bytes remain unchanged.

Actual proof `.repo-review/t018c_settings_mount_rehearsal.py`: reproduced hidden
settings, passed corrected candidate and fallback settings probes with real
Compose/images and synthetic dollar/multiline values, verified original config
source unchanged, and confirmed all temporary container/network cleanup.
Focused release/diagnostic23 passed; full normal164 passed/35 optional skips/two
existing warnings in5.39seconds. No further SOL work or owner inventory is needed.
Retry must use the new pinned script commit; the earlier failed release has no
cutover marker, so a fresh guarded attempt is permitted. Do not reuse its override.

## Original release acceptance

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
