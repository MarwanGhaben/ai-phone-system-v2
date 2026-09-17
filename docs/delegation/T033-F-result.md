# T033-F diagnostics release result

Lead review completed on 2026-09-17; see `T033-F-review.md`. Lead corrected the
old Settings-field preflight mismatch and the synthetic Docker baseline fixture.
Corrected offline suites34 pass/one skip; explicit Docker rehearsal passed with
candidate speech tests, parent/nested mounts, replacement and exact rollback.
All36 frozen assets remain unchanged. Original implementer evidence follows.

Implemented locally on `codex/phase-4-arabic-voice-quality` at unchanged HEAD
`3b062d435e7bfc907f7feda8bef5949e7ffd843c`. Nothing was committed, staged,
pushed, deployed or sent to a server/provider. Background filtering and runtime
diagnostics remain disabled.

## Implementation

`scripts/deploy-barge-in-diagnostics.py` is an import-safe owner-run app-only
release. It validates the established source/image, checkout, nginx hotfix,
private override, effective settings and mounts, dependency health, public HTTPS,
capacity and the read-only schema-0004 contract before cutover. It requires root,
the shared deployment lock and one future reviewed 40-hex commit. The candidate
must descend from the deployed source and have exactly the five application paths
declared by the immutable manifest; all six new release files must exist in that
same commit.

The release reads and verifies all 36 normalized manifest entries from the pinned
commit. It derives a candidate from a verified local tag of the exact running
image with no pull or dependency install, copies only the four runtime modules,
checks those four and six protected runtime hashes, and imports the reviewed
runtime modules. A separate read-only, network-disabled container mounts only the
accepted tests and three support assets and runs all four speech test directories.
Named one-shot containers are registered before creation and verified removed on
success, failure and timeout.

The candidate override preserves the server's existing private configuration and
changes only the app image, `BARGE_IN_DIAGNOSTICS_ENABLED=true`,
`ELEVENLABS_STT_FILTER_BACKGROUND_AUDIO=false`, and one exact read-only nested
mount from the protected release's `candidate-settings.py` to
`/app/config/settings.py`. The parent `/app/config` mount and every unrelated
setting, mount field, duplicate, network and port remain significant. A no-lifespan
one-shot probe verifies effective settings and all ten runtime/protected hashes
through the actual mounts.

Cutover rechecks all captured fingerprints, stops ingress and app, atomically
installs the candidate override and replaces app only with no build, pull or
dependency recreation. It then verifies readiness, Docker health, source hashes,
settings, mounts, schema history, nginx and certificate-validated HTTPS. Any
failure after traffic stops attempts the exact prior image, override and settings.
A TLS failure closes candidate ingress before recovery; failed recovery forces
ingress closed and never emits the success marker. No migration, database restore,
checkout reset, provider operation or generic Docker cleanup is performed.

`scripts/read-barge-in-diagnostics.py` uses a fixed bounded `docker logs` command
for `ai-voice-app` (last 30 minutes, tail 2000), a 20-second subprocess deadline,
2 MiB total capture cap and 32 KiB line cap. It terminates/kills the subprocess,
closes both pipes and joins both reader threads on timeout or a limit. It accepts
only schema-1 `barge_in_diagnostics_summary` JSON with an opaque lowercase 32-hex
ID, exact field/category allowlists, finite nonnegative numbers and at most 16
windows. It rebuilds safe output, retains at most five summaries and never echoes
raw stdout, stderr, malformed objects or exception text.

## Failing-first evidence

The first two release contract tests were added before the release corrections.
The run failed as expected:

```text
ERROR: Release had no settings_overlay
FAIL: corrupted protected runtime was not rejected
Ran 2 tests
FAILED (failures=1, errors=1)
```

The completed implementation adds the protected nested settings overlay and
verifies the six protected runtime assets, resolving both failures. Later failing
runs were test-fixture corrections for Windows mode reporting and mocked command
semantics; they were not product regressions.

## Validation

Available standard-library validation passed:

```text
C:\Python314\python.exe -B -m unittest \
  tests.integration.test_barge_in_diagnostics_release \
  tests.integration.test_barge_in_diagnostics_readout
Ran 34 tests
OK (skipped=1)

AST_OK_4
```

The 33 executed cases cover import safety; exact baseline/scope/new-file guards;
all 36 accepted hashes; missing and changed protected/test-support assets; hidden
runtime mounts; order-insensitive full mount comparison with changed fields and
duplicate counts retained; exact diagnostic environment/settings exceptions;
preservation of unknown fields, dollars and newlines; parent-plus-nested settings
mount construction; four-directory isolated test invocation; pre-cutover test
failure; read-only exact schema history; timeout and cleanup behavior; actual
replacement contract checks; successful cutover; readiness and TLS recovery; and
failed recovery with ingress closed.

Reader cases use real-format synthetic summaries beside secret-like lines and
cover logger prefixes, unknown fields, booleans, invalid categories/IDs, NaN,
malformed JSON, more than 16 windows, oversized lines, five-summary output caps,
bounded-scan status, fixed Docker arguments, subprocess failure, real timeout and
real byte-limit cleanup. The opt-in Docker/Compose case is present under
`T033_F_RUN_DOCKER_TESTS=1` and was explicitly skipped because Docker is absent.
Running the reader CLI in that environment returned only its fixed sanitized JSON
with `status: read_failed` and exit code 1; it printed no command error or raw log.

The environment has Python 3.14 only, without pytest or project dependencies, and
has no Git or Docker executable. The prescribed Python 3.11 focused/speech/full
pytest runs, `git diff --check`, and the opt-in derived-image/Compose rehearsal
could not run here. No dependency was installed. The lead should run:

```powershell
C:\Python311\python.exe -m pytest -q tests/integration/test_barge_in_diagnostics_release.py tests/integration/test_barge_in_diagnostics_readout.py
C:\Python311\python.exe -m pytest -q tests/stt tests/tts tests/telephony tests/conversation
C:\Python311\python.exe -m pytest -q
$env:T033_F_RUN_DOCKER_TESTS = '1'
C:\Python311\python.exe -m pytest -q tests/integration/test_barge_in_diagnostics_release.py
Remove-Item Env:T033_F_RUN_DOCKER_TESTS
git diff --check
```

The immutable manifest still has SHA-256
`f2e9581422ffebaf364019cd3c42ae6d2a3e55581bbcba3c42f28f34c626ef7b`,
and all 36 CRLF-to-LF-normalized entries match after implementation. The branch
ref still points to the required unchanged HEAD. Standard-library tests created no
Docker/network/server/provider resources; temporary directories and child
processes were cleaned up, and bytecode writing was disabled.

## Remaining limitations

The opt-in rehearsal uses an already-available local base image and synthetic
settings. It can exercise the parent-plus-nested mount, candidate imports,
diagnostics-on/filter-off configuration, Compose replacement and exact rollback;
it cannot prove the DigitalOcean image digest, private production mounts, nginx,
public TLS, real PostgreSQL, phone timing or cleanup on the server.

The release has not been published or run, and the runbook deliberately retains
the conspicuous `PUBLISHED_T033_F_COMMIT` placeholder. Numeric summaries cannot
identify a speaker, prove packet loss, establish a safe threshold or show that
diagnostics solve unwanted interruptions. No filtering change should follow until
the lead reviews supervised call evidence.
