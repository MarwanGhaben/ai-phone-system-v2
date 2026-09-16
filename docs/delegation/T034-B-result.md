# T034-B implementation result

Lead review completed on2026-09-16. The original evidence below is historical;
see [independent acceptance and corrections](T034-B-review.md) for final Docker,
clean-release tests and the explicitly amended test-only manifest hash.

T034-B is implemented locally on `codex/phase-4-arabic-voice-quality` at unchanged
HEAD `3e1a7496dfd596a5294ab0124027dce3ac3724d5`. Nothing was committed, pushed,
deployed or sent to a provider. The accepted-hashes manifest was treated as
immutable input; all 25 normalized runtime/test hashes still match.

## Release package

`scripts/deploy-voice-beta.py` is an import-safe, owner-run app-only release. It
requires root, the shared deployment lock and a future reviewed 40-hex commit. It
verifies ancestry, the exact eight-file application scope, the unchanged phase-2
checkout/nginx hotfix, the deployed source/image, protected override, healthy
dependencies, active observation/notification settings, certificate-validated
public health and the exact read-only schema-0004 history and compatibility gate.

The script reads the manifest, eight runtime files and 17 accepted test-support
files from that same commit. It normalizes only CRLF to LF, rejects any hash drift,
and builds a derived image from a verified local tag of the running image with no
pulls or dependency installation. It copies only the eight reviewed runtime files,
checks their image hashes/imports, then runs the accepted suite in a read-only,
network-disabled container with synthetic credentials and tracked cleanup.

The candidate override changes only the app image. Full rendered Compose data is
compared after normalizing that image and mount ordering only, so unknown values
and duplicate mounts remain significant. A no-lifespan probe checks effective
settings and source hashes through the real mounts. Cutover rechecks the baseline,
stops nginx and app before replacement, atomically installs the override, replaces
only app with `--no-deps --no-build --pull never`, verifies readiness, settings,
runtime contract and schema, then restores nginx and verifies public HTTPS.

Failures after traffic stops attempt the exact prior override/image. Recovery
stops the candidate before starting the old worker, verifies the old app and HTTPS,
and leaves ingress closed with distinct markers when recovery cannot be verified.
Database migration, restore, provider probes and unrelated Docker cleanup are not
part of this release.

## Tests and evidence

The focused suite has 20 tests. The available standard-library run completed with
**19 passed and one explicit Docker rehearsal skipped**:

```text
C:\Python314\python.exe -m unittest -v tests.integration.test_voice_release
Ran 20 tests
OK (skipped=1)
```

The passing offline cases exercise import safety; exact source scope and manifest
hash rejection; hidden mounts; semantic mount reordering versus changed fields and
duplicates; preservation of dollar signs, newlines and unknown Compose fields;
candidate-test refusal before cutover; settings and source mismatch; stale baseline;
read-only exact-0004 compatibility; partial-creation timeout cleanup; preservation
of an original error when cleanup also fails; successful replacement; candidate
readiness recovery; post-restart TLS recovery; and failed recovery with ingress
closed. Python 3.14 compiled the release, test module and both embedded probe
programs successfully without writing bytecode as a validation dependency.

The first focused execution had two temporary-directory fixture construction
errors. Those were test-harness errors, not behavioral failing-first evidence, and
are not counted as a product failure. After correcting the fixture, the behavioral
suite passed. No claim of behavioral test-first failure is made because the release
implementation existed before this new harness was first executed.

The exact standalone preservation command was:

```powershell
C:\Python314\python.exe -B -c "import hashlib,json; from pathlib import Path; r=Path.cwd(); m=json.loads((r/'docs/delegation/T034-B-accepted-hashes.json').read_text(encoding='utf-8')); e={**m['runtime'],**m['tests']}; b=[p for p,h in e.items() if hashlib.sha256((r/p).read_bytes().replace(bytes((13,10)),bytes((10,)))).hexdigest()!=h]; print('ACCEPTED_HASHES_OK count='+str(len(e)) if not b else 'HASH_MISMATCH '+repr(b)); raise SystemExit(bool(b))"
```

Result: `ACCEPTED_HASHES_OK count=25`. The immutable manifest itself remained at
SHA-256 `ede100ffdf63da329de34fcd405beac1b25bcafd0a942933e7f819a83c1935c2`.

## Remaining lead acceptance

This shell has Python 3.14 without pytest or the application dependencies, and it
has no Git or Docker executable. The full 342-test baseline, `git diff --check`,
and the opt-in rehearsal therefore remain for the lead's accepted local toolchain:

```powershell
$env:T034_RUN_DOCKER_TESTS = '1'
C:\Python311\python.exe -m pytest -q tests/integration/test_voice_release.py
Remove-Item Env:T034_RUN_DOCKER_TESTS
C:\Python311\python.exe -m pytest -q
git diff --check
```

The opt-in case uses the existing local `ai-phone-t049a-candidate:20260913` image
by default (or `T034_BASE_IMAGE`), synthetic credentials, an internal network and
no public ports. It exercises the actual derived-image build, accepted tests,
source probe, Compose replacement, mount/environment contract and rollback, then
checks container/network/image cleanup. It does not exercise DigitalOcean's exact
image digest, private mounts, real PostgreSQL, nginx, public TLS, phone audio or
providers. Those server/runtime boundaries remain preflight and supervised rollout
checks after lead publication.

The runbook intentionally contains an unfilled commit placeholder. Arabic audio
quality, full concurrent turn intake, correction preemption, endpointing and public
launch acceptance remain outside T034-B.
