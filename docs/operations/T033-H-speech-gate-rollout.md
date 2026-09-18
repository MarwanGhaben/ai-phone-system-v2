# T033-H speech-aware barge-in beta rollout

Status: independently reviewed for supervised beta rollout; see
[lead evidence](../delegation/T033-H-review.md). The actual Linux candidate,
speech suite, nested settings mounts and exact image/settings rollback passed.
Server execution is still owner-mediated. `PUBLISHED_T033_H_COMMIT` below is a
template: use the exact published 40-character commit returned by the lead.

## Release boundary

This is an app-only beta release from deployed source
`4c135d883554496a250e135368e7e6eade90cc7a` and image
`sha256:089447b0a1acb36c1ffd9d5a14251dd3357f19e75bacd8ebed919df3ffae8913`.
It preserves schema 0004, the active observation/notification workers, numeric
barge-in diagnostics, provider filtering off, private settings and runtime mounts,
the parent config bind, networks, ports, credentials and every existing release
directory. It changes only the app image, the reviewed nested settings overlay and
the six manifest-pinned speech-gate settings.

The release checks the actual running image's Python and architecture before any
download. The accepted target is CPython 3.12 on Linux x86_64. It downloads three
exact binary wheels from the official PyPI package host, verifies their pinned
SHA-256 values, and installs them offline into a derivative of the exact local
image without resolving or changing any other package:

- `onnxruntime==1.30.0`
- `flatbuffers==25.12.19`
- `protobuf==7.36.1`

NumPy remains exactly `1.26.4`. The protected release evidence records platform,
wheel URLs/names/hashes, complete before/after package inventories and the warmed
model measurement. The candidate copies only four runtime modules and the three
raw model assets. It verifies all 51 frozen assets, the model checksum/license,
the real model's positive/negative tests, package preservation and explicit model
loading under the effective parent-plus-nested mounts before stopping traffic.

## Future owner command after publication

Finish beta calls first. In the DigitalOcean browser console, the lead-published
command will have the placeholder replaced with the reviewed commit:

```bash
bash <<'BASH'
set -euo pipefail
cd /opt/ai-phone-system-v2
release='PUBLISHED_T033_H_COMMIT'
test "$release" != 'PUBLISHED_T033_H_COMMIT' && test "${#release}" -eq 40
git fetch --no-tags origin codex/phase-4-arabic-voice-quality
git show "$release:scripts/deploy-speech-aware-barge-in.py" | sudo python3 - "$release"
BASH
```

Do not run this placeholder form. Do not substitute a branch name, abbreviated
SHA or worktree file. Do not rerun on failure. Return only the protected release
path and fixed public markers; never paste `private.log`, settings, environment,
package inventories, credentials, provider output, transcripts, phone numbers,
CallSid or stream IDs.

A successful release ends with:

```text
SPEECH_AWARE_BARGE_IN_DEPLOYED_READY_HTTPS_OK
DEPLOYED_COMMIT=<the published 40-character commit>
DEPLOYED_IMAGE=sha256:<candidate image ID>
```

Keep the printed `/opt/ai-phone-speech-gate-release.*` directory. Its reviewed
`candidate-settings.py` remains mounted by the running app and its root-only files
are recovery evidence. Keep every older release directory, including
`/opt/ai-phone-diagnostics-release.ckajbnls`.

## Failure and recovery

The script performs no migration, database dump/restore, business write, provider
call, checkout/reset, whole-project replacement, image pruning, volume pruning or
swap change. It rejects inadequate memory/disk rather than bypassing capacity.
Before cutover it rechecks the exact old image, override, settings, mounts, source,
schema, dependencies, nginx and public HTTPS.

After ingress stops, any settings, replacement, model-load, readiness, health or
TLS failure restores the exact old image, old override/settings mount and old
source. Successful recovery prints the fixed previous-app marker. If exact recovery
cannot be proved, ingress remains closed and the script prints
`RECOVERY_UNVERIFIED_MANUAL_REVIEW_REQUIRED`; stop and return the fixed markers and
release path to the lead without a retry.

## One controlled post-success check

Only after the reviewed release succeeds, make one short controlled non-booking
phone check containing quiet foreground speech and a music-only interval matching
the owner's normal setup. Stop before any booking confirmation, cancellation or
customer SMS. Report the approximate time and whether playback stopped for each
interval. Do not make or upload another recording corpus.

This candidate is not speaker identification, echo cancellation or proof against
nearby voices, singing or the owner's exact music. Distant Arabic speech also
triggered in local evidence. The current diagnostics schema does not count VAD
probabilities or gate firings, so zero legacy trigger counts must not be interpreted
as zero enabled interruptions. Use the owner's observed behavior together with
the fixed lifecycle evidence; do not add an observability feature during rollout.
