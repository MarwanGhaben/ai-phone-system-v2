# T033-H independent lead review — 2026-09-17

Accepted for a supervised beta rollout after independent review and two bounded
packaging/test corrections. This is not acceptance of live music rejection,
speaker identification, echo handling, or the remaining turn/mutation work.
Server execution remains owner-mediated; no server or speech provider was contacted.

## Reproduced failure and correction

The first real Linux rehearsal stopped at wheel preparation because the script
required SymPy in the baseline image. That image has NumPy 1.26.4 and packaging
26.3, but no SymPy. Installed ONNX Runtime 1.30.0 metadata declares
`sympy; extra == "symbolic"`; this release does not request that extra.

A new regression failed against the submitted script. The lead removed only the
optional SymPy requirement, retaining packaging, exact NumPy, three pinned wheels,
wheel integrity checks and complete before/after distribution equality. No
additional dependency was installed into the application or global host Python.

The next Linux candidate built, imported the actual model and passed all staged
speech tests. The Windows rehearsal driver then failed by reading Windows
`/proc/meminfo`. The driver now reads real available memory from the isolated
Docker Linux environment. The production Linux capacity check is unchanged.

## Independent validation

- Submitted offline release suite: 24 passed, one opt-in Docker skip.
- Corrected release suite with explicit Docker opt-in: **26 passed**, no skips.
- Real pinned-wheel download, offline derivative image build, complete package
  inventory preservation, 51-asset checks, model load and staged speech tests passed.
- Actual Compose parent config plus nested settings overlay, candidate replacement,
  and exact previous-image/settings rollback passed on Linux/Python 3.12.14.
- Warmed real model: 500 frames; mean 0.172 ms, p95 0.257 ms, maximum 0.572 ms;
  measured model/runtime RSS delta 71,344,128 bytes (about 68 MiB). These are local
  Docker measurements, not DigitalOcean performance guarantees. The server release
  repeats measurement and checks four-worker memory reserve before stopping calls.
- Python 3.11 full working-tree run: 603 passed, 52 skipped, seven failures.
  Six failures are historical T033-F/T034-B release manifests correctly rejecting
  newer frozen inputs. The seventh is the previously reported 30 ms Windows probe
  timing test; it passed independently and in the actual Linux candidate suite.
  None was hidden by weakening frozen assertions. The full suite is not all green.
- Clean export of the proposed Git tree: **604 passed, 52 skipped, six historical
  release-manifest failures**, with no timing failure. The new manifest validates
  all 51 assets and all four frozen booking methods in the exported tree too.
- Historical model/runtime/booking inputs remain frozen; all 51 manifest asset
  hashes match. Manifest canonical SHA-256 remains
  `4d1aeb6f59b5291c88094493457edcd11806d49cddd56a4f1b7e129c489390a4`.
- `git diff --check` passed (existing Windows line-ending notices only).
- Rehearsal containers, network and candidate/baseline tags were removed; existing
  local images were retained. Global dependencies, live database and server remain unchanged.

## Release boundary and remaining acceptance

Publish only the accepted T033-G assets, manifest, T033-H procedure/tests/runbook
and evidence. The ten-path application source scope stays exact. Keep historical
manifests unchanged. Runtime defaults stay off; the owner rollout enables only the
six pinned local speech-gate settings, preserving diagnostics on, provider filtering
off, schema 0004, notification workers and private mounts.

After successful rollout, the owner makes one short non-booking check: music while
the agent speaks, then a deliberate spoken interruption, including quiet Arabic.
Vocal music and nearby people can still classify as speech. This model does not
identify which person owns the phone. A successful deployment is not evidence of
that broader capability.
