# Restored-flow availability replies

Owner requested a narrow fix after choosing to retain the restored caller flow.
No model, voice, policy, schema, notification worker, or phase-5 route change.

## Behavior

The legacy calendar reader formerly returned an empty list both for a completed
search with no slots and for a failed request. The restored booking handler
classified both as `AVAILABILITY_UNVERIFIED`.

The reader now raises a fixed `AvailabilityReadError` for failed, malformed,
unknown-status, missing-staff, or continued responses. Completed empty results
produce `STAFF_UNAVAILABLE` with the selected consultant's configured name and
an offer to check up to two other configured consultants. Arabic uses configured
Arabic names. Neither reply promises another consultant is free, infers vacation,
or supplies an unverified return date. Wording refers only to the period checked.

Available-slot calculation, the existing seven-day query, two-business-day policy,
requested-day handling, confirmation, persistence, and notification behavior are
preserved. Choosing another consultant requires a new availability check.
All checks still clear stale pending authorization before provider work.

## Effective runtime and rollout

The production orchestrator was restored from `792952b2965e85513bf0971a41bfaff7bd3377c4`.
The branch's ordinary orchestrator remains parked phase-5 code. Do not use a generic
branch rebuild for this release.

`scripts/deploy-restored-availability.py` accepts only restoration revision
`ca46f311e05e4a0c2f3093ed7e3781077c57f7ed`, with either of its two supported,
hash-verified effective-source variants. It derives an image from the running
image and changes only the legacy calendar reader, two new helpers, and the
historical orchestrator's empty-result branch. The historical source and final
patched source are both pinned. The verified phone route must remain disabled.

The procedure reuses the reviewed restoration mechanics: protected override and
source checks, offline image probe, effective settings/mount verification, an
unresolved-future-operation gate, app-only replacement, readiness/HTTPS checks,
and exact predecessor recovery. It runs no migration and preserves operation
evidence. Protected release files stay on the server.

After publication, the owner finishes active calls, fetches
`codex/phase-5-booking-reliability`, and runs this script from the exact published
commit using the same full commit as its argument. Success is
`RESTORED_AVAILABILITY_DEPLOYED_READY_HTTPS_OK`, followed by commit/image lines.
If it stops, retain the protected directory and report the fixed stage marker;
do not bypass baseline or operation guards.

## Validation

- Original source: 11 failing behavioral cases, 2 passing controls.
- Final targeted calendar/release selection: 43 passed, 1 opt-in Docker skip.
- Docker-enabled release suite: 19 passed, including actual Linux build,
  English/Arabic blocked-consultant → other consultant → date correction →
  decline/recheck/confirm journeys, and real PostgreSQL 16 schema-0005 booking
  and held-notification persistence. Provider responses are synthetic.
- Broader calendar/scheduling/language selection: 491 passed, one historical
  T007/T013 hash assertion failed on unchanged contracts/test files.
- AST comparison confirms only `_check_booking` changes in the historical
  orchestrator; all other methods, including create/cancel and speech, match.
- Source pins, Python compilation, and whitespace checks passed.
- Rehearsal containers, networks, and image tags were removed and independently
  checked absent. No server or live provider was contacted.
- Ruff could not run because the installed Python wrapper has no Ruff executable.

The local Linux parent supplies dependencies; it is not the remote production
image. The owner rollout checks the actual running parent's sources and settings.
These checks do not establish live LLM phrasing or telephone pronunciation.

## Owner acceptance

Ask for Hussam during a period with no appointments. Expect a named no-availability
reply and an offer to check another consultant, in the caller's language. Choose
Rami or Abdul and verify a fresh check occurs. An availability-only call needs no
booking confirmation. If performing a new booking, verify the exact consultant,
date/time, one Microsoft appointment, matching local record, and expected SMS.
No broader booking, concurrency, cancellation-identity, or voice backlog item is
closed by this change.
