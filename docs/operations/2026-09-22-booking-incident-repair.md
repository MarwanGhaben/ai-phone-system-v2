# Duplicate availability repair and create diagnostics

## Verified defect and scope

The September 22 owner-run check evaluated the same Microsoft response using
the deployed parser and the corrected parser, without changing server files:

| Result | Deployed parser | Corrected parser |
| --- | --- | --- |
| Availability | invalid_response | available |
| Free intervals | 0 | 7 |
| Eligible Rami slots | 0 | 10 |

The deployed parser rejected an exact repeated interval. The corrected parser
counts matching staff/classification/instants once. Every timestamp and window
is still validated. Duplicates still count toward response size limits. Missing
staff, pagination, unknown statuses, duplicate staff envelopes and conflicting
free/busy evidence remain unsuccessful. Gaps are never treated as free time.

Only three runtime files change: the availability decoder, create transport
diagnostics, and create-coordinator diagnostics. The create request, single-POST
rule, approval checks, operation store, schema 0005, SMS workers, language logic
and voice settings are unchanged.

`BookingCreateDiagnostic` records constant stages/reasons, HTTP status and a
small allowlist of provider error codes. It does not print provider messages,
response bodies, tokens, customer data or operation/provider identifiers. Logging
failure cannot change the booking result; cancellation still propagates.

This does **not** establish the cause of the September 19 failed create. That
operation remains dispatched with no saved receipt or local booking. A complete
current calendar view had no matching appointment, which is not proof about the
historical POST outcome. Do not replay, release or delete that operation. No
Arabic audio-quality fix is included or claimed.

## Validation

- Duplicate regressions: 14 failed / 4 passed before the correction; 18 passed after.
- Focused reader, live-shape and caller-policy checks: 169 passed.
- Create transport/coordinator diagnostics: 26 passed, including redaction,
  cancellation, failed logging, denied send and no automatic retry.
- Broader calendar/scheduling/booking selection: 550 passed, two failed. One is
  an existing frozen-file hash assertion. The other is an unchanged dialogue test
  with a hardcoded September 21 request; it also fails with deployed HEAD runtime
  sources on September 22. Neither failure is treated as passing acceptance.
- Exact owner comparison program: four synthetic checks passed; the actual live
  owner comparison then recovered seven intervals and ten eligible Rami slots.
- Release checks: 14 passed with the opt-in local Linux image build/probe enabled.
  The actual builder used available local dependency image content with accepted
  predecessor Git sources layered onto it. The exact production parent is remote.
  The owner rollout derives from the actual inspected, source-verified server image.
- The Linux probe uses `--network none`, synthetic HTTP and synthetic customers.
  It does not establish live create or audible Arabic acceptance.

## Owner rollout

Publication must finish before running the release. Use the published exact SHA
for both `git show` and the Python argument; no branch-tip execution. The entry
point is `scripts/deploy-booking-incident-repair.py`.

The release accepts only revision `3e492696348c4c4ce617d69b6b7cd817f56d7127`.
It verifies predecessor sources/settings, derives an image locally, runs the
offline probe, checks effective source and configuration, and replaces only the
app image. It does not run migrations or write/reconcile booking records. Normal
application workers resume in their existing configuration.

Finish active calls before rollout. Expected final marker:
`BOOKING_INCIDENT_REPAIR_DEPLOYED_READY_HTTPS_OK`, followed by the deployed commit
and image. If replacement fails, the procedure attempts to restore the exact
previous app image and private override. Keep the protected release directory.

After deployment, first check Rami availability without confirming a booking.
A new controlled create test, if needed, must be separate from the unresolved old
operation and must be explicitly approved by the caller. Read only the fixed
`BookingCreateDiagnostic` entries when investigating its result; do not paste raw
application logs. An existing failed-call recording is still needed to investigate
the reported Arabic sound rather than infer quality from text-only tests.
