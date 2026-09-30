# Equal-endpoint Graph availability repair

The owner reported that both Rami and Abdul searches returned a calendar error
on Luna release e3f96437fcb1b4af57ef82c9be8752371ea983d9. The owner-run diagnostic
confirmed successful token/staff/availability HTTP responses, correct mappings,
and the restored orchestrator with verified phone booking disabled.

Rami's response contained 54 entries (13 available), Abdul's 25 (8 available).
Each contained one entry with identical start and end. Both reads failed at
legacy_availability.py line 49, the newly introduced `end <= start` rejection.
The error discarded all usable intervals. This establishes a calendar-validator
failure independently of the model's synthetic compatibility checks.

## Correction

Change that condition to `end < start`. Equal endpoints cover no time; the
existing 30-minute slot loop emits no slot for them. Positive intervals are
unchanged. Reversed/malformed intervals, unknown statuses, incomplete collections,
and wrong/missing staff remain rejected. No gaps are inferred to be available.

Only services/calendar/legacy_availability.py changes in the deployed image.
Luna, prompts, restored orchestrator, calendar request format, staff mapping,
SQL, workers, notification policy, mounts, and all settings remain unchanged.

## Rollout

Owner finishes active calls, fetches the phase-5 branch, then pipes
scripts/deploy-zero-interval-repair.py from the exact published commit into
python3 with the identical full SHA. The script accepts only e3f96437's supported
restored-availability source variants, requiring Luna and verified route disabled.

It derives from the exact running image, verifies source and settings, and runs
an offline synthetic caller-flow probe. Before stopping traffic, a separate
candidate process reads Rami and Abdul through the real get_available_slots path
using existing server settings. It permits only token acquisition, staff GET,
and availability read POST; it cannot create or cancel appointments. Each read
has a 20-second limit and the probe a 48-second overall budget. Output is fixed
status, consultant labels, and slot counts. A failed read stops before cutover.
A completed empty result is permitted: the release does not invent availability.

Success markers:

- ZERO_INTERVAL_REPAIR_LIVE_CALENDAR_READS_OK
- ZERO_INTERVAL_REPAIR_DEPLOYED_READY_HTTPS_OK
- DEPLOYED_COMMIT and DEPLOYED_IMAGE

A failed cutover restores the exact preceding image and override. Schema and
unknown operation evidence remain intact. No migration runs. Protected release
files remain under /opt/ai-phone-zero-interval-repair.*.

## Verification

Nine new zero-duration tests failed first at the same validator line as production.
After correction, 29 restored-reader/phone tests pass. They exercise all three
recognized statuses, English/Arabic pending offers, zero-only responses producing
no slots, and existing malformed/reversed/incomplete rejection controls.

The packaged synthetic probe runs 12 restored-handler journeys: English/Arabic,
Rami/Abdul, and each recognized zero-duration status. Each journey checks a blocked
consultant, a different-day rejection, correct offered slot, decline without
creation, fresh confirmation, and exactly one synthetic provider create plus
local persistence/notification handoff. No real provider or database is contacted.

The combined calendar, conversation, LLM, STT, TTS and telephony selection passed
609 tests with the two existing audioop/optional-ffmpeg warnings. Its first run
exposed an existing test-fixture cleanup bug: borrowed IsolatedAsyncioTestCase
patch cleanups ran without its async runner, leaving global mocks installed.
The fixture now explicitly executes its synchronous cleanups; runtime code was
not changed for that test isolation issue.

All 24 Docker-enabled release tests passed, including the actual Linux build.
Owned rehearsal containers and tags were removed. The Linux release rehearsal compares the new probe on old and repaired sources,
checks exact source hashes, and runs restored bilingual journeys in the actual
built candidate. The old handler returns AVAILABILITY_UNVERIFIED, so the initial
rehearsal assertion expecting an uncaught AvailabilityReadError was corrected.
The local dependency parent is available locally rather than the remote exact
production image; the owner rollout verifies and derives from the actual server
image. Live Microsoft reads remain an owner-run pre-cutover gate.

This repair does not establish complete live booking/audio acceptance. After
successful rollout, verify Rami/Abdul availability and one explicitly approved
new booking against Microsoft, dashboard, and SMS. Preserve old unresolved
operation evidence; do not retry prior ambiguous booking attempts.
