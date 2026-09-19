# Requested-day and bilingual booking dialogue repair

The owner reported that asking for Abdul on Monday produced Tuesday options,
and that the Arabic conversation used English "staff member" acknowledgements.
Four tests reproduced matching defects in the actual local caller path before
the correction. The precise live transcript was not collected; these findings
are code reproductions, not a reconstruction of that call.

## Findings and changes

The verified route had an exact-time check but no day-only availability tool.
It encouraged the model to invent a clock time when the caller asked for a day.
The handler then offered alternatives from the whole policy window, and it did
not bind model dates to a weekday stated by the caller. Separately, short STT
English metadata could switch an Arabic conversation after a consultant name,
and a model holding sentence could be spoken without a corresponding lookup.

- Added a read-only day search. It lists up to three verified options on the
  requested Toronto calendar date without creating a proposal or booking.
- Bound recognized caller dates to the search and retained the day when changing
  consultant or selecting a time. Ambiguous or conflicting dates ask for
  clarification; out-of-window dates ask before changing days.
- Restricted exact-time alternatives to that same day. No availability on Monday
  now produces a Monday-specific response and asks before another day is searched.
- Kept short names, times, email spellings and yes/no answers from changing the
  conversation language. Explicit language requests remain supported.
- Allowed an unqualified English yes/no in an Arabic conversation and vice versa,
  while preserving the existing exact-presentation, later-receipt, freshness and
  one-use mutation guards. Qualified answers still do not authorize a booking.
- Added one bounded model retry for an invalid-language response or a promise to
  search without a tool. The first invalid response is never spoken. Actual
  availability calls use the existing owned, localized acknowledgement.

The review covered caller admission and language, model context and tool protocol,
date selection, consultant selection, policy/availability assessment, proposal
presentation, approval and durable creation. The Graph readers, approved business
policy, operation store, migrations, cancellation and notification implementation
are unchanged. Earlier Arabic date/time rendering is included in the package
when upgrading directly from the Graph repair.

## Validation

- Four actual-path regressions failed before the changes: cross-day alternatives,
  model weekday disagreement, short-name language switching and phantom lookup.
- Conversation, scheduling, calendar, STT, TTS, telephony and LLM selection:
  **898 passed, one failed**. The failure is the existing
  `test_accepted_t007_and_t013_files_remain_frozen` assertion, which pins the
  calendar decoder before the published Graph repair. That decoder and the old
  assertion were not changed here. Two known audioop/optional-ffmpeg warnings remain.
- Real PostgreSQL phone journeys: **9 passed**, including day search, time choice,
  exact readback, later approval and one synthetic-provider booking with durable
  notification records; also English approval in an Arabic conversation.
- Release selection with the opt-in Linux Docker build: **16 passed**. Checks
  include actual offline day/language behavior, packaged hashes, both accepted
  predecessors, failure refusal and exact rollback boundaries.
- Standalone network-free probe: `BOOKING_DIALOGUE_OFFLINE_DAY_LANGUAGE_OK`.
- Python compilation, reviewed source/probe pins and `git diff --check` passed.

One prior test fixture said only Monday but asserted a 10 AM proposal; it was
corrected to explicitly request a time. A separate journey now exercises the
previously missing day-only dialogue. Changed no-availability assertions reflect
the intentional same-day and read-failure distinctions, retaining no-write checks.

The date recognizer is deliberately bounded; it is not a complete natural-language
date parser. Unsupported ambiguous dates require clarification. These tests do
not establish live Arabic pronunciation, STT accuracy or every possible model
wording. No live provider, customer SMS or production server was contacted.

## App-only rollout

`scripts/deploy-booking-dialogue-repair.py` accepts only the published Graph repair
`20c917bb5a44b3ec596b8db1f50400218832d95a` or Arabic speech repair
`a3d2d50aa39e6f7928b1f7dd7c2e31dfa06ed425` as predecessor. The owner confirmed the
former; deployment output for the latter was not supplied. Each predecessor has
its own pinned effective source contract. The procedure verifies settings,
protected mounts, source and model assets, captures the actual immutable running
image for rollback, and builds from that local image without downloading packages.

The local Linux rehearsal used an available dependency image with accepted source
overlaid, not the unavailable exact server image. The owner procedure verifies
the actual server image and effective mounted source before replacement.

After publishing the reviewed commit, finish active test calls, fetch the phase-5
branch, and pipe that commit's `scripts/deploy-booking-dialogue-repair.py` into
`python3 -` with the identical full commit SHA as its argument. The procedure does
not run a migration. Schema 0005, notifications, voice configuration and private
mounts remain in place. Preserve all protected release directories.

Success is `BOOKING_DIALOGUE_REPAIR_DEPLOYED_READY_HTTPS_OK`, followed by
`DEPLOYED_COMMIT` and `DEPLOYED_IMAGE`. On failure, return the sanitized markers;
do not paste protected logs or repeatedly rerun the release.

After success, ask in Arabic for Abdul on Monday without specifying a clock.
Expect Monday-only options or an explicit explanation that Monday has none. Then
change to another consultant without changing the day: the day and Arabic should
remain. Select one offered time, listen to the complete Arabic readback, and
decline unless a test booking is intended. A different day requires a new request.
