# Arabic booking speech regression repair

The owner reports that the newly enabled verified booking route sounds robotic,
mixing English service/time labels into Arabic. This is reproduced in the actual
proposal and response paths: fixed readbacks bypass the previous LLM's Arabic
date/time instructions and send `appointment`, ISO dates, numeric clocks,
`America/Toronto`, and English alternatives directly to speech.

## Change

One pure display module renders Arabic weekdays, calendar dates, clock times,
duration, timezone and exact configured consultant-name aliases. It is used by
proposal presentation, alternative suggestions, successful confirmation and
existing-appointment readback. Unknown names and services retain their supplied
identity. Caller/contact/address facts remain intact. English rendering remains
unchanged. No provider setting, voice ID, model, dependency, schema, availability
decision or mutation behavior is changed.

Example: a Tuesday ten-to-ten-thirty proposal now says `الثلاثاء` and
`من الساعة العاشرة صباحاً إلى العاشرة والنصف صباحاً، بتوقيت تورونتو`.
It no longer reads `appointment`, `2030-01-08`, `10:00–10:30` or the IANA zone ID.

The proposal state machine is byte-equivalent at the AST level. The availability
method differs only in its textual alternatives assignment. Exact presentation,
Twilio acknowledgement, later caller approval and one-use claim remain required.

## Evidence

- Three actual-path tests failed before the correction: proposal, alternatives,
  and final confirmation. All pass afterward.
- Focused final speech/proposal/phone/release selection: 87 passed, one opt-in
  Docker skip. Covers Arabic clocks, winter/summer UTC conversion, unknown lookup
  times and preservation of exact proposal/contact facts.
- Broader calendar, scheduling, conversation, STT, TTS, telephony and LLM run:
  838 passed, one existing frozen-parser-hash failure. That assertion still pins
  the parser before the already-published Graph repair. It was not rewritten.
- Explicit Docker release selection: 10 passed, including the actual offline
  Linux build, source hashes, Arabic and English proposal playback/approval/claim
  probe and temporary-resource cleanup. Two additional predecessor rejection
  tests passed in the final offline selection.
- Python 3.11 parsing, source/probe pins, preservation checks and whitespace pass.

These are text, lifecycle and packaging checks. No live TTS audio, provider,
booking or customer notification was requested during this repair. Audible
quality still needs the owner's short call after rollout.

## Release

`scripts/deploy-arabic-booking-speech.py` is an app-only successor of the reviewed
Graph repair procedure. Its only accepted predecessor revision is
`20c917bb5a44b3ec596b8db1f50400218832d95a`, which the owner confirmed deployed.
The owner did not provide its resulting image SHA. The procedure therefore
captures the running immutable image ID, requires the protected override to match
it, verifies the exact predecessor revision and all protected effective source,
model and settings evidence before building from that same local image. It saves
that exact image/configuration for recovery. A failed candidate restores it.

The local Linux rehearsal uses the available dependency parent with accepted
predecessor Git source overlaid; the exact server image is not available locally.
The owner-side procedure verifies the actual running parent and effective mounts.
No migration runs; schema 0005, notifications, voice and private mounts remain.

Publish the reviewed commit, then use that same full SHA for both `git show`
and the procedure argument. Success is
`ARABIC_BOOKING_SPEECH_REPAIR_DEPLOYED_READY_HTTPS_OK` followed by commit/image.
Keep the protected release directory. After success, ask in Arabic for an
available accountant/time and listen to the alternatives/readback; decline the
booking unless a test appointment is intended.
