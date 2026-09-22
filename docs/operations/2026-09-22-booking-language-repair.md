# English caller / Arabic response regression

## Observed production state

The owner confirmed that Rami availability works after deployment of
`5a22b1b1713156e888342e65cc56ea3e22183704`, image
`sha256:0c653159b71ec1bac58e42ab0c292e7ded478df99184461b364a0efc31709307`.
During an English call, replies remained Arabic. An explicit English request
produced repeated language-choice questions. No recording or transcript of that
call was retrieved; the exact utterance is unknown.

## Reproduced causes and repair

Canonical ElevenLabs committed transcripts can lack provider language detection;
the accepted adapter then returns `auto` or its configured-language fallback.
The verified booking route required an `en` label in addition to substantive
English text. Consequently an Arabic context could remain Arabic despite a clear
English request. The explicit-request matcher also missed common polite forms,
and an English turn did not reject an Arabic model reply.

The three-file application repair:

- Recognizes substantive English text without requiring a provider `en` label.
  Names, times, email-only input and brief confirmations retain the current language.
- Handles complete explicit switch requests directly, including “Can you please
  switch to English?”. These turns acknowledge the chosen language without a model
  language-choice loop or an immediate preference database write. Existing explicit
  save/remember/always requests retain the preference-tool route. Existing end-call
  preference persistence is unchanged.
- Rejects predominantly Arabic model replies on English turns, using the existing
  bounded model correction and localized fallback.
- Rejects stale turn ownership before changing the current response language.

Only `_process_verified_turn` and `_get_verified_response` change in the
orchestrator. The four booking method ASTs remain identical. Calendar contracts,
provider mutations, operation store, SQL, notifications, STT/TTS adapters, audio
gates and settings are unchanged.

## Evidence

- Initial language reproduction: 22 failed, 2 passed before production changes.
- Added ownership regression reproduced a stale turn changing language; it passed
  after the early generation check.
- Final new language regressions: 40 passed. Actual canonical adapter events cover
  `auto` and configured Arabic, explicit requests, negative/ambiguous forms,
  English availability playback, both switch directions, brief answers, isolated
  sessions, persistent preference, stale/closed/replaced ownership, and wrong-language
  model replies.
- Combined language/day/caller-policy selection: 149 passed.
- Conversation/STT/TTS/telephony/calendar/scheduling selection: 959 passed,
  2 existing failures. `test_enabled_availability_check_keeps_owned_acknowledgement`
  uses a hardcoded September 21 appointment with a moving current date;
  `test_accepted_t007_and_t013_files_remain_frozen` expects hashes predating the
  deployed calendar corrections. These failures were not changed or called green.
- Application release suite with actual offline Linux builder: 14 passed. The
  candidate probe drives real canonical STT normalization, booking-session admission,
  enabled phone handling, scripted model results, typed synthetic calendar evidence,
  and captured speech output. Docker uses `--network none`; no providers contacted.
- The exact production image is remote. Local Linux rehearsal layers the accepted
  predecessor Git source onto the available dependency image, then invokes the actual
  repair builder. The owner release verifies effective predecessor sources/settings
  and derives from the running local image.
- All four booking method ASTs unchanged; `git diff --check` passed. Disposable
  Docker probe containers and image tags were removed and absence asserted.

## Owner rollout

After publication of the reviewed commit, use that same full commit for both
`git show COMMIT:scripts/deploy-booking-language-repair.py` and its `python3 - COMMIT`
argument. Fetch `codex/phase-5-booking-reliability` first and enable `pipefail`.
Do not substitute a moving branch reference. Finish active calls before cutover.

The release accepts the deployed 5a22 predecessor, verifies inherited source and
settings, changes only the app image, and restores the predecessor if replacement
fails. It never runs a migration. Expected success marker:
`BOOKING_LANGUAGE_REPAIR_DEPLOYED_READY_HTTPS_OK`, followed by deployed commit/image.

After that marker, make one availability-only English call for Rami on a specified
future day. Expect English acknowledgement and English slots. Ask to speak Arabic,
then “Can you please switch to English?”; expect an immediate acknowledgement in
the requested language and continued English after a brief consultant-name answer.
Stop before booking confirmation during this language check.

## Still open

Live telephone acceptance, Arabic pronunciation/audio quality, and the original
unresolved September 19 booking attempt remain open. This repair does not retry,
resolve or release that dispatched operation, and does not establish successful
provider booking creation. No live provider/server contact occurred during local
implementation and verification.
