# Booking confirmation playback repair — September 22, 2026

The owner requested Rami on September 23, 4:30–5:00 PM Toronto, heard the
appointment readback, and answered yes. The agent could not confirm. The supplied
database query returned no operation created in the preceding 24 hours, and the
bounded log search returned no create diagnostic. These observations narrow the
failure to before durable admission or an unobserved failure; they do not establish
the exact cause of that call.

## Reproduced defect and correction

The actual Twilio handler, orchestrator speech method and proposal session reproduce
a confirmation race. A yes after the matching playback mark but during STT reset
or TTS cleanup invalidates the presentation. The same answer after cleanup works.
An additional regression reproduced the interval between receiving the mark and
resuming its awaiting coroutine.

The handler now delivers a synchronous, exact-mark completion callback before
resuming the playback waiter. The booking session records completed presentation
at that boundary. Cleanup can finish afterwards without rejecting a later caller
approval. Marks from retired playback, clears, missing acknowledgements, early
answers and corrections cannot grant approval. The callback belongs to the exact
proposal, playback and call session. Fixed internal diagnostic classifications
make pre-admission rejection visible without logging caller or appointment data.

This corrects a demonstrated defect consistent with the incident. A successful
supervised production booking is still needed to establish the live outcome.

## Verification

- Original controlled reproduction: one passed, one failed before correction.
- Immediate-mark regression: one failed, one passed before synchronous delivery.
- Confirmation boundary regressions: 17 passed.
- Combined confirmation, safe-flow and playback tests: 47 passed.
- Speech, calendar and scheduling selection: 977 passed, one historical frozen
  calendar hash assertion failed. Its expected hashes predate deployed parser
  repairs; the assertion was not changed. This is not a fully green repository run.
- Real PostgreSQL phone journeys: 11 passed, including English and Arabic yes
  during post-readback STT reset. These use synthetic provider responses and verify
  one durable booking and the expected held notification records.
- The first database run had five passes and four reminder-count failures because
  the fixture's September 21 appointment had become historical. The notification
  clock is now frozen to the journey clock in the test only; production scheduling
  and assertions are unchanged.
- Release checks including actual offline Linux build and runtime probe: 14 passed.
- The standalone offline probe covers five arrival boundaries with both approval
  and correction; it contacts no provider.
- Source/probe pins, syntax and whitespace checked. Only `_speak_to_caller` changes
  among orchestrator methods; booking methods and the accepted language repair
  remain intact. Test containers were removed; unrelated development containers
  were left running.

## App-only rollout

Baseline: `e85a50599fb29a3bafcd3ec2f7f05e3ffeafcb11`, owner-reported running image
`sha256:75c5921e2eb6efafe6397eb01d664d0f0357b1454792927d367eb6a19f2e8cd6`.

Publish the reviewed repair commit only after owner approval. Use its exact SHA
for both `git show <SHA>:scripts/deploy-booking-confirmation-repair.py` and the
argument to `python3 - <SHA>` after fetching the phase-5 branch. Do not substitute
a moving branch reference.

The procedure verifies the predecessor's effective runtime source and private
configuration, builds from its local image, runs the offline probe, preserves
settings and mounts, and replaces only the application. Failed cutover restores
the predecessor. No database migration or provider mutation is performed.
The local Linux rehearsal uses an available dependency parent with accepted source
layered onto it; the production parent itself is only available on the server.

Required success marker: `BOOKING_CONFIRMATION_REPAIR_DEPLOYED_READY_HTTPS_OK`,
followed by the deployed commit and image. After success, perform one supervised
readback and explicit confirmation, then check the Microsoft appointment, local
booking and confirmation notification. If it fails, collect the new fixed
confirmation classifications before attempting another booking.

The separate September 19 operation with an unknown provider outcome remains
unresolved. This repair neither retries nor releases that operation.
