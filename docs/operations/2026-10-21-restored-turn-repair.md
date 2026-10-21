# Restored caller input repair

The last owner-confirmed application is f70eeb0812ef4166a2067e90b4515e03c738bae3,
image sha256:6643b086c7ae831260a747e0a3391c28697f24f9803a2be9ce496d01a81b2e67.
Its effective orchestrator is the restored historical source, not the branch's
parked phase-five orchestrator. This repair targets that exact effective source.

## Behavior

- A canonical final committed before current playback waits in the existing
  serial consumer until the same call is listening and its echo guard expires.
  The complete utterance then reaches dialogue once, including corrections.
- Ambiguous playback overlap cannot authorize booking. It invalidates the pending
  offer and requests repetition once; a subsequent fresh listening turn resets
  the clarification latch. This is recovery, not speaker identification.
- The wait ends on cancellation, call/STT replacement, call end, or a 30-second
  deadline. Timeout invalidates the offer and emits a fixed category. It does not
  force-clear playback or automatically retry a booking.
- Explicit language requests can be independent clauses in a longer request,
  including supported unpunctuated English and Arabic task starts. The full
  booking request is retained. Conflicting, quoted and negated compound language
  requests remain for dialogue clarification.

Only the restored transcript consumer, the shared language policy, and a new
admission helper change in the deployed application. Existing audio/VAD gating,
four booking methods, model, voice, database and notification settings are retained.
The parked verified route remains disabled.

## Evidence and limits

The initial language tests reproduced five failures. Four actual restored-consumer
regressions also fail when the source-builder repair is bypassed; all four pass
with the repair. Tests cover actual Twilio playback marks and cleanup, post-playback
deferral, duplicate identities, ambiguous English/Arabic approval, late consumption,
replacement sessions, cancellation, timeout, and clarification-loop suppression.

Final speech/conversation/calendar selection passed 606 tests; the focused
language selection passed 78 tests. Two existing audioop / ffmpeg warnings remain. Release simulation:
eight passed. Final Docker-enabled release selection: nine passed, including the
actual Linux image builder, seven installed-consumer journeys, and the existing
zero-interval availability/confirmation probe. The exact server image is remote;
the local Linux rehearsal layers the accepted sources on an available dependency
parent. The server procedure still requires its exact predecessor image and runs
the installed-image probe before cutover.

An early offline probe blocked Windows event-loop self-pipe creation; socket
blocking now starts after loop creation. Early Linux runs timed out during cold
dependency imports. Temporary traceback instrumentation was removed, and the
standalone probe no longer runs the inherited HTTP healthcheck (it is not an HTTP
server). The actual application retains readiness and its Docker healthcheck.
The final Linux selection completed in 33.72 seconds, including cleanup.
Compilation and whitespace checks passed. Ruff could not run because its local
Python launcher references a missing executable; no lint pass is claimed.

These are synthetic caller-final tests. They do not establish the owners' live
root cause, microphone/STT accuracy, acoustic speaker identity, spoken Arabic
quality, or overall booking acceptance. The original direct-handler diagnostic
is intentionally not an acceptance test: the fix acts at the actual consumer and
does not blindly bypass its speaking/echo protection for arbitrary direct calls.

## Rollout boundary

`scripts/deploy-restored-turn-repair.py` is an owner-run app-only procedure.
It requires the exact predecessor image, revision, effective sources, restored
route and Luna setting. It validates the installed-image offline probe before
cutover, preserves mounted settings, and restores the predecessor on cutover
failure. It makes no calendar-provider requests and applies no migration.
Unresolved future booking operations block release; evidence is never deleted.

Publication needs approval for the concrete reviewed commit. Do not run the
procedure from an unreviewed branch tip or substitute the branch orchestrator.
After deployment, start with one availability-only English/Arabic language and
date-correction call; assess a complete new booking separately.
