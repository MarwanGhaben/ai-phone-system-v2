# Barge-in diagnostics

The T033-E observer records bounded numeric evidence about the existing local
barge-in path. It is disabled by default through
`BARGE_IN_DIAGNOSTICS_ENABLED=false`. Enabling it requires a separately reviewed
beta release; this document does not authorize or provide a deployment command.

The observer does not change the RMS threshold, eight-callback gate, grace period,
audio routing, STT resets, playback ownership, provider filtering, turn state, or
booking behavior. It does not identify a speaker and does not retain audio or
transcripts.

## Summary boundary

When enabled, each admitted call receives a fresh random diagnostic ID. The ID is
not derived from its CallSid, stream ID, phone number, or caller. At teardown the
application emits at most one compact JSON object with the fixed event label
`barge_in_diagnostics_summary`.

Only numeric counters, durations, fixed categories, and the opaque diagnostic ID
are present. The observer keeps the last 16 speaking windows and records how many
older windows overflowed. It admits at most 64 active calls; later calls continue
normally without diagnostics until capacity is available. It has no background
worker and performs no per-frame logging, filesystem write, database write, or
provider request.

Each retained speaking window reports:

- callback and byte totals received while the application considered the call to
  be speaking;
- bytes passed to the existing mu-law decoder, maximum computed RMS, and maximum
  existing consecutive count;
- fixed counts for empty input, grace suppression, decode failure,
  below-threshold reset, above-threshold accumulation, and gate firing;
- bytes forwarded to STT or dropped during speaking;
- interrupt and reset invocation outcomes;
- speaking-window elapsed time and bounded callback dispatch-gap minimum, mean,
  and maximum.

The incoming callback captures the exact speaking window before it enters the
detector or awaits playback interruption, STT reset, or forwarding. Delayed
outcomes remain attached to that retained window even if its speech coroutine has
already closed or a newer speech invocation has started. A delayed observation
cannot create a window, mutate a foreign call, restore one of the 16-window
history entries that has been evicted, or reopen a finished call.

Teardown permanently retires the admitted state and releases its capacity slot
even if closing the current window, taking the snapshot, formatting JSON, or
emitting the summary fails. Such failures use only the fixed
`observer_failure` label. A repeated teardown does not retry formatting or
emission and cannot produce a second summary.

Dispatch gaps describe when this process received callbacks. They do not prove
packet loss, network jitter, or provider behavior. Send-to-stop timing is not a
speaker-identity signal.

## Later supervised-call interpretation

After lead review and a separate default-off beta release, one supervised call can
help distinguish several paths:

- Repeated `above_threshold_count`, a maximum consecutive count reaching eight,
  and `gate_trigger_count` show that sustained decoded input energy crossed the
  current local gate. An `interrupt_succeeded_count` then distinguishes a gate
  decision from a confirmed owned-playback clear.
- High `grace_suppressed_count`, unusual callback byte sizes, or widened dispatch
  gaps show where grace or callback timing coincided with the observation. Those
  measurements do not establish loss or its cause.
- Window replacement/overflow and isolated opaque summaries can show whether an
  observation belonged to the current speaking window. Old-session callbacks and
  repeated teardown cannot alter or duplicate the replacement call's summary.
- If a reported stop has no gate firing or no successful interrupt in the matching
  summary, the stop came from another lifecycle path or evidence is incomplete;
  the summary does not guess which source produced the audio.

The owner should report the approximate call time and whether the unwanted stop
was heard. Reviewers correlate that operational observation with the single
sanitized summary. They must not request raw logs containing customer data, audio,
provider payloads, tokens, CallSids, stream IDs, or phone numbers.

Delayed outcomes require their explicitly captured window. If window acquisition
failed, the observation is discarded; it must not adopt a newer active window.
Closing a window does not prevent a still-owned delayed outcome from updating it
while retained, but evicted, foreign and finalized-call windows cannot be updated.

Filtering remains off. These diagnostics do not fix nearby speech, certify the
local RMS gate, validate echo/pre-roll behavior, or complete T031/T033/T034/T037.
