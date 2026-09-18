# Speech-aware barge-in candidate

This candidate adds a local speech-evidence gate to the existing playback
interruption path. It is disabled by default. When enabled, caller audio can clear
owned playback only after the packaged Silero VAD model reports sustained speech;
the legacy RMS detector is not used as a fallback.

## Pinned detector

- Silero VAD release `v6.2.1`, immutable revision
  `7e30209a3e901f9842f81b225f3e93d8199902b1`
- Model SHA-256
  `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`
- Model size: 2,327,524 bytes
- License: MIT; the unchanged upstream license is packaged beside the model
- Runtime: `onnxruntime==1.30.0`, `CPUExecutionProvider`, one intra-op and one
  inter-op thread
- Production compatibility: the ONNX Runtime package metadata supports Python
  3.11 and requires NumPy 1.21.6 or newer, which includes the existing NumPy
  1.26.4 pin

The adapter verifies the size and SHA-256 before opening a session, selects only
the CPU provider, and validates the model's names, types and dynamic shapes. At
8 kHz it passes 256 new samples plus 32 context samples as `[1, 288]`, carries a
`[2, 1, 128]` recurrent state, and validates `[1, 1]` probability output and the
replacement recurrent state. Session resources are shared inside one process;
context and recurrent arrays belong to one speaking window.

The image already copies the repository with `COPY . .`, so the model under
`models/vad/` is packaged without a Dockerfile change. The disabled path does not
import ONNX Runtime or load the model.

## Gate behavior

Incoming Twilio mu-law bytes are reassembled into 256-sample frames, decoded to
mono PCM16 and normalized by the model adapter. The starting policy is probability
`>= 0.5` for five consecutive frames. Each frame represents 32 ms, so the 160 ms
candidate requires exactly five frames. A non-speech frame clears the candidate;
an input gap over 96 ms also clears the partial frame, evidence and recurrent
state. Callback count and RMS do not participate in enabled admission.

The gate keeps at most 2,560 mu-law bytes, or 320 ms, of consecutive candidate
audio. When it accepts a window, the callback performs the owned clear, resets
that call's STT session, forwards the retained onset once, and then forwards bytes
that arrived during reset in order. The reset-following buffer is independently
bounded to 320 ms. The original 0.8-second post-playback-start grace period remains
and uses a monotonic clock in the enabled path.

Gate state is attached to both the exact conversation context and playback owner.
Ownership is rechecked after classification, clear, reset and forwarding awaits.
Same-ID replacement calls, old callbacks and late teardown cannot use or clear the
replacement state. No state lock is held during VAD inference or STT/TTS provider
I/O, and the implementation does not create per-frame tasks.

Missing or altered model bytes, runtime/load failure, invalid output, inference
failure and an inference result over the configured budget retire that speaking
window. Playback then finishes normally and the existing completion path returns
the call to listening. These failures emit one warning with a fixed category and
contain no call identifier, audio, probability, transcript or exception value.

Existing barge-in diagnostic schemas are unchanged. In enabled mode they continue
to record honest dispatch, route, clear/reset and speaking-lifecycle observations.
They do not expose VAD probabilities and their legacy RMS/consecutive fields do
not describe enabled admission. In particular, the old diagnostic `trigger_count`
is a legacy detector value and is not a count of VAD acceptances.

## Configuration

| Environment variable | Default | Accepted range |
|---|---:|---:|
| `SPEECH_AWARE_BARGE_IN_ENABLED` | `false` | boolean |
| `LOCAL_VAD_MODEL_PATH` | `models/vad/silero_vad.onnx` | non-empty `.onnx` path, at most 512 characters |
| `LOCAL_VAD_PROBABILITY_THRESHOLD` | `0.5` | finite `0.0` through `1.0` |
| `LOCAL_VAD_SPEECH_DURATION_MS` | `160` | 32 through 320 ms; rounded up to 32 ms frames |
| `LOCAL_VAD_MAX_INPUT_GAP_MS` | `96` | 32 through 1,000 ms |
| `LOCAL_VAD_MAX_INFERENCE_MS` | `20.0` | finite, greater than 0 through 100 ms |

Changing the model path cannot authorize different model bytes because the adapter
still enforces the pinned size and digest. Leave the enable flag false until the
candidate receives independent code review and a separate protected release
rehearsal.

## Local measurements and audio evidence

The isolated Windows/Python 3.14 development run used ONNX Runtime 1.30.0 and
NumPy 2.5.3. Across repeated warmed 8 kHz frames, wall time averaged 0.508 ms,
the 95th percentile was 0.616 ms, and one scheduler outlier reached 93.419 ms.
The default 20 ms budget deliberately retires a window on such an overload rather
than queueing work. Production Linux/Python 3.11 timing remains to be measured.

Loading one session increased process RSS by 18,776,064 bytes (17.9 MiB) in that
run. Four independent app workers would therefore add about 71.6 MiB if their
measured deltas are fully resident and unshared. Per-call numeric state is 1,152
bytes, plus at most 2,560 candidate bytes, 255 partial-frame bytes, and at most
2,560 bytes temporarily waiting behind an accepted reset, excluding Python object
overhead.

The real-model offline test rejects generated silence; 100, 440, 1,000, 2,000 and
3,000 Hz tones at several amplitudes; a three-tone mixture; and seeded white noise
at four amplitudes after mu-law conversion. It detects an unmodified segment from
the upstream MIT-licensed `examples/c++/aepyx_8k.wav` fixture. The fixture source,
source digest, exact frame interval and embedded PCM digest are recorded in
`models/vad/provenance.json`.

These checks do not reproduce the owner's private music, nearby voices or singing.
Silero VAD can classify nearby speech or vocal music as speech, and it does not
identify which person spoke. The private approved Arabic normal/quiet files were
not present in this workspace, so Arabic and quiet-speech acceptance remains a
later opt-in evaluation requirement.

## Lead review supplement (2026-09-17)

The Python 3.11 and local Linux checks, direct recovery corrections and private
Arabic replay evidence are recorded in `docs/delegation/T033-G-review.md`.
Failure or cancellation after owned playback removal now releases a still-current
call to LISTENING instead of leaving it permanently SPEAKING. Classifier reset
failure and cancellation retire partial gate evidence.

The local Linux image uses Python 3.12.14. Its measured model/runtime RSS delta was
55.6 MiB per process, with 0.424 ms warmed p95 inference; four independent deltas
would be roughly 222.4 MiB. This supersedes using the Windows estimate alone for
release planning. Real server capacity and end-to-end interruption timing remain
release/supervised-call checks. The owner's normal and quiet Arabic clips triggered
successfully offline, but distant speech also triggered. No music-rejection or
speaker-identity guarantee follows from these results.

## Reversal procedure

Setting `SPEECH_AWARE_BARGE_IN_ENABLED` to false restores the deployed byte/order
and decision path through the legacy detector. No database, provider setting or
diagnostic schema rollback is involved. Removing the runtime/model package belongs
to a later release change; this implementation performs no deployment.
