# T033-C provider compatibility probe

`scripts/check-stt-provider-compatibility.py` is a standalone diagnostic for the
reviewed ElevenLabs Scribe v2 query. It is offline by default:

```powershell
python3 scripts/check-stt-provider-compatibility.py
python3 scripts/check-stt-provider-compatibility.py --help
```

Those invocations load no Settings, open no socket and do not read a credential.
The only execution mode is an explicit opt-in:

```powershell
python3 scripts/check-stt-provider-compatibility.py --allow-provider-contact
```

The owner must use that command only after the lead publishes a reviewed release
containing this diagnostic. The future command opens six short sequential
connections (auto, English and Arabic, each baseline and filtering candidate),
waits only for `session_started`, sends no audio, and closes each socket. It may
consume provider quota and is not a free local check. No server command or commit
is published here; any future release reference remains a lead-filled
`<REVIEWED_COMMIT>` placeholder until publication.

Each result has a fixed label, language, filtering request, outcome, allowlisted
provider error category, filtering echo state and numeric elapsed time. A socket
opening is not acceptance. A missing filtering echo is `unknown`, never proof that
the option was applied. The aggregate succeeds only when all six arms receive a
valid `session_started` event. Rejection, malformed/oversized input, remote close,
unknown-event flood, timeout or local configuration failure produces a nonzero
result without retry or downgrade.

The query preserves the reviewed endpoint and parameters: `scribe_v2_realtime`,
`ulaw_8000`, `8000`, VAD commit strategy, `0.3` silence threshold and language
detection. Auto detection omits `language_code`; English and Arabic add only their
language hint. The candidate adds only `filter_background_audio=true`, and never
adds `include_timestamps`. This matches the [official realtime API reference](https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime),
which documents the `xi-api-key` header, `session_started` response and the
filtering parameter, and the [official event reference](https://elevenlabs.io/docs/eleven-api/guides/how-to/speech-to-text/realtime/event-reference).

This handshake does not test language metadata delivery, transcription accuracy,
noise suppression, caller recognition, local RMS/barge-in behavior, audio
endpointing or the business call path. Even six accepted arms require the paired
audio evaluation before any filtering activation is considered. Filtering remains
default-off and no production settings are changed by this diagnostic.

## Reviewed execution procedure

The lead supplies the exact published diagnostic commit. In the DigitalOcean
browser console, fetch `codex/phase-4-arabic-voice-quality`, then pipe that commit's
`scripts/check-stt-provider-compatibility.py` into
`docker exec -i -w /app ai-voice-app python -B - --allow-provider-contact`.
Use `set -o pipefail` first. Do not switch the server checkout, rebuild, restart,
edit environment files or reuse an app deployment script for this check.

Expect one JSON object, normally within 80 seconds, with `status`,
`aggregate_success` and the six arm results. Return that object even if an arm
fails. Do not retry, change options or paste private logs in response to failure.
A `cleanup_failed` result is not accepted even if the handshake had succeeded.
Completed results remain visible if the overall budget expires. No customer data
or audio is used, and no appointment/SMS/database operation is performed.
