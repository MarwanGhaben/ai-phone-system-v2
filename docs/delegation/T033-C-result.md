# T033-C provider compatibility probe result

Lead addendum: reviewed and corrected locally. Focused16 and actual websockets12
loopback six-arm checks passed; clean diagnostic-only archive408 passed/50 skipped.
See [T033-C independent review](T033-C-review.md) for failing-first findings,
corrections, preservation and remaining live-evidence boundary. Original
implementer evidence follows unchanged.

Implemented only the standalone diagnostic scope on branch
`codex/phase-4-arabic-voice-quality` at unchanged HEAD
`2ba16c1aeea787bd8de6a89a65e24f571e7d14dd`. T033-A/B production files, tests,
Settings, Compose and manifests were preserved. Filtering remains default-off.
Nothing was committed, pushed, deployed or contacted at the provider.

## Implementation

`scripts/check-stt-provider-compatibility.py` is offline by default and has one
explicit `--allow-provider-contact` opt-in. Settings and the API key are loaded
only inside that execution path, and only the reviewed model/key fields are used.
The command performs six sequential handshake-only arms: auto, English and Arabic,
each baseline and `filter_background_audio=true`. It preserves the reviewed query,
uses the `xi-api-key` header, sends no audio or end-of-stream message, and closes
each socket with bounded child-task ownership.

Only a valid `session_started` object is accepted. Provider errors are limited to
the official allowlist; malformed, oversized, unknown-flood, close and timeout
outcomes are fixed local categories. Filtering echo is `unknown` when absent, so
acceptance is never overstated. Output contains no URLs, headers, secrets, raw
events, exception classes, provider bodies, session IDs or caller data. Aggregate
success requires all six arms to be accepted; failures return nonzero.

## Offline evidence

The synthetic runtime harness exercised the actual probe functions and synthetic
sockets:

```text
T033C_OFFLINE_RUNTIME_CHECKS_OK=9
ACTUAL_ADAPTER_QUERY_MATCH_OK=6
```

Those scenarios covered offline default/help and incompatible-configuration behavior,
all six exact query arms and no audio, open versus `session_started`, honest missing
echo, named error redaction and no retry, remote close, malformed and oversized
events, unknown-event bounds, and cancellation-resistant receive cleanup. The
adapter comparison used the actual local `ElevenLabsSTT` constructor for all six
arms; no independent expected-query fixture replaced it.

Additional checks completed:

```text
AST_OK
DEFAULT_EXIT=0
```

The prescribed pytest suite could not run in this environment: Python 3.11 is
unavailable and Python 3.14 has no pytest or project dependencies. No provider
contact was attempted. The opt-in command was not executed.

## Remaining limitations

This is a handshake compatibility diagnostic, not a filtering or speech-quality
acceptance test. It does not measure language metadata, transcription accuracy,
noise suppression, caller recognition, local RMS/barge-in behavior, endpointing or
business-call outcomes. The lead must review and publish a release before any
owner runs the opt-in command. Even six accepted arms require the paired-audio
evaluation before enabling filtering.
