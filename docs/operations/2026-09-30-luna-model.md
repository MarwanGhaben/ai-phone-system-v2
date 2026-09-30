# GPT-6 Luna on the restored phone flow

Owner requested implementing the model switch. This release changes the OpenAI
adapter and the app's `OPENAI_MODEL` override only. ElevenLabs, prompts, calendar
handlers, schema, notification workers and the disabled phase-5 route remain as
they were in the accepted running predecessor.

## Compatibility

The exact model is `gpt-6-luna`, with `reasoning_effort=none`. OpenAI's model page
requires this setting for Luna function calling through Chat Completions:
https://developers.openai.com/api/docs/models/gpt-6-luna

The adapter sends the existing per-request voice budget as `max_completion_tokens`.
It retains temperature, which is supported with reasoning disabled. SDK 1.10 does
not expose these new fields as named arguments, so the adapter forwards them via
`extra_body`, merging rather than replacing the existing single-tool flag.
Normal chat, streaming, and tool requests share these Luna parameters. GPT-4o wire
parameters and the repository's default model remain unchanged. The existing
ordinary token-cost estimate now recognizes Luna's published standard input/output
prices; it is not a cache-aware billing reconciliation.

API parameter reference:
https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create

The stream is explicitly closed on completion or early termination; the initial
SDK regression run exposed a pending async-generator cleanup warning.

## Deployment

Use `scripts/deploy-luna-model.py` from the exact published release commit and
pass that same full commit as its argument. Finish active calls first. The script
accepts either restored revision `ca46f311e05e4a0c2f3093ed7e3781077c57f7ed` or
availability-reply revision `02c3995f196b74a84e8c71ecd1c167d546955530`. It verifies
the exact supported effective source variant, preserves that caller flow, and
does not reapply the availability change or replace the orchestrator.

The image derives from the running app and replaces only
`services/llm/openai_service.py`. The controlled override changes only its image
and `OPENAI_MODEL=gpt-6-luna`. No dependencies are installed, no migration runs,
and no API key is printed or passed in command-line arguments. Existing protected
settings, mounts, database, and unresolved operation evidence remain in place.

Before stopping traffic, the candidate passes the offline SDK probe and effective
source/settings checks. It then runs `scripts/probe-luna-compatibility.py --live`
using the server's existing OpenAI API key. This makes at most six sequential,
synthetic requests: chat, tool call, and linked streamed tool-result response in
both English and Arabic. No customer conversation is sent; no calendar, SMS or
telephone action is executed. SDK retries are disabled for this probe, individual
requests have a 15-second timeout, and the entire probe has a 100-second budget.

Access denial, API rejection, timeout, malformed arguments, incomplete response,
or failed basic language checks stop before cutover. Output contains only fixed
classifications and an HTTP status where available. Do not respond to a failed
probe by changing keys or repeatedly deploying without examining that status.

After a successful probe, the script rechecks the predecessor, performs the
app-only replacement, and verifies settings, readiness, health and HTTPS. A failed
cutover restores the exact previous app image and override. Protected release files
and the previous image remain available for recovery.

Success marker: `LUNA_MODEL_DEPLOYED_READY_HTTPS_OK`, followed by commit, image,
and `OPENAI_MODEL=gpt-6-luna`.

## Evidence and limits

- Original adapter: 8 failing Luna wire cases, 8 passing GPT-4o controls.
- Installed SDK tests cover English/Arabic normal, streaming, tool, and
  single-tool requests with preserved messages, IDs, arguments and token budgets.
- Final focused adapter/probe/release selection: 44 passed, 1 Docker skip.
- Docker-enabled release suite: 24 passed, including actual Linux image building,
  the offline six-request SDK probe, and unchanged restored bilingual booking
  journeys. Baseline variants, source drift, settings preservation, failed API
  gates and exact rollback are covered by release regressions.
- Broader LLM/conversation/STT/TTS/telephony suite: 575 passed, two known optional
  audioop/ffmpeg warnings. No full-repository green claim is made.
- Source pins, Python compilation and whitespace checks passed. Owned rehearsal
  containers/images were removed and independently checked absent.

No live OpenAI call was made locally. Account access and real response compatibility
remain gated on the owner-run server probe. Local Docker uses an available dependency
parent layered with pinned accepted source; the remote production image is verified
by the rollout. Synthetic language checks do not establish natural Arabic, dialect
recognition, real call latency, or full booking quality.

After deployment, test English and Arabic availability, a consultant/date correction,
and an explicit language switch. A new supervised booking should still be verified
in Microsoft, the local dashboard and SMS. Keep the current model/flow baseline
available if Luna performs worse. This release closes no parked booking backlog.
