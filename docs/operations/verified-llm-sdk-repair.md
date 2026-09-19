# Verified phone SDK compatibility repair

The first owner call after af3873c could not check Hussam's availability in
English or Arabic and repeatedly offered human help. The pinned OpenAI 1.10.0
SDK rejects the new `parallel_tool_calls` keyword before making an HTTP request.
The verified orchestrator catches that error and produces exactly that fallback.
Earlier phone tests replaced the LLM adapter, hiding this compatibility boundary.

Only the LLM adapter changes at runtime: send `parallel_tool_calls: false` through
the SDK's supported `extra_body`. This retains the server-side single-tool policy
and the local rejection of ambiguous tool calls. No dependency, prompt, booking
policy, Graph operation, notification, schema or migration changes.

## Evidence

- Actual verified orchestrator plus OpenAI 1.10.0 and HTTP MockTransport: four
  failures before the correction, including the exact Arabic/English fallback
  and zero HTTP requests. All four pass afterward, verifying the transmitted
  request body and linked availability tool handling.
- Focused LLM/booking checks: 42 passed.
- Conversation/STT/TTS/telephony/LLM regressions: 419 passed, two existing warnings.
- App-only release checks: six passed, including refusal before cutover and
  exact previous override/image restoration on failed candidate readiness.
- Embedded SDK probe passed in a disposable Linux container with networking
  disabled and synthetic HTTP responses. It used an available older local app
  image plus the repaired adapter, not the exact server-only parent image.
- The actual server image and private configuration will be checked by the
  owner rollout. Live calls and provider replies have not been retested yet.

## Rollout

After publication, use the exact reviewed full commit supplied by the lead with
`scripts/deploy-verified-llm-repair.py`. Finish active calls first.

The script accepts only predecessor af3873c and image
`sha256:79de88be290df1c4895c9e2cc7af642737eb68133f0f495b1da93469c2e8c68a`.
It loads immutable helper methods from af3873c, checks the effective frozen
runtime/model files and settings, and permits exactly the reviewed adapter edit.
Its isolated build context contains only that module and its Dockerfile. No
private settings or logs enter the build context. No dependencies are installed.

Before stopping traffic, the candidate must pass the actual SDK offline probe
and effective mounted source/settings verification. Compose changes only the app
image; private mounts, workers, environment and other services remain unchanged.
Cutover replaces only the application and verifies health, readiness, effective
source, settings and HTTPS. It never invokes database migrations or restores.
Failure during cutover attempts exact previous image/override restoration; that
restores the preceding service state, which still has the SDK bug. If recovery
cannot be verified, ingress is stopped where possible and manual review is needed.

Success prints `LLM_SDK_REPAIR_DEPLOYED_READY_HTTPS_OK`, commit and image.
On failure, preserve the protected release folder and return only printed
markers. Do not rerun or share private logs. After success, repeat a short
Hussam availability request before attempting a confirmed booking.
