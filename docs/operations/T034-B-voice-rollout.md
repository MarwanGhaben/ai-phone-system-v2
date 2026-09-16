# T034-B voice beta rollout

Status: prepared for lead review only. Nothing in this document has been published
or run on the server. The lead must replace both placeholders below with the same
new reviewed 40-character commit before giving this procedure to the owner.

## 1. Finish calls, then run the pinned release

Finish any beta calls first. In the **DigitalOcean browser console**, verify that
`PUBLISHED_T034_B_COMMIT` has been filled with the lead-published commit, then run:

```bash
bash <<'BASH'
set -euo pipefail
cd /opt/ai-phone-system-v2
release='PUBLISHED_T034_B_COMMIT'
test "$release" != 'PUBLISHED_T034_B_COMMIT' && test "${#release}" -eq 40
git fetch origin codex/phase-4-arabic-voice-quality
git show "$release:scripts/deploy-voice-beta.py" | sudo python3 - "$release"
BASH
```

Do not substitute the currently deployed source SHA: that commit does not contain
this release script. A successful run ends with all three markers:

```text
VOICE_BETA_DEPLOYED_READY_HTTPS_OK
DEPLOYED_COMMIT=<the published commit>
DEPLOYED_IMAGE=sha256:<candidate image ID>
```

Keep the printed `/opt/ai-phone-voice-release.*` directory. It contains protected
rollback evidence. If the command stops or prints a recovery-failure marker, do
not rerun it. Give the lead only the public stage/type markers and release path;
do not paste `private.log`, settings, environment values or provider responses.

## 2. Run the language-choice regression smoke

Make one short, non-booking call. Ask an FAQ whose approved answer is known, ask
for English, then ask for Arabic. Confirm that the agent follows those explicit
language choices. If a before-release recording or note exists, use the same
prompts for the comparison. Record the approximate time, requested languages and
any incorrect behavior without publishing caller audio or private logs.

This is a regression smoke check. It does not establish Arabic recognition,
pronunciation or dialect acceptance, and it is not public-launch approval.

## 3. Check interruption, repetition and existing operations

Make a second short call. Interrupt one response and ask a follow-up that could
otherwise cause the previous answer to repeat. Record the exact symptom, language
and approximate time. Then check the existing dashboard and confirm that the
already-working booking observation and automatic-notification status still look
normal. No booking is required for this smoke check.

The speech consumer remains serial and broader correction handling is still open.
This release does not claim that every interrupted utterance is preserved.
