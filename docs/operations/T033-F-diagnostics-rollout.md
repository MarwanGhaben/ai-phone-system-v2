# T033-F numeric diagnostics rollout

Status: prepared for lead review only. Nothing here has been published or run on
the server. The lead must replace every `PUBLISHED_T033_F_COMMIT` placeholder with
the same reviewed 40-character commit before giving these commands to the owner.

## 1. Finish beta calls, then run the pinned app-only release

Finish any beta calls first. In the **DigitalOcean browser console**, run this
single pinned fetch/show block only after the lead has filled the placeholder:

```bash
bash <<'BASH'
set -euo pipefail
cd /opt/ai-phone-system-v2
release='PUBLISHED_T033_F_COMMIT'
test "$release" != 'PUBLISHED_T033_F_COMMIT' && test "${#release}" -eq 40
git fetch --no-tags origin codex/phase-4-arabic-voice-quality
git show "$release:scripts/deploy-barge-in-diagnostics.py" | sudo python3 - "$release"
BASH
```

Do not use the current `3b062d435e7bfc907f7feda8bef5949e7ffd843c`
worktree HEAD as the release commit; it does not contain this package. Success
ends with:

```text
BARGE_IN_DIAGNOSTICS_DEPLOYED_READY_HTTPS_OK
DEPLOYED_COMMIT=<the published commit>
DEPLOYED_IMAGE=sha256:<candidate image ID>
```

Keep the printed `/opt/ai-phone-diagnostics-release.*` directory. Its
`candidate-settings.py` remains mounted by the running app, and the directory
also holds protected recovery evidence. If the command stops or prints a
recovery-failure marker, do not rerun it. Return the release path and fixed public
stage/type markers to the lead. Do not paste `private.log`, settings, environment
values, provider responses, tokens, transcripts, phone numbers, CallSid or stream
IDs. The script changes only the app image, the reviewed nested settings-file
mount and the two declared flags. It performs no database migration or restore.

## 2. Make two short supervised non-booking calls

After the success marker, make one short call with quiet foreground speech. Then
make one short call in the usual nearby-speech situation that previously caused
an unwanted interruption. Do not book, cancel or change an appointment, and do
not trigger a customer SMS. Report only each call's approximate time and what the
agent did. Do not make or upload another recording-file pilot.

These calls collect numeric observations. They do not identify who spoke and do
not show that diagnostics solve interruptions. Background filtering must remain
off.

## 3. Run the pinned sanitized reader and return its JSON

In the **DigitalOcean browser console**, run the reader from the same published
commit and return only the JSON it prints:

```bash
bash <<'BASH'
set -euo pipefail
cd /opt/ai-phone-system-v2
release='PUBLISHED_T033_F_COMMIT'
test "$release" != 'PUBLISHED_T033_F_COMMIT' && test "${#release}" -eq 40
git show "$release:scripts/read-barge-in-diagnostics.py" | sudo python3 -
BASH
```

The reader examines only bounded recent logs from the fixed `ai-voice-app`
container, reconstructs allowlisted numeric summaries and never prints raw log
lines. `scan_limited` or `output_limited` means some older data may be absent.
`no_valid_summaries` does not prove diagnostics ran, and `read_failed` requires
lead review without pasting logs. The lead must interpret the returned evidence
before any threshold or filtering change. Do not enable background filtering,
contact a provider, or claim speaker identification from these summaries.
