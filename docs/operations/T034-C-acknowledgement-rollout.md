# Availability acknowledgement rollout

Use only the new commit published by the lead for T034-C. The previous deployed
voice commit is `2f7e15e1b38f051615fc63c11f7bf6a66d827ad9`, with image
`sha256:9cecc2353a29f4fefdd95566a7c1365fca8cbd380e9db8777ce08914cf3b14f5`.
The revised `scripts/deploy-voice-beta.py` requires that exact starting image.

1. Finish beta calls. Run the lead's two pinned fetch/show commands in the
   DigitalOcean browser console, with pipefail enabled. No local PowerShell or
   GitHub website action. This is an app-only update: no database migration,
   provider change, background filter or notification policy change.
2. Expect `VOICE_BETA_DEPLOYED_READY_HTTPS_OK`, followed by the new commit/image.
   On failure return only public stage/type markers and release directory; do not
   rerun blindly or paste private logs. Keep all existing/new release directories:
   private mounted settings and recovery evidence depend on them.
3. Make a short call in Arabic, then English: ask to check availability and listen
   for the brief acknowledgement before the answer. In a separate lookup, interrupt
   the acknowledgement to change consultant. The old lookup must not proceed from
   that interrupted phrase; the next correction should be processed. Do not confirm
   a booking. Report the phrase, requested consultant and result. Background voices
   remain a separate pending task; no noise-performance acceptance is claimed.

The script automatically attempts the exact prior image/configuration on failed
cutover. Schema0004 and current automatic notification settings remain required.
No manual compose rebuild/down, checkout reset or deletion of protected files.
