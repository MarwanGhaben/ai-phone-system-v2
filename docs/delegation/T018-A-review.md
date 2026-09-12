# T018-A lead review

Reviewed 2026-09-11 on codex/phase-3-safe-booking, starting at 2535759.
Accepted for owner-run, read-only baseline evidence collection; no synchronization
or application deployment is included.

Lead corrections after inspecting SOL's actual diff:
- Removed the fixed UTC-04 timezone substitute and fake imported HTTP/database/
  settings modules from tests. Real ZoneInfo rules, HTTP exception classes and
  application settings import are exercised; external I/O remains synthetic.
- Added winter readback to prove Toronto changes to UTC-05, while fixed September
  test expectations remain explicitly separate from local/provider agreement.
- Reject date-only strings as malformed provider timestamps; cover genuinely
  missing pair members as well as null values.
- Preserve opaque saved provider IDs exactly instead of silently stripping them.
- Restore standalone Loguru sink suppression alongside standard logging suppression.

Verification:
- Focused: 12 passed in 0.77 seconds.
- Full normal suite: 120 passed, 27 opt-in skips, two existing warnings, 4.58 seconds.
- git diff --check: no whitespace errors (Windows line-ending notices only).

The tests exercise read-only SQL selection and bounded synthetic HTTP responses,
not live Graph or PostgreSQL execution. No migration/application/Compose changes
require repeating the previously accepted image/DB deployment suites. This does
not independently verify the owner's new booking until server readback returns.

Next owner action: run the published standalone script in ai-voice-app. It selects
the newest canonical-time row, outputs the numeric local ID and strict comparison
results. Reuse that ID on every subsequent change check. Keep the appointment
unchanged until baseline review. A Graph 404 remains unavailable/unknown rather
than proof of cancellation. Parent T017/T018 and Outlook identity mapping remain
open. Server image stays 075d8cf/f65fd1b4; this script does not replace it.
