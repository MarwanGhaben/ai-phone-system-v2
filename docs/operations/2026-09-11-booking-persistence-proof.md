# Reproduced: provider success without a local booking row

Lead execution on 2026-09-11, local application source at
`4d87c3a3633680350552fe1fd9a1f10498adce08`. The relevant confirmation method is
unchanged from the deployed b1c4dc4 booking release. Real PostgreSQL 16 and asyncpg;
actual bootstrap SQL and actual `_confirm_booking`; synthetic identity/date;
mocked calendar success, speech and SMS. No real provider or server access.

Command (local only):

```powershell
& C:/Python311/python.exe -B docs/evaluation/reproduce-booking-persistence.py --local-docker
```

Observed result:

```json
{
  "simulated_provider_create_calls": 1,
  "returned_booking_success": true,
  "persisted_booking_rows": 0,
  "scheduled_confirmation_sms": 1,
  "bug_reproduced": true
}
```

The driver reports DataError: `can't subtract offset-naive and offset-aware
datetimes`. The real INSERT sends an aware pending datetime to
`bookings.appointment_time TIMESTAMP WITHOUT TIME ZONE`. Its exception is caught;
normal confirmation continues. The independent codec probe also reproduced this
failure with `SELECT $1::timestamp`.

First attempt stopped at settings import; corrected the harness to use a temporary
working directory and synthetic-only environment, then reran successfully (exit 0).
Both temporary database containers were removed with verified absence. Final:
`t005-migrations-e5c5619839ec4a838e4f8c2660af18b4` removed. No downloads or installs.

This proves a reachable application defect. It does not prove the exact cause of
the owner's September call or establish whether deleting the consultant's Outlook
copy cancelled the Bookings appointment. The June record from the latest-row
diagnostic is not evidence that the September appointment was saved locally.

The reproduction is historical evidence, not a test whose required future behavior
is to preserve the bug. The T005-D implementation must add a failing-first
regression that requires successful local persistence.

Next coding assignment: [T005-D](../delegation/T005-D-booking-persistence.md).
External synchronization remains explicitly open in
[calendar-change-sync-gap.md](calendar-change-sync-gap.md).
