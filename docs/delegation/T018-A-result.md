# T018-A implementation result

Date: 2026-09-11

## Scope and starting point

- Verified branch: `codex/phase-3-safe-booking`.
- Verified HEAD: `2535759ac5321ca80e37cf821c17274c4d53bd83`.
- Production code changed only in `scripts/check-latest-booking.py`.
- No application, migration, settings, dependency, fixture, deployment-script or
  task-checkbox change was made. Existing uncommitted work was preserved.
- No live database, Microsoft Graph appointment or other provider was contacted.

## Implemented behavior

The standalone probe now accepts an optional positive `--row-id`. An explicit ID
uses one `WHERE id = $1` query and never falls back. With no ID, the query selects
the newest row whose `appointment_time_utc` is non-NULL. Both paths select only
the local numeric ID, saved provider ID, canonical UTC time and status inside a
read-only transaction with connection, command, statement and overall time bounds.

The Graph request reads the exact URL-encoded business and appointment IDs from
the fixed Microsoft Graph host. Redirect following is disabled. The only non-GET
request is the OAuth client-credentials POST. The probe has no database write,
schema operation, Graph mutation, enumeration, subscription, retry or redirect
path.

Appointment parsing accepts either the documented `start`/`end` pair or the
tenant-observed `startDateTime`/`endDateTime` pair. If both complete pairs appear,
their parsed instants must agree. Partial pairs, malformed values, missing or
unknown timezones, nonzero offsets labelled UTC, conflicting pairs and nonpositive
durations are rejected. UTC naive, `Z`, zero-offset and seven-digit fractional
timestamps are supported without assigning a zone to legacy local timestamps.

Output is one sanitized JSON document. A successful observation contains the
numeric local row ID, allowlisted local status, local canonical UTC time, provider
times converted to `America/Toronto`, duration, boolean provider-ID and local-start
matches, and fixed-test comparisons for Hussam, September 14, 2026 at 11 AM
Toronto, 30 minutes, the expected service and in-person location. Optional fields
use `match`, `mismatch` or `unknown`; absent fields cannot produce a verified match.

Failures have fixed classifications and stages for absent/unverified local rows,
authentication, Graph 404, authorization, throttling, server failures, timeouts,
network errors, malformed successes, identity mismatches and valid observed
mismatches. A Graph 404 reports `cancellation_state: unknown`; it does not change
or infer local cancellation. Raw exceptions, response bodies, identifiers,
settings, URLs and contact data are never emitted.

## Same-row readback procedure

The baseline probe result supplies `local_row_id`. The lead can retain that number
and, after reviewing baseline evidence, use `--row-id` on a later run so the probe
re-reads the same local row and its saved Bookings appointment. This prevents a
newer booking from changing the target between observations.

No Outlook edit or deletion is requested by this result. The owner has not yet
been asked to change the test appointment. Any later change requires separate
owner authorization after baseline evidence and lead review. A Bookings
appointment ID is not assumed to equal a staff Outlook event ID, and a missing
resource or deleted attendee copy is not classified as cancellation.

## Test evidence

Failing first, before the production-script change:

```text
python -B tests/integration/test_booking_readback_diagnostic.py -v
Ran 11 tests
FAILED (errors=26)
Representative failure: report() got an unexpected keyword argument 'row_id'
```

The first implementation run passed nine test methods and exposed two incorrect
fractional-second values in the new test fixtures. After correcting those fixtures,
the final focused run passed:

```text
python -B tests/integration/test_booking_readback_diagnostic.py -v
Ran 11 tests in 0.840s
OK
```

The 11 methods include parameterized/subtest coverage for both time-field shapes,
equal and conflicting dual pairs, missing/partial/malformed values, UTC offset
rules, invalid duration, local/provider mismatch, fixed-test mismatch/unknown,
explicit/default selection, legacy exclusion, safe HTTP/network classifications,
privacy sentinels, read-only SQL, allowed HTTP methods and same-row reuse.

Additional checks:

```text
python -B -m compileall -q scripts/check-latest-booking.py \
  tests/integration/test_booking_readback_diagnostic.py
compile_exit=0

python -B scripts/check-latest-booking.py --help
exit=0; help lists optional --row-id

Python 3.11 AST parse and trailing-whitespace check
PASS

python -B -m pytest tests/integration/test_booking_readback_diagnostic.py -q
C:\Python314\python.exe: No module named pytest
```

The focused test is written with standard-library `unittest` so it runs directly
here and remains discoverable by pytest. Focused pytest and full pytest could not
be run because this environment lacks pytest. No package was installed. The lead
still needs to run the requested focused and full pytest suites in the normal test
environment before publishing an owner command.

## Remaining questions

There are no open implementation questions within T018-A. Baseline provider
readback and any later owner-authorized comparison remain operational evidence,
not work performed or claimed by this assignment. Full synchronization,
notification subscriptions, Outlook/Bookings identity correlation and local
status reconciliation remain outside this bounded task.
