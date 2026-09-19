# T013-A implementation result

Date: 2026-09-18

Branch: `codex/phase-5-booking-reliability`

Required and retained HEAD: `792952b2965e85513bf0971a41bfaff7bd3377c4`

## Outcome

Implemented the bounded typed Microsoft availability reader:

```python
async def MSBookingsService.get_availability(
    query: AvailabilityQuery,
) -> AvailabilityResult
```

The method validates the accepted query and configured scope before network
contact, sends the exact requested staff tuple and UTC window to the fixed Graph
v1.0 `getStaffAvailability` action, and decodes only complete validated evidence.
It reuses the service's injected/pooled `httpx.AsyncClient`, preserves
`CancelledError`, follows no redirects, performs no retries or sleeps, and has a
20-second whole-call budget covering token acquisition and streamed response
reading. The budget is a class attribute so offline tests can shorten it.

This is deliberately not connected to the legacy list-based caller route. The
running phone agent still uses `get_available_slots`; therefore this local method
alone does not fix live booking behavior.

## Changed paths

- `services/calendar/contracts.py` (new): pure, clock-free and I/O-free wire
  decoder.
- `services/calendar/ms_bookings_service.py`: typed read method plus small bounded
  response/auth and safe classification helpers; `_get_client` now states the
  already-default no-redirect behavior explicitly.
- `tests/calendar/test_graph_availability_contracts.py` (new): decoder and actual
  `httpx.MockTransport` adapter coverage.
- `docs/delegation/T013-A-result.md` (new): this record.

No mutation method, shared calendar abstraction, setting, dependency, manifest,
task checkbox, migration, voice path, notification path, or deployment artifact
was changed.

## Wire compatibility and fail-closed boundaries

- Accepts exactly one of the `value` or `staffAvailabilityItem` collection
  envelopes. A dual envelope is invalid; a continuation marker is incomplete and
  is never followed.
- Requires exactly one structurally valid result entry for every requested staff
  ID. Missing staff is incomplete. Duplicate, unexpected, or malformed staff is
  invalid. Every item is validated, including non-free items.
- Supports the documented lower-camel statuses and the title-case spellings shown
  by the method documentation. Only `available`/`Available` creates free evidence.
  `busy` and `outOfOffice` are understood non-free evidence.
  `slotsAvailable`, `unknownFutureValue`, and unknown future spellings yield
  `INCOMPLETE` with no partial intervals.
- Normalizes timestamps to UTC. Supports `UTC`, valid IANA zone names, `Eastern
  Standard Time` -> `America/Toronto`, `Pacific Standard Time` ->
  `America/Los_Angeles`, and explicit Pacific display-label aliases. Named-zone
  gaps and folds are rejected unless an explicit offset unambiguously agrees with
  the named zone. Missing/unsupported zones and conflicting offsets fail closed.
- Accepts one through six fractional digits. A seventh digit is accepted only
  when it is zero and therefore exactly representable at Python microsecond
  precision; a nonzero seventh digit is rejected.
- Requires `start < end` and full containment in the query window. Evidence is
  never clipped. Exact duplicates and overlapping free/non-free contradictions
  are invalid. Distinct overlapping free intervals are retained in provider order;
  adjacent intervals are not treated as overlapping.
- Complete understood evidence with no free intervals returns
  `NO_AVAILABILITY`. Missing capability/evidence returns `INCOMPLETE`; malformed
  evidence returns `INVALID_RESPONSE`; provider/auth/transport failures return
  `UNAVAILABLE` with a fixed safe category. No failure becomes a successful empty
  calendar.
- Local defensive limits are 1 MiB for the streamed Graph body, 64 KiB for the
  streamed token body, and 10,000 availability items across all staff. These are
  not asserted Microsoft service limits. Limit breaches return `INCOMPLETE` with
  no intervals and streamed responses are closed.
- The token and action responses are each single-attempt. Token HTTP 400/401 and
  Graph 401 are authentication failures; 403 and 429 are permission/throttling
  failures at either boundary. Timeout, transport errors, other unsuccessful
  statuses, and unexpected 2xx responses are classified without using response
  bodies or exception text. HTTP 404 is provider failure, never empty
  availability.
- Newly added paths emit no raw token, URL, identifier, body, customer data, or
  exception text in logs, errors, or result representations.

The implementation follows these wire references retained from the assignment:

- [getStaffAvailability](https://learn.microsoft.com/en-us/graph/api/bookingbusiness-getstaffavailability?view=graph-rest-1.0)
- [staffAvailabilityItem](https://learn.microsoft.com/en-us/graph/api/resources/staffavailabilityitem?view=graph-rest-1.0)
- [availabilityItem](https://learn.microsoft.com/en-us/graph/api/resources/availabilityitem?view=graph-rest-1.0)
- [dateTimeTimeZone](https://learn.microsoft.com/en-us/graph/api/resources/datetimetimezone?view=graph-rest-1.0)

No live tenant call was made, so neither envelope is claimed as the observed live
shape.

## Behavioral and test evidence

Tests were written before implementation. The first focused run failed during
collection as intended with:

```text
ModuleNotFoundError: No module named 'services.calendar.contracts'
```

All requested commands used the already-present Python 3.12.14 runtime and its
already-extracted cached packages, with bytecode writes disabled. No dependency
was installed.

1. Focused contract tests:

   ```text
   python -B -m pytest tests/calendar/test_graph_availability_contracts.py tests/scheduling/test_booking_result_contracts.py -q
   215 passed in 4.57s
   exit code 0
   ```

2. Calendar, booking-safety, and persistence regression:

   ```text
   python -B -m pytest tests/calendar tests/conversation/test_booking_safety_guards.py tests/integration/test_booking_persistence.py -q
   86 passed, 3 skipped, 1 warning in 13.20s
   exit code 0
   ```

3. Full suite:

   ```text
   python -B -m pytest -q
   7 failed, 817 passed, 53 skipped, 8 warnings in 68.52s
   exit code 1
   ```

   Six failures are the documented historical release-manifest hash failures:
   four in `test_barge_in_diagnostics_release.py` and two in
   `test_voice_release.py`. The additional failure is the previously identified
   10 ms STT timing test
   `test_connection_and_close_timeouts_are_bounded_and_logs_are_safe`; it failed
   again in isolation (`1 failed in 5.69s`) and emitted its existing abandoned
   connection/send pending-task warnings. It is not claimed fixed and is outside
   the allowed T013-A paths.

4. Scoped lint:

   ```text
   ruff check services/calendar/contracts.py services/calendar/ms_bookings_service.py tests/calendar/test_graph_availability_contracts.py
   exit code 0
   ```

5. Repository checks:

   ```text
   git diff --check
   exit code 0

   git status --short --branch  # T013-A entries shown below
   ## codex/phase-5-booking-reliability
    M services/calendar/ms_bookings_service.py
   ?? services/calendar/contracts.py
   ?? tests/calendar/test_graph_availability_contracts.py
   ?? docs/delegation/T013-A-result.md
   ```

   The full status also retains the owner's pre-existing untracked work. The
   command confirmed the unchanged branch and HEAD; no unrelated path was edited
   for T013-A. Git emitted its existing LF-to-CRLF working-copy warning for the
   tracked adapter, but the whitespace check returned success.

The T013-A timeout, size-limit, invalid-response, and explicit cancellation tests
all close their owned synthetic streams/clients, and the focused/scoped runs emit
no retained-task diagnostics. The pending-task diagnostics above originate only
from the unrelated failing STT test. No provider, server, database, container, or
deployment resource was contacted or created.

## Preservation evidence

Raw SHA-256 checks for the accepted T007-A files still exactly match the frozen
review values:

```text
576d6627208d2a621edf2a41c01493da04284d6a94f16174347a48002f1f931d  services/scheduling/models.py
ceaefbc1b4b8372b6aab614800a1acb4f360ab9dca687be2b5c48f06f4e260ad  tests/scheduling/test_booking_result_contracts.py
f6ff3d65f12d40a559b47f7313989d4409b63887bd93c3b4a5a4bd8ba9020e3b  docs/delegation/T007-A-result.md
```

AST source-segment SHA-256 checks for the four legacy methods match their
pre-implementation values exactly:

```text
55568c10a8b1244bc2a126c1693743c42071771808a9c19686e3fcde323f0376  get_available_slots
cde0339b8e24b0050fd8cde7811fe8b5ee4384d436afc26c97e21c4f308c2bae  create_booking
21d1978a7be0e64a03f88e0a87377ee1e421a0fac419f00625e9933aeb8e95ad  get_customer_appointments
527807302bed4d10dc21ca01dc65889ba06353fba6596abaa69581352e938978  cancel_customer_appointment
```

## Remaining work

The legacy phone-agent/tool path still needs an explicitly reviewed integration
slice before it can consume `AvailabilityResult`. Service/staff policy, notice and
horizon enforcement, proposal/readback/confirmation contracts, retries/backoff,
and durable mutation/concurrency/unknown-outcome safeguards remain separate
T010-T020 work. This implementation does not waive any of those gates and does not
authorize an appointment write.

No commit, push, deployment, provider contact, or branch change was performed.

## 2026-09-18 focused correction addendum

The three findings in `T013-A-review.md` were corrected without changing the
accepted domain model, the four legacy booking method bodies, or the unconnected
phone-agent route.

### Corrections

1. Streamed token and Graph exchanges now run in individually owned tasks. The
   20-second request deadline remains unchanged. If deadline or owner cancellation
   reaches response cleanup, the worker is cancelled and given a separately named
   one-second `AVAILABILITY_CLEANUP_GRACE_SECONDS`; public completion no longer
   waits indefinitely for cancellation-resistant `aclose` behavior.

   A worker still pending after that grace is retained by the service, its late
   exception is consumed, and its availability admission slot remains occupied
   until it completes. `AVAILABILITY_CLEANUP_CAPACITY` limits active plus retained
   availability work to eight, so repeated hostile transports cannot create an
   unbounded task set. New availability reads wait within their own request budget;
   legacy appointment lookup remains outside this availability-only admission
   boundary. Service shutdown waits at most the same cleanup grace for retained
   availability workers and creates no replacement cleanup tasks. The subsequent
   existing HTTP-client close behavior is unchanged; this does not establish a
   deadline for the complete service shutdown. A permanently hostile transport can remain
   quarantined under this bound and is not claimed closed.

2. Continuation detection now covers the active top-level `value` or
   `staffAvailabilityItem` collection, its collection-qualified
   `<envelope>@odata.nextLink` annotation, and each recognized nested
   `availabilityItems` collection including
   `availabilityItems@odata.nextLink`. Presence is incomplete evidence regardless
   of marker value. Unrelated nested metadata is not recursively scanned, and
   explicit empty item collections remain valid no-free evidence.

3. Token JSON now rejects non-standard numeric constants during parsing and then
   requires a finite, positive, non-boolean numeric `expires_in`. NaN, either
   infinity, numeric overflow, booleans, nulls, strings, and containers return
   `INVALID_RESPONSE` before Graph contact or cache mutation. A replacement
   failure preserves prior cache fields. Tokens whose lifetime cannot accommodate
   the existing 60-second safety margin are used only for the current request and
   are not cached beyond their actual lifetime; ordinary finite caching is
   preserved.

### Failing-first evidence

The focused correction selection failed against the reviewed implementation with:

```text
19 failed, 6 passed, 66 deselected in 7.81s
exit code 1
```

Failures reproduced both collection-qualified envelopes, nested continuation,
NaN/Infinity/overflow acceptance, short-lifetime over-caching, unbounded token and
Graph cleanup, repeated cancellation, late cleanup ownership, missing admission
capacity, and lookup isolation while cleanup was stalled. Every event gate was
released and every public test task was joined in `finally`.

### Final validation

The commands ran with the already-present Python 3.12.14 runtime and extracted
cached dependencies; no package was installed.

```text
python -B -m pytest tests/calendar/test_graph_availability_contracts.py tests/scheduling/test_booking_result_contracts.py -q
241 passed in 5.75s
exit code 0

python -B -m pytest tests/calendar tests/conversation/test_booking_safety_guards.py tests/integration/test_booking_persistence.py -q
112 passed, 3 skipped, 1 warning in 15.05s
exit code 0

python -B -m pytest -q
7 failed, 843 passed, 53 skipped, 8 warnings in 67.40s
exit code 1
```

The full-suite failures remain outside T013-A: four historical diagnostics-release
manifest checks, two historical voice-release manifest checks, and the known 10 ms
STT connection timing test. The latter again emitted its existing pending cleanup
diagnostics in this Python 3.12 environment and is not claimed fixed.

The unchanged lead reproducer exited zero and reported:

```text
TOKEN nan INVALID_RESPONSE requests 1
TOKEN infinity INVALID_RESPONSE requests 1
NESTED_CONTINUATION INCOMPLETE
TIMEOUT_DURING_CLOSE token returned True
CLEANUP token False True
TIMEOUT_DURING_CLOSE graph returned True
CLEANUP graph False True
```

The immediate `CLEANUP ... False` samples occur before that reproducer yields to
the retained cleanup workers after opening their gates. Maintained regressions now
wait for the service-owned cleanup set to drain and verify delayed normal close,
late close-exception consumption, admission release, repeated cancellation,
bounded retention, bounded repeated service close, and unrelated lookup progress.

Scoped Ruff validation passed, and `git diff --check` exited zero with only the
existing Windows LF-to-CRLF notice for the tracked adapter. The three frozen T007-A
raw hashes and four legacy method source-segment hashes were rechecked and still
exactly match the values in the preservation section above. The correction changed
only the same four allowed T013-A paths. No provider/server/database/container
contact, installation, task-checkbox change, staging, commit, push, deployment, or
branch change occurred.
