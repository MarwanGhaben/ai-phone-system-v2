# Booking provider observations

T018-B records read-only Microsoft Bookings checks for existing locally confirmed
bookings. It does not cancel, reschedule, hard-delete, update the booking row,
send messages, change availability or suppress reminders. A Bookings appointment
GET returning 404 is **unavailable and needs review**, not proof of cancellation.
Bookings appointment IDs are not assumed to equal Outlook staff event IDs.

## Schema and migration

`migrations/0003_booking_provider_observations.sql` creates one observation row
per numeric `bookings.id`, with `ON DELETE CASCADE` only to remain compatible with
the existing local delete endpoint. The migration has no data backfill and does
not assign a timezone to historical `bookings.appointment_time`. The table stores
the local provider ID/start/status snapshot, the exact observed provider identity
on a valid matching GET, provider interval and optional staff/service/location
metadata. It stores no names, contacts, notes, raw responses, tokens or URLs.

`checked_at` is the **last attempt**, including 404 and failed checks. Provider
metadata is from that same attempt and is NULL when the attempt is unavailable
or failed; it is not a separately retained last-success snapshot. Finite outcome
and error categories, paired positive provider intervals, identity agreement and
the booking foreign key are SQL constraints. Migration runner status/preparation
and application readiness require the exact 0003 checksum, history ordering and
table contract. `--apply` remains the historical narrow 0001 repair; `--prepare`
is the explicit full-schema upgrade. No application startup DDL occurs.

The unapplied 0003 asset also creates one singleton control row for this app's
currently configured single Bookings provider. It holds the next allowed request
time and the exact PostgreSQL CHECK definitions captured when the checksum-pinned
migration runs. Readiness compares each current definition with that trusted
migration-time baseline, so renamed, weakened or reversed predicates fail closed.
The primary-key catalog flag `connoinherit` is not treated as a CHECK flag. The
interval CHECKs require both provider values to be absent on unavailable/failed
attempts and both present with `end > start` on present/changed attempts.

The new migration needs a separate lead-reviewed release and fallback rehearsal.
The T005-D release script and previous server commands must not be reused.

## Polling and provider failure

Configuration defaults are:

| Setting | Default | Validation |
| --- | --- | --- |
| `BOOKING_OBSERVATION_ENABLED` | `false` | Boolean; explicit rollout switch |
| `BOOKING_OBSERVATION_INTERVAL_SECONDS` | `60` | Integer from 10 through 3600 |
| `BOOKING_OBSERVATION_FRESHNESS_SECONDS` | `180` | Integer from 11 through 7200 and greater than poll interval |

The poller starts only after database schema readiness. Four app workers may each
have a local task, but a stable PostgreSQL **session advisory lock on one dedicated
acquired connection** admits only one tick. That connection remains acquired until
the tick ends. No transaction is held during HTTP. Lock release is attempted in
`finally`; a failed unlock terminates the connection. A lost or changed database
session stops the tick before further writes. Shutdown cancels and awaits the task
and closes its HTTP client before the shared pool closes.

Each tick selects at most 20 locally confirmed bookings with nonempty saved
provider IDs and canonical UTC starts no older than seven days. Missing canonical
time is skipped. Never-checked rows come first, then oldest last-attempt time,
with numeric ID as deterministic tie-break. A row checked within the configured
interval is ineligible unless its local snapshot changed. The whole tick has a
50-second deadline. Each complete OAuth/GET observation has an 8-second budget;
the tick reserves time for recording a fixed timeout attempt and cleaning up the
connection. Timed-out rows receive `checked_at`, allowing later due rows to make
progress. Unfinished rows remain eligible. A conditional upsert takes
a row lock and requires the current local ID, provider ID, canonical start and
status to equal the selected snapshot. A late response cannot overwrite a newer
local state. Failed attempts get `checked_at`, so one repeatedly failing row does
not indefinitely starve others.

Authentication uses the existing configured tenant/client/business and caches a
token for a bounded lifetime. The only provider actions are OAuth POST and one
URL-encoded exact Bookings appointment GET per selected row. Redirects and Graph
mutations are disabled. The parser accepts the official `start`/`end` or observed
`startDateTime`/`endDateTime` UTC pair, requires equal complete pairs if both are
present, rejects offset conflicts/invalid duration, and verifies the returned
identity. `present` means the appointment exists at the local canonical start;
`changed` means the same ID exists at another valid start. These do **not** prove
caller confirmation or verify optional consultant/service/location details.
Missing optional fields remain unknown. A 404 is `unavailable`; authentication,
authorization, throttling, server/network/timeout/malformed/identity errors are
`check_failed` with fixed categories, never error-as-empty. Throttling,
authentication and Graph server errors stop the batch. The lock owner persists
the provider's next allowed request time before releasing the lock; all workers
and replacement processes read it before another request. Valid Retry-After
seconds and HTTP dates, including an hour, are honored. A pause longer than
100 years becomes an indefinite hold rather than an early retry. A Graph 401
invalidates the cached token; only a later allowed tick fetches another. A 403
does not trigger a refresh loop. Internal poller failures emit fixed categories
without exception text, tokens, IDs or customer fields. Remote outage does not
fail call readiness.

## Dashboard interpretation

The authenticated bookings endpoint joins stored observations only. It does not
call Graph on page load. It retains the original local appointment and separately
shows provider state, last attempt and observed time when changed. Snapshot
mismatch or age beyond freshness is stale. A green confirmed badge appears only
for a fresh `present` observation whose local snapshot and provider start still
agree. Other local confirmed rows say **Confirmed locally** with a warning badge
and their review/stale/unchecked Microsoft state. Provider IDs never reach the
browser; text is inserted through text nodes.

`local_recorded_upcoming` retains the local confirmed count, including records
whose provider state is unknown. `microsoft_verified_upcoming` counts only fresh
matching `present` observations with a current local snapshot. Neither is a
total cancellation count. The existing `upcoming` JSON key is retained for
compatibility but the UI labels it **Local recorded upcoming**.

## Remaining dependency

The current reminder scheduler persists phone-based reminder records without a
durable appointment identity. T018-B does not suppress an obsolete reminder after
an external change. Appointment-scoped notification/outbox work must follow
verified identity and reconciliation policy. This slice also does not infer
Outlook-copy deletion, install subscriptions/webhooks/delta feeds, or update local
booking status.
