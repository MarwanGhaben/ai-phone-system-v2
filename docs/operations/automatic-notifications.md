# Automatic appointment notifications (T049-A)

This is an opt-in implementation. `AUTOMATIC_NOTIFICATIONS_ENABLED`
defaults to `false`; `BOOKING_OBSERVATION_ENABLED` must also be `true`
when it is enabled. Migration 0004 adds durable reconciliation and SMS outbox
tables without changing existing booking rows or sending messages. It must be
reviewed and applied separately before enabling the flag. The current server
release and its settings-file mount are not changed by this assignment.

The policy manages one local numeric booking ID, its Microsoft tenant/business
scope, exact appointment ID and aware UTC start time. An existing appointment
enrolls only after a new matching exact-ID read in the current scope while the
feature is enabled. An already missing old booking has no prior enrollment and
cannot generate a retroactive removal notice. Existing enrolled appointments
may get a future-due reminder; their initial confirmation is never replayed.
New bookings create the local record plus confirmation and reminder jobs in
one transaction. In enabled mode the legacy JSON scheduler and fire-and-forget
SMS path are inactive. With the flag off, the old JSON behavior remains and
does not acquire appointment-scoped protection.

An enrolled future booking enters a hold after its first exact-ID 404. A
second independent 404 must follow 60-600 seconds later. The poller then
checks the exact configured business identity and the complete unfiltered
appointments collection, capped at 20 pages, 2000 entries and 30 seconds.
A matching ID on any page vetoes removal. Bad identity, pagination, auth,
network, timeout, throttling, malformed response or exhausted budget leaves
the booking unresolved. A failed or contradictory complete inventory clears
the two-404 sequence with a fixed reason; the next 404 starts a new sequence.
A found ID does not prove that its time matches the local booking, so it does
not release held reminders. Evidence from a read that began before a newer
read cannot overwrite the newer result. SMS-worker reads may break a missing
sequence but cannot count as the second independent 404. The shared provider
backoff applies to these reads.
Only a fully absent inventory permits one atomic local transition to
`removed_externally` with reason `inferred_provider_removal`, suppression
of that booking's pending confirmation/reminder jobs, and one removal notice.
This is an application policy, not Microsoft proof of cancellation reason or
which person removed the appointment. A later reappearance is a discrepancy;
it does not silently restore the local booking.

Before dispatch, each job checks the current booking scope/version under its
per-booking lock. Confirmation and reminder require a fresh matching exact-ID
read. Removal requires a fresh exact-ID 404 after the complete-inventory
decision. Provider failure holds the job; an overdue reminder is suppressed.
Each job retains its original `due_at` and a separate durable
`next_attempt_at`. Held provider checks use bounded exponential waits (one
minute to one hour) and the shared Microsoft pause; eligible jobs are ordered
by next attempt, so repeatedly failing notices cannot occupy the first ten
slots indefinitely. `retry_count` counts held checks. `attempts` counts
committed possible carrier POST intents, not Graph checks or proven delivery.
After a fresh matching positive read, still-valid held confirmations and reminders
become eligible again without waiting out an old provider-failure retry delay.
The original due time remains the expiry reference: confirmation validity is
15 minutes, reminder validity is five minutes, and neither may dispatch after
the appointment starts. Expired, removed or mismatched snapshots are not revived.
The worker commits a dispatch-intent token before Telnyx POST and keeps a
session-level per-booking lock through submission and result storage, without
holding a SQL transaction across the network. A crashed or uncertain intent
becomes `unknown` and is never automatically reposted. A valid 2xx response
with a message ID means `accepted`, not delivered. Invalid recipient,
missing configuration and definite provider rejection are visible failures.
There is no carrier-level exactly-once guarantee or delivery-receipt handling.
A reminder already submitted before a later cancellation cannot be recalled.

The observer holds its global tick lock, then takes the provider-admission
session lock only for a provider call and releases it before taking a booking
transaction lock. The SMS worker holds its booking session lock, then takes
the same admission lock for the provider call. This order has no admission-to-
booking wait cycle. No SQL transaction spans Microsoft HTTP or Telnyx POST.
The admission lock checks the persisted control row before each exact read,
OAuth refresh, business read and inventory page. A Retry-After response updates
that row before another request is admitted. An already in-flight request
cannot be recalled; a later request waits. An auth 401 clears the cached token,
and token refresh occurs only after the shared pause ends.

The dashboard keeps the original appointment time and distinguishes Microsoft
observation time from notification attempt time. It labels automatic removal,
reminders on hold and queued/accepted/failed/unknown notices using fixed
categories. Managed bookings and their audit rows cannot be hard-deleted by
the existing dashboard routes. Review `held`, `failed`, `unknown`, and
reappeared cases individually; there is no automatic resend control for
uncertain carrier submissions. Logs must not include recipient numbers,
message bodies, provider IDs or raw provider errors.

Caller cancellation uses the existing provider mutation path. After reported
success, the enabled notification path maps exactly one local booking and uses
the same booking/version removal event key. An ambiguous or missing mapping is
reported as unresolved. This work does not certify the provider mutation,
direct Outlook-copy deletion propagation, delivery receipts, Arabic speech
quality or a production release. Any future reschedule/rebook must advance
the snapshot version or create a new local booking lifecycle. A later fallback
image must remain compatible with the applied 0004 schema. Disabling the feature
flag reactivates legacy messaging, so it is not an acceptable recovery procedure
after notification activation. T049-B adds `AUTOMATIC_NOTIFICATION_WORKERS_PAUSED`:
with feature mode enabled, this pauses both observation and notification workers
on restart, preserves the outbox, keeps legacy messaging off, and retains the
complete 0004 readiness gate. New bookings still enqueue durable jobs. It cannot
recall submitted messages. See `T049-B-notification-rollout.md` for the manual
deployment and degraded recovery procedure; implementation is not server activation.

Readiness accepts the two equivalent CHECK serializations reproduced during real
PostgreSQL backup/restore: literal-array cast distribution and the leading
missing-count AND flattening. Stored baselines and SQL hashes remain unchanged;
changed rule values, bounds and Boolean expressions continue to fail readiness.
