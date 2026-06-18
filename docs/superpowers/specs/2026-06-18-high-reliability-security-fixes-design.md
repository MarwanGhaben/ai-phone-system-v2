# High Reliability and Security Fixes Design

## Scope

Fix six validated or partially validated defects in booking time handling, password hashing, request throttling, Twilio audio delivery, call transfer timing, and appointment cancellation authorization. Tenant isolation remains excluded because this deployment serves one customer, but individual callers must not be able to cancel each other's appointments.

## Design

### Booking timezone consistency

Use `zoneinfo.ZoneInfo("America/Toronto")` as the single business timezone. Parsed caller input remains timezone-aware internally: naive input is interpreted as Toronto wall-clock time, while input containing an offset is converted to Toronto time. Microsoft Graph timestamps are converted before their offsets are removed for APIs that require a separate Windows timezone field. Replace the remaining naive `datetime.now()` call in appointment lookup with Toronto-aware time.

### Mandatory bcrypt password hashing

Make bcrypt a direct pinned runtime dependency and remove password creation through SHA-256. Application startup must fail clearly if bcrypt cannot be imported. Keep verification-only compatibility for existing `sha256:` hashes and replace a verified legacy hash with bcrypt during successful login, preventing deployment breakage while eliminating the weak format over time.

### Trusted, distributed rate limiting

Resolve forwarded client addresses only when the direct peer belongs to a configured trusted proxy network. Configure Nginx to replace, rather than append to, the forwarded address it sends upstream. Store rate-limit counters in Redis with atomic updates and expiry so limits are shared by all Uvicorn workers and stale clients cannot create permanent process-memory entries. Dashboard login throttling uses the same bounded mechanism instead of its separate in-memory dictionary.

The application service must not be publicly reachable around Nginx in the production container topology. Direct requests that bypass a trusted proxy ignore forwarded headers.

### Serialized Twilio WebSocket writes

Delete the unused full-audio queue, sender task, and chunked legacy sender. Route media frames, clear events, marks, and other control messages through one lock-protected WebSocket send method. A clear event may wait for one in-flight frame, but no two coroutines may call Starlette's `send_json` concurrently.

### Playback-confirmed transfer

After the transfer announcement's final media frame, send a unique Twilio mark and wait for its inbound acknowledgement. Bound the wait by the audio duration plus a small network allowance so a missing acknowledgement cannot hang the call. Remove the fixed five-second sleep. Barge-in and disconnected-call paths cancel pending playback waits.

### Caller-authorized appointment cancellation

Require caller identity when cancelling an appointment. Fetch the appointment, normalize both phone numbers consistently, and delete only when the appointment belongs to that caller. Invalid or missing phone numbers fail closed. Remove the orchestrator fallback that accepts an appointment ID supplied by the language model; cancellation must select an appointment returned by the caller's prior lookup.

`get_customer_appointments` remains the ownership-filtered lookup API, but it will reject invalid phone numbers and use the same normalization function as cancellation.

## Error handling

- Invalid or ambiguous booking times return the existing request-for-clarification result instead of fabricating a time.
- Redis failures do not silently disable throttling; protected requests return a temporary service error and log the infrastructure failure.
- Missing bcrypt prevents startup instead of selecting a weaker algorithm.
- Missing Twilio mark acknowledgements time out, log a warning, and allow the call flow to continue without waiting indefinitely.
- Appointment ownership mismatches are logged without exposing whether an appointment ID exists.

## Testing

Add focused regression tests proving:

- naive, UTC, and offset-aware booking inputs normalize to Toronto time across spring and autumn DST boundaries;
- Microsoft Graph lookup filters use Toronto-aware wall-clock time;
- password creation cannot select SHA-256 and verified legacy hashes are upgraded;
- forwarded headers are ignored from untrusted peers and accepted from the configured proxy;
- rate-limit state expires and is shared through the Redis-backed implementation;
- simultaneous Twilio media and control sends never overlap;
- transfer waits for a matching mark and has a bounded timeout;
- cancellation rejects an appointment owned by another caller and rejects language-model IDs not produced by lookup.

Run the focused tests first, then the full available test suite, Python compilation, and repository lint checks. Review production changes with `clean-code-guard` and test changes with `test-guard` before committing.

## Operational impact

Valid booking and call flows remain unchanged. Deployments must provide Redis and route public traffic through the trusted Nginx proxy, both of which are already represented in the repository's container setup. Existing bcrypt users are unaffected. Legacy SHA-256 users can still sign in once and are upgraded immediately. Transfers wait for actual playback completion rather than an arbitrary delay, and cancellation becomes bound to the caller's phone number.
