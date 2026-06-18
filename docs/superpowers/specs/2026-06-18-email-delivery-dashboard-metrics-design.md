# Email Delivery and Dashboard Metrics Design

## Scope

Remove false-positive email delivery results from dashboard authentication and replace fabricated dashboard metric zeros with database-backed values or explicit unavailable states.

## Email delivery

Email delivery remains asynchronous from the route through Microsoft Graph. Remove the synchronous wrapper that schedules background work and immediately reports success. `send_mfa_code` and `send_login_alert` will await the Graph response and return its real delivery result.

An unconfigured email service returns failure and never logs an MFA code. If MFA delivery fails, the login endpoint invalidates the generated code and returns HTTP 503 without setting the MFA cookie. A login-alert failure is awaited and logged but does not revoke an already verified login session because the alert is informational rather than an authentication factor.

Network failures are caught only for the timeout and `aiohttp` client exceptions that the service can classify as delivery failures. Unexpected programming errors propagate.

## Dashboard metrics

The visible call statistics endpoint will read `call_logs` for all-time total calls, unique callers, returning callers, average duration, transfer rate, and language distribution. Average duration uses only rows containing a duration. Transfer rate uses recorded calls and returns a percentage.

Counts may legitimately be zero. Metrics that have no samples return `null` with an accompanying sample count of zero, allowing the dashboard to render “Not available” instead of a fabricated `0s` or `0%`.

Pipeline latency is not currently instrumented. Its endpoint will return `available: false`, `null` timing fields, and an explanatory message instead of zero measurements. Legacy dashboard-service responses that still contain unimplemented cost, error-rate, or analytics metrics will likewise identify themselves as unavailable and use `null` for unmeasured values.

## User interface

The dashboard renders actual zero counts normally. Nullable duration, percentage, latency, cost, and error-rate measurements render as “Not available” or “Not tracked”; they are never formatted as numeric zero.

## Testing

Regression tests will prove:

- unconfigured email returns failure without exposing the MFA code;
- Graph rejection and network failure return failure;
- a running event loop cannot produce optimistic email success;
- failed MFA delivery returns 503, invalidates the code, and sets no MFA cookie;
- verified login remains successful when an informational login alert fails;
- call statistics calculate duration and transfer rate from `call_logs`;
- empty call logs return zero counts but unavailable sampled metrics;
- pipeline and legacy uninstrumented metrics are explicitly unavailable;
- the dashboard renders nullable metrics honestly.

Run focused tests, the full pytest suite, Python compilation, JavaScript syntax validation, and `clean-code-guard` before committing and pushing the active branch.

## Operational impact

Dashboard password validation can no longer appear successful when the MFA email was not delivered. Production dashboard login therefore requires complete Microsoft Graph email configuration and a successful Graph send response. Existing call records immediately populate the wired metrics; no schema migration is required.
