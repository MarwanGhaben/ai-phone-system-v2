# Critical Runtime Fixes Design

## Scope

Fix the five validated defects and two partially valid defects identified in the dashboard, booking, STT, Whisper, and OpenAI services. Tenant isolation is explicitly excluded because this deployment serves one customer and is not a multi-tenant SaaS product.

## Design

### Dashboard XSS

Stop inserting API values into HTML strings. Render attacker-reachable caller, booking, call, and SMS fields with DOM elements and `textContent`. Replace data-bearing inline `onclick` attributes with event listeners so names and phone numbers never enter executable HTML or JavaScript contexts. Static empty-state markup may continue using fixed HTML.

### Dashboard administrator initialization

Remove the seeded administrator from `services/dashboard/db_init.sql`. Direct trusted deployment tooling to `AuthService.create_user`, which writes the dashboard's `admin_users` table. Existing database users are unaffected because this changes initialization only.

### Booking date parsing

When the requested date/time is absent or cannot be parsed, return a structured tool result instructing the model to ask the caller to repeat the date and time. Do not create a pending booking or substitute a fabricated appointment. Preserve existing handling for valid dates, weekends, unavailable slots, and explicit confirmation.

### ElevenLabs STT reset synchronization

Add a per-instance asynchronous connection lock. Connection replacement and audio sends use the same lock, ensuring an audio send either completes on the old socket before reset or waits and uses the new socket afterward. The lock is local to one call's STT instance, so one caller cannot block another caller.

### Whisper audio processing

Build the mu-law decode table once and use NumPy indexing for energy calculation and PCM conversion. Preserve the current threshold and byte-for-byte little-endian PCM contract. Add NumPy as an explicit pinned dependency because the current optional import is not backed by `requirements.txt`.

### OpenAI request timeout

Configure the `AsyncOpenAI` client with a bounded timeout appropriate for a voice call. Apply it centrally so normal chat, streaming setup, and tool calls share the same policy. Timeout errors continue through the existing error path, allowing the orchestrator to use its established fallback behavior instead of leaving a call waiting for the SDK's long default timeout.

## Testing

Add focused pytest regression tests that prove:

- dashboard rendering no longer interpolates attacker-controlled values into HTML;
- dashboard SQL does not seed an administrator;
- invalid or absent booking date/time requests a repeat and creates no pending booking;
- audio submitted during an STT reset is sent after the new socket becomes ready;
- vectorized mu-law conversion matches a known scalar reference and energy decisions remain unchanged;
- the OpenAI client receives the bounded timeout configuration.

Run the focused suite, Python compilation, and any existing repository checks. Review production changes with `clean-code-guard` and tests with `test-guard` before committing.

## Operational impact

The fixes preserve valid booking and voice flows. Intentional behavior changes are limited to rejecting ambiguous appointment times, failing stalled OpenAI requests promptly, removing automatic dashboard-admin creation for fresh databases, and preventing unsafe dashboard rendering. No existing administrator row is deleted, and no database migration is required.
