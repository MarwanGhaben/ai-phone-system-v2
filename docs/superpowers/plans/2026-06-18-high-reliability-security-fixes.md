# High Reliability and Security Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the six validated reliability and security defects without changing successful booking, login, or voice-call behavior.

**Architecture:** Centralize Toronto timezone conversion and trusted client-IP resolution, replace process-local throttles with Redis TTL counters, serialize every Twilio WebSocket write, and enforce appointment ownership inside the calendar service. Each production change is introduced by a focused failing regression test.

**Tech Stack:** Python 3.11, FastAPI/Starlette, redis-py 5, bcrypt 4, Microsoft Graph, Twilio Media Streams, pytest, pytest-asyncio.

---

### Task 1: Timezone-safe booking datetimes

**Files:**
- Create: `services/calendar/business_time.py`
- Modify: `services/conversation/orchestrator.py`
- Modify: `services/calendar/ms_bookings_service.py`
- Modify: `tests/conversation/test_booking_time_validation.py`
- Create: `tests/calendar/test_ms_bookings_service.py`

- [ ] Write tests asserting naive caller input receives Toronto tzinfo, offset input converts to Toronto, UTC Graph timestamps convert before slot comparison, and appointment lookup emits a Toronto wall-clock filter.
- [ ] Run `pytest tests/conversation/test_booking_time_validation.py tests/calendar/test_ms_bookings_service.py -v` and confirm failures expose naive/aware comparison and offset stripping.
- [ ] Add `BUSINESS_TIMEZONE`, `as_business_time(datetime)`, `parse_graph_datetime(str)`, and `business_now()` in `business_time.py`; use them from both services and keep aware datetimes internally.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Mandatory bcrypt with legacy upgrade

**Files:**
- Modify: `requirements.txt`
- Modify: `services/dashboard/auth_service.py`
- Create: `tests/dashboard/test_auth_service.py`

- [ ] Write tests asserting new hashes always begin with bcrypt, malformed hashes fail verification, and a successful login with a valid legacy `sha256:` hash updates that row to bcrypt.
- [ ] Run `pytest tests/dashboard/test_auth_service.py -v` and confirm the legacy-upgrade assertion fails.
- [ ] Import bcrypt directly, pin the installed compatible release in `requirements.txt`, remove SHA-256 hash creation, retain constant-time legacy verification, and update the stored hash after successful legacy authentication.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Trusted Redis-backed throttling

**Files:**
- Create: `services/security/client_ip.py`
- Create: `services/security/rate_limiter.py`
- Modify: `services/security/middleware.py`
- Modify: `services/dashboard/auth_service.py`
- Modify: `services/dashboard/dashboard_routes.py`
- Modify: `config/settings.py`
- Modify: `nginx/nginx.conf`
- Modify: `docker-compose.yml`
- Create: `tests/security/test_rate_limiter.py`

- [ ] Write tests asserting untrusted peers cannot set their address with forwarded headers, trusted proxies can, Redis keys receive expiry, minute and burst limits return bounded retry values, and dashboard login failures share Redis state.
- [ ] Run `pytest tests/security/test_rate_limiter.py -v` and confirm the tests fail against the in-memory implementation.
- [ ] Implement `ClientIPResolver`, an atomic Lua-backed `RedisRateLimiter`, and a TTL-backed `LoginAttemptLimiter`; await them from HTTP middleware and authentication.
- [ ] Replace dashboard IP parsing with the shared resolver, set Nginx `X-Forwarded-For` to `$remote_addr`, and replace public app port mappings with Docker `expose`.
- [ ] Run focused security and dashboard tests and confirm they pass.

### Task 4: Serialized Twilio output and playback marks

**Files:**
- Modify: `services/telephony/twilio_service.py`
- Modify: `services/conversation/orchestrator.py`
- Create: `tests/telephony/test_twilio_stream.py`
- Create: `tests/conversation/test_transfer_playback.py`

- [ ] Write tests with a WebSocket that detects overlapping sends, a matching inbound mark that releases playback waiting, a bounded missing-mark timeout, and a transfer path that performs no fixed sleep.
- [ ] Run the focused tests and confirm concurrent sends overlap and transfer still calls `sleep(5)`.
- [ ] Delete the unused full-audio queue path, add one `_send_message` method guarded by an `asyncio.Lock`, track mark futures by unique name, and resolve/cancel them from inbound events and cleanup.
- [ ] Make `_speak_to_caller` await the playback mark with an audio-duration-based timeout and remove the transfer sleep.
- [ ] Run the focused Twilio and transfer tests and confirm they pass.

### Task 5: Caller-owned appointment cancellation

**Files:**
- Modify: `services/calendar/ms_bookings_service.py`
- Modify: `services/conversation/orchestrator.py`
- Modify: `tests/calendar/test_ms_bookings_service.py`
- Create: `tests/conversation/test_booking_cancellation.py`

- [ ] Write tests asserting invalid phones return no appointments, another caller's appointment cannot be deleted, an owned appointment can be deleted, and an LLM-supplied ID is rejected when lookup context is absent.
- [ ] Run the focused tests and confirm arbitrary IDs reach the current DELETE path.
- [ ] Add one canonical NANP phone normalizer, require `cancel_customer_appointment(appointment_id, customer_phone)`, fetch before delete, compare ownership, and remove the orchestrator argument fallback.
- [ ] Run the focused tests and confirm they pass.

### Task 6: Guard review and release verification

**Files:**
- Review all files changed by Tasks 1-5.

- [ ] Run `clean-code-guard` against the production diff and remove dead code, broad error swallowing introduced by the change, duplicate rules, and unverified APIs.
- [ ] Run `test-guard` against new and modified tests; remove mock-only assertions and test duplication.
- [ ] Run focused tests, then `pytest -q`, `python -m compileall services config tests`, and the repository's available lint checks.
- [ ] Run `git diff --check`, review `git status --short`, commit only intended files, and push `claude/fix-voice-barge-in-1skvY` to `origin`.
