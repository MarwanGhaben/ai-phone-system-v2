# Email Delivery and Dashboard Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make dashboard email delivery fail honestly and replace fabricated metric zeros with measured values or explicit unavailable states.

**Architecture:** Keep Microsoft Graph delivery asynchronous through the route boundary, with MFA failing closed and informational alerts failing visibly in logs. Query persisted call logs for measurable statistics and represent uninstrumented metrics as nullable values carrying availability metadata.

**Tech Stack:** Python 3.11, FastAPI, aiohttp, asyncpg, pytest, vanilla JavaScript.

---

### Task 1: Honest asynchronous email delivery

**Files:**
- Modify: `services/dashboard/email_service.py`
- Modify: `services/dashboard/auth_service.py`
- Modify: `services/dashboard/dashboard_routes.py`
- Create: `tests/dashboard/test_email_service.py`
- Modify: `tests/dashboard/test_auth_service.py`

- [ ] Add failing tests proving unconfigured delivery returns false without logging the code, Graph rejection returns false, MFA route failure invalidates the code and returns 503, and login-alert failure does not invalidate a verified session.
- [ ] Run `pytest tests/dashboard/test_email_service.py tests/dashboard/test_auth_service.py -v` and verify failures come from optimistic success and synchronous call sites.
- [ ] Delete `send_email`, make both message-specific methods async, return false when configuration is incomplete, and catch only `aiohttp.ClientError` plus `asyncio.TimeoutError` around provider I/O.
- [ ] Add `AuthService.invalidate_mfa_codes(user_id)` and await email delivery from both authentication routes; fail MFA initiation closed while treating login alerts as non-blocking notifications.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Measured and explicitly unavailable metrics

**Files:**
- Modify: `services/dashboard/dashboard_routes.py`
- Modify: `services/dashboard/dashboard_service.py`
- Modify: `templates/dashboard.html`
- Create: `tests/dashboard/test_dashboard_metrics.py`
- Modify: `tests/dashboard/test_dashboard_security.py`

- [ ] Add failing tests for populated and empty `call_logs`, nullable pipeline latency, unavailable legacy placeholders, and “Not available” rendering for a null average duration.
- [ ] Run `pytest tests/dashboard/test_dashboard_metrics.py tests/dashboard/test_dashboard_security.py -v` and verify the hardcoded zeros cause the expected failures.
- [ ] Query `call_logs` for totals, caller counts, sampled average duration, transfer rate, and language distribution. Return zero only for real counts; return null plus sample counts for unsampled measurements.
- [ ] Mark latency, error-rate, and untracked cost metrics unavailable, and update dashboard JavaScript to format nullable measurements as “Not available” or “Not tracked”.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Guard review and release

**Files:**
- Review every production and test file changed by Tasks 1-2.

- [ ] Apply `clean-code-guard` to the production diff and `test-guard` to test changes; remove dead synchronous email code, broad new exception handling, duplicate metric calculations, and mock-only assertions.
- [ ] Run focused tests, `pytest -q`, Python compilation, dashboard JavaScript syntax validation, new-code lint, and `git diff --check`.
- [ ] Commit only intended files and push `claude/fix-voice-barge-in-1skvY` to `origin`.
- [ ] Verify the remote branch SHA exactly matches local `HEAD`.
