# Critical Runtime Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the validated dashboard, booking, STT, audio-processing, and LLM timeout defects without adding multi-tenant behavior.

**Architecture:** Apply six focused fixes at existing boundaries: safe browser DOM construction, secure SQL initialization, explicit booking parse failure, per-call STT synchronization, vectorized mu-law decoding, and a central OpenAI client timeout. Regression tests exercise observable behavior or security invariants before each implementation change.

**Tech Stack:** Python 3.12, pytest, pytest-asyncio, FastAPI dashboard template JavaScript, asyncio, NumPy, OpenAI Python SDK 1.10.

---

### Task 1: Dashboard injection and administrator seed

**Files:**
- Modify: `templates/dashboard.html`
- Modify: `services/dashboard/db_init.sql`
- Create: `tests/dashboard/test_dashboard_security.py`

- [ ] **Step 1: Write failing security regression tests**

Read both files and assert that attacker-controlled caller, booking, and SMS values are not directly interpolated into HTML strings, caller actions do not embed values in `onclick`, and dashboard SQL contains no `INSERT INTO admin_users` seed.

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest tests/dashboard/test_dashboard_security.py -v`
Expected: failures identify the raw `${caller.name}`, `${booking.client}`, `${sms.message}`, inline caller action attributes, and seeded administrator.

- [ ] **Step 3: Implement safe rendering and secure initialization**

Add focused DOM helpers that assign dynamic values through `textContent`. Render dynamic rows and action buttons with `document.createElement`, and register button listeners with `addEventListener`. Retain fixed empty-state HTML only. Delete the administrator `INSERT` and direct trusted provisioning tooling to `AuthService.create_user`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/dashboard/test_dashboard_security.py -v`
Expected: all dashboard security tests pass.

### Task 2: Invalid booking time handling

**Files:**
- Modify: `services/conversation/orchestrator.py`
- Create: `tests/conversation/test_booking_time_validation.py`

- [ ] **Step 1: Write failing behavior tests**

Create a real `ConversationContext` with an existing pending booking. Call `_check_booking` with missing and malformed `date_time` values. Assert the result starts with `INVALID_DATE_TIME` and `pending_booking` is cleared without contacting the calendar service.

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest tests/conversation/test_booking_time_validation.py -v`
Expected: current code proceeds to calendar availability or substitutes tomorrow at 10 AM.

- [ ] **Step 3: Implement minimal parse rejection**

Extract the current ISO/dateutil parsing into a small `_parse_booking_datetime` query. In `_check_booking`, parse before calendar I/O; when parsing returns `None`, clear stale pending state and return `INVALID_DATE_TIME: Ask the caller to repeat the requested date and time.` Preserve valid-date behavior.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/conversation/test_booking_time_validation.py -v`
Expected: missing and malformed values request repetition; a valid ISO value still parses.

### Task 3: ElevenLabs reset synchronization

**Files:**
- Modify: `services/stt/elevenlabs_stt_service.py`
- Create: `tests/stt/test_elevenlabs_reset.py`

- [ ] **Step 1: Write the failing concurrency regression test**

Use fake old/new WebSockets at the external WebSocket boundary. Pause reconnection, submit audio concurrently, and assert the audio task waits and the new socket receives the chunk after reset completes.

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest tests/stt/test_elevenlabs_reset.py -v`
Expected: `stream_audio` returns while no socket exists and the new socket receives nothing.

- [ ] **Step 3: Serialize socket lifecycle and sends**

Initialize `self._connection_lock = asyncio.Lock()`. Hold it around the socket replacement in `reset_for_listening` and around connection validation plus send in `stream_audio`. Send through a captured local socket reference.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/stt/test_elevenlabs_reset.py -v`
Expected: audio waits during reset and is sent exactly once on the replacement socket.

### Task 4: Vectorized Whisper mu-law processing

**Files:**
- Modify: `requirements.txt`
- Modify: `services/stt/whisper_service.py`
- Create: `tests/stt/test_whisper_audio.py`

- [ ] **Step 1: Write failing equivalence tests**

Implement scalar reference decoders inside the test and compare all 256 byte values against `_mulaw_to_pcm`. Parameterize silence/speech samples to verify `_has_speech_energy` preserves threshold decisions, including empty input.

- [ ] **Step 2: Run tests and verify the performance requirement is RED**

Run: `pytest tests/stt/test_whisper_audio.py -v`
Expected: the test requiring direct NumPy-backed conversion fails against the per-byte implementation.

- [ ] **Step 3: Implement vectorized conversion**

Pin `numpy==1.26.4`. Import NumPy directly, precompute the existing energy and PCM lookup tables once, index them with `np.frombuffer`, calculate energy with `np.mean`, and emit little-endian signed 16-bit PCM with `.astype('<i2').tobytes()`.

- [ ] **Step 4: Verify GREEN and benchmark**

Run: `pytest tests/stt/test_whisper_audio.py -v`
Expected: scalar-equivalence and threshold tests pass. Run the bounded local benchmark and confirm a 30-second buffer conversion no longer blocks for the previous approximately 129 ms.

### Task 5: OpenAI voice-call timeout

**Files:**
- Modify: `services/llm/openai_service.py`
- Create: `tests/llm/test_openai_timeout.py`

- [ ] **Step 1: Write the failing client-configuration test**

Replace `AsyncOpenAI` only at the external SDK boundary, call `_get_client`, and assert it receives `timeout=30.0` while preserving the configured API key and optional base URL.

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest tests/llm/test_openai_timeout.py -v`
Expected: the timeout keyword is absent.

- [ ] **Step 3: Configure the timeout centrally**

Define `OPENAI_REQUEST_TIMEOUT_SECONDS = 30.0` and pass it when constructing `AsyncOpenAI`. Do not add a second timeout wrapper or change retry behavior.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/llm/test_openai_timeout.py -v`
Expected: timeout configuration tests pass.

### Task 6: Integrated review, verification, and publication

**Files:**
- Review all modified production and test files.

- [ ] **Step 1: Run focused and repository checks**

Run: `pytest -v`
Run: `python -m compileall api services tests`
Run: `ruff check services tests`
Expected: all commands pass without new warnings or errors.

- [ ] **Step 2: Run quality guards**

Walk `clean-code-guard` against the production diff and `test-guard` against every new test. Remove broad swallowed exceptions, dead code, generic names, duplicated setup, and implementation-detail assertions that do not enforce a security invariant.

- [ ] **Step 3: Inspect publication scope**

Run: `git status --short --branch`
Run: `git diff --check`
Run: `git diff --stat origin/claude/fix-voice-barge-in-1skvY...HEAD`
Expected: only the approved spec, plan, production fixes, dependency pin, and regression tests are included; `.repo-review/` and `.vs/` remain untracked and unstaged.

- [ ] **Step 4: Commit and push**

Stage explicit approved paths, commit with `Fix critical dashboard and voice runtime issues`, and push `claude/fix-voice-barge-in-1skvY` to `origin`.

- [ ] **Step 5: Confirm remote state**

Run: `git ls-remote --heads origin claude/fix-voice-barge-in-1skvY`
Expected: the remote branch hash equals local `HEAD`.
