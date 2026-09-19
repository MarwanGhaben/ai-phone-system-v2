# T020-C contention correction

The owner-run c60e774 release aborted before cutover. Its restored-copy migration
and preservation checks completed, but the separate 20-request create journey
returned zero VERIFIED results. The original live app remained healthy on image
14589fc and schema 0004.

The targeted server replay established the failure sequence: one admission and
dispatch, one simulated Graph create receipt, then failed receipt persistence.
It reported two PostgreSQL deadlocks (40P01), one admission store error, 18
interval conflicts, and one pending create. This is a real database-contention
failure, not evidence of a bad Microsoft credential or a harmless test timeout.
No actual Microsoft endpoint participated in that replay.

## Correction

Only `OperationStore.admit` changes in runtime code. Its transaction acquires a
stable, transaction-scoped advisory lock for tenant/business/staff, then performs
the existing exact-intent/idempotency lookup and a nonlocking committed-overlap
check before inserting a new claim. This keeps competing application admissions
out of speculative GiST insertions while the accepted operation persists its
receipt or finalizes. Service IDs deliberately do not partition the lock, matching
the cross-service staff exclusion. The lock covers admission only, never provider
I/O. A hash collision can conservatively serialize unrelated admissions; it
cannot permit an overlap. Existing database command timeouts bound lock waits.

The database exclusion constraint remains authoritative, including for direct SQL.
No provider retry, receipt retry, timeout relaxation, schema change or migration
checksum change is introduced. Exact-intent replay, fencing, provider verification,
atomic local persistence and held notifications remain unchanged. Higher-isolation
or uncooperative external writers can still cause database errors; those retain
the existing fail-closed outcome rather than permitting another provider POST.

## Regression evidence

A real PostgreSQL test holds a successful receipt UPDATE open in one transaction
and admits a competing intent through a second connection. Before the correction,
the competing insert waits on the receipt transaction and fails the two-second
completion assertion. After the correction it returns INTERVAL_CONFLICT while
the receipt transaction is still open. This deterministically reproduces the
relevant blocking boundary; the exact two-way deadlock was observed on the server,
not claimed as reproduced by this smaller local regression.

A second test keeps one admission transaction open, verifies another consultant
can still admit, cancels a same-consultant waiter, checks transaction cleanup,
then verifies admission after rollback. Existing real database cases still check
20 identical requests, 20 competing intents, direct-SQL exclusion, buffer and
adjacency rules, cross-service scope, recovery and mutation fencing.

Validation:

- Failing-first receipt-writer regression: one failure at the intended assertion.
- Operation/create selection: 46 passed (collected before the second new test).
- Separate staff-scope/cancellation regression: one passed.
- Release offline checks: 31 passed, four expected platform/Docker skips.
- Remaining repository selection: 1064 passed, 56 skipped, 52 database cases
  deselected for their separate execution; ten unchanged historical manifest
  assertions fail. No other failures or setup errors in that selection.
- An earlier all-repository invocation with Docker removed from PATH produced
  23 database setup errors (WinError 2). It is not claimed as a passing run;
  the real database selection above ran with Docker available.
- Packaged Linux database journeys: 51 executed, zero skipped, all passed,
  using the actual release staging/build/database-journey methods. The original
  competing-slot test and both new regressions are included.
- Python 3.11 syntax, Ruff F checks and git diff --check passed. Default Ruff
  additionally reports ten preexisting exact-type E721 checks outside the change;
  those strict validation checks were not altered.

## Release integrity

The new `T020-C-contention-hashes.json` preserves the previous two manifests and
changes exactly two of 108 entries: operation_store.py and test_booking_operations.py.
All other 106 assets, all SQL, and every other operation-store method are unchanged.
The application source scope remains the original 29 paths relative to the actual
server predecessor. The release script changes only the manifest path and checksum.

Manifest SHA-256:
`c3f89ece10d06f9edf14ea191fbad9f2b12c1eaa7cc8533b0ec528607a46de4a`.

No production action or provider call was made during this correction. The exact
production parent remains unavailable locally; local packaging uses the previously
documented substitute with frozen protected runtime/model assets. The server
procedure still requires the exact original parent and all rehearsal gates.
Publish a newly reviewed commit before another owner rollout; never rerun c60e774
or mark the verified phone route active based on local tests alone.
