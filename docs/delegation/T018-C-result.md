# T018-C implementation result (local, uncommitted)

Created only the requested release script, offline tests and this pair of
documents. The accepted T018-B runtime, SQL and tests were not edited.

- `scripts/deploy-provider-observations.py` uses a 40-hex pinned commit,
  accepted server/image/schema/HTTPS preflight, whitelisted no-network candidate
  and compatible fallback builds, complete effective Compose/settings checks,
  protected backups, isolated restored-copy 0002-to-0003 rehearsal, app-only
  cutover and schema-aware recovery. It emits fixed public markers and keeps
  raw command output/private settings in a root-protected release directory.
- `tests/integration/test_observation_release.py` exercises copy scope,
  preflight refusal, effective environment and mount preservation, aware and
  naive rows, repeat-safe migration, partial cleanup, pre-stop failure,
  predecessor/fallback/ambiguous recovery, original error retention and marker
  privacy through mocked boundaries.
- `docs/operations/T018-C-observation-rollout.md` records the eventual pinned
  owner command, expected markers, stop/review conditions and one dashboard check.

Failing-first local `python -B -m unittest discover -s tests/integration -p
test_observation_release.py -v`: 11 tests, 2 failures and 2 errors. The real
failure was order-sensitive mount comparison during replacement. The other
three were test fixture mistakes (a repeated `xb` path, escaped newline
assertion and a sliced build argument). The mount check now compares every
mount field without depending on Docker inspect order. After adding further
cases, final command: **16 tests passed**. Mocked partial-create/timeout cases
observed both container and network cleanup calls. No real container or
network was created by these tests.

Static Python 3.11-compatible syntax and the four embedded probe strings
compile. The local CLI used for this assignment has Python 3.14, but no
`pytest`, Docker or Git executable; it did not run a real PostgreSQL rehearsal,
the existing pytest suite, image build or `git diff --check`. The lead's local
Python 3.11/Docker environment must run those before acceptance and publication.
No server or provider was contacted, and there was no commit, push or deploy.

Preservation SHA-256 values frozen before the release edit and verified unchanged
afterward (15/15). Branch remains `codex/phase-3-safe-booking` at required HEAD
`50c45cefa4c88fdb2e0827381d3825507dbdd82b`:

| Accepted path | SHA-256 |
| --- | --- |
| `api/main.py` | `fd1386a50242eabcb4e82a536f30584473f22f1891f29c9791e9be9f5f94bbba` |
| `config/settings.py` | `41f302c1cd594d112671f3a240efa812404f8f3a47523fd01bc7173b6dd003d4` |
| `docker-compose.yml` | `3ec9d352cccf4e4501733eeb71786254b05df46413bd0e9c67e20d6379ed0e62` |
| `migrations/0003_booking_provider_observations.sql` | `b3abaa09c340f89c32778b0b17b5c2a584845983b13eb95dbb9c366f3d6c5830` |
| `migrations/runner.py` | `f93987d40e58e1588d546e463f871348e13e25ad08cebbd5b7666cfa02d6ff3b` |
| `migrations/schema_contract.py` | `4913ca2ebfb33d12ab7efd4cb86015ffe91d236aebc807989be3f850a049c39d` |
| `services/calendar/booking_readback.py` | `e489e9464bef06846bf189a5c36efa1e6e02f77b682a60b0ef1d342903e3662e` |
| `services/dashboard/dashboard_routes.py` | `d8598d085f15771124f96d65df45d2c873806cfc4c1f0e50a8c44b68780f050e` |
| `services/scheduling/provider_observations.py` | `3e6c1bd03fe238d3e4e72773bacda6718cc1f8d7d73996a16cefda999a7b03f7` |
| `templates/dashboard.html` | `53167d702f3cc18a68d47b09b468463e77cc496973e55bbe77369b683cbf008a` |
| `tests/integration/test_booking_persistence.py` | `013090f7fcc78899bd1395505e11d0b084a9a0c5addb8d78b24f5e9eebe1c430` |
| `tests/integration/test_database_startup.py` | `83ca3afc3ad86e5243abb45806bc546731e43de86ee07ec0baa2054651e1f9ae` |
| `tests/integration/test_image_contents.py` | `0ae019c7676b2ffcc1425b1ae572e5427d650e85fce1dd60c333d3a7d315493e` |
| `tests/integration/test_migrations.py` | `b96c9927f17a1dd55a14c85e3d4b2c642d351f9b8126166d8491f4acf4606805` |
| `tests/integration/test_provider_observations.py` | `17b85733831aafc1102a1095a119d7436e5cf2fd84ff152e022996e1e4f58a34` |

Remaining acceptance: independent lead review, real local image/restore/migration
and recovery exercise, normal pytest suites, and final source-scope/diff review.
The final published release commit is intentionally unknown; the owner command
remains a placeholder. No new business decision or provider access is needed.
