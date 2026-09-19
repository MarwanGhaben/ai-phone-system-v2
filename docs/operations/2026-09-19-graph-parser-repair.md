# Graph availability parser and caller reply repair

The deployed ac02a49 application rejects actual Microsoft response shapes before
it can offer appointments. This repair accepts the observed shapes and gives
different English/Arabic replies for a verified empty schedule and a failed read.

## Scope

- Accept empty businessHours for bookWhenStaffAreFree.
- Accept a hours list on notBookable while preserving the entire inclusive closed
  date range. Never derive openings from that list; reject nonlist hours.
- Recognize the exact (UTC) Coordinated Universal Time display alias.
- Keep populated general free-mode hours, offset conflicts, unknown zones,
  malformed dates, incomplete responses and other unsafe evidence rejected.
- Change only the two parsers and the no-slot caller reply. Preserve all
  scheduling/mutation safeguards, schema 0005, and the existing SDK repair.

The owner ran these exact parsers in memory: service verified, no parser errors.
For September 19–22 Toronto, both request representations returned three
outOfOffice and five busy intervals, none available. The owner inspected the
calendar Busy strip and confirmed Hussam is blocked through October 13.
Another accountant failing on the still-deployed old parsers is not a corrected
live test. No successful live Abdul/Rami search is claimed yet.

The agent searches only the approved booking window. Its new no-slot reply says
there are no bookable appointments in the checked period and offers another
accountant. It does not invent a leave reason or return date. Failed checks retain
their separate error reply. The experimental Eastern request exposed another
unsupported display alias; production stays UTC, with no guessed offset.

## Validation

- Parser regressions: 15 failed/13 passed before; 28 passed after.
- Caller reply tests with valid tool arguments: 2 failed/2 passed before; 4 passed
  after, including no invented return date.
- Focused parser/policy/caller checks: 82 passed.
- Calendar/conversation/STT/TTS/telephony/LLM: 579 passed, existing audioop and
  optional ffmpeg warnings only.
- Release: 10 passed including actual Linux image build, synthetic HTTP/policy
  checks and installed source hashes. Temporary containers and image tags removed.
- Earlier scheduling selection: 461 passed, 3 database skips, one old frozen
  parser-hash assertion failed because contracts.py intentionally changed. The
  historical manifest was preserved. Full-suite acceptance is not claimed.
- git diff --check passed.

The exact production image is remote. The local Docker rehearsal layered accepted
predecessor Git runtime onto the available dependency parent, then exercised the
actual candidate builder and offline probe. It does not claim the server's
private mounts or live readiness were reproduced locally. No provider contact.

## Rollout

Use scripts/deploy-graph-parser-repair.py at the published reviewed commit,
passing the same full SHA. Publication must precede the owner command.

Only predecessor ac02a494dd33614b606b325bd9753b90a62e5a67 and image
sha256:cfd3e52a29e5518909a37073903d8a1bbaff025e031ce02694abf0e90add92ac are accepted.
The procedure derives from that exact local image, verifies pinned repair sources,
runs the offline transport/policy probe, checks effective source/settings and
preserved mounts, and changes only the app image. Readiness and HTTPS are required.
Cutover failure restores the previous app image and override; no migration occurs.

Success: GRAPH_PARSER_REPAIR_DEPLOYED_READY_HTTPS_OK plus commit/image.
Preserve protected release directories. Then test an actually available Abdul
or Rami slot in the approved window, first stopping before confirmation. Hussam
should receive the scoped no-slot response.
