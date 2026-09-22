# Restore the pre-phase-5 caller flow

The owner rejected the repeated phase-5 repairs and requested a review of the
entire flow or a return to the earlier working version. This release restores the
caller-flow implementation from `792952b2965e85513bf0971a41bfaff7bd3377c4`.
It is a rollback of the conversation and booking integration, not another repair
of individual phase-5 symptoms.

## What the flow review established

Phase 5 replaced the caller intake, model/tool loop, availability composition,
proposal presentation and confirmation path. These are coupled in the new
orchestrator and BookingSession. The new response dispatcher maps multiple
booking failures to generic caller replies, which cannot by themselves identify
the failed boundary. Previous isolated passing tests did not establish acceptable
end-to-end production dialogue or speech quality.

The proposal `_readback` explicitly speaks the full location, customer phone and
customer email, or "no email" / "غير مسجل". That directly explains the unwanted
contact-detail recital. This source evidence does not establish that a company
phone number was spoken: the template field is the customer phone. The latest
availability failure has not been attributed to a specific live provider result.

Merely disabling `VERIFIED_PHONE_BOOKING_ENABLED` on the current source would
leave the phase-5 `_check_booking` implementation active in its legacy branch.
Starting the original phase-4 image alone would also bring its older database
compatibility checker back. Neither is the restoration implemented here.

## Restoration scope

The release copies the exact historical `services/conversation/orchestrator.py`
into a derived image of the current application and sets only
`VERIFIED_PHONE_BOOKING_ENABLED=false`. That restores the prior intake, natural
dialogue/tool-result processing, two-step booking, acknowledgements, language
handling and speech ownership. It removes the phase-5 mandatory deterministic
contact readback from the active caller path.

The legacy calendar methods used by that caller flow remain present; comparison
with the historical source found their business-method bodies unchanged. New
typed calendar methods remain installed but are not used by the restored flow.
Existing persistence code retains its original `persist_booking_record` body.

The current API, schema-0005 compatibility checker, migrations, booking records,
operation evidence, notification outbox, observation/cancellation workers, voice
model, credentials and mounts are retained. No SQL migration, row rewrite or
provider mutation is part of restoration. The previous image and exact private
override are retained for recovery.

This also disables phase 5's new durable admission, concurrency exclusion and
exact-playback confirmation path for new phone bookings. The earlier two-step
guards are retained, but this is a temporary beta restoration, not acceptance of
all production booking-reliability requirements. Existing unresolved operations
are not retried, released or deleted.

Before building and again after stopping writers, a bounded read-only query must
find no unresolved operation whose protected interval extends into the future.
Otherwise restoration stops; if writers were stopped, the predecessor is restored.
Historical unresolved evidence remains in place.

## Validation

- Restoration release suite: 19 passed, including the actual Linux derived build.
- The actual candidate ran all 12 historical booking-guard tests without skips.
  Provider traffic is synthetic; the app probe has networking disabled.
- The candidate also ran on an isolated internal network against PostgreSQL 16:
  schema 0005 preparation and compatibility succeeded, and both English and Arabic
  restored confirmation handlers created one local booking with the correct
  provider identifier/instant and two held notification jobs. Direct legacy SMS
  submission remained disabled when the notification outbox was enabled.
- Recovery tests cover exact prior image/override restoration, both supported
  predecessors, configuration drift, source mismatch, and a new unresolved
  operation appearing between preflight and writer shutdown.
- Historical source, probe and both predecessor hashes match. Syntax and
  whitespace checks pass. Owned Docker containers, networks and image tags were
  independently confirmed absent; unrelated development containers were retained.

The production parent image is remote. Local tests layered the accepted source
onto an available dependency parent. Actual telephone audio, real model-generated
dialogue and live provider booking were not exercised locally. A supervised call
after restoration remains necessary; test counts alone do not prove voice quality.

## Owner rollout

Publish the reviewed restoration commit after exact owner approval. Finish active
calls, fetch `codex/phase-5-booking-reliability`, then execute
`scripts/restore-pre-phase5-call-flow.py` from that exact commit with the identical
SHA as its argument. The script accepts only the verified effective source of
`e85a50599fb29a3bafcd3ec2f7f05e3ffeafcb11` or
`6b3babf0eee0ed611c05d7031caacebe0e9dd652`.

Success requires `CALL_FLOW_RESTORE_DEPLOYED_READY_HTTPS_OK` and the resulting
commit/image. After success, first check ordinary English/Arabic conversation and
one availability request. Then perform one supervised booking if the flow sounds
correct, and verify Microsoft, dashboard and SMS results.

The application image deliberately uses the historical orchestrator source pinned
by the restoration script. The repository's phase-5 runtime remains parked for
review; a generic rebuild from branch HEAD is not this restoration. Do not resume
phase-5 rollout until the whole caller journey is reviewed and accepted, including
spoken output and real provider boundaries.
