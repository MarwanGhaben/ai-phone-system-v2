-- T049-A: opt-in, appointment-scoped notification state. No backfill or sends.
-- Historical 0001-0003 SQL and existing booking rows remain unchanged.
CREATE TABLE public.booking_notification_reconciliation (
    booking_id INTEGER PRIMARY KEY REFERENCES public.bookings(id) ON DELETE RESTRICT,
    snapshot_tenant_id TEXT NOT NULL,
    snapshot_business_id TEXT NOT NULL,
    snapshot_provider_id VARCHAR(255) NOT NULL,
    snapshot_start TIMESTAMP WITH TIME ZONE NOT NULL,
    snapshot_version INTEGER NOT NULL,
    enrolled_at TIMESTAMP WITH TIME ZONE,
    last_present_at TIMESTAMP WITH TIME ZONE,
    last_evidence_started_at TIMESTAMP WITH TIME ZONE,
    first_missing_at TIMESTAMP WITH TIME ZONE,
    last_missing_at TIMESTAMP WITH TIME ZONE,
    missing_count INTEGER NOT NULL,
    last_result VARCHAR(20),
    disposition VARCHAR(32) NOT NULL,
    reason VARCHAR(40),
    transitioned_at TIMESTAMP WITH TIME ZONE,
    last_error_category VARCHAR(32),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    CONSTRAINT booking_notification_scope_check CHECK (
        snapshot_tenant_id <> '' AND snapshot_business_id <> ''
        AND snapshot_provider_id <> '' AND snapshot_version > 0),
    CONSTRAINT booking_notification_missing_check CHECK (
        missing_count BETWEEN 0 AND 2
        AND ((missing_count = 0 AND first_missing_at IS NULL AND last_missing_at IS NULL)
          OR (missing_count > 0 AND enrolled_at IS NOT NULL
              AND first_missing_at IS NOT NULL AND last_missing_at IS NOT NULL
              AND last_missing_at >= first_missing_at))),
    CONSTRAINT booking_notification_disposition_check CHECK (
        (disposition = 'active' AND reason IS NULL AND transitioned_at IS NULL)
        OR (disposition = 'removed_externally'
            AND reason = 'inferred_provider_removal' AND transitioned_at IS NOT NULL)
        OR (disposition = 'cancelled_by_caller'
            AND reason = 'caller_confirmed' AND transitioned_at IS NOT NULL)),
    CONSTRAINT booking_notification_result_check CHECK (
        last_result IS NULL OR last_result IN
        ('present','unavailable','changed','check_failed','reappeared'))
);

CREATE INDEX booking_notification_reconciliation_scope_idx
    ON public.booking_notification_reconciliation
    (snapshot_tenant_id,snapshot_business_id,snapshot_provider_id);

CREATE TABLE public.booking_notification_outbox (
    id UUID PRIMARY KEY,
    booking_id INTEGER NOT NULL REFERENCES public.bookings(id) ON DELETE RESTRICT,
    snapshot_version INTEGER NOT NULL,
    snapshot_tenant_id TEXT NOT NULL,
    snapshot_business_id TEXT NOT NULL,
    snapshot_provider_id VARCHAR(255) NOT NULL,
    snapshot_start TIMESTAMP WITH TIME ZONE NOT NULL,
    event_key TEXT NOT NULL UNIQUE,
    kind VARCHAR(20) NOT NULL,
    due_at TIMESTAMP WITH TIME ZONE NOT NULL,
    next_attempt_at TIMESTAMP WITH TIME ZONE NOT NULL,
    state VARCHAR(20) NOT NULL,
    attempts INTEGER NOT NULL,
    retry_count INTEGER NOT NULL,
    recipient VARCHAR(50) NOT NULL,
    language VARCHAR(10) NOT NULL,
    consultant VARCHAR(255) NOT NULL,
    body TEXT NOT NULL,
    error_category VARCHAR(32),
    provider_message_id VARCHAR(255),
    claim_token UUID,
    dispatch_started_at TIMESTAMP WITH TIME ZONE,
    last_attempt_at TIMESTAMP WITH TIME ZONE,
    accepted_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    CONSTRAINT booking_notification_outbox_identity_unique
        UNIQUE (booking_id,snapshot_version,kind),
    CONSTRAINT booking_notification_outbox_scope_check CHECK (
        snapshot_version > 0 AND snapshot_tenant_id <> ''
        AND snapshot_business_id <> '' AND snapshot_provider_id <> ''
        AND event_key <> '' AND recipient <> '' AND consultant <> '' AND body <> ''),
    CONSTRAINT booking_notification_outbox_kind_check CHECK (
        kind IN ('confirmation','reminder','removal')),
    CONSTRAINT booking_notification_outbox_state_check CHECK (
        state IN ('pending','held','dispatching','accepted','failed','unknown','suppressed')),
    CONSTRAINT booking_notification_outbox_attempts_check CHECK (
        attempts >= 0 AND retry_count >= 0),
    CONSTRAINT booking_notification_outbox_claim_check CHECK (
        (state = 'dispatching') = (claim_token IS NOT NULL AND dispatch_started_at IS NOT NULL)),
    CONSTRAINT booking_notification_outbox_acceptance_check CHECK (
        (state = 'accepted') = (provider_message_id IS NOT NULL AND accepted_at IS NOT NULL))
);

CREATE INDEX booking_notification_outbox_due_idx
    ON public.booking_notification_outbox (state,next_attempt_at,due_at,id);
CREATE INDEX booking_notification_outbox_booking_idx
    ON public.booking_notification_outbox (booking_id,state);

-- Store PostgreSQL's exact deparsed CHECK definitions at migration time.
-- Readiness compares later definitions to this checksum-pinned baseline.
CREATE TABLE public.booking_notification_contract (
    singleton INTEGER PRIMARY KEY,
    check_definitions JSONB NOT NULL,
    CONSTRAINT booking_notification_contract_singleton_check CHECK (singleton = 1)
);
INSERT INTO public.booking_notification_contract (singleton,check_definitions)
SELECT 1,pg_catalog.jsonb_object_agg(k.conname,pg_catalog.pg_get_constraintdef(k.oid,false))
FROM pg_catalog.pg_constraint AS k
WHERE k.conrelid IN (
    'public.booking_notification_reconciliation'::regclass,
    'public.booking_notification_outbox'::regclass,
    'public.booking_notification_contract'::regclass)
  AND k.contype='c';
