-- T017-A additive, opt-in create-operation ledger. The whole asset runs inside
-- the runner's advisory-locked transaction. No application booking rows change.
CREATE EXTENSION IF NOT EXISTS btree_gist WITH SCHEMA public;

CREATE TABLE public.booking_operations (
    operation_id uuid PRIMARY KEY,
    tenant_id text NOT NULL,
    business_id text NOT NULL,
    staff_id text NOT NULL,
    service_id text NOT NULL,
    action text NOT NULL,
    proposal_id text NOT NULL,
    proposal_revision bigint NOT NULL,
    payload_hash text NOT NULL,
    payload_snapshot jsonb NOT NULL,
    confirmation_identity jsonb NOT NULL,
    confirmed_at timestamptz NOT NULL,
    issued_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    admitted_at timestamptz NOT NULL,
    starts_at timestamptz NOT NULL,
    ends_at timestamptz NOT NULL,
    pre_buffer interval NOT NULL,
    post_buffer interval NOT NULL,
    claim_span tstzrange NOT NULL,
    state text NOT NULL DEFAULT 'pending',
    fence bigint NOT NULL DEFAULT 0,
    dispatch_count smallint NOT NULL DEFAULT 0,
    owner_token uuid,
    lease_until timestamptz,
    ownership_started_at timestamptz,
    provider_id text,
    receipt_provider_id text,
    booking_id integer REFERENCES public.bookings(id),
    verified_snapshot jsonb,
    settlement_source text,
    settlement_observed_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT booking_operations_intent_key UNIQUE
      (tenant_id,business_id,action,proposal_id,proposal_revision),
    CONSTRAINT booking_operations_receipt_identity_key UNIQUE
      (tenant_id,business_id,receipt_provider_id),
    CONSTRAINT booking_operations_booking_link_key UNIQUE (booking_id),
    CONSTRAINT booking_operations_action_check CHECK (action = 'create'),
    CONSTRAINT booking_operations_identity_check CHECK (
      length(tenant_id) BETWEEN 1 AND 255 AND length(business_id) BETWEEN 1 AND 255
      AND length(staff_id) BETWEEN 1 AND 255 AND length(service_id) BETWEEN 1 AND 255
      AND length(proposal_id) BETWEEN 1 AND 255 AND proposal_revision > 0),
    CONSTRAINT booking_operations_payload_check CHECK (
      payload_hash ~ '^[0-9a-f]{64}$' AND jsonb_typeof(payload_snapshot) = 'object'
      AND jsonb_typeof(confirmation_identity) = 'object'),
    CONSTRAINT booking_operations_clock_check CHECK (
      isfinite(confirmed_at) AND isfinite(issued_at) AND isfinite(expires_at)
      AND isfinite(admitted_at)
      AND isfinite(starts_at) AND isfinite(ends_at) AND isfinite(created_at)
      AND isfinite(updated_at) AND issued_at <= confirmed_at
      AND confirmed_at < expires_at AND expires_at - issued_at > interval '0'
      AND expires_at - issued_at <= interval '2 minutes'
      AND confirmed_at <= admitted_at AND admitted_at < expires_at
      AND starts_at > issued_at),
    CONSTRAINT booking_operations_interval_check CHECK (
      starts_at < ends_at AND ends_at - starts_at <= interval '4 hours'
      AND pre_buffer >= interval '0' AND pre_buffer <= interval '2 hours'
      AND post_buffer >= interval '0' AND post_buffer <= interval '2 hours'
      AND NOT isempty(claim_span) AND lower_inc(claim_span) AND NOT upper_inc(claim_span)
      AND NOT lower_inf(claim_span) AND NOT upper_inf(claim_span)
      AND lower(claim_span) = starts_at - pre_buffer
      AND upper(claim_span) = ends_at + post_buffer),
    CONSTRAINT booking_operations_state_check CHECK (
      state IN ('pending','dispatched','unresolved','applied','released','manual_review')
      AND fence >= 0 AND dispatch_count BETWEEN 0 AND 1
      AND ((owner_token IS NULL AND lease_until IS NULL AND ownership_started_at IS NULL)
           OR (owner_token IS NOT NULL AND lease_until IS NOT NULL
               AND ownership_started_at IS NOT NULL AND isfinite(lease_until)
               AND isfinite(ownership_started_at)
               AND ownership_started_at < lease_until))
      AND ((settlement_source IS NULL AND settlement_observed_at IS NULL)
           OR (settlement_source IS NOT NULL AND length(settlement_source) BETWEEN 1 AND 255
               AND settlement_observed_at IS NOT NULL
               AND isfinite(settlement_observed_at)))
      AND (provider_id IS NULL OR length(provider_id) BETWEEN 1 AND 255)
      AND (receipt_provider_id IS NULL OR length(receipt_provider_id) BETWEEN 1 AND 255)
      AND ((booking_id IS NULL AND verified_snapshot IS NULL)
           OR (booking_id IS NOT NULL AND verified_snapshot IS NOT NULL
               AND jsonb_typeof(verified_snapshot) = 'object'
               AND receipt_provider_id IS NOT NULL AND provider_id IS NOT NULL
               AND state = 'applied' AND provider_id = receipt_provider_id))
      AND (receipt_provider_id IS NULL OR state IN
           ('dispatched','unresolved','manual_review','applied'))
      AND (
        (state = 'pending' AND fence = 0 AND dispatch_count = 0
         AND owner_token IS NULL AND lease_until IS NULL AND ownership_started_at IS NULL
         AND provider_id IS NULL AND settlement_source IS NULL
         AND settlement_observed_at IS NULL)
        OR
        (state = 'dispatched' AND fence > 0 AND dispatch_count = 1
         AND owner_token IS NOT NULL AND lease_until IS NOT NULL
         AND ownership_started_at IS NOT NULL AND provider_id IS NULL
         AND settlement_source IS NULL AND settlement_observed_at IS NULL)
        OR
        (state = 'unresolved' AND fence > 0 AND dispatch_count = 1
         AND owner_token IS NOT NULL AND lease_until IS NOT NULL
         AND ownership_started_at IS NOT NULL AND provider_id IS NULL
         AND settlement_source IS NULL AND settlement_observed_at IS NULL)
        OR
        (state = 'applied' AND fence > 0 AND dispatch_count = 1
         AND owner_token IS NOT NULL AND lease_until IS NOT NULL
         AND ownership_started_at IS NOT NULL AND provider_id IS NOT NULL
         AND settlement_source IS NOT NULL AND settlement_observed_at IS NOT NULL
         AND settlement_observed_at >= ownership_started_at)
        OR
        (state = 'manual_review' AND fence > 0 AND dispatch_count = 1
         AND owner_token IS NOT NULL AND lease_until IS NOT NULL
         AND ownership_started_at IS NOT NULL AND provider_id IS NULL
         AND settlement_source IS NOT NULL AND settlement_observed_at IS NOT NULL
         AND settlement_observed_at >= ownership_started_at)
        OR
        (state = 'released' AND (
          (fence = 0 AND dispatch_count = 0 AND owner_token IS NULL
           AND lease_until IS NULL AND ownership_started_at IS NULL
           AND provider_id IS NULL AND settlement_source IS NULL
           AND settlement_observed_at IS NULL)
          OR
          (fence > 0 AND dispatch_count = 1 AND owner_token IS NOT NULL
           AND lease_until IS NOT NULL AND ownership_started_at IS NOT NULL
           AND provider_id IS NULL AND settlement_source IS NOT NULL
           AND settlement_observed_at IS NOT NULL
           AND settlement_observed_at >= ownership_started_at))))),
    CONSTRAINT booking_operations_staff_exclusion EXCLUDE USING gist
      (tenant_id WITH =, business_id WITH =, staff_id WITH =, claim_span WITH &&)
      WHERE (state <> 'released')
);

CREATE INDEX booking_operations_unresolved_idx
  ON public.booking_operations (state,lease_until,operation_id);

CREATE TABLE public.booking_operation_contract (
    singleton integer PRIMARY KEY,
    constraint_definitions jsonb NOT NULL,
    CONSTRAINT booking_operation_contract_singleton_check CHECK (singleton = 1)
);

INSERT INTO public.booking_operation_contract (singleton,constraint_definitions)
SELECT 1, jsonb_object_agg(conname,pg_catalog.pg_get_constraintdef(oid,false))
FROM pg_catalog.pg_constraint
WHERE conrelid='public.booking_operations'::regclass;
