-- T018-B: one last-attempt observation per canonical local booking.
-- No historical time conversion, booking status update, or data backfill.
CREATE TABLE public.booking_provider_observations (
    booking_id INTEGER PRIMARY KEY REFERENCES public.bookings(id) ON DELETE CASCADE,
    snapshot_provider_id VARCHAR(255) NOT NULL,
    snapshot_start TIMESTAMP WITH TIME ZONE NOT NULL,
    snapshot_status VARCHAR(50) NOT NULL,
    checked_at TIMESTAMP WITH TIME ZONE NOT NULL,
    outcome VARCHAR(20) NOT NULL,
    error_category VARCHAR(32),
    http_status INTEGER,
    provider_start TIMESTAMP WITH TIME ZONE,
    provider_end TIMESTAMP WITH TIME ZONE,
    staff_member_ids TEXT[],
    service_id TEXT,
    is_location_online BOOLEAN,
    observed_provider_id VARCHAR(255),
    CONSTRAINT booking_observation_snapshot_status_check
        CHECK (snapshot_status = 'confirmed'),
    CONSTRAINT booking_observation_outcome_check
        CHECK (outcome IN ('present', 'changed', 'unavailable', 'check_failed')),
    CONSTRAINT booking_observation_error_check
        CHECK ((outcome = 'check_failed') = (error_category IS NOT NULL)),
    CONSTRAINT booking_observation_error_category_check
        CHECK (error_category IS NULL OR error_category IN (
            'configuration', 'authentication', 'authorization', 'throttled',
            'server_error', 'timeout', 'network', 'malformed',
            'identity_mismatch', 'http_error')),
    CONSTRAINT booking_observation_interval_pair_check
        CHECK (((outcome IN ('present', 'changed'))
                AND provider_start IS NOT NULL AND provider_end IS NOT NULL)
               OR ((outcome IN ('unavailable', 'check_failed'))
                   AND provider_start IS NULL AND provider_end IS NULL)),
    CONSTRAINT booking_observation_interval_order_check
        CHECK (provider_start IS NULL
               OR (provider_end IS NOT NULL AND provider_end > provider_start)),
    CONSTRAINT booking_observation_http_status_check
        CHECK (http_status IS NULL OR http_status BETWEEN 100 AND 599),
    CONSTRAINT booking_observation_identity_presence_check
        CHECK ((observed_provider_id IS NOT NULL)
               = (outcome IN ('present', 'changed'))),
    CONSTRAINT booking_observation_identity_match_check
        CHECK (observed_provider_id IS NULL
               OR observed_provider_id = snapshot_provider_id)
);

CREATE INDEX booking_provider_observations_checked_at_idx
    ON public.booking_provider_observations (checked_at, booking_id);

-- One Bookings business is configured for this application. The row holds the
-- provider-wide pause shared by all workers and the exact CHECK definitions
-- produced by PostgreSQL from this checksum-pinned migration. Later readiness
-- compares these definitions byte-for-byte; it does not learn a new baseline.
CREATE TABLE public.booking_provider_observation_control (
    singleton INTEGER PRIMARY KEY,
    next_request_at TIMESTAMP WITH TIME ZONE NOT NULL,
    check_definitions JSONB NOT NULL,
    CONSTRAINT booking_observation_control_singleton_check CHECK (singleton = 1)
);

INSERT INTO public.booking_provider_observation_control
    (singleton, next_request_at, check_definitions)
SELECT 1, TIMESTAMPTZ '2000-01-01 00:00:00+00',
       pg_catalog.jsonb_object_agg(k.conname, pg_catalog.pg_get_constraintdef(k.oid, false))
FROM pg_catalog.pg_constraint AS k
WHERE k.conrelid IN (
    'public.booking_provider_observations'::regclass,
    'public.booking_provider_observation_control'::regclass)
  AND k.contype = 'c';
