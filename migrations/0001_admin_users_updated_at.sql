-- T005-A: preserve unknown historical update instants.
-- Deliberately two statements: an initial DEFAULT would populate old rows.
ALTER TABLE public.admin_users
    ADD COLUMN updated_at TIMESTAMP WITHOUT TIME ZONE;
ALTER TABLE public.admin_users
    ALTER COLUMN updated_at SET DEFAULT CURRENT_TIMESTAMP;
