-- T005-D: preserve historical wall times while adding a canonical instant.
ALTER TABLE public.bookings
    ADD COLUMN appointment_time_utc TIMESTAMP WITH TIME ZONE;
