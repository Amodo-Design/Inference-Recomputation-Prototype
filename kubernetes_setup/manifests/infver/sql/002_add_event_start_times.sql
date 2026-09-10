-- Event *start* timestamps (ts keeps meaning completion time). Nullable:
-- rows written by pre-upgrade writers stay valid.
ALTER TABLE inference_event    ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ;
ALTER TABLE verification_event ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ;
