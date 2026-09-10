-- Per-event GPU activity integrated from Prometheus/DCGM by the gpu-enricher
-- service (the only writer). Owned by gpu-enricher: created by its startup
-- DDL, never by ledger migrations. Prefix = owning component; "enrichment"
-- because it covers ALL traffic, organic included.
CREATE TABLE IF NOT EXISTS enrichment_gpu_activity (
    event_type            TEXT NOT NULL CHECK (event_type IN ('inference', 'verification')),
    event_id              UUID NOT NULL,
    -- Wall-clock span of the event's GPU-attributable window (ts - started_at).
    window_s              DOUBLE PRECISION NOT NULL,
    -- Number of Prometheus samples inside the window. 0-sample events still
    -- get a row so the poller does not re-query them forever; the UI filters
    -- on this at display time (MIN_GPU_SAMPLES).
    sample_count          INTEGER NOT NULL,
    -- ∫ DCGM_FI_PROF_PIPE_TENSOR_ACTIVE dt over the window: tensor-pipe
    -- busy-seconds, the headline efficiency quantity.
    tensor_active_time_s  DOUBLE PRECISION,
    -- Mean (not integrated) DCGM_FI_PROF_SM_OCCUPANCY over the window.
    sm_occupancy_mean     DOUBLE PRECISION,
    -- Integrals for the remaining pipe metrics, keyed by verbatim DCGM field
    -- name. JSONB so adding/dropping a collected metric needs no migration.
    pipe_activity_s       JSONB,
    -- Other events overlapping this window on the same pod (node when the
    -- pod is unknown). NULL = concurrency unknown (no identity to match on).
    concurrent_events     INTEGER,
    enriched_at           TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (event_type, event_id)
);
