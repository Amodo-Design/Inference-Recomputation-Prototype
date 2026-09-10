-- Pod identity per event: inf-proxy (sidecar, Downward API) for inference
-- events; inf-ver-runner (EndpointSlice lookup) for verification events.
-- Nullable: rows written before this migration have neither.
ALTER TABLE inference_event
    ADD COLUMN IF NOT EXISTS pod_name  TEXT,
    ADD COLUMN IF NOT EXISTS node_name TEXT;
ALTER TABLE verification_event
    ADD COLUMN IF NOT EXISTS pod_name  TEXT,
    ADD COLUMN IF NOT EXISTS node_name TEXT;

-- Per-event GPU activity is NOT part of the core ledger schema. It lives in
-- enrichment_gpu_activity, owned and created by the gpu-enricher service
-- (gpu-enricher/sql/enrichment_gpu_activity.sql) at its own startup.
