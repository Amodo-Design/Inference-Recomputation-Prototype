-- ----------------------------------------------------------
-- HardwareOwner
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS hardware_owner (
    owner_id          UUID NOT NULL PRIMARY KEY,
    organisation_name TEXT NOT NULL,
    is_trusted        BOOLEAN NOT NULL DEFAULT FALSE

);

-- ----------------------------------------------------------
-- Hardware
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS hardware (
    hardware_id          UUID NOT NULL PRIMARY KEY,
    hostname             TEXT NOT NULL,
    gpu_product_id       TEXT,
    cpu_product_id       TEXT,
    owner_id             UUID,                    -- optional
    gpu_firmware_version TEXT,

    CONSTRAINT fk_hardware_owner
        FOREIGN KEY (owner_id) REFERENCES hardware_owner (owner_id)
        ON DELETE NO ACTION ON UPDATE NO ACTION
);

-- ----------------------------------------------------------
-- Model
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS model (
    model_id                   UUID NOT NULL PRIMARY KEY,
    model_name                 TEXT NOT NULL,
    temperature                DOUBLE PRECISION,
    top_k                      INTEGER,
    top_p                      DOUBLE PRECISION,
    seed                       BIGINT,             -- verifier defined
    decoding_algorithm         TEXT,               -- e.g. 'gumbel_max'
    -- Operator-set pass mark (mean logit-difference) for this model's
    -- verification. Mutable and deliberately NOT part of the model_id
    -- identity hash. NULL = verification paused for this model (the
    -- orchestrator won't spawn runners until it is set).
    verification_threshold     DOUBLE PRECISION,
    -- Operator-set margin cap (difr "delta max"): per-token margins are
    -- clipped to this value, and it substitutes for the +inf margin when the
    -- prover's token falls outside the verifier's candidate set. Mutable,
    -- NOT part of the identity hash. NULL = the runner's default (10.0).
    delta_max                  DOUBLE PRECISION

);

-- ----------------------------------------------------------
-- InferenceEvent
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS inference_event (
    id                         UUID NOT NULL PRIMARY KEY,
    session_id                 TEXT NOT NULL,          -- chat id / agent session id
    ts                         TIMESTAMPTZ NOT NULL,   -- ISO8601, >= ms precision
    model_id                   UUID NOT NULL,
    input_raw_logits           BYTEA,
    output_raw_logits          BYTEA,
    hash_input_raw_logits      BYTEA NOT NULL,         -- e.g. 32 bytes (SHA-256)
    hash_output_raw_logits     BYTEA NOT NULL,
    input_text_representation  TEXT,                   -- optional
    output_text_representation TEXT,                   -- optional
    hardware_id                UUID NOT NULL,

    CONSTRAINT fk_inference_event_model
        FOREIGN KEY (model_id) REFERENCES model (model_id)
        ON DELETE NO ACTION ON UPDATE NO ACTION,
    CONSTRAINT fk_inference_event_hardware
        FOREIGN KEY (hardware_id) REFERENCES hardware (hardware_id)
        ON DELETE NO ACTION ON UPDATE NO ACTION
);

-- ----------------------------------------------------------
-- VerificationEvent
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS verification_event (
    id                       UUID NOT NULL PRIMARY KEY,
    inference_event_id       UUID NOT NULL,
    logit_difference_margins BYTEA,                    -- per-token array
    mean_logit_difference    DOUBLE PRECISION,
    std_dev_logit_difference DOUBLE PRECISION,
    verifier_raw_logits      BYTEA,                    -- optional
    exact_match_level_pct    DOUBLE PRECISION,
    hardware_id              UUID NOT NULL,
    ts                       TIMESTAMPTZ NOT NULL,     -- ISO8601
    result                   TEXT,                     -- 'pass' | 'fail' | 'unverifiable'
    result_detail            TEXT,                     -- human-readable explanation of the result
    error_code               TEXT,                     -- set when result = 'unverifiable'
    -- Pass threshold in force for this verdict; set dynamically per test run.
    verification_threshold   DOUBLE PRECISION,
    -- The model the verifier declared at startup (see /model-deployments/declare).
    verifier_model_id        UUID NOT NULL,
    -- Rich display payload for the results UI (token comparison, texts,
    -- metrics). JSONB, not BYTEA: display data, not hash-verified bytes.
    verifier_detail          JSONB,

    CONSTRAINT fk_verification_event_inference
        FOREIGN KEY (inference_event_id) REFERENCES inference_event (id)
        ON DELETE NO ACTION ON UPDATE NO ACTION,
    CONSTRAINT fk_verification_event_hardware
        FOREIGN KEY (hardware_id) REFERENCES hardware (hardware_id)
        ON DELETE NO ACTION ON UPDATE NO ACTION,
    CONSTRAINT fk_verification_event_verifier_model
        FOREIGN KEY (verifier_model_id) REFERENCES model (model_id)
        ON DELETE NO ACTION ON UPDATE NO ACTION
);

-- ----------------------------------------------------------
-- ModelDeployment
--
-- A `model` row is a reusable verification *config* (name + verifier-defined
-- params). A deployment records that a given config is running on a specific
-- piece of hardware for a time window. The same model_id may be deployed on
-- many hosts concurrently (many deployments, one shared config); each host runs
-- at most one active model at a time.
--
-- Inference events resolve their model_id/hardware_id via these deployments:
--   hostname -> hardware -> active deployment at the inference timestamp.
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_deployment (
    deployment_id UUID NOT NULL PRIMARY KEY,
    model_id      UUID NOT NULL,
    hardware_id   UUID NOT NULL,
    started_at    TIMESTAMPTZ NOT NULL,        -- when the model was declared started
    ended_at      TIMESTAMPTZ,                 -- NULL = still active

    CONSTRAINT fk_model_deployment_model
        FOREIGN KEY (model_id) REFERENCES model (model_id)
        ON DELETE NO ACTION ON UPDATE NO ACTION,
    CONSTRAINT fk_model_deployment_hardware
        FOREIGN KEY (hardware_id) REFERENCES hardware (hardware_id)
        ON DELETE NO ACTION ON UPDATE NO ACTION
);

-- ----------------------------------------------------------
-- Indexes
-- ----------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_inference_event_session  ON inference_event (session_id);
CREATE INDEX IF NOT EXISTS idx_inference_event_ts       ON inference_event (ts);
CREATE INDEX IF NOT EXISTS idx_inference_event_model    ON inference_event (model_id);
CREATE INDEX IF NOT EXISTS idx_inference_event_hardware ON inference_event (hardware_id);
CREATE INDEX IF NOT EXISTS idx_verification_event_infer ON verification_event (inference_event_id);
CREATE INDEX IF NOT EXISTS idx_verification_event_ts    ON verification_event (ts);
CREATE INDEX IF NOT EXISTS idx_verification_event_result ON verification_event (result);

-- Hostname uniquely identifies a hardware instance (partitions included).
CREATE UNIQUE INDEX IF NOT EXISTS uq_hardware_hostname ON hardware (hostname);

-- At most one active (un-ended) deployment per hardware instance.
CREATE UNIQUE INDEX IF NOT EXISTS uq_model_deployment_active_per_hardware
    ON model_deployment (hardware_id) WHERE ended_at IS NULL;

-- Resolution lookups: active deployment for a hardware at a timestamp.
CREATE INDEX IF NOT EXISTS idx_model_deployment_hardware_window
    ON model_deployment (hardware_id, started_at);
CREATE INDEX IF NOT EXISTS idx_model_deployment_model
    ON model_deployment (model_id);