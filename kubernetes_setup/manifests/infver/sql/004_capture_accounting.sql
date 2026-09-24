-- ----------------------------------------------------------
-- Capture accounting: what the tapped link carried, per window.
--
-- frame-processor classifies every frame on the tapped link and checks it
-- against a per-direction whitelist. Until now the result lived in its log.
-- These two tables make it a record: one capture_window row per fixed
-- wall-clock window per capture host, and one capture_finding row per (kind,
-- class, direction, sender) that departed from the whitelist inside it.
--
-- The window row is the primary object and it exists when nothing happened.
-- A findings-only record cannot tell "clean" from "the capture was down",
-- and the capture being down while inference continues is the failure that
-- matters most. So a window with zero frames is still a row, and a window
-- that is MISSING means the tap was blind for it.
--
-- Prefix = owning concern, as with enrichment_gpu_activity. Written by
-- frame-processor through the ledger API (POST /capture-windows); read by
-- the UI and by the inference_event_capture view below.
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS capture_window (
    -- uuid5 of (capture_host, ifaces, window_start): a retry after a crash is
    -- idempotent, and the same window can never be recorded twice.
    window_id               UUID NOT NULL PRIMARY KEY,
    capture_host            TEXT NOT NULL,          -- where frame-processor ran, e.g. the capture node's hostname
    ifaces                  TEXT NOT NULL,          -- capture interfaces, comma-joined
    -- The TAPPED node, resolved from the hostname its model deployment
    -- declared. NULL when that hostname is unknown to the ledger; the row is
    -- still kept, it just cannot be joined to inference events.
    hardware_id             UUID,
    window_start            TIMESTAMPTZ NOT NULL,   -- aligned to the window length, kernel packet clock
    window_end              TIMESTAMPTZ NOT NULL,
    -- The frame-processor build that wrote the row (FRAME_PROCESSOR_VERSION).
    -- Named generically so either tap implementation can populate it.
    tap_version             TEXT NOT NULL,
    process_epoch           TIMESTAMPTZ,            -- when the reporting capture process started
    observed                BIGINT NOT NULL,        -- frames handed over by the kernel
    classified              BIGINT NOT NULL,        -- frames placed in exactly one class (= observed when balanced)
    kernel_dropped          BIGINT NOT NULL,        -- frames the kernel never handed over
    truncated               INTEGER NOT NULL,       -- frames too large for the ring slot
    errors                  INTEGER NOT NULL,       -- accountant exceptions (should be 0)
    finding_count           BIGINT NOT NULL,        -- every finding raised in the window
    finding_groups          INTEGER NOT NULL,       -- capture_finding rows written for it
    finding_groups_overflow INTEGER NOT NULL DEFAULT 0,  -- findings beyond the per-window group cap, counted not kept
    -- balanced AND kernel_dropped = 0 AND finding_count = 0 AND every capture
    -- interface reported. The one flag a verifier needs.
    complete                BOOLEAN NOT NULL,
    -- {"<frame class>": {"frames": N, "bytes": N, "out": N, "in": N, "unknown": N}, ...}
    classes                 JSONB NOT NULL,
    -- {"<mac>/<frame class>": {"digest": "…", "declared": "…"|null,
    --   "frames": N, "distinct": N, "payload_b64": "…"}, ...}
    -- The payload is kept (first 512 bytes) so a pin can be re-seeded after a
    -- restart and compared as bytes.
    pins                    JSONB NOT NULL,

    CONSTRAINT fk_capture_window_hardware
        FOREIGN KEY (hardware_id) REFERENCES hardware (hardware_id)
        ON DELETE NO ACTION ON UPDATE NO ACTION,
    CONSTRAINT uq_capture_window UNIQUE (capture_host, ifaces, window_start),
    CONSTRAINT ck_capture_window_span CHECK (window_end > window_start)
);

CREATE TABLE IF NOT EXISTS capture_finding (
    finding_id   UUID NOT NULL PRIMARY KEY,
    window_id    UUID NOT NULL,
    kind         TEXT NOT NULL,   -- class-not-allowed | pin-mismatch | mac-unknown | unexpected-exchange | capture-gap | beat-missed | beat-unscheduled | ...
    frame_class  TEXT NOT NULL,   -- arp | ipv4-tcp | http-exchange | capture | ...
    direction    TEXT NOT NULL,   -- from the tapped node's side: out (it emitted) | in (it was sent) | unknown
    source_mac   TEXT,            -- rendered MAC, NULL when the finding has no sender
    count        BIGINT NOT NULL,
    first_ts     TIMESTAMPTZ NOT NULL,
    last_ts      TIMESTAMPTZ NOT NULL,
    -- Up to a few {"ts", "detail", "frame_b64"}: the first bytes of the
    -- offending frames, bounded by frame-processor. Enough to see what it
    -- was; not a capture.
    samples      JSONB NOT NULL,

    CONSTRAINT fk_capture_finding_window
        FOREIGN KEY (window_id) REFERENCES capture_window (window_id)
        ON DELETE CASCADE ON UPDATE NO ACTION,
    CONSTRAINT ck_capture_finding_direction CHECK (direction IN ('in', 'out', 'unknown')),
    CONSTRAINT uq_capture_finding_group UNIQUE (window_id, kind, frame_class, direction, source_mac)
);

CREATE INDEX IF NOT EXISTS idx_capture_window_span    ON capture_window (window_start, window_end);
CREATE INDEX IF NOT EXISTS idx_capture_window_hw_span ON capture_window (hardware_id, window_start, window_end);
CREATE INDEX IF NOT EXISTS idx_capture_finding_window ON capture_finding (window_id);
CREATE INDEX IF NOT EXISTS idx_capture_finding_kind   ON capture_finding (kind, frame_class);

-- ----------------------------------------------------------
-- Capture status of each inference event, computed at read time.
--
-- Deliberately NOT a column on verification_event. The verdict answers "did
-- the prover run the declared model on these tokens" and stays true when the
-- link was noisy; capture integrity is a second axis on the inference event.
-- Computing it at read time also avoids a race: windows are written minutes
-- after they end, and the runner takes newest events first.
--
--   complete  — windows cover the inference's whole span and every one is complete
--   tainted   — covered, but a window had drops, findings, or a missing interface
--   partial   — some window overlaps but the span is not fully covered
--   uncovered — no window overlaps the span at all (the tap was blind, or the
--               window has not been written yet: allow FRAME_PROCESSOR_WINDOW_FLUSH_SECONDS
--               past the inference before reading this as a gap)
-- ----------------------------------------------------------
CREATE OR REPLACE VIEW inference_event_capture AS
WITH spans AS (
    SELECT ie.id,
           ie.hardware_id,
           COALESCE(ie.started_at, ie.ts) AS span_start,
           ie.ts                          AS span_end
    FROM inference_event ie
),
joined AS (
    SELECT s.id,
           s.span_start,
           s.span_end,
           COUNT(cw.window_id)                                            AS windows,
           BOOL_AND(cw.complete)                                          AS all_complete,
           COALESCE(SUM(cw.kernel_dropped), 0)                            AS kernel_dropped,
           COALESCE(SUM(cw.finding_count), 0)                             AS findings,
           -- Overlap of each window with the event's span, summed. The CASE is
           -- load-bearing: LEAST and GREATEST IGNORE nulls rather than
           -- propagating them, so on a LEFT JOIN row that matched no window
           -- the clipping collapses to (span_end - span_start) and an event
           -- the tap never saw reports as FULLY covered. Guard on the join
           -- key instead; the COALESCE alone cannot catch it.
           COALESCE(SUM(
               CASE WHEN cw.window_id IS NULL THEN 0
                    ELSE EXTRACT(EPOCH FROM (
                        LEAST(cw.window_end, s.span_end)
                      - GREATEST(cw.window_start, s.span_start)))
               END
           ), 0)                                                          AS covered_seconds
    FROM spans s
    LEFT JOIN capture_window cw
           ON cw.hardware_id = s.hardware_id
          AND cw.window_start <= s.span_end
          AND cw.window_end   >  s.span_start
    GROUP BY s.id, s.span_start, s.span_end
)
SELECT id AS inference_event_id,
       CASE
         WHEN windows = 0                                                        THEN 'uncovered'
         WHEN covered_seconds + 0.001 < EXTRACT(EPOCH FROM (span_end - span_start)) THEN 'partial'
         WHEN all_complete                                                       THEN 'complete'
         ELSE                                                                         'tainted'
       END                                            AS capture_status,
       windows,
       EXTRACT(EPOCH FROM (span_end - span_start))    AS span_seconds,
       covered_seconds,
       kernel_dropped,
       findings
FROM joined;
