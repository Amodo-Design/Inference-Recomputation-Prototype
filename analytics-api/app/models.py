"""Read-only mappings of the core event tables (owned by ledger-api) plus
the enrichment table (owned by gpu-enricher). Only the columns the
`/analysis` query touches are mapped — this service never writes any of
these tables."""

from __future__ import annotations

import datetime
import uuid

from sqlalchemy import Boolean, DateTime, Double, ForeignKey, Integer, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Model(Base):
    __tablename__ = "model"

    model_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    temperature: Mapped[float | None] = mapped_column(Double)
    top_k: Mapped[int | None] = mapped_column(Integer)
    top_p: Mapped[float | None] = mapped_column(Double)


class InferenceEvent(Base):
    __tablename__ = "inference_event"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    session_id: Mapped[str] = mapped_column(Text, nullable=False)
    ts: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    model_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("model.model_id"), nullable=False)
    hardware_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    pod_name: Mapped[str | None] = mapped_column(Text)
    node_name: Mapped[str | None] = mapped_column(Text)
    input_text_representation: Mapped[str | None] = mapped_column(Text)
    output_text_representation: Mapped[str | None] = mapped_column(Text)


class VerificationEvent(Base):
    __tablename__ = "verification_event"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    inference_event_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("inference_event.id"), nullable=False
    )
    ts: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    result: Mapped[str | None] = mapped_column(Text)
    result_detail: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(Text)
    mean_logit_difference: Mapped[float | None] = mapped_column(Double)
    exact_match_level_pct: Mapped[float | None] = mapped_column(Double)
    verification_threshold: Mapped[float | None] = mapped_column(Double)
    verifier_model_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("model.model_id"), nullable=False
    )
    verifier_detail: Mapped[dict | None] = mapped_column(JSONB)
    pod_name: Mapped[str | None] = mapped_column(Text)
    node_name: Mapped[str | None] = mapped_column(Text)
    hardware_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)


class EnrichmentGpuActivity(Base):
    __tablename__ = "enrichment_gpu_activity"

    event_type: Mapped[str] = mapped_column(Text, primary_key=True)
    event_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    window_s: Mapped[float] = mapped_column(Double, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    tensor_active_time_s: Mapped[float | None] = mapped_column(Double)
    sm_occupancy_mean: Mapped[float | None] = mapped_column(Double)
    pipe_activity_s: Mapped[dict | None] = mapped_column(JSONB)
    concurrent_events: Mapped[int | None] = mapped_column(Integer)
    enriched_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class BenchmarkingRun(Base):
    """Mirrors prompt-runner/app/db.py's DDL — prompt-runner is the sole
    writer, this service only reads it (run cohort/economics)."""

    __tablename__ = "benchmarking_run"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    state: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)
    settings: Mapped[dict] = mapped_column(JSONB, nullable=False)
    total_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class BenchmarkingRunResult(Base):
    """Mirrors prompt-runner/app/db.py's DDL — prompt-runner is the sole
    writer, this service only reads it."""

    __tablename__ = "benchmarking_run_result"

    run_id: Mapped[str] = mapped_column(Text, ForeignKey("benchmarking_run.id"), primary_key=True)
    request_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    prompt_id: Mapped[str] = mapped_column(Text, nullable=False)
    sampling_id: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    status_code: Mapped[int | None] = mapped_column(Integer)
    stream_completed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    elapsed_s: Mapped[float] = mapped_column(Double, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
