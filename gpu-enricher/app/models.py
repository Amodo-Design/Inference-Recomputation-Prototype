"""Read-only mappings of the core event tables (owned by ledger-api) plus
the enrichment table this service owns. Only the columns the poller touches
are mapped — the enricher never writes core tables."""

from __future__ import annotations

import datetime
import uuid

from sqlalchemy import DateTime, Double, Integer, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class InferenceEvent(Base):
    __tablename__ = "inference_event"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    ts: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    pod_name: Mapped[str | None] = mapped_column(Text)
    node_name: Mapped[str | None] = mapped_column(Text)
    hardware_id: Mapped[uuid.UUID] = mapped_column(Uuid)


class VerificationEvent(Base):
    __tablename__ = "verification_event"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    ts: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    pod_name: Mapped[str | None] = mapped_column(Text)
    node_name: Mapped[str | None] = mapped_column(Text)
    hardware_id: Mapped[uuid.UUID] = mapped_column(Uuid)


class Hardware(Base):
    __tablename__ = "hardware"

    hardware_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    hostname: Mapped[str] = mapped_column(Text)


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
