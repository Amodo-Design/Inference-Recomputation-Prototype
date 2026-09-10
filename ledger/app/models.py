import datetime
import uuid

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Double,
    ForeignKey,
    Integer,
    LargeBinary,
    Text,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class HardwareOwner(Base):
    __tablename__ = "hardware_owner"

    owner_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    organisation_name: Mapped[str] = mapped_column(Text, nullable=False)
    is_trusted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )


class Hardware(Base):
    __tablename__ = "hardware"

    hardware_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    hostname: Mapped[str] = mapped_column(Text, nullable=False)
    gpu_product_id: Mapped[str | None] = mapped_column(Text)
    cpu_product_id: Mapped[str | None] = mapped_column(Text)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("hardware_owner.owner_id")
    )
    gpu_firmware_version: Mapped[str | None] = mapped_column(Text)


class Model(Base):
    __tablename__ = "model"

    model_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    temperature: Mapped[float | None] = mapped_column(Double)
    top_k: Mapped[int | None] = mapped_column(Integer)
    top_p: Mapped[float | None] = mapped_column(Double)
    seed: Mapped[int | None] = mapped_column(BigInteger)
    decoding_algorithm: Mapped[str | None] = mapped_column(Text)
    # Mutable operator setting — NOT part of the model_id identity hash.
    verification_threshold: Mapped[float | None] = mapped_column(Double)
    delta_max: Mapped[float | None] = mapped_column(Double)


class ModelDeployment(Base):
    __tablename__ = "model_deployment"

    deployment_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    model_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("model.model_id"), nullable=False
    )
    hardware_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("hardware.hardware_id"), nullable=False
    )
    started_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    ended_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )


class InferenceEvent(Base):
    __tablename__ = "inference_event"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    session_id: Mapped[str] = mapped_column(Text, nullable=False)
    ts: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Event start; ts is the completion time. Nullable: legacy rows have none.
    started_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("model.model_id"), nullable=False
    )
    input_raw_logits: Mapped[bytes | None] = mapped_column(LargeBinary)
    output_raw_logits: Mapped[bytes | None] = mapped_column(LargeBinary)
    hash_input_raw_logits: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False
    )
    hash_output_raw_logits: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False
    )
    input_text_representation: Mapped[str | None] = mapped_column(Text)
    output_text_representation: Mapped[str | None] = mapped_column(Text)
    hardware_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("hardware.hardware_id"), nullable=False
    )
    # Serving pod identity, captured at write time by the observer (inf-proxy
    # sidecar / inf-ver-runner). NULL outside Kubernetes.
    pod_name: Mapped[str | None] = mapped_column(Text)
    node_name: Mapped[str | None] = mapped_column(Text)


class VerificationEvent(Base):
    __tablename__ = "verification_event"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    inference_event_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("inference_event.id"), nullable=False
    )
    logit_difference_margins: Mapped[bytes | None] = mapped_column(LargeBinary)
    mean_logit_difference: Mapped[float | None] = mapped_column(Double)
    std_dev_logit_difference: Mapped[float | None] = mapped_column(Double)
    verifier_raw_logits: Mapped[bytes | None] = mapped_column(LargeBinary)
    exact_match_level_pct: Mapped[float | None] = mapped_column(Double)
    hardware_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("hardware.hardware_id"), nullable=False
    )
    ts: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Event start; ts is the completion time. Nullable: legacy rows have none.
    started_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    result: Mapped[str | None] = mapped_column(Text)
    result_detail: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(Text)
    verification_threshold: Mapped[float | None] = mapped_column(Double)
    verifier_model_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("model.model_id"), nullable=False
    )
    verifier_detail: Mapped[dict | None] = mapped_column(JSONB)
    # Serving pod identity, captured at write time by the observer (inf-proxy
    # sidecar / inf-ver-runner). NULL outside Kubernetes.
    pod_name: Mapped[str | None] = mapped_column(Text)
    node_name: Mapped[str | None] = mapped_column(Text)
