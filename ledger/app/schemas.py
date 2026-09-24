"""Pydantic v2 request/response schemas.

BYTEA columns are exposed over JSON as base64-encoded strings. On input a base64
string is decoded to ``bytes``; on output ``bytes`` are encoded back to base64.
"""

import base64
import datetime
import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer


def _b64_decode(value: object) -> object:
    """Accept a base64 string (JSON input) or raw bytes (ORM value).

    ``validate=True`` so malformed base64 is rejected rather than silently
    stripped — these bytes are hashed, so silent mangling would be an integrity
    hole for the verification ledger.
    """
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        return base64.b64decode(value, validate=True)
    return value


def _b64_encode(value: object) -> object:
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(value).decode("ascii")
    return value


# bytes that round-trip as base64 strings in JSON.
Base64Bytes = Annotated[
    bytes,
    BeforeValidator(_b64_decode),
    PlainSerializer(_b64_encode, return_type=str, when_used="json"),
]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------
# HardwareOwner
# --------------------------------------------------------------------------
class HardwareOwnerCreate(BaseModel):
    owner_id: uuid.UUID
    organisation_name: str
    is_trusted: bool = False


class HardwareOwnerRead(ORMModel):
    owner_id: uuid.UUID
    organisation_name: str
    is_trusted: bool


# --------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------
class HardwareCreate(BaseModel):
    hardware_id: uuid.UUID
    hostname: str
    gpu_product_id: str | None = None
    cpu_product_id: str | None = None
    owner_id: uuid.UUID | None = None
    gpu_firmware_version: str | None = None


class HardwareRead(ORMModel):
    hardware_id: uuid.UUID
    hostname: str
    gpu_product_id: str | None
    cpu_product_id: str | None
    owner_id: uuid.UUID | None
    gpu_firmware_version: str | None


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class ModelCreate(BaseModel):
    model_id: uuid.UUID
    model_name: str
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None
    decoding_algorithm: str | None = None


class ModelRead(ORMModel):
    model_id: uuid.UUID
    model_name: str
    temperature: float | None
    top_k: int | None
    top_p: float | None
    seed: int | None
    decoding_algorithm: str | None
    verification_threshold: float | None
    delta_max: float | None


class ModelThresholdUpdate(BaseModel):
    """Set (or clear, with null) a model's verification pass mark. Mutable
    and outside the model's identity hash by design."""

    verification_threshold: float | None


class ModelDeltaMaxUpdate(BaseModel):
    """Set (or clear, with null) a model's margin cap (difr delta max).
    Mutable and outside the model's identity hash by design; NULL means the
    runner's built-in default."""

    delta_max: float | None


# --------------------------------------------------------------------------
# ModelDeployment / declaration / resolution
# --------------------------------------------------------------------------
class ModelDeclarationRequest(BaseModel):
    """A model process declares that it has started on a host.

    The ledger owns all ids: the `model_id` is **derived from the config**
    (name + params), so an identical config always resolves to the same row;
    `hardware_id` is matched by unique `hostname`; `deployment_id` is generated
    per declaration. The caller sends only the natural spin-up data.

    Declaring upserts the hardware and model config, closes any still-active
    deployment on that host, and opens a new one.
    """

    # Hardware the model is running on (matched/created by hostname).
    hostname: str
    gpu_product_id: str | None = None
    cpu_product_id: str | None = None
    owner_id: uuid.UUID | None = None
    # Convenience alternative to owner_id: upserts a hardware_owner by
    # organisation name and links the hardware to it. The verify-taps declare
    # "prover", the runners "verifier".
    owner_name: str | None = None
    gpu_firmware_version: str | None = None

    # Model config (verification params). model_id is derived from these.
    model_name: str
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None
    decoding_algorithm: str | None = None

    # When the model started; defaults to server receive time if omitted.
    started_at: datetime.datetime | None = None


class ModelDeploymentRead(ORMModel):
    deployment_id: uuid.UUID
    model_id: uuid.UUID
    hardware_id: uuid.UUID
    started_at: datetime.datetime
    ended_at: datetime.datetime | None


class ModelDeploymentCloseRequest(BaseModel):
    """A model process declares that it has stopped on a host.

    Mirrors declare's resolution semantics: the host's ACTIVE deployment
    (ended_at IS NULL) gets its end timestamp set. Sent by the verify-taps
    (prover shutdown) and the inf-ver-runners (drain/exit).
    """

    hostname: str
    ended_at: datetime.datetime | None = None


class ModelResolution(BaseModel):
    """Result of resolving a tapped inference event to ledger ids."""

    deployment_id: uuid.UUID
    model_id: uuid.UUID
    hardware_id: uuid.UUID
    model_name: str
    # False when the declared model name differs from the one seen by the proxy.
    model_name_matches: bool


# --------------------------------------------------------------------------
# InferenceEvent
# --------------------------------------------------------------------------
class InferenceEventCreate(BaseModel):
    id: uuid.UUID
    session_id: str
    ts: datetime.datetime
    started_at: datetime.datetime | None = None
    model_id: uuid.UUID
    input_raw_logits: Base64Bytes | None = None
    output_raw_logits: Base64Bytes | None = None
    hash_input_raw_logits: Base64Bytes
    hash_output_raw_logits: Base64Bytes
    input_text_representation: str | None = None
    output_text_representation: str | None = None
    hardware_id: uuid.UUID
    pod_name: str | None = None
    node_name: str | None = None


class InferenceEventRead(ORMModel):
    id: uuid.UUID
    session_id: str
    ts: datetime.datetime
    started_at: datetime.datetime | None = None
    model_id: uuid.UUID
    input_raw_logits: Base64Bytes | None
    output_raw_logits: Base64Bytes | None
    hash_input_raw_logits: Base64Bytes
    hash_output_raw_logits: Base64Bytes
    input_text_representation: str | None
    output_text_representation: str | None
    hardware_id: uuid.UUID
    pod_name: str | None
    node_name: str | None


# --------------------------------------------------------------------------
# VerificationEvent
# --------------------------------------------------------------------------
VerificationResult = Literal["pass", "fail", "unverifiable"]


class VerificationEventCreate(BaseModel):
    id: uuid.UUID
    inference_event_id: uuid.UUID
    logit_difference_margins: Base64Bytes | None = None
    mean_logit_difference: float | None = None
    std_dev_logit_difference: float | None = None
    verifier_raw_logits: Base64Bytes | None = None
    exact_match_level_pct: float | None = None
    hardware_id: uuid.UUID
    ts: datetime.datetime
    started_at: datetime.datetime | None = None
    result: VerificationResult
    result_detail: str | None = None
    error_code: str | None = None
    verification_threshold: float | None = None
    verifier_model_id: uuid.UUID
    verifier_detail: dict[str, Any] | None = None
    pod_name: str | None = None
    node_name: str | None = None


class VerificationEventRead(ORMModel):
    id: uuid.UUID
    inference_event_id: uuid.UUID
    logit_difference_margins: Base64Bytes | None
    mean_logit_difference: float | None
    std_dev_logit_difference: float | None
    verifier_raw_logits: Base64Bytes | None
    exact_match_level_pct: float | None
    hardware_id: uuid.UUID
    ts: datetime.datetime
    started_at: datetime.datetime | None = None
    # Nullable on read: rows written before the verdict columns existed.
    result: VerificationResult | None = None
    result_detail: str | None = None
    error_code: str | None = None
    verification_threshold: float | None = None
    verifier_model_id: uuid.UUID
    verifier_detail: dict[str, Any] | None = None
    pod_name: str | None
    node_name: str | None


# --------------------------------------------------------------------------
# Verifier polling / UI views
# --------------------------------------------------------------------------
class UnverifiedInferenceEvent(BaseModel):
    """An inference event with no verification_event row yet, plus the model
    config the verifier needs (sampling params + model name)."""

    event: InferenceEventRead
    model: ModelRead


class UnverifiedInferenceEventView(BaseModel):
    """Lightweight inference_event + model join for the awaiting-verification
    UI table — no raw-logit payloads."""

    id: uuid.UUID
    session_id: str
    ts: datetime.datetime
    model_id: uuid.UUID
    model_name: str
    sampling_config: dict[str, Any]
    verification_threshold: float | None
    input_text_representation: str | None
    output_text_representation: str | None
    hardware_id: uuid.UUID


class UnverifiedInferenceEventViewPage(BaseModel):
    items: list[UnverifiedInferenceEventView]
    total: int
    limit: int
    offset: int


class VerificationEventView(BaseModel):
    """Flat verification_event + inference_event + model join for the UI."""

    id: uuid.UUID
    inference_event_id: uuid.UUID
    ts: datetime.datetime
    result: VerificationResult | None
    result_detail: str | None
    error_code: str | None
    verification_threshold: float | None
    verifier_model_id: uuid.UUID
    verifier_model_name: str | None
    model_name: str
    sampling_config: dict[str, Any]
    exact_match_level_pct: float | None
    mean_logit_difference: float | None
    std_dev_logit_difference: float | None
    difr_margins: list[float]
    session_id: str
    input_text_representation: str | None
    output_text_representation: str | None
    verifier_detail: dict[str, Any] | None
    # Was the tapped link fully accounted for while this inference crossed it
    # (inference_event_capture view, sql/004): complete | tainted | partial |
    # uncovered. A second axis beside the verdict, never part of it.
    capture_status: str | None = None
    capture_findings: int = 0
    # When the INFERENCE ran (its span on the tapped link), as distinct from
    # ``ts``, which is when it was verified. Capture status follows the former.
    inference_started_at: datetime.datetime | None = None
    inference_ts: datetime.datetime | None = None


class VerificationEventViewPage(BaseModel):
    items: list[VerificationEventView]
    total: int
    limit: int
    offset: int


class VerificationStats(BaseModel):
    total: int
    pass_count: int
    fail_count: int
    unverifiable_count: int
    average_match_level: float | None


# --------------------------------------------------------------------------
# Capture windows (frame-processor's account of the tapped link, sql/004)
# --------------------------------------------------------------------------
CaptureDirection = Literal["in", "out", "unknown"]


class CaptureFindingCreate(BaseModel):
    kind: str
    frame_class: str
    direction: CaptureDirection = "unknown"
    source_mac: str | None = None
    count: int
    first_ts: datetime.datetime
    last_ts: datetime.datetime
    # [{"ts": ..., "detail": ..., "frame_b64": ... | null}, ...]
    samples: list[dict[str, Any]] = []


class CaptureFindingRead(ORMModel):
    finding_id: uuid.UUID
    window_id: uuid.UUID
    kind: str
    frame_class: str
    direction: str
    source_mac: str | None
    count: int
    first_ts: datetime.datetime
    last_ts: datetime.datetime
    samples: list[dict[str, Any]]


class CaptureWindowCreate(BaseModel):
    """One window of the link's account, findings included.

    The ledger owns the id (derived from capture_host, ifaces and
    window_start, so a retry cannot double-record) and resolves the tapped
    node from ``tapped_hostname`` — the hostname its model deployment
    declared — to a hardware row. The caller sends only natural data.
    """

    capture_host: str
    ifaces: str
    tapped_hostname: str | None = None
    window_start: datetime.datetime
    window_end: datetime.datetime
    tap_version: str
    process_epoch: datetime.datetime | None = None
    observed: int
    classified: int
    kernel_dropped: int
    truncated: int
    errors: int
    finding_count: int
    finding_groups: int
    finding_groups_overflow: int = 0
    complete: bool
    classes: dict[str, Any]
    pins: dict[str, Any]
    findings: list[CaptureFindingCreate] = []


class CaptureWindowRead(ORMModel):
    window_id: uuid.UUID
    capture_host: str
    ifaces: str
    hardware_id: uuid.UUID | None
    window_start: datetime.datetime
    window_end: datetime.datetime
    tap_version: str
    process_epoch: datetime.datetime | None
    observed: int
    classified: int
    kernel_dropped: int
    truncated: int
    errors: int
    finding_count: int
    finding_groups: int
    finding_groups_overflow: int
    complete: bool
    classes: dict[str, Any]
    pins: dict[str, Any]


class CaptureWindowDetail(CaptureWindowRead):
    findings: list[CaptureFindingRead]


class CaptureWindowPage(BaseModel):
    items: list[CaptureWindowRead]
    total: int
    limit: int
    offset: int


CaptureStatus = Literal["complete", "tainted", "partial", "uncovered"]


class InferenceEventCapture(BaseModel):
    """Capture status of one inference event, from the inference_event_capture
    view: was the link fully accounted for while this inference crossed it."""

    inference_event_id: uuid.UUID
    capture_status: CaptureStatus
    windows: int
    span_seconds: float
    covered_seconds: float
    kernel_dropped: int
    findings: int
