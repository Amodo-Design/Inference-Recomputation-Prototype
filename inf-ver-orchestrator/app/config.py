"""Environment configuration for the orchestrator."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _require_pinned_image(value: str | None, env_name: str) -> str:
    """The verifier vLLM image must be explicitly and immovably pinned.

    difr's verifier re-implements the serving vLLM version's seeded sampler;
    an image that floats (missing env, no tag, or :latest) can silently
    diverge from the provers and turn every verdict into `fail`. Fail at
    startup instead.
    """
    if not value:
        raise RuntimeError(
            f"{env_name} must be set to the pinned vLLM image used by the serving "
            "LLMISvcs (see infver-config); refusing to default to a floating tag."
        )
    last_segment = value.rsplit("/", 1)[-1]
    tag = last_segment.rsplit(":", 1)[-1] if ":" in last_segment else None
    if tag is None or tag == "latest":
        raise RuntimeError(
            f"{env_name}={value!r} is not pinned; use an explicit version tag or "
            "digest matching the serving vLLMs (a floating verifier image breaks "
            "prover/verifier sampler correspondence)."
        )
    return value


def _require_ge_one(value: int, name: str) -> int:
    if value < 1:
        raise RuntimeError(f"{name} must be >= 1 (got {value})")
    return value


@dataclass(frozen=True)
class PlacementOverride:
    """Per-model verifier placement: pin the vLLM to a node (selector) and
    optionally to that node's own model-cache PVC. A node-local PVC pins
    the pod to one node, so cached placement is per-node, not per-pool."""

    node_selector: dict[str, str] = field(default_factory=dict)
    cache_pvc: str | None = None


@dataclass(frozen=True)
class Settings:
    ledger_api_url: str
    namespace: str
    poll_seconds: float
    runner_image: str
    vllm_image: str
    # --max-logprobs on the spawned verifier vLLMs; must stay >= the taps'
    # pinned top_k (50) — the runner requests prompt_logprobs=top_k.
    vllm_max_logprobs: int
    vllm_max_model_len: int
    # Verifier vLLMs are single-sequence: cap the KV pool well below the
    # default ~0.92 so the prompt_logprobs path has transient headroom
    # (log_softmax materializes fp32 over the full vocab per prefill chunk;
    # observed 5.24GiB allocation vs 4GiB free at defaults — engine death).
    vllm_gpu_memory_utilization: float = 0.85
    # Bounds the prefill chunk and therefore the prompt_logprobs transient
    # (~vocab * 4 bytes * chunk tokens; 2048 -> ~1.6GiB for a 201k vocab).
    vllm_max_num_batched_tokens: int = 2048
    # JSON object, e.g. '{"kubernetes.io/hostname": "gpu-node-1"}'.
    vllm_node_selector: dict[str, str] = field(default_factory=dict)
    # Per-model verifier placement (JSON via ORCH_VLLM_PLACEMENT_OVERRIDES).
    # Empty = every verifier uses vllm_node_selector + model_cache_pvc.
    vllm_placement_overrides: dict[str, PlacementOverride] = field(
        default_factory=dict
    )
    job_backoff_limit: int = 3
    # Failed runner Jobs are kept for inspection by default; their vLLM is
    # still reaped (the GPU is the expensive part, the Job pod logs are not).
    keep_failed_jobs: bool = True
    # How long the runner waits for its paired verifier vLLM to come up.
    # Cold start re-downloads the model from HF into an emptyDir every spawn;
    # large repos (gpt-oss ~40GiB) on a slow/flaky link far exceed the
    # runner's built-in 900s default.
    runner_vllm_ready_timeout_seconds: int = 3600
    # GPUs for the runner Job. The runner replays vLLM's per-request CUDA
    # Philox exponential stream; without a GPU it draws CPU MT19937 noise
    # that can never match the prover's. 0 disables (sets
    # RUNNER_REQUIRE_CUDA=false on the Job) for CPU-only smoke deployments.
    runner_gpus: int = 1
    # ServiceAccount for runner Jobs: grants the EndpointSlice get/list the
    # pod-identity resolver needs. RBAC manifests live in the kube repo
    # (manifests/infver/inf-ver-runner-rbac.yaml).
    runner_service_account: str = "inf-ver-runner"   # env ORCH_RUNNER_SERVICE_ACCOUNT
    # PVC holding the node-local HF model cache. None (unset/empty env)
    # disables caching entirely: no seed Jobs, hf:// URIs as before.
    model_cache_pvc: str | None = None
    # Concurrent verifications per runner Job (RUNNER_MODEL_CONCURRENCY on
    # the Job -> the runner's verify semaphore). 1 = today's serial drains.
    # Raising it stresses the prompt_logprobs prefill path (bounded by
    # vllm_max_num_batched_tokens); step to 2-4 first.
    runner_model_concurrency: int = 1   # env ORCH_RUNNER_MODEL_CONCURRENCY


def load_settings() -> Settings:
    raw_selector = os.getenv("ORCH_VLLM_NODE_SELECTOR")
    node_selector: dict[str, Any] = json.loads(raw_selector) if raw_selector else {}

    raw_overrides = os.getenv("ORCH_VLLM_PLACEMENT_OVERRIDES")
    placement_overrides: dict[str, PlacementOverride] = {}
    if raw_overrides:
        for name, cfg in json.loads(raw_overrides).items():
            placement_overrides[name] = PlacementOverride(
                node_selector=dict(cfg.get("node_selector") or {}),
                cache_pvc=cfg.get("cache_pvc") or None,
            )
    model_cache_pvc = os.getenv("ORCH_MODEL_CACHE_PVC") or None
    if model_cache_pvc is None and any(
        o.cache_pvc for o in placement_overrides.values()
    ):
        raise RuntimeError(
            "ORCH_VLLM_PLACEMENT_OVERRIDES sets a cache_pvc but "
            "ORCH_MODEL_CACHE_PVC is unset; per-model caches require the "
            "model cache feature to be enabled globally."
        )

    return Settings(
        ledger_api_url=os.getenv("LEDGER_API_URL", "http://ledger-api:8000"),
        namespace=os.getenv("ORCH_NAMESPACE", "infver"),
        poll_seconds=_as_float(os.getenv("ORCH_POLL_SECONDS"), 15.0),
        runner_image=os.getenv(
            "ORCH_RUNNER_IMAGE",
            "harbor.infver.local/infver_images/inf-ver-runner:latest",
        ),
        vllm_image=_require_pinned_image(os.getenv("ORCH_VLLM_IMAGE"), "ORCH_VLLM_IMAGE"),
        vllm_max_logprobs=_as_int(os.getenv("ORCH_VLLM_MAX_LOGPROBS"), 200),
        vllm_max_model_len=_as_int(os.getenv("ORCH_VLLM_MAX_MODEL_LEN"), 8192),
        vllm_gpu_memory_utilization=_as_float(
            os.getenv("ORCH_VLLM_GPU_MEMORY_UTILIZATION"), 0.85
        ),
        vllm_max_num_batched_tokens=_as_int(
            os.getenv("ORCH_VLLM_MAX_NUM_BATCHED_TOKENS"), 2048
        ),
        vllm_node_selector=node_selector,
        vllm_placement_overrides=placement_overrides,
        job_backoff_limit=_as_int(os.getenv("ORCH_JOB_BACKOFF_LIMIT"), 3),
        keep_failed_jobs=_as_bool(os.getenv("ORCH_KEEP_FAILED_JOBS"), True),
        runner_gpus=_as_int(os.getenv("ORCH_RUNNER_GPUS"), 1),
        runner_vllm_ready_timeout_seconds=_as_int(
            os.getenv("ORCH_RUNNER_VLLM_READY_TIMEOUT_SECONDS"), 3600
        ),
        runner_service_account=os.getenv(
            "ORCH_RUNNER_SERVICE_ACCOUNT", "inf-ver-runner"
        ),
        model_cache_pvc=model_cache_pvc,
        runner_model_concurrency=_require_ge_one(
            _as_int(os.getenv("ORCH_RUNNER_MODEL_CONCURRENCY"), 1),
            "ORCH_RUNNER_MODEL_CONCURRENCY",
        ),
    )
