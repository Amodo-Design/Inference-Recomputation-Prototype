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


@dataclass(frozen=True)
class VerifierModelConfig:
    id: str
    display_name: str
    revision: str | None = None
    tokenizer_revision: str | None = None
    dtype: str = "bfloat16"
    quantization: str | None = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.70
    max_model_len: int = 32768
    max_logprobs: int = 50
    # Width of the serving vLLM's logits tensor (the HF config's padded
    # vocab_size — 152064 for Qwen2.5, NOT len(tokenizer)). The Gumbel replay
    # draws one full noise row per generated token, so this must match the
    # prover exactly or the RNG stream desyncs after the first token. None =
    # resolve from the model's HF config at first use.
    vocab_size: int | None = None
    extra_vllm_args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerifierModelConfig":
        model_id = data["id"]
        raw_vocab_size = data.get("vocab_size")
        return cls(
            id=model_id,
            display_name=data.get("display_name") or model_id,
            revision=data.get("revision"),
            tokenizer_revision=data.get("tokenizer_revision"),
            dtype=data.get("dtype", "bfloat16"),
            quantization=data.get("quantization"),
            tensor_parallel_size=int(data.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(data.get("gpu_memory_utilization", 0.70)),
            max_model_len=int(data.get("max_model_len", 32768)),
            max_logprobs=int(data.get("max_logprobs", 50)),
            vocab_size=int(raw_vocab_size) if raw_vocab_size is not None else None,
            extra_vllm_args=dict(data.get("extra_vllm_args") or {}),
        )

    def vllm_args(self) -> dict[str, Any]:
        args = {
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "enforce_eager": True,
            "dtype": self.dtype,
            "logprobs_mode": "raw_logits",
            "max_logprobs": self.max_logprobs,
        }
        if self.revision:
            args["revision"] = self.revision
        if self.tokenizer_revision:
            args["tokenizer_revision"] = self.tokenizer_revision
        if self.quantization:
            args["quantization"] = self.quantization
        args.update(self.extra_vllm_args)
        return args


def _as_opt_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _as_opt_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class LaunchSampling:
    """Sampling config the orchestrator injects at launch, copied from the
    ledger model row it is spawning this runner for. Used as a cross-check
    against the ledger (which stays the source of truth for the actual
    verification requests); fields left unset are not checked."""

    seed: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None

    def mismatches(self, model_row: dict[str, Any]) -> list[str]:
        """Human-readable list of launch-vs-ledger disagreements."""
        problems = []
        for field_name in ("seed", "temperature", "top_k", "top_p"):
            launched = getattr(self, field_name)
            declared = model_row.get(field_name)
            if launched is not None and declared != launched:
                problems.append(
                    f"{field_name}: launched with {launched}, ledger has {declared}"
                )
        return problems


@dataclass(frozen=True)
class Settings:
    ledger_api_url: str
    ledger_timeout_seconds: int
    poll_interval_seconds: float
    poll_batch_size: int
    max_verify_attempts: int
    retry_backoff_seconds: float
    max_ledger_failures: int
    max_declare_attempts: int
    # How long to wait for the paired vLLM to come up before consuming any
    # events. Cold-start (weight download + load) is minutes, and verifying
    # against a booting vLLM burns the retry budget into wrong
    # `unverifiable` verdicts.
    vllm_ready_timeout_seconds: float
    runner_hostname: str
    preload_models: bool
    verification_timeout_seconds: int
    max_concurrent_verifications_per_model: int
    margin_clip: float
    enable_activation_verification: bool
    vllm_url: str
    vllm_api_key: str | None
    # The ledger model_id whose pending events this runner drains. The pass
    # threshold lives on that model row (NULL = verification paused).
    runner_model_id: str
    launch_sampling: LaunchSampling
    # The model this runner's paired vLLM serves; declared to the ledger at
    # startup so verification events can link to the verifier's model_id.
    model: VerifierModelConfig
    # The Gumbel replay must draw noise with the same RNG as the prover's
    # vLLM (CUDA Philox); on CPU torch draws MT19937 noise that can never
    # match, so a missing GPU is a hard startup error unless explicitly
    # waived (RUNNER_REQUIRE_CUDA=false, e.g. CPU-only smoke deployments).
    require_cuda: bool = True


DEFAULT_MODEL = {"id": "openai/gpt-oss-20b", "display_name": "gpt-oss-20b", "dtype": "bfloat16"}


def load_settings() -> Settings:
    # RUNNER_MODEL_JSON gives full control over the verifier model config;
    # the common path is just RUNNER_MODEL_NAME (the HF id) plus defaults.
    raw_model = os.getenv("RUNNER_MODEL_JSON")
    model_name = os.getenv("RUNNER_MODEL_NAME")
    if raw_model:
        model_data = json.loads(raw_model)
    elif model_name:
        model_data = {"id": model_name}
    else:
        model_data = dict(DEFAULT_MODEL)
    # The paired vLLM must return at least top_k logprobs per position; 200
    # comfortably covers the taps' pinned top_k=50.
    model_data.setdefault("max_logprobs", _as_int(os.getenv("RUNNER_MAX_LOGPROBS"), 200))

    runner_model_id = os.getenv("RUNNER_MODEL_ID")
    if not runner_model_id:
        raise RuntimeError("RUNNER_MODEL_ID is required (the ledger model to verify)")

    return Settings(
        ledger_api_url=os.getenv("LEDGER_API_URL", "http://ledger-api:8000"),
        ledger_timeout_seconds=_as_int(os.getenv("RUNNER_LEDGER_TIMEOUT_SECONDS"), 10),
        poll_interval_seconds=_as_float(os.getenv("RUNNER_POLL_INTERVAL_SECONDS"), 5.0),
        poll_batch_size=_as_int(os.getenv("RUNNER_POLL_BATCH"), 10),
        max_verify_attempts=_as_int(os.getenv("RUNNER_MAX_ATTEMPTS"), 3),
        retry_backoff_seconds=_as_float(os.getenv("RUNNER_RETRY_BACKOFF_SECONDS"), 15.0),
        max_ledger_failures=_as_int(os.getenv("RUNNER_MAX_LEDGER_FAILURES"), 10),
        max_declare_attempts=_as_int(os.getenv("RUNNER_MAX_DECLARE_ATTEMPTS"), 30),
        vllm_ready_timeout_seconds=_as_float(
            os.getenv("RUNNER_VLLM_READY_TIMEOUT_SECONDS"), 900.0
        ),
        runner_hostname=os.getenv("RUNNER_HOSTNAME", "inf-ver-runner"),
        preload_models=_as_bool(os.getenv("RUNNER_PRELOAD_MODELS"), False),
        verification_timeout_seconds=_as_int(os.getenv("RUNNER_TIMEOUT_SECONDS"), 120),
        max_concurrent_verifications_per_model=_as_int(os.getenv("RUNNER_MODEL_CONCURRENCY"), 1),
        margin_clip=_as_float(os.getenv("RUNNER_MARGIN_CLIP"), 10.0),
        enable_activation_verification=_as_bool(os.getenv("RUNNER_ENABLE_ACTIVATIONS"), False),
        vllm_url=os.getenv("RUNNER_VLLM_URL", "http://vllm:8000/v1"),
        vllm_api_key=os.getenv("RUNNER_VLLM_API_KEY"),
        runner_model_id=runner_model_id,
        launch_sampling=LaunchSampling(
            seed=_as_opt_int(os.getenv("RUNNER_SEED")),
            temperature=_as_opt_float(os.getenv("RUNNER_TEMPERATURE")),
            top_k=_as_opt_int(os.getenv("RUNNER_TOP_K")),
            top_p=_as_opt_float(os.getenv("RUNNER_TOP_P")),
        ),
        model=VerifierModelConfig.from_dict(model_data),
        require_cuda=_as_bool(os.getenv("RUNNER_REQUIRE_CUDA"), True),
    )
