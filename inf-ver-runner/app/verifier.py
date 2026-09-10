from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import math
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import torch
from openai import AsyncOpenAI
from transformers import AutoConfig, AutoTokenizer

from difr.gumbel_verify import (
    SimpleTokenMetrics,
    TokenSequence,
    get_probs,
    verify_vllm_gumbel_max,
)

from .config import Settings, VerifierModelConfig
from .models import (
    ErrorCode,
    ModelInfo,
    OutputTokenComparison,
    TokenMetricSummary,
    VerificationStatus,
    VerifyRequest,
    VerifyResponse,
)

log = logging.getLogger(__name__)

# Per-call margin cap (the ledger model row's delta_max); None = use the
# verifier's configured default.
_ACTIVE_MARGIN_CLIP: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "active_margin_clip", default=None
)


def _torch_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    return torch.bfloat16


def _empty_metrics(prompt_tokens: int = 0, output_tokens: int = 0) -> TokenMetricSummary:
    return TokenMetricSummary(
        token_count=output_tokens,
        exact_match_count=0,
        exact_match_ratio=0.0,
        margins=[],
        min_probability=None,
        mean_probability=None,
        mean_margin=None,
        max_margin=None,
        max_logit_rank=None,
        max_gumbel_rank=None,
    )


class CachedModelVerifier:
    def __init__(
        self,
        model_config: VerifierModelConfig,
        vllm_url: str,
        vllm_api_key: str | None,
        concurrency: int,
        margin_clip: float,
        require_cuda: bool = True,
    ):
        self.model_config = model_config
        self.vllm_url = vllm_url
        self.vllm_api_key = vllm_api_key
        self.margin_clip = margin_clip if math.isfinite(margin_clip) and margin_clip > 0 else 10.0
        self._client: AsyncOpenAI | None = None
        self._tokenizer: Any | None = None
        self._verify_lock = asyncio.Semaphore(concurrency)
        # The serving vLLM's logits width; resolved lazily, see
        # _resolve_vocab_size.
        self._vocab_size: int | None = model_config.vocab_size
        # The Gumbel replay must draw noise from the same RNG the prover's
        # vLLM used — a per-request CUDA Philox generator. CPU torch draws
        # MT19937 noise with the same seed, which never matches, so margins
        # silently degrade from "seeded replay" to "entropy of the output
        # distribution". Fail fast rather than degrade silently.
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif require_cuda:
            raise RuntimeError(
                "CUDA is required to replay the prover's Philox RNG stream "
                "(the Gumbel-margin replay is meaningless with CPU noise). "
                "Schedule this runner on a GPU node, or set "
                "RUNNER_REQUIRE_CUDA=false to accept entropy-level margins."
            )
        else:
            log.warning(
                "CUDA unavailable and RUNNER_REQUIRE_CUDA=false: Gumbel replay "
                "will use CPU noise that cannot match the prover's; margins "
                "will sit at the model's entropy level, not near zero."
            )
            self._device = torch.device("cpu")

    @property
    def loaded(self) -> bool:
        # vLLM server connection is established on-demand; consider always "ready" if configured
        return True

    async def preload(self) -> None:
        # No-op for server-based verification
        pass

    async def verify(
        self,
        request: VerifyRequest,
        pass_threshold: float,
        margin_clip: float | None = None,
    ) -> VerifyResponse:
        async with self._verify_lock:
            # Per-call margin cap (the ledger model row's delta_max): a
            # ContextVar so concurrent verifies (semaphore > 1) can't clobber
            # each other's cap.
            token = _ACTIVE_MARGIN_CLIP.set(margin_clip)
            try:
                return await self._verify_async(request, pass_threshold)
            finally:
                _ACTIVE_MARGIN_CLIP.reset(token)

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.vllm_url,
                api_key=self.vllm_api_key or "not-used",
            )
        return self._client

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            kwargs: dict[str, Any] = {}
            revision = self.model_config.tokenizer_revision or self.model_config.revision
            if revision:
                kwargs["revision"] = revision
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_config.id, **kwargs)
        return self._tokenizer

    async def _verify_async(self, request: VerifyRequest, pass_threshold: float) -> VerifyResponse:
        started = time.monotonic()
        prompt_token_ids = request.prompt_token_ids or []
        output_token_ids = request.output_token_ids or []
        sampling = request.sampling_config

        log.info(
            "Starting verification request_id=%s model=%s prompt_tokens=%s output_tokens=%s "
            "seed=%s temperature=%s top_k=%s top_p=%s max_output_tokens=%s match_target=%.6f",
            request.request_id,
            request.model.model_id or request.model.name,
            len(prompt_token_ids),
            len(output_token_ids),
            sampling.seed,
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            sampling.max_output_tokens,
            pass_threshold,
        )

        if request.constrained_decoding:
            log.info(
                "Verification unverifiable request_id=%s error_code=%s",
                request.request_id,
                ErrorCode.CONSTRAINED_DECODING.value,
            )
            return self._unverifiable(
                request,
                started,
                ErrorCode.CONSTRAINED_DECODING,
                "Request used constrained decoding (forced tool_choice or "
                "response_format grammar); generation was masked beyond the "
                "declared sampling contract.",
                pass_threshold,
            )

        if (
            request.usage_completion_tokens is not None
            and len(output_token_ids) != request.usage_completion_tokens
        ):
            log.info(
                "Verification unverifiable request_id=%s error_code=%s "
                "captured=%s usage=%s",
                request.request_id,
                ErrorCode.TOKEN_CAPTURE_INCOMPLETE.value,
                len(output_token_ids),
                request.usage_completion_tokens,
            )
            return self._unverifiable(
                request,
                started,
                ErrorCode.TOKEN_CAPTURE_INCOMPLETE,
                f"Tap captured {len(output_token_ids)} output token ids but the "
                f"provider reported generating {request.usage_completion_tokens}; "
                "the capture is incomplete and cannot be replayed.",
                pass_threshold,
            )

        cfg_missing = [
            name
            for name, value in {
                "seed": sampling.seed,
                "temperature": sampling.temperature,
                "top_k": sampling.top_k,
                "top_p": sampling.top_p,
                "max_output_tokens": sampling.max_output_tokens,
            }.items()
            if value is None
        ]
        if cfg_missing:
            log.info(
                "Verification unverifiable request_id=%s error_code=%s missing=%s",
                request.request_id,
                ErrorCode.SAMPLING_CONFIG_MISSING.value,
                ",".join(cfg_missing),
            )
            return self._unverifiable(
                request,
                started,
                ErrorCode.SAMPLING_CONFIG_MISSING,
                f"Missing sampling configuration: {', '.join(cfg_missing)}",
                pass_threshold,
            )

        if not prompt_token_ids or not output_token_ids:
            log.info(
                "Verification unverifiable request_id=%s error_code=%s prompt_tokens=%s output_tokens=%s",
                request.request_id,
                ErrorCode.TOKENIZATION_MISMATCH.value,
                len(prompt_token_ids),
                len(output_token_ids),
            )
            return self._unverifiable(
                request,
                started,
                ErrorCode.TOKENIZATION_MISMATCH,
                "Verifier requires exact prompt and output token ID sequences.",
                pass_threshold,
            )

        try:
            token_metrics, vllm_call_started_at, vllm_call_completed_at = (
                await self._verify_token_sequence(
                    TokenSequence(
                        prompt_token_ids=prompt_token_ids,
                        output_token_ids=output_token_ids,
                    ),
                    temperature=float(sampling.temperature),
                    top_k=int(sampling.top_k),
                    top_p=float(sampling.top_p),
                    seed=int(sampling.seed),
                )
            )
        except Exception as exc:
            latency_ms = int((time.monotonic() - started) * 1000)
            log.exception(
                "Verification error request_id=%s error_code=%s latency_ms=%s",
                request.request_id,
                ErrorCode.INTERNAL_ERROR.value,
                latency_ms,
            )
            return VerifyResponse(
                request_id=request.request_id,
                status=VerificationStatus.UNVERIFIABLE,
                reason=f"Verification error: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                match_target=pass_threshold,
                metrics=_empty_metrics(len(prompt_token_ids), len(output_token_ids)),
                latency_ms=latency_ms,
                verifier_model_id=self.model_config.id,
            )

        summary = self._summarize(token_metrics)
        output_artifacts = self._output_artifacts(output_token_ids, token_metrics)
        status = (
            VerificationStatus.PASS
            if summary.mean_margin is not None and summary.mean_margin < pass_threshold
            else VerificationStatus.FAIL
        )
        reason = (
            "All tokens matched deterministic vLLM Gumbel-Max verification."
            if status == VerificationStatus.PASS
            else (
                "Generated token sequence did not satisfy the configured "
                f"logit difference threshold {pass_threshold:.3f}."
            )
        )
        latency_ms = int((time.monotonic() - started) * 1000)
        log.info(
            "Verification completed request_id=%s status=%s match_level=%.6f match_target=%.6f "
            "exact_match_count=%s token_count=%s latency_ms=%s min_probability=%s "
            "mean_probability=%s mean_margin=%s max_margin=%s max_logit_rank=%s max_gumbel_rank=%s",
            request.request_id,
            status.value,
            summary.exact_match_ratio,
            pass_threshold,
            summary.exact_match_count,
            summary.token_count,
            latency_ms,
            summary.min_probability,
            summary.mean_probability,
            summary.mean_margin,
            summary.max_margin,
            summary.max_logit_rank,
            summary.max_gumbel_rank,
        )
        return VerifyResponse(
            request_id=request.request_id,
            status=status,
            reason=reason,
            match_target=pass_threshold,
            metrics=summary,
            latency_ms=latency_ms,
            vllm_call_started_at=vllm_call_started_at,
            vllm_call_completed_at=vllm_call_completed_at,
            verifier_model_id=self.model_config.id,
            prover_output=output_artifacts["prover_output"],
            verifier_output=output_artifacts["verifier_output"],
            prover_output_token_ids=output_artifacts["prover_output_token_ids"],
            verifier_output_token_ids=output_artifacts["verifier_output_token_ids"],
            output_token_comparison=output_artifacts["output_token_comparison"],
            raw={"token_metrics": [asdict(item) for item in token_metrics]},
        )

    async def _verify_token_sequence(
        self,
        sequence: TokenSequence,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int,
    ) -> tuple[list[SimpleTokenMetrics], datetime, datetime]:
        gen_ids = sequence.output_token_ids
        prompt_len = len(sequence.prompt_token_ids)
        prompt_and_output = sequence.prompt_token_ids + gen_ids

        if top_k > self.model_config.max_logprobs:
            raise ValueError(
                f"top_k={top_k} requires prompt_logprobs={top_k}, but verifier model config "
                f"allows max_logprobs={self.model_config.max_logprobs}."
            )

        # Call vLLM's OpenAI-compatible API with token IDs and request prompt
        # logprobs for the concatenated prompt + output token sequence.
        client = self._get_client()
        log.info(
            "Requesting vLLM prompt_logprobs model=%s prompt_tokens=%s output_tokens=%s top_k=%s top_p=%s temperature=%s seed=%s",
            self.model_config.id,
            prompt_len,
            len(gen_ids),
            top_k,
            top_p,
            temperature,
            seed,
        )
        vllm_call_started_at = datetime.now(timezone.utc)
        response = await client.completions.create(
            model=self.model_config.id,
            prompt=prompt_and_output,
            max_tokens=1,
            temperature=temperature,
            top_p=top_p,
            extra_body={
                "prompt_logprobs": top_k,
                "top_k": top_k,
            },
        )
        vllm_call_completed_at = datetime.now(timezone.utc)

        if not response.choices:
            raise ValueError("vLLM did not return completion choices for verification.")

        choice = response.choices[0]
        logprobs_data = getattr(choice, "prompt_logprobs", None)
        if logprobs_data is None and getattr(choice, "model_extra", None):
            logprobs_data = choice.model_extra.get("prompt_logprobs")
        if logprobs_data is None and hasattr(choice, "model_dump"):
            logprobs_data = choice.model_dump().get("prompt_logprobs")
        if logprobs_data is None and getattr(response, "model_extra", None):
            logprobs_data = response.model_extra.get("prompt_logprobs")
        if not logprobs_data:
            response_keys = list(response.model_dump().keys()) if hasattr(response, "model_dump") else []
            choice_keys = list(choice.model_dump().keys()) if hasattr(choice, "model_dump") else []
            choice_extra_keys = list(getattr(choice, "model_extra", {}) or {})
            raise ValueError(
                "vLLM did not return prompt_logprobs for verification "
                f"(response_keys={response_keys}, choice_keys={choice_keys}, choice_extra_keys={choice_extra_keys})."
            )

        token_logprobs = logprobs_data[prompt_len : prompt_len + len(gen_ids)]
        if len(token_logprobs) != len(gen_ids):
            raise ValueError(
                f"Expected {len(gen_ids)} generated-token logprob rows, got {len(token_logprobs)}."
            )
        log.info(
            "Received vLLM prompt_logprobs model=%s total_rows=%s generated_rows=%s",
            self.model_config.id,
            len(logprobs_data),
            len(token_logprobs),
        )

        def _token_id(value: Any) -> int:
            return int(value)

        def _logprob(value: Any) -> float:
            if isinstance(value, dict):
                return float(value.get("logprob"))
            return float(getattr(value, "logprob"))

        all_token_ids = set(gen_ids)
        for row in token_logprobs:
            if row:
                all_token_ids.update(_token_id(token_id) for token_id in row.keys())
        # The noise tensor must have EXACTLY the serving vLLM's logits width
        # (the HF config's padded vocab_size): the replay consumes one full
        # generator row per token, so any other width (max(token_id)+1,
        # len(tokenizer)) desyncs the RNG stream after the first token.
        vocab_size = self._resolve_vocab_size(observed_max_id=max(all_token_ids))

        device = self._device

        logits_jv = torch.full((len(gen_ids), vocab_size), float("-inf"), device=device, dtype=torch.float32)
        for pos, row in enumerate(token_logprobs):
            if row:
                for token_id, logprob in row.items():
                    logits_jv[pos, _token_id(token_id)] = _logprob(logprob)

        token_ids_j = torch.as_tensor(gen_ids, device=device, dtype=torch.long)
        rows_j = torch.arange(len(gen_ids), device=device)
        top_k_j = torch.full((len(gen_ids),), top_k, device=device, dtype=torch.long)
        top_p_j = torch.full((len(gen_ids),), top_p, device=device, dtype=logits_jv.dtype)

        probs_jv = get_probs(logits_jv, temperature, top_k_j, top_p_j)
        gold_logits_j = logits_jv[rows_j, token_ids_j]
        logit_ranks_j = (logits_jv > gold_logits_j.unsqueeze(1)).sum(dim=1).float()
        probs_gold_j = probs_jv.gather(1, token_ids_j.view(-1, 1)).squeeze(1)

        pred_ids_j, gumbel_ranks_j, margins_j = verify_vllm_gumbel_max(
            temperature=temperature,
            seed=seed,
            logits_JV=logits_jv,
            probs_JV=probs_jv,
            gold_col_idx_J=token_ids_j,
            top_k_tensor_J=top_k_j,
            top_p_tensor_J=top_p_j,
        )

        metrics = []
        for idx, actual_id in enumerate(gen_ids):
            predicted_id = int(pred_ids_j[idx])
            metrics.append(
                SimpleTokenMetrics(
                    exact_match=bool(predicted_id == int(actual_id)),
                    prob=float(probs_gold_j[idx].item()),
                    margin=self._clip_margin(float(margins_j[idx].item())),
                    logit_rank=float(logit_ranks_j[idx].item()),
                    gumbel_rank=float(gumbel_ranks_j[idx].item()),
                    actual_token_id=int(actual_id),
                    predicted_token_id=predicted_id,
                )
            )
        return metrics, vllm_call_started_at, vllm_call_completed_at

    def _resolve_vocab_size(self, observed_max_id: int) -> int:
        """Width of the serving vLLM's logits tensor.

        Prefers the explicit model-config value; otherwise resolved once from
        the model's HF config (config.json `vocab_size`, the padded width the
        model — and therefore vLLM's noise rows — actually uses). A token id
        at or beyond that width means the configured width is wrong; growing
        the tensor silently (the old max(token_id)+1 behavior) would desync
        the Gumbel replay, so it is an error instead.
        """
        if self._vocab_size is None:
            hf_config = AutoConfig.from_pretrained(
                self.model_config.id, revision=self.model_config.revision
            )
            self._vocab_size = int(hf_config.vocab_size)
            log.info(
                "Resolved vocab_size=%s for model=%s from HF config",
                self._vocab_size,
                self.model_config.id,
            )
        if observed_max_id >= self._vocab_size:
            raise ValueError(
                f"Observed token id {observed_max_id} is outside the configured "
                f"vocab width {self._vocab_size} for {self.model_config.id}; the "
                "vocab_size config (or the model's HF config) does not match the "
                "serving model."
            )
        return self._vocab_size

    def _summarize(self, token_metrics: list[SimpleTokenMetrics]) -> TokenMetricSummary:
        if not token_metrics:
            return _empty_metrics()

        probs = [item.prob for item in token_metrics if math.isfinite(item.prob)]
        margins = [self._clip_margin(item.margin) for item in token_metrics]
        logit_ranks = [item.logit_rank for item in token_metrics if math.isfinite(item.logit_rank)]
        gumbel_ranks = [item.gumbel_rank for item in token_metrics if math.isfinite(item.gumbel_rank)]
        exact_count = sum(1 for item in token_metrics if item.exact_match)

        return TokenMetricSummary(
            token_count=len(token_metrics),
            exact_match_count=exact_count,
            exact_match_ratio=exact_count / len(token_metrics),
            margins=margins,
            min_probability=min(probs) if probs else None,
            mean_probability=statistics.fmean(probs) if probs else None,
            mean_margin=statistics.fmean(margins) if margins else None,
            max_margin=max(margins) if margins else None,
            max_logit_rank=max(logit_ranks) if logit_ranks else None,
            max_gumbel_rank=max(gumbel_ranks) if gumbel_ranks else None,
        )

    def _clip_margin(self, margin: float) -> float:
        clip = _ACTIVE_MARGIN_CLIP.get()
        if clip is None or not math.isfinite(clip) or clip <= 0:
            clip = self.margin_clip
        if not math.isfinite(margin):
            return clip
        return min(margin, clip)

    def _output_artifacts(
        self,
        prover_token_ids: list[int],
        token_metrics: list[SimpleTokenMetrics],
    ) -> dict[str, Any]:
        verifier_token_ids = [
            int(item.predicted_token_id)
            for item in token_metrics
            if item.predicted_token_id is not None
        ]
        if len(verifier_token_ids) != len(prover_token_ids):
            verifier_token_ids = []

        try:
            prover_text = self._decode_token_ids(prover_token_ids)
            verifier_text = self._decode_token_ids(verifier_token_ids) if verifier_token_ids else None
            prover_pieces = self._decode_token_pieces(prover_token_ids)
            verifier_pieces = self._decode_token_pieces(verifier_token_ids) if verifier_token_ids else []
        except Exception:
            log.exception(
                "Failed to decode verifier comparison tokens model=%s token_count=%s",
                self.model_config.id,
                len(prover_token_ids),
            )
            prover_text = " ".join(str(token_id) for token_id in prover_token_ids)
            verifier_text = " ".join(str(token_id) for token_id in verifier_token_ids) if verifier_token_ids else None
            prover_pieces = [str(token_id) for token_id in prover_token_ids]
            verifier_pieces = [str(token_id) for token_id in verifier_token_ids]

        comparison: list[OutputTokenComparison] = []
        for index, metric in enumerate(token_metrics):
            if index >= len(prover_token_ids) or index >= len(verifier_token_ids):
                break
            comparison.append(
                OutputTokenComparison(
                    index=index,
                    prover_token_id=int(prover_token_ids[index]),
                    verifier_token_id=int(verifier_token_ids[index]),
                    prover_text=prover_pieces[index] if index < len(prover_pieces) else str(prover_token_ids[index]),
                    verifier_text=(
                        verifier_pieces[index] if index < len(verifier_pieces) else str(verifier_token_ids[index])
                    ),
                    exact_match=bool(metric.exact_match),
                    margin=self._clip_margin(metric.margin) if math.isfinite(metric.margin) else None,
                )
            )

        return {
            "prover_output": prover_text,
            "verifier_output": verifier_text,
            "prover_output_token_ids": prover_token_ids,
            "verifier_output_token_ids": verifier_token_ids,
            "output_token_comparison": comparison,
        }

    def _decode_token_ids(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        tokenizer = self._get_tokenizer()
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _decode_token_pieces(self, token_ids: list[int]) -> list[str]:
        tokenizer = self._get_tokenizer()
        return [
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for token_id in token_ids
        ]

    def _unverifiable(
        self,
        request: VerifyRequest,
        started: float,
        code: ErrorCode,
        reason: str,
        match_target: float,
    ) -> VerifyResponse:
        latency_ms = int((time.monotonic() - started) * 1000)
        log.info(
            "Verification completed request_id=%s status=%s error_code=%s match_level=0.000000 "
            "match_target=%.6f latency_ms=%s reason=%s",
            request.request_id,
            VerificationStatus.UNVERIFIABLE.value,
            code.value,
            match_target,
            latency_ms,
            reason,
        )
        return VerifyResponse(
            request_id=request.request_id,
            status=VerificationStatus.UNVERIFIABLE,
            reason=reason,
            error_code=code,
            match_target=match_target,
            metrics=_empty_metrics(
                len(request.prompt_token_ids or []),
                len(request.output_token_ids or []),
            ),
            latency_ms=latency_ms,
            verifier_model_id=self.model_config.id,
        )


class VerificationService:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Single-model verifier: one config, one cached verifier. The pass
        # threshold is not held here: it lives on the ledger's model row and
        # is passed into verify() per call.
        self.model_config = settings.model
        self._verifier = CachedModelVerifier(
            settings.model,
            vllm_url=settings.vllm_url,
            vllm_api_key=settings.vllm_api_key,
            concurrency=max(1, settings.max_concurrent_verifications_per_model),
            margin_clip=settings.margin_clip,
            require_cuda=settings.require_cuda,
        )

    async def startup(self) -> None:
        if self.settings.preload_models:
            await self._verifier.preload()

    def list_models(self) -> list[ModelInfo]:
        model = self.model_config
        return [
            ModelInfo(
                id=model.id,
                display_name=model.display_name,
                revision=model.revision,
                tokenizer_revision=model.tokenizer_revision,
                dtype=model.dtype,
                quantization=model.quantization,
                ready=self._verifier.loaded,
                max_model_len=model.max_model_len,
                max_logprobs=model.max_logprobs,
            )
        ]

    async def verify(
        self,
        request: VerifyRequest,
        pass_threshold: float,
        margin_clip: float | None = None,
    ) -> VerifyResponse:
        started = time.monotonic()
        model_config = self._resolve_model(request.model.model_id or request.model.name)
        if model_config is None:
            latency_ms = int((time.monotonic() - started) * 1000)
            log.info(
                "Verification rejected request_id=%s status=%s error_code=%s model=%s match_target=%.6f latency_ms=%s",
                request.request_id,
                VerificationStatus.UNVERIFIABLE.value,
                ErrorCode.UNSUPPORTED_MODEL.value,
                request.model.model_id or request.model.name,
                pass_threshold,
                latency_ms,
            )
            return VerifyResponse(
                request_id=request.request_id,
                status=VerificationStatus.UNVERIFIABLE,
                reason=f"Model is not configured on verifier: {request.model.model_id or request.model.name}",
                error_code=ErrorCode.UNSUPPORTED_MODEL,
                match_target=pass_threshold,
                metrics=_empty_metrics(
                    len(request.prompt_token_ids or []),
                    len(request.output_token_ids or []),
                ),
                latency_ms=latency_ms,
                verifier_model_id=None,
            )

        try:
            return await asyncio.wait_for(
                self._verifier.verify(request, pass_threshold, margin_clip),
                timeout=self.settings.verification_timeout_seconds,
            )
        except asyncio.TimeoutError:
            latency_ms = int((time.monotonic() - started) * 1000)
            log.exception(
                "Verification timed out request_id=%s model=%s timeout_seconds=%s match_target=%.6f latency_ms=%s",
                request.request_id,
                model_config.id,
                self.settings.verification_timeout_seconds,
                pass_threshold,
                latency_ms,
            )
            return VerifyResponse(
                request_id=request.request_id,
                status=VerificationStatus.UNVERIFIABLE,
                reason="Verification timed out.",
                error_code=ErrorCode.VERIFICATION_TIMEOUT,
                match_target=pass_threshold,
                metrics=_empty_metrics(
                    len(request.prompt_token_ids or []),
                    len(request.output_token_ids or []),
                ),
                latency_ms=latency_ms,
                verifier_model_id=model_config.id,
            )
        except Exception as exc:
            latency_ms = int((time.monotonic() - started) * 1000)
            log.exception(
                "Verifier internal error request_id=%s model=%s match_target=%.6f latency_ms=%s",
                request.request_id,
                model_config.id,
                pass_threshold,
                latency_ms,
            )
            return VerifyResponse(
                request_id=request.request_id,
                status=VerificationStatus.UNVERIFIABLE,
                reason=f"Verifier internal error: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                match_target=pass_threshold,
                metrics=_empty_metrics(
                    len(request.prompt_token_ids or []),
                    len(request.output_token_ids or []),
                ),
                latency_ms=latency_ms,
                verifier_model_id=model_config.id,
            )

    def _resolve_model(self, model_id_or_name: str) -> VerifierModelConfig | None:
        if model_id_or_name in (self.model_config.id, self.model_config.display_name):
            return self.model_config
        return None
