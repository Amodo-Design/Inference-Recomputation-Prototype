"""Gumbel-Max verification core.

Pure-torch building blocks for verifying sampled tokens against a model's
logprobs. This module intentionally depends on nothing but torch so that
services (e.g. inf-ver-api) can use it without installing the full difr
research stack (vllm, datasets, transformers, ...).

The heavyweight generation/verification drivers that need a local vLLM
instance live in token_difr_vllm.py, which re-exports these symbols.
"""

from dataclasses import dataclass

import torch


@dataclass
class TokenSequence:
    """A prompt and its generated output as token IDs."""

    prompt_token_ids: list[int]
    output_token_ids: list[int]


@dataclass
class SimpleTokenMetrics:
    exact_match: bool
    prob: float
    margin: float
    logit_rank: float
    gumbel_rank: float
    actual_token_id: int | None = None
    predicted_token_id: int | None = None


def exponential_to_gumbel(random_exponentials: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Convert exponential noise E ~ Exp(1) to Gumbel noise G = -log(E).

    Args:
        random_exponentials: Tensor of exponential random variables E ~ Exp(1)
        epsilon: Small constant to prevent log(0)

    Returns:
        Gumbel noise tensor with same shape as input
    """
    return -torch.log(random_exponentials.clamp(min=epsilon))


def apply_top_k_only(
    logits: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.

    The logits tensor may be updated in-place.

    NOTE: this is directly copy pasted from vllm: https://github.com/vllm-project/vllm/blob/10d765482d19abfab6c66b5f815720a66aa9de42/vllm/v1/sample/ops/topk_topp_sampler.py#L164
    They use 2D.
    """

    # probably not necessary, keeping it for now.
    assert len(logits.shape) == 2
    assert k.shape[0] == logits.shape[0], f"k.shape: {k.shape}, logits.shape: {logits.shape}"

    no_top_k_mask = k == logits.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = int(k.max().item())
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits.masked_fill_(logits < top_k_mask, -float("inf"))
    return logits


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.

    NOTE: this is directly copy pasted from vllm: https://github.com/vllm-project/vllm/blob/10d765482d19abfab6c66b5f815720a66aa9de42/vllm/v1/sample/ops/topk_topp_sampler.py#L164
    They use 2D.
    """
    if p is None:
        if k is None:
            return logits

        # Avoid sorting vocab for top-k only case.
        return apply_top_k_only(logits, k)

    # probably not necessary, keeping it for now.
    assert len(logits.shape) == 2

    if k is not None:
        assert k.shape[0] == logits.shape[0], f"k.shape: {k.shape}, logits.shape: {logits.shape}"
    if p is not None:
        assert p.shape[0] == logits.shape[0], f"p.shape: {p.shape}, logits.shape: {logits.shape}"

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None and (k > 0).all():
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits


def keep_one_token(scores: torch.Tensor, tok_idx: torch.Tensor) -> torch.Tensor:
    """
    Keep exactly one token per row along the last dimension.

    Args:
        scores: shape (..., V) - logits/scores tensor
        tok_idx: shape (...) - must match scores.shape[:-1]

    Returns:
        shape (..., V) with all -inf except at chosen indices
    """
    # Simple rule: tok_idx shape must match all dims except last
    assert tok_idx.shape == scores.shape[:-1], (
        f"tok_idx.shape {tok_idx.shape} must match scores.shape[:-1] {scores.shape[:-1]}"
    )
    out = torch.full_like(scores, float("-inf"))

    idx = tok_idx.unsqueeze(-1)

    values = torch.gather(scores, dim=-1, index=idx)
    out.scatter_(dim=-1, index=idx, src=values)

    return out


def get_probs(logits: torch.Tensor, temperature: float, top_k: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """
    logits: shape [..., V]
    returns: probabilities with same shape, normalized along the last dim
    """

    assert len(logits.shape) == 2, f"Expected 2D logits, got shape {logits.shape}"

    if temperature > 0.0:
        x = logits / max(temperature, 1e-8)
    else:
        # greedy: pick argmax per row
        idx = torch.argmax(logits, dim=-1)
        x = keep_one_token(logits, idx)

    x = apply_top_k_top_p(x, top_k, top_p)
    probs = torch.nn.functional.softmax(x, dim=-1, dtype=torch.float32)
    return probs


def compute_margin_batch(
    logits_JV: torch.Tensor,
    random_exponentials_JV: torch.Tensor,
    neg_inf_mask_JV: torch.Tensor,
    temperature: float,
    gold_idx_J: torch.Tensor,
) -> torch.Tensor:
    """
    Compute max - gold margins for a batch where gold_idx_J indexes logits_JV.
    """
    assert logits_JV.dim() == 2, f"Expected [J, V] logits, got {logits_JV.shape}"
    J, V = logits_JV.shape

    # Add Gumbel noise to logits and re-apply mask
    # epsilon is 0 because torch.exponential_() will not generate 0
    random_gumbels_JV = exponential_to_gumbel(random_exponentials_JV.float(), epsilon=0)
    noised_logits_JV = logits_JV + (random_gumbels_JV * temperature)
    noised_logits_JV[neg_inf_mask_JV] = float("-inf")

    max_idx_J = noised_logits_JV.argmax(dim=-1)  # [J]
    row_J = torch.arange(J, device=logits_JV.device)
    max_vals_J = noised_logits_JV[row_J, max_idx_J]
    gold_vals_J = noised_logits_JV[row_J, gold_idx_J]
    logit_diff_J = max_vals_J - gold_vals_J

    return logit_diff_J


def verify_vllm_gumbel_max(
    temperature: float,
    seed: int,
    logits_JV: torch.Tensor,
    probs_JV: torch.Tensor,
    gold_col_idx_J: torch.Tensor,
    top_k_tensor_J: torch.Tensor,
    top_p_tensor_J: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Verify the outputs against vLLM's Gumbel-Max sampling.
    """

    filtered_logits_JV = logits_JV.clone()

    # vllm scales logits by 1/temperature before top-k/top-p, so we do it here too
    # note: We do NOT scale the real logits by temperature - otherwise the noise would vary significantly
    # with temperature as well
    if temperature > 0.0:
        filtered_logits_JV = filtered_logits_JV / max(temperature, 1e-8)

    # Apply top-k/top-p to build the mask; requires per-row k/p tensors.
    filtered_logits_JV = apply_top_k_top_p(filtered_logits_JV, top_k_tensor_J, top_p_tensor_J)
    neg_inf_mask_JV = ~torch.isfinite(filtered_logits_JV)

    # Create per-request generator for verification
    generator = torch.Generator(device=logits_JV.device)

    J = logits_JV.shape[0]
    row_idx_J = torch.arange(J, device=logits_JV.device)

    # Sample all Exponential(1) noises with consistent generator usage
    # Must be done row-by-row to match vLLM's token-by-token RNG order
    generator.manual_seed(seed)
    exponential_rows = []
    for _ in range(J):
        exp_v = torch.empty_like(probs_JV[0])
        exp_v.exponential_(generator=generator)
        exponential_rows.append(exp_v)
    random_exponentials_JV = torch.stack(exponential_rows, dim=0)

    gumbel_max_scores_JV = probs_JV / random_exponentials_JV
    # With full vocab, column index is the token ID
    pred_ids_J = gumbel_max_scores_JV.argmax(dim=-1)

    # Track whether the gold token was removed by top-k/top-p filtering.
    gold_filtered_J = ~torch.isfinite(filtered_logits_JV[row_idx_J, gold_col_idx_J])

    # Rank of the gold token in the Gumbel-Max scores (0 = highest score).
    gold_gumbel_scores_J = gumbel_max_scores_JV[row_idx_J, gold_col_idx_J]
    gumbel_ranks_J = torch.full((J,), float("inf"), device=logits_JV.device)
    valid_mask_J = ~gold_filtered_J
    if valid_mask_J.any():
        higher_scores_counts = (
            gumbel_max_scores_JV[valid_mask_J] > gold_gumbel_scores_J[valid_mask_J].unsqueeze(1)
        ).sum(dim=1)
        gumbel_ranks_J[valid_mask_J] = higher_scores_counts.float()

    margins_J = compute_margin_batch(
        logits_JV,
        random_exponentials_JV,
        neg_inf_mask_JV=neg_inf_mask_JV,
        temperature=temperature,
        gold_idx_J=gold_col_idx_J,
    )

    return pred_ids_J, gumbel_ranks_J, margins_J
