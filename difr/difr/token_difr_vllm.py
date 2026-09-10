import os

# Enable vLLM v1 features (prompt logprobs)
os.environ["VLLM_USE_V1"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
)
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

# Verification math lives in the torch-only gumbel_verify module so that
# services can use it without the vllm/datasets stack. Re-exported here to
# keep existing imports working. The except branch covers running these
# modules script-style (python token_difr_vllm.py) where the difr package
# root is not on sys.path.
try:
    from difr.gumbel_verify import (  # noqa: F401
        SimpleTokenMetrics,
        TokenSequence,
        apply_top_k_only,
        apply_top_k_top_p,
        compute_margin_batch,
        exponential_to_gumbel,
        get_probs,
        keep_one_token,
        verify_vllm_gumbel_max,
    )
except ImportError:
    from gumbel_verify import (  # noqa: F401
        SimpleTokenMetrics,
        TokenSequence,
        apply_top_k_only,
        apply_top_k_top_p,
        compute_margin_batch,
        exponential_to_gumbel,
        get_probs,
        keep_one_token,
        verify_vllm_gumbel_max,
    )


@dataclass
class AttestationConfig:
    seed: int = 42
    trusted_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_decode_tokens: int = 512
    n_samples: int = 100
    max_ctx_len: int = 512
    vllm_args: dict[str, Any] = field(default_factory=dict)

    dataset_name: str = "lmsys/lmsys-chat-1m"
    dataset_split: str = "train"
    dataset_config: str | None = None

    # Sampling parameters
    sampling_temperature: float = 1.0
    verification_temperature: float = 1.0
    top_k: int = 50
    sampling_top_p: float = 0.95
    verification_top_p: float = 0.95
    sampling_seed: int = 42
    verification_seed: int = 42

    external_prompt_filename: str | None = None

    save_dir: str = "token_difr_attestation_results"
    save_filename: str = "attestation_results.json"


def _as_list(x) -> list[int]:
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, tuple):
        return list(x)
    return list(x)


def _prompt_logprobs_to_tensor(
    prompt_logprobs: list[dict[int, Any] | None],
    start_idx: int,
    n_tokens: int,
    device: torch.device,
    vocab_size: int,
) -> torch.Tensor:
    """
    Convert a slice of prompt_logprobs into a dense full-vocabulary tensor.

    Args:
        prompt_logprobs: Full prompt_logprobs list from vLLM RequestOutput.
        start_idx: Starting index (inclusive) of the slice to convert.
        n_tokens: Number of tokens to convert starting at start_idx.
        device: Target torch device.
        vocab_size: Full vocabulary size. Positions not in top-k are set to -inf.

    Returns:
        logits_JV: Tensor of shape [J, vocab_size] with logprobs/logits.
    """

    slice_rows = prompt_logprobs[start_idx : start_idx + n_tokens]
    if len(slice_rows) != n_tokens:
        raise ValueError(f"Expected {n_tokens} prompt logprob rows, got {len(slice_rows)}")

    # Initialize full vocab tensor with -inf
    logits = torch.full((n_tokens, vocab_size), float("-inf"), device=device)

    for j, row in enumerate(slice_rows):
        if not row:
            continue

        token_ids = torch.tensor([int(tok_id) for tok_id in row.keys()], device=device, dtype=torch.long)
        logprobs = torch.tensor([float(val.logprob) for val in row.values()], device=device)

        # Scatter logprobs into the correct positions
        logits[j].scatter_(0, token_ids, logprobs)

    return logits


def generate_outputs_vllm(
    cfg: AttestationConfig,
    prompt_token_ids: list[list[int]],
    dtype: torch.dtype,
) -> list[TokenSequence]:
    vllm_args = dict(cfg.vllm_args)

    requested_tensor_parallel = vllm_args.pop("tensor_parallel_size", None)
    available_gpus = torch.cuda.device_count()
    tensor_parallel_size = max(1, available_gpus)

    if requested_tensor_parallel is not None:
        if requested_tensor_parallel < 1:
            raise ValueError(f"tensor_parallel_size must be >= 1, got {requested_tensor_parallel}")
        if available_gpus and requested_tensor_parallel > available_gpus:
            raise ValueError(
                f"tensor_parallel_size {requested_tensor_parallel} exceeds available GPUs ({available_gpus})"
            )
        tensor_parallel_size = requested_tensor_parallel

    model = LLM(
        model=cfg.trusted_model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=(cfg.max_ctx_len + cfg.max_decode_tokens) * 2,
        enforce_eager=True,
        dtype=dtype,  # type: ignore[arg-type]
        **vllm_args,
    )

    # Use vLLM's native Gumbel-Max sampling by passing seed directly
    sampling_params = SamplingParams(
        temperature=cfg.sampling_temperature,
        max_tokens=cfg.max_decode_tokens + 1,
        top_k=cfg.top_k,
        top_p=cfg.sampling_top_p,
        seed=cfg.sampling_seed,  # This enables deterministic Gumbel-Max sampling
    )

    token_prompts: list[TokensPrompt] = [{"prompt_token_ids": ids} for ids in prompt_token_ids]

    vllm_outputs = model.generate(token_prompts, sampling_params=sampling_params)

    # Convert vLLM outputs to TokenSequence
    outputs = [
        TokenSequence(
            prompt_token_ids=_as_list(req.prompt_token_ids),
            output_token_ids=_as_list(req.outputs[0].token_ids),
        )
        for req in vllm_outputs
    ]

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return outputs


@torch.inference_mode()
def verify_outputs(
    outputs: list[TokenSequence],
    cfg: AttestationConfig,
    dtype: torch.dtype,
) -> list[list[SimpleTokenMetrics]]:
    """
    Verifies outputs using Gumbel-Max verification.

    Returns: list[list[TokenMetrics]]
    """

    model = LLM(
        model=cfg.trusted_model_name,
        tensor_parallel_size=1,
        max_model_len=(cfg.max_ctx_len + cfg.max_decode_tokens) * 2,
        enforce_eager=True,
        dtype=dtype,  # type: ignore[arg-type]
        gpu_memory_utilization=0.7,
        logprobs_mode="raw_logits",
        max_logprobs=cfg.top_k,
    )

    # Get vocab size from the model's tokenizer (use len() to include added tokens)
    tokenizer = model.get_tokenizer()
    vocab_size = len(tokenizer)

    prompt_logprob_params = SamplingParams(
        prompt_logprobs=cfg.top_k,
        max_tokens=1,
        logprobs=0,
        detokenize=False,
    )

    device_for_inputs = torch.device("cuda")

    # === First pass: build all prompts for batched generation ===
    verify_prompts: list[TokensPrompt] = []
    request_metadata: list[tuple[int, int, list[int]]] = []  # (original_idx, prompt_len, gen_ids)

    for i, req in enumerate(tqdm(outputs, desc="Preparing verification prompts")):
        prompt_token_ids: list[int] = _as_list(req.prompt_token_ids)
        gen_ids: list[int] = _as_list(req.output_token_ids)

        if len(gen_ids) == 0:
            continue

        seq_concat: list[int] = prompt_token_ids + gen_ids
        verify_prompts.append({"prompt_token_ids": seq_concat})
        request_metadata.append((i, len(prompt_token_ids), gen_ids))

    # === Batched generation ===
    print(f"Running batched verification for {len(verify_prompts)} sequences...")
    verify_results = model.generate(verify_prompts, sampling_params=prompt_logprob_params)

    # === Second pass: compute metrics from results ===
    # Pre-initialize with empty lists to maintain positional correspondence
    all_token_metrics: list[list[SimpleTokenMetrics]] = [[] for _ in outputs]

    for batch_idx, (orig_idx, prompt_len, gen_ids) in enumerate(
        tqdm(request_metadata, desc="Computing verification metrics")
    ):
        verify_req = verify_results[batch_idx]
        gen_len = len(gen_ids)

        if verify_req.prompt_logprobs is None:
            raise ValueError("vLLM did not return prompt_logprobs; enable prompt_logprobs in SamplingParams.")

        logits_JV = _prompt_logprobs_to_tensor(
            verify_req.prompt_logprobs,
            start_idx=prompt_len,
            n_tokens=gen_len,
            device=device_for_inputs,
            vocab_size=vocab_size,
        )

        J = logits_JV.shape[0]
        # With full vocab tensor, token ID is the column index
        gold_col_idx_J = torch.as_tensor(gen_ids, device=device_for_inputs, dtype=torch.long)

        top_k_tensor_J = torch.full((J,), cfg.top_k, device=logits_JV.device, dtype=torch.long)
        top_p_tensor_J = torch.full((J,), cfg.verification_top_p, device=logits_JV.device, dtype=logits_JV.dtype)

        logits_JV = logits_JV.float()
        probs_JV = get_probs(logits_JV, cfg.verification_temperature, top_k_tensor_J, top_p_tensor_J)

        # Rank of the gold token in the raw logits (0 = highest logit).
        row_idx_J = torch.arange(J, device=device_for_inputs)
        gold_logits_J = logits_JV[row_idx_J, gold_col_idx_J]
        logit_ranks_J = (logits_JV > gold_logits_J.unsqueeze(1)).sum(dim=1).float()

        probs_gold_J = probs_JV.gather(1, gold_col_idx_J.view(-1, 1)).squeeze(1)

        pred_ids_J, gumbel_ranks_J, margins_J = verify_vllm_gumbel_max(
            temperature=cfg.verification_temperature,
            seed=cfg.verification_seed,
            logits_JV=logits_JV,
            probs_JV=probs_JV,
            gold_col_idx_J=gold_col_idx_J,
            top_k_tensor_J=top_k_tensor_J,
            top_p_tensor_J=top_p_tensor_J,
        )

        seq_token_metrics: list[SimpleTokenMetrics] = []
        for j in range(J):
            actual_id = int(gen_ids[j])

            token_metrics = SimpleTokenMetrics(
                exact_match=bool(int(pred_ids_J[j]) == actual_id),
                prob=float(probs_gold_J[j].item()),
                margin=float(margins_J[j].item()),
                logit_rank=float(logit_ranks_J[j].item()),
                gumbel_rank=float(gumbel_ranks_J[j].item()),
            )
            seq_token_metrics.append(token_metrics)

        all_token_metrics[orig_idx] = seq_token_metrics

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return all_token_metrics


def construct_dataset(cfg: AttestationConfig) -> tuple[list[list[int]], list[str], list[list[dict[str, str]]]]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.trusted_model_name)

    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)

    # prepare raw prompts (chat template -> text -> token ids)

    tokenized_prompts = []
    hf_prompts = []  # used because we need to add padding to prompts with hf
    conversation_prompts = []
    unique_prompts = set()

    system_prompt = []

    count = 0
    while len(tokenized_prompts) < cfg.n_samples:
        raw_prompt = ds[count]["conversation"]  # type: ignore[index]

        # Check language before processing
        # do english so I can read the results
        if ds[count]["language"].lower() != "english":  # type: ignore[index]
            count += 1
            continue

        count += 1

        # Only include conversations that end with a user message
        # If it ends with assistant, remove the last assistant message(s) to ensure
        # the model has something to respond to
        conversation = list(raw_prompt)  # Convert to list to allow modification
        conversation = system_prompt + conversation
        while conversation and conversation[-1].get("role") == "assistant":
            conversation = conversation[:-1]

        # Skip if conversation is empty or doesn't end with user
        if not conversation or conversation[-1].get("role") != "user":
            continue

        rendered_prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        tokenized_prompt = tokenizer.encode(rendered_prompt, add_special_tokens=False, return_tensors=None)
        if len(tokenized_prompt) <= cfg.max_ctx_len:
            if tuple(tokenized_prompt) not in unique_prompts:
                unique_prompts.add(tuple(tokenized_prompt))
                tokenized_prompts.append(tokenized_prompt)
                hf_prompts.append(rendered_prompt)
                conversation_prompts.append(conversation)

    del tokenizer

    return tokenized_prompts, hf_prompts, conversation_prompts


def load_external_tokens(data_path: str) -> list[TokenSequence]:
    """Load external tokenized data from JSON file.

    Supports the following format:
        {"samples": [{"prompt_token_ids": [...], "output_token_ids": [...]}, ...]}


    Args:
        data_path: Path to the JSON file containing tokenized data

    Returns:
        List of TokenSequence objects
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "samples" in data, "Samples not found in data"

    outputs = []
    for sample in data["samples"]:
        assert "prompt_token_ids" in sample, "prompt_token_ids not found in sample"

        output_ids = sample["output_token_ids"]

        outputs.append(
            TokenSequence(
                prompt_token_ids=sample["prompt_token_ids"],
                output_token_ids=output_ids,
            )
        )

    return outputs


def run(
    cfg: AttestationConfig,
    external_responses: list[TokenSequence] | None = None,
) -> None:
    dtype = torch.bfloat16
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    save_path = Path(cfg.save_dir) / cfg.save_filename

    if save_path.exists():
        print(f"Skipping {save_path} because it already exists")
        return

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    if external_responses is None:
        tokenized_prompts, hf_prompts, conversation_prompts = construct_dataset(cfg)

        untrusted_outputs = generate_outputs_vllm(
            cfg,
            tokenized_prompts,
            dtype,
        )
    else:
        untrusted_outputs = external_responses

    verifier_scores = verify_outputs(untrusted_outputs, cfg, dtype)

    # Convert dataclasses to dicts for JSON serialization
    scores_as_dicts = [[asdict(token) for token in seq] for seq in verifier_scores]

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scores": scores_as_dicts,
                "config": asdict(cfg),
            },
            f,
        )


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-8B"
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model_name_str = model_name.replace("/", "_").replace(".", "_")
    cfgs = []

    # 1. Correct configuration (baseline)
    cfgs.append(
        AttestationConfig(
            save_filename=f"verification_{model_name_str}_vllm_bf16.json",
            trusted_model_name=model_name,
        )
    )

    # cfgs.append(
    #     AttestationConfig(
    #         save_filename=f"verification_{model_name_str}_vllm_fp8.json",
    #         trusted_model_name=model_name,
    #         vllm_args={"quantization": "fp8"},
    #     )
    # )

    # cfgs.append(
    #     AttestationConfig(
    #         save_filename=f"verification_{model_name_str}_vllm_4bit.json",
    #         trusted_model_name=model_name,
    #         vllm_args={"quantization": "bitsandbytes"},
    #     )
    # )

    # cfgs.append(
    #     AttestationConfig(
    #         save_filename=f"verification_{model_name_str}_vllm_fp8_kv.json",
    #         trusted_model_name=model_name,
    #         vllm_args={"kv_cache_dtype": "fp8", "calculate_kv_scales": True},
    #     )
    # )

    # external_prompt_filename = (
    #     "openrouter_responses/openrouter_groq_meta-llama_llama-3_1-8b-instruct_token_difr_prompts.json"
    # )

    # external_save_filename = external_prompt_filename.split("/")[-1].split(".")[0]

    # cfgs.append(
    #     AttestationConfig(
    #         save_filename=f"verification_{external_save_filename}.json",
    #         trusted_model_name=model_name,
    #         external_prompt_filename=external_prompt_filename,
    #     )
    # )

    # external_prompt_filename = (
    #     "openrouter_responses/openrouter_hyperbolic_meta-llama_llama-3_1-8b-instruct_token_difr_prompts.json"
    # )

    # external_save_filename = external_prompt_filename.split("/")[-1].split(".")[0]

    # cfgs.append(
    #     AttestationConfig(
    #         save_filename=f"verification_{external_save_filename}.json",
    #         trusted_model_name=model_name,
    #         external_prompt_filename=external_prompt_filename,
    #     )
    # )

    # external_prompt_filename = (
    #     "openrouter_responses/"
    #     "openrouter_siliconflow_fp8_meta-llama_llama-3_1-8b-instruct_token_difr_prompts.json"
    # )

    # external_save_filename = external_prompt_filename.split("/")[-1].split(".")[0]

    # cfgs.append(
    #     AttestationConfig(
    #         save_filename=f"verification_{external_save_filename}.json",
    #         trusted_model_name=model_name,
    #         external_prompt_filename=external_prompt_filename,
    #     )
    # )

    # external_prompt_filename = (
    #     "openrouter_responses/openrouter_cerebras_meta-llama_llama-3_1-8b-instruct_token_difr_prompts.json"
    # )

    # external_save_filename = external_prompt_filename.split("/")[-1].split(".")[0]

    # cfgs.append(
    #     AttestationConfig(
    #         save_filename=f"verification_{external_save_filename}.json",
    #         trusted_model_name=model_name,
    #         external_prompt_filename=external_prompt_filename,
    #     )
    # )

    # external_prompt_filename = (
    #     "openrouter_responses/openrouter_deepinfra_meta-llama_llama-3_1-8b-instruct_token_difr_prompts.json"
    # )

    # external_save_filename = external_prompt_filename.split("/")[-1].split(".")[0]

    # cfgs.append(
    #     AttestationConfig(
    #         save_filename=f"verification_{external_save_filename}.json",
    #         trusted_model_name=model_name,
    #         external_prompt_filename=external_prompt_filename,
    #     )
    # )

    # Apply smoketest settings to all configs
    filenames = []
    for i in range(len(cfgs)):
        cfgs[i].n_samples = 200
        cfgs[i].max_decode_tokens = 200
        cfgs[i].save_dir = "token_difr_results"
        filenames.append(cfgs[i].save_filename)

    # Check for duplicates
    assert len(filenames) == len(set(filenames)), f"Duplicate filenames: {filenames}"

    print(f"Running {len(cfgs)} configurations:")
    for i, cfg in enumerate(cfgs, 1):
        print(f"  {i}. {cfg.save_filename}")
    print("=" * 80)
    print()

    for i, cfg in enumerate(cfgs, 1):
        print(f"[{i}/{len(cfgs)}] Running: {cfg.save_filename}")

        external_responses = None
        if cfg.external_prompt_filename is not None:
            external_responses = load_external_tokens(cfg.external_prompt_filename)

        run(cfg, external_responses)
        print(f"Complete: {cfg.save_filename}")
        print()
