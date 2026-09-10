"""Deterministic k8s resource names for a model's verifier pair.

Both resources also carry a `infver.io/model-id` label with the ledger's
model_id (a UUID) — the label, not the slug, is what maps resources back to
models, so slug collisions between similar model names only risk a name
clash, never a wrong pairing.
"""

from __future__ import annotations

import re

# LLMISvc names spawn child resources with long suffixes
# (-kserve-workload-svc etc.), so keep our own budget well under 63.
MAX_SLUG = 40

VLLM_PREFIX = "verify-"
JOB_PREFIX = "inf-ver-runner-"
SEED_PREFIX = "seed-"


def slug(model_name: str) -> str:
    """Sanitize a HF model name (e.g. Qwen/Qwen2.5-7B-Instruct) into a DNS-1123
    label fragment."""
    s = model_name.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:MAX_SLUG].rstrip("-")


def vllm_name(model_name: str) -> str:
    return f"{VLLM_PREFIX}{slug(model_name)}"


def job_name(model_name: str) -> str:
    return f"{JOB_PREFIX}{slug(model_name)}"


def seed_job_name(model_name: str) -> str:
    return f"{SEED_PREFIX}{slug(model_name)}"


def vllm_service_url(model_name: str) -> str:
    """The KServe workload Service the LLMISvc controller creates for the
    router-less deployment (same naming as the serving stack)."""
    return f"http://{vllm_name(model_name)}-kserve-workload-svc:8000/v1"
