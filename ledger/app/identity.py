"""Deterministic identity for model configs.

A model's identity is derived from its verification config (name + params) so
the same config always maps to the same `model_id` — declaring it again, or on
another host, reuses the existing row instead of creating a duplicate. Changing
any param yields a different id (a genuinely different verification config).
"""

import json
import uuid

# Fixed namespace for model-config UUIDv5 derivation. Do not change — it would
# re-key every existing model_id.
MODEL_NAMESPACE = uuid.UUID("b9c1e6a2-1f4d-5e7a-9c3b-2d8f0a6b4e11")


def derive_model_id(
    *,
    model_name: str,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    seed: int | None,
    decoding_algorithm: str | None,
) -> uuid.UUID:
    """UUIDv5 over the canonical (stable, sorted) config JSON.

    The verification threshold is deliberately NOT part of a model's identity:
    it is set dynamically per test run and recorded on each verification_event
    as `verification_threshold`.
    """
    canonical = json.dumps(
        {
            "model_name": model_name,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "seed": seed,
            "decoding_algorithm": decoding_algorithm,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return uuid.uuid5(MODEL_NAMESPACE, canonical)
