"""Capture-integrity contract: the verifier must refuse (unverifiable, not
FAIL) events whose captured token stream is provably incomplete or whose
generation was constrained beyond the declared sampling contract.

Motivating incident: gpt-oss harmony responses on vLLM < v0.11.1 omit the
analysis-channel tokens from return_token_ids; the verifier teacher-forced
the partial sequence against the wrong context and produced ~100% spurious
FAILs that were indistinguishable from a cheating prover.
"""

from __future__ import annotations

import base64
import json

import pytest

from app.mapper import inference_event_to_verify_request
from app.models import ErrorCode, VerificationStatus

from helpers import MODEL_ID


def _b64(obj) -> str:
    return base64.b64encode(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).decode()


def make_item(output_payload: dict) -> dict:
    return {
        "event": {
            "id": "11111111-0000-0000-0000-000000000001",
            "input_raw_logits": _b64({"prompt_token_ids": [1, 2, 3]}),
            "output_raw_logits": _b64(output_payload),
            "input_text_representation": "hi",
            "output_text_representation": "there",
        },
        "model": {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "seed": 42,
            "temperature": 1.0,
            "top_k": 200,
            "top_p": 0.95,
        },
    }


def test_mapper_reads_capture_integrity_fields():
    req = inference_event_to_verify_request(
        make_item(
            {
                "output_token_ids": [5, 6],
                "output_logprobs": None,
                "usage_completion_tokens": 7,
                "constrained_decoding": True,
            }
        )
    )
    assert req.usage_completion_tokens == 7
    assert req.constrained_decoding is True


def test_mapper_tolerates_legacy_payloads():
    req = inference_event_to_verify_request(
        make_item({"output_token_ids": [5, 6], "output_logprobs": None})
    )
    assert req.usage_completion_tokens is None
    assert req.constrained_decoding is None


@pytest.fixture
def verifier():
    torch = pytest.importorskip("torch")  # noqa: F841
    pytest.importorskip("transformers")
    from app.config import VerifierModelConfig
    from app.verifier import CachedModelVerifier

    return CachedModelVerifier(
        model_config=VerifierModelConfig(id="Qwen/Qwen2.5-7B-Instruct", display_name="q"),
        vllm_url="http://vllm:8000/v1",
        vllm_api_key=None,
        concurrency=1,
        margin_clip=10.0,
        require_cuda=False,
    )


async def test_incomplete_capture_is_unverifiable(verifier):
    req = inference_event_to_verify_request(
        make_item(
            {
                "output_token_ids": [5, 6],
                "output_logprobs": None,
                "usage_completion_tokens": 50,
            }
        )
    )
    resp = await verifier.verify(req, pass_threshold=0.3)
    assert resp.status == VerificationStatus.UNVERIFIABLE
    assert resp.error_code == ErrorCode.TOKEN_CAPTURE_INCOMPLETE


async def test_constrained_decoding_is_unverifiable(verifier):
    req = inference_event_to_verify_request(
        make_item(
            {
                "output_token_ids": [5, 6],
                "output_logprobs": None,
                "usage_completion_tokens": 2,
                "constrained_decoding": True,
            }
        )
    )
    resp = await verifier.verify(req, pass_threshold=0.3)
    assert resp.status == VerificationStatus.UNVERIFIABLE
    assert resp.error_code == ErrorCode.CONSTRAINED_DECODING


async def test_matching_usage_proceeds_past_integrity_checks(verifier):
    # usage == captured count: the request must NOT be rejected by the
    # integrity checks (it proceeds to the vLLM call, which fails here since
    # no server exists — anything except the two integrity codes is fine).
    req = inference_event_to_verify_request(
        make_item(
            {
                "output_token_ids": [5, 6],
                "output_logprobs": None,
                "usage_completion_tokens": 2,
                "constrained_decoding": False,
            }
        )
    )
    resp = await verifier.verify(req, pass_threshold=0.3)
    assert resp.error_code not in (
        ErrorCode.TOKEN_CAPTURE_INCOMPLETE,
        ErrorCode.CONSTRAINED_DECODING,
    )
