"""inference_event -> VerifyRequest mapping and the verification_event
payload builder."""

from __future__ import annotations

import base64
import json
import statistics

from app.mapper import build_verification_event_payload, inference_event_to_verify_request
from app.models import ErrorCode, VerificationStatus

from helpers import make_item, make_response


def test_maps_token_ids_and_sampling_config():
    item = make_item(prompt_token_ids=[1, 2, 3], output_token_ids=[7, 8])
    request = inference_event_to_verify_request(item)

    assert request.request_id == item["event"]["id"]
    assert request.prompt_token_ids == [1, 2, 3]
    assert request.output_token_ids == [7, 8]
    assert request.model.name == "Qwen/Qwen3-8B"
    assert request.sampling_config.seed == 42
    assert request.sampling_config.temperature == 1.0
    assert request.sampling_config.top_k == 50
    assert request.sampling_config.top_p == 0.95
    # Validated-but-unused by the verifier; filled so the config check passes.
    assert request.sampling_config.max_output_tokens == 2
    assert request.metadata.user_prompt == "hello"
    assert request.metadata.prover_output == "world"


def test_prompt_text_fallback_leaves_ids_none():
    item = make_item(prompt_token_ids=None)
    request = inference_event_to_verify_request(item)
    assert request.prompt_token_ids is None


def test_missing_payloads_leave_ids_none():
    item = make_item(output_token_ids=None)
    item["event"]["input_raw_logits"] = None
    request = inference_event_to_verify_request(item)
    assert request.prompt_token_ids is None
    assert request.output_token_ids is None
    assert request.sampling_config.max_output_tokens is None


def test_undecodable_payload_leaves_ids_none():
    item = make_item()
    item["event"]["output_raw_logits"] = "not base64!!!"
    request = inference_event_to_verify_request(item)
    assert request.output_token_ids is None


def test_null_sampling_values_pass_through():
    item = make_item(seed=None)
    request = inference_event_to_verify_request(item)
    # The verifier itself classifies this as sampling_config_missing.
    assert request.sampling_config.seed is None


def test_payload_builder_pass():
    item = make_item()
    request = inference_event_to_verify_request(item)
    margins = [0.1, 0.2, 0.3]
    response = make_response(request.request_id, VerificationStatus.PASS, margins=margins)

    payload = build_verification_event_payload(request, response, "hw-1", "vm-1")

    assert payload["inference_event_id"] == item["event"]["id"]
    assert payload["hardware_id"] == "hw-1"
    assert payload["result"] == "pass"
    assert payload["error_code"] is None
    assert payload["verification_threshold"] == 0.1
    assert payload["verifier_model_id"] == "vm-1"
    assert payload["exact_match_level_pct"] == 1.0
    assert payload["std_dev_logit_difference"] == statistics.pstdev(margins)
    decoded = json.loads(base64.b64decode(payload["logit_difference_margins"]))
    assert decoded == margins
    detail = payload["verifier_detail"]
    assert detail["prompt_token_count"] == 3
    assert detail["output_token_count"] == 2
    assert detail["latency_ms"] == 12
    assert detail["prover_output_text"] == "world"
    assert detail["metrics"]["exact_match_ratio"] == 1.0


def test_verifier_detail_records_runner_concurrency():
    item = make_item()
    request = inference_event_to_verify_request(item)
    margins = [0.1, 0.2, 0.3]
    response = make_response(request.request_id, VerificationStatus.PASS, margins=margins)

    payload = build_verification_event_payload(
        request, response, "hw-1", "vm-1", runner_concurrency=4
    )

    assert payload["verifier_detail"]["runner_concurrency"] == 4


def test_runner_concurrency_defaults_to_one():
    item = make_item()
    request = inference_event_to_verify_request(item)
    margins = [0.1, 0.2, 0.3]
    response = make_response(request.request_id, VerificationStatus.PASS, margins=margins)

    payload = build_verification_event_payload(request, response, "hw-1", "vm-1")

    assert payload["verifier_detail"]["runner_concurrency"] == 1


def test_payload_builder_unverifiable():
    item = make_item()
    request = inference_event_to_verify_request(item)
    response = make_response(
        request.request_id,
        VerificationStatus.UNVERIFIABLE,
        error_code=ErrorCode.TOKENIZATION_MISMATCH,
    )

    payload = build_verification_event_payload(request, response, "hw-1", "vm-1")

    assert payload["result"] == "unverifiable"
    assert payload["verifier_model_id"] == "vm-1"
    assert payload["error_code"] == "tokenization_mismatch"
    assert payload["exact_match_level_pct"] is None  # no tokens verified
    assert payload["mean_logit_difference"] is None
    assert payload["std_dev_logit_difference"] is None
    assert payload["logit_difference_margins"] is None
