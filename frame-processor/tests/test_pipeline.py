"""End-to-end: synthetic TCP segments in, contract-shaped tap messages out."""

from __future__ import annotations

import json

from app.main import process_segments
from tests.helpers import conversation, http_request, http_response_json, http_response_sse

REQUEST_BODY = {
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "What is a network tap?"}],
    "temperature": 1.0,
    "seed": 42,
    "top_p": 0.95,
    "stream": True,
}
SSE_CHUNKS = [
    {"id": "cmpl-1", "model": "openai/gpt-oss-120b",
     "choices": [{"delta": {"content": "A passive"}, "token_ids": [5, 6], "finish_reason": None}]},
    {"id": "cmpl-1",
     "choices": [{"delta": {"content": " device."}, "token_ids": [7], "finish_reason": "stop"}],
     "usage": {"completion_tokens": 3}},
]


def _messages():
    client = http_request(
        "/infver/gpt-oss-120b/v1/chat/completions",
        json.dumps(REQUEST_BODY).encode(),
        headers={"X-OpenWebUI-Chat-Id": "chat-9", "X-OpenWebUI-User-Id": "user-3"},
    )
    server = http_response_sse([json.dumps(c).encode() for c in SSE_CHUNKS])
    return list(process_segments(iter(conversation(client, server))))


def test_pipeline_reconstructs_one_llm_exchange():
    messages = _messages()
    assert len(messages) == 1
    msg = messages[0]

    assert msg["tap"]["source"] == "frame-processor"
    assert msg["tap"]["endpoint"] == "/v1/chat/completions"
    assert msg["tap"]["streamed"] is True
    assert msg["tap"]["received_at"] < msg["tap"]["completed_at"]

    assert msg["session"]["chat_id"] == "chat-9"
    assert msg["session"]["user_id"] == "user-3"
    assert msg["session"]["response_id"] == "cmpl-1"

    assert msg["model"]["name"] == "openai/gpt-oss-120b"
    assert msg["sampling"]["seed"] == 42
    assert msg["sampling"]["temperature"] == 1.0
    assert msg["sampling"]["top_p"] == 0.95

    assert msg["request"]["prompt_text"] == "user: What is a network tap?"
    assert msg["response"]["output_text"] == "A passive device."
    assert msg["response"]["output_token_ids"] == [5, 6, 7]
    assert msg["response"]["finish_reason"] == "stop"
    assert msg["response"]["usage_completion_tokens"] == 3


def test_tap_message_matches_contract_shape():
    """Key-for-key against docs/tap-message-contract.md — drift fails here."""
    msg = _messages()[0]

    assert set(msg) == {"tap", "session", "model", "sampling", "request", "response"}
    # tap adds observed_on (frame-processor only) alongside the inf-proxy keys.
    assert set(msg["tap"]) == {
        "source", "proxy_version", "hostname", "upstream_base_url", "observed_on",
        "endpoint", "streamed", "received_at", "completed_at", "pod_name", "node_name",
    }
    assert set(msg["session"]) == {"session_id", "user_id", "chat_id", "message_id", "response_id"}
    assert set(msg["model"]) == {"name", "revision", "tokenizer_revision"}
    assert set(msg["sampling"]) == {
        "seed", "temperature", "top_k", "top_p", "max_output_tokens", "reporting_flags_added",
    }
    assert set(msg["request"]) == {"prompt_text", "prompt_token_ids"}
    assert set(msg["response"]) == {
        "output_text", "output_token_ids", "output_logprobs", "finish_reason",
        "tool_calls", "usage_completion_tokens", "constrained_decoding",
    }


def test_non_llm_traffic_produces_nothing():
    client = http_request("/metrics", b"")
    server = b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nok"
    assert list(process_segments(iter(conversation(client, server)))) == []


def test_verifier_replay_never_reaches_a_tap_message():
    """The verifier's replay of an event must not become a new event."""
    body = {"model": "openai/gpt-oss-120b", "prompt": [1, 2, 3], "max_tokens": 1,
            "temperature": 1.0, "logprobs": 5}
    client = http_request("/v1/completions", json.dumps(body).encode()).replace(
        b"host: 10.0.0.2:8080",
        b"host: verify-openai-gpt-oss-120b-kserve-workload-svc:8000",
        1,
    )
    server = http_response_json(
        json.dumps({"id": "cmpl-1", "choices": [{"text": "", "finish_reason": "length"}]}).encode()
    )
    assert list(process_segments(iter(conversation(client, server)))) == []


def test_token_id_prompt_is_captured_from_the_request():
    """A prompt submitted as token IDs is never echoed by the response."""
    body = {"model": "openai/gpt-oss-120b", "prompt": [200006, 17360, 200008], "max_tokens": 1}
    client = http_request("/v1/completions", json.dumps(body).encode())
    server = http_response_json(
        json.dumps(
            {"id": "cmpl-7", "model": "openai/gpt-oss-120b",
             "choices": [{"text": "", "finish_reason": "length"}]}
        ).encode()
    )
    msg = list(process_segments(iter(conversation(client, server))))[0]

    assert msg["request"]["prompt_token_ids"] == [200006, 17360, 200008]
    # No tokenizer here, so there is no honest text rendering of token IDs.
    assert msg["request"]["prompt_text"] is None


# The two completions cases above deliberately generate nothing (max_tokens=1,
# finish_reason=length), so neither exercises the response shape. A legacy
# completions response puts its text in choices[0].text with the older
# parallel-array logprobs; reading the chat shape out of it nulls both while
# leaving the token IDs intact, which no amount of capture loss can produce and
# no downstream check would flag.
_COMPLETIONS_REQUEST = {
    "model": "openai/gpt-oss-120b", "prompt": "The capital of France is",
    "max_tokens": 3, "logprobs": 5,
}
_COMPLETIONS_LOGPROBS = {
    "tokens": [" Paris", "."], "token_logprobs": [-0.01, -0.5],
    "top_logprobs": [{" Paris": -0.01}, {".": -0.5}], "text_offset": [0, 6],
}


def test_completions_response_text_and_logprobs_are_captured():
    client = http_request("/v1/completions", json.dumps(_COMPLETIONS_REQUEST).encode())
    server = http_response_json(
        json.dumps(
            {"id": "cmpl-9", "object": "text_completion", "model": "openai/gpt-oss-120b",
             "choices": [{"index": 0, "text": " Paris.", "token_ids": [12095, 13],
                          "logprobs": _COMPLETIONS_LOGPROBS, "finish_reason": "length"}],
             "usage": {"completion_tokens": 2}}
        ).encode()
    )
    msg = list(process_segments(iter(conversation(client, server))))[0]

    assert msg["tap"]["endpoint"] == "/v1/completions"
    assert msg["response"]["output_text"] == " Paris."
    assert msg["response"]["output_logprobs"] == _COMPLETIONS_LOGPROBS
    assert msg["response"]["output_token_ids"] == [12095, 13]
    assert msg["response"]["finish_reason"] == "length"
    assert msg["response"]["usage_completion_tokens"] == 2


def test_streamed_completions_response_is_captured():
    body = dict(_COMPLETIONS_REQUEST, stream=True)
    client = http_request("/v1/completions", json.dumps(body).encode())
    chunks = [
        {"id": "cmpl-9", "model": "openai/gpt-oss-120b",
         "choices": [{"index": 0, "text": " Paris", "token_ids": [12095], "finish_reason": None}]},
        {"id": "cmpl-9",
         "choices": [{"index": 0, "text": ".", "token_ids": [13], "finish_reason": "length"}],
         "usage": {"completion_tokens": 2}},
    ]
    server = http_response_sse([json.dumps(c).encode() for c in chunks])
    msg = list(process_segments(iter(conversation(client, server))))[0]

    assert msg["tap"]["streamed"] is True
    assert msg["response"]["output_text"] == " Paris."
    assert msg["response"]["output_token_ids"] == [12095, 13]
    assert msg["response"]["finish_reason"] == "length"
