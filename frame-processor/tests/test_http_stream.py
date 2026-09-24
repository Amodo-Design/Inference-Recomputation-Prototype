"""HTTP parsing: requests, JSON responses, chunked SSE streams."""

from __future__ import annotations

import json

from app.http_stream import (
    ExchangeAssembler,
    merge_stream_events,
    parse_exchanges,
    parse_sse_events,
)
from tests.helpers import http_request, http_response_json, http_response_sse


def test_plain_json_exchange():
    body = json.dumps({"model": "openai/gpt-oss-120b", "messages": []}).encode()
    response = json.dumps({"id": "resp-1", "choices": []}).encode()

    exchanges = parse_exchanges(
        http_request("/infver/gpt-oss-120b/v1/chat/completions", body),
        http_response_json(response),
    )
    assert len(exchanges) == 1
    ex = exchanges[0]
    assert ex.request.method == "POST"
    assert ex.request.path == "/infver/gpt-oss-120b/v1/chat/completions"
    assert ex.request.json()["model"] == "openai/gpt-oss-120b"
    assert ex.response.status == 200
    assert ex.response.json()["id"] == "resp-1"


def test_chunked_sse_response_parses_and_merges():
    chunks = [
        {"id": "r1", "model": "m", "choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]},
        {"id": "r1", "choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
        {"id": "r1", "choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"completion_tokens": 2}},
    ]
    raw = http_response_sse([json.dumps(c).encode() for c in chunks])

    exchanges = parse_exchanges(
        http_request("/v1/chat/completions", json.dumps({"model": "m"}).encode()), raw
    )
    response = exchanges[0].response
    assert response.is_sse

    events = parse_sse_events(response.body)
    assert len(events) == 3  # [DONE] excluded

    merged = merge_stream_events(events, "chat")
    assert merged["id"] == "r1"
    assert merged["choices"][0]["message"]["content"] == "Hello"
    assert merged["choices"][0]["finish_reason"] == "stop"
    assert merged["usage"]["completion_tokens"] == 2


def test_streamed_token_ids_concatenate():
    chunks = [
        {"choices": [{"delta": {"content": "a"}, "token_ids": [11, 12], "finish_reason": None}]},
        {"choices": [{"delta": {"content": "b"}, "token_ids": [13], "finish_reason": "stop"}]},
    ]
    merged = merge_stream_events(chunks, "chat")
    assert merged["choices"][0]["token_ids"] == [11, 12, 13]


def test_truncated_chunked_body_yields_partial_output():
    raw = http_response_sse([b'{"choices":[{"delta":{"content":"partial"}}]}'])
    cut = raw[: len(raw) - 12]  # capture ended mid-stream
    exchanges = parse_exchanges(
        http_request("/v1/chat/completions", json.dumps({"model": "m"}).encode()), cut
    )
    events = parse_sse_events(exchanges[0].response.body)
    assert merge_stream_events(events, "chat")["choices"][0]["message"]["content"] == "partial"


def test_completions_stream_merges_the_text_shape():
    """Legacy completions stream `choices[0].text`, and a whole logprobs object
    per chunk rather than per-token entries under `content`."""
    chunks = [
        {"choices": [{"text": " Pa", "token_ids": [11],
                      "logprobs": {"tokens": [" Pa"], "token_logprobs": [-0.1]},
                      "finish_reason": None}]},
        {"choices": [{"text": "ris", "token_ids": [12],
                      "logprobs": {"tokens": ["ris"], "token_logprobs": [-0.2]},
                      "finish_reason": "length"}]},
    ]
    merged = merge_stream_events(chunks, "text")
    assert merged["choices"][0]["text"] == " Paris"
    assert merged["choices"][0]["token_ids"] == [11, 12]
    assert merged["choices"][0]["finish_reason"] == "length"
    assert merged["choices"][0]["logprobs"] == [
        {"tokens": [" Pa"], "token_logprobs": [-0.1]},
        {"tokens": ["ris"], "token_logprobs": [-0.2]},
    ]


def test_a_stream_that_carried_no_text_is_null_not_empty():
    """Capture loss must not read as a legitimately empty generation, and it
    must read the same on both response shapes."""
    assert merge_stream_events([], "chat")["choices"][0]["message"]["content"] is None
    assert merge_stream_events([], "text")["choices"][0]["text"] is None


def test_dead_request_direction_does_not_hoard_response_bytes():
    """h11 must not retain responses it can never frame.

    Waiting for a request method is correct while one is still in flight — the
    two directions are read by independent processes, so a response can overtake
    its request. But once the request direction is dead no method is coming, and
    h11 keeps every byte it cannot frame. On the pooled keep-alive connections
    this path uses, nothing reclaims that until a close that may never happen.
    """
    assembler = ExchangeAssembler()
    # Kill the request direction the way a capture joined mid-message does.
    assembler.feed_client(b"\x00\x01\x02 not http at all\r\n\r\n")

    response = http_response_json(json.dumps({"id": "c-1", "choices": []}).encode())
    for _ in range(200):
        assembler.feed_server(response)

    retained = len(assembler._resp_conn.trailing_data[0])
    assert retained < 4 * len(response), (
        f"h11 retained {retained} bytes of unframeable responses"
    )


def test_response_still_waits_for_a_request_that_is_merely_late():
    """The bound must not break the legitimate reorder it exists alongside.

    A response arriving before its request is normal here and must be held, not
    framed against a guessed method and paired with nothing.
    """
    request = http_request("/v1/chat/completions", json.dumps({"model": "m"}).encode())
    response = http_response_json(json.dumps({"id": "c-1", "choices": []}).encode())

    assembler = ExchangeAssembler()
    assert assembler.feed_server(response) == []  # held, not dropped
    exchanges = assembler.feed_client(request)

    assert len(exchanges) == 1
    assert exchanges[0].request.path == "/v1/chat/completions"
    assert exchanges[0].response.json()["id"] == "c-1"
