"""frame-processor's own extractors (the shared ones are inf-proxy's, tested there).

- prompt token IDs submitted by the client rather than echoed by the model,
  which only a passive tap needs — see extract_request_prompt_token_ids;
- the constrained-decoding flag, a copy of inf-proxy's because its module
  isn't loadable here — see detect_constrained_decoding.
"""

from __future__ import annotations

from app.extract import detect_constrained_decoding, extract_request_prompt_token_ids


def test_token_id_prompt_is_extracted():
    assert extract_request_prompt_token_ids({"prompt": [1, 2, 3]}) == [1, 2, 3]


def test_batched_token_id_prompt_takes_the_first():
    assert extract_request_prompt_token_ids({"prompt": [[1, 2], [3, 4]]}) == [1, 2]


def test_explicit_prompt_token_ids_field_wins():
    payload = {"prompt": "hello", "prompt_token_ids": [7, 8]}
    assert extract_request_prompt_token_ids(payload) == [7, 8]


def test_text_prompts_yield_nothing():
    assert extract_request_prompt_token_ids({"prompt": "hello"}) is None
    assert extract_request_prompt_token_ids({"prompt": ["hello", "there"]}) is None
    assert extract_request_prompt_token_ids({"messages": [{"role": "user", "content": "hi"}]}) is None
    assert extract_request_prompt_token_ids({}) is None


def test_malformed_prompts_yield_nothing():
    assert extract_request_prompt_token_ids({"prompt": [1, "two", 3]}) is None
    assert extract_request_prompt_token_ids({"prompt": []}) is None
    assert extract_request_prompt_token_ids({"prompt": None}) is None


# The cases below mirror inf-proxy/tests/test_capture_integrity.py one for one.
# The flag is hashed into the ledger event, so the two taps disagreeing on any
# of these files two different events for one inference — and the false
# positives (`tool_choice: "auto"`, `response_format: {"type": "text"}`) are
# the ones Open WebUI actually sends.


def test_forced_tool_choice_is_constrained():
    assert detect_constrained_decoding(
        {"tool_choice": {"type": "function", "function": {"name": "f"}}}
    )
    assert detect_constrained_decoding({"tool_choice": "required"})


def test_auto_tool_choice_is_not_constrained():
    assert not detect_constrained_decoding({"tool_choice": "auto"})
    assert not detect_constrained_decoding({"tool_choice": "none"})
    assert not detect_constrained_decoding({"tools": [{"type": "function"}]})


def test_response_format_grammar_is_constrained():
    assert detect_constrained_decoding({"response_format": {"type": "json_object"}})
    assert detect_constrained_decoding(
        {"response_format": {"type": "json_schema", "json_schema": {}}}
    )


def test_plain_text_response_format_is_not_constrained():
    assert not detect_constrained_decoding({"response_format": {"type": "text"}})
    assert not detect_constrained_decoding({})
