"""Tap-side capture-integrity contract: usage accounting is requested on
streams, and constrained-decoding requests are flagged."""

from __future__ import annotations

from app.payload import add_tap_report_options, detect_constrained_decoding


def test_stream_requests_usage_accounting():
    payload = add_tap_report_options({"stream": True}, object_kind="chat")
    assert payload["stream_options"] == {"include_usage": True}


def test_non_stream_leaves_stream_options_alone():
    payload = add_tap_report_options({}, object_kind="chat")
    assert "stream_options" not in payload


def test_client_stream_options_win():
    payload = add_tap_report_options(
        {"stream": True, "stream_options": {"include_usage": False}}, object_kind="chat"
    )
    assert payload["stream_options"] == {"include_usage": False}


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
