"""Tests for GPU efficiency enrichment fields: started_at and pod identity.

Covers Task 4: persisting started_at (from tap.received_at) and pod identity
(pod_name, node_name) in inference events for the ledger API. Tests both
presence and absence (legacy taps) scenarios.
"""

from __future__ import annotations

import pytest

from app import transform
from app.schemas import TapMessage

SAMPLE = {
    "tap": {
        "hostname": "h",
        "completed_at": "2026-01-01T00:00:00Z",
        "received_at": "2026-01-01T00:00:00Z",
    },
    "session": {"session_id": "chat-1", "response_id": "resp-1"},
    "model": {"name": "m"},
    "request": {"prompt_text": "hi", "prompt_token_ids": [1, 2, 3]},
    "response": {
        "output_text": "yo",
        "output_token_ids": [40, 41],
        "output_logprobs": None,
        "finish_reason": "stop",
        "tool_calls": False,
    },
}


def _msg(**tree) -> TapMessage:
    merged = {**SAMPLE, **tree}
    return TapMessage.model_validate(merged)


def _build(msg, model_id="MID", hardware_id="HID"):
    return transform.build_inference_event(msg, model_id=model_id, hardware_id=hardware_id)


def test_started_at_and_pod_identity_when_present():
    """When pod identity and started_at are in tap, they flow to event."""
    event = _build(
        _msg(
            tap={
                "hostname": "h",
                "completed_at": "2026-01-01T00:00:00Z",
                "received_at": "2026-01-01T00:00:00Z",
                "pod_name": "model-serving-abc123",
                "node_name": "gpu-node-1",
            }
        )
    )
    assert event["started_at"] == "2026-01-01T00:00:00Z"
    assert event["pod_name"] == "model-serving-abc123"
    assert event["node_name"] == "gpu-node-1"


def test_started_at_and_pod_identity_absent_for_legacy_tap_with_received_at():
    """When pod identity fields absent but received_at present, started_at has value."""
    event = _build(
        _msg(
            tap={
                "hostname": "h",
                "completed_at": "2026-01-01T00:00:00Z",
                "received_at": "2026-01-01T00:00:00Z",
            }
        )
    )
    assert event["started_at"] == "2026-01-01T00:00:00Z"
    assert event["pod_name"] is None
    assert event["node_name"] is None


def test_legacy_tap_all_gpu_fields_absent_produces_none():
    """When received_at, pod_name, node_name all absent (legacy tap), all are None."""
    event = _build(
        _msg(
            tap={
                "hostname": "h",
                "completed_at": "2026-01-01T00:00:00Z",
            }
        )
    )
    assert event["started_at"] is None
    assert event["pod_name"] is None
    assert event["node_name"] is None


def test_started_at_maps_from_received_at():
    """started_at is populated from tap.received_at."""
    event = _build(
        _msg(
            tap={
                "hostname": "h",
                "received_at": "2026-03-03T12:34:56Z",
            }
        )
    )
    assert event["started_at"] == "2026-03-03T12:34:56Z"
