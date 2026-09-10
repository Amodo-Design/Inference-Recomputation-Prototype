"""Pod identity (pod_name, node_name) from Kubernetes Downward API env vars.

The tap reads pod_name and node_name from INF_PROXY_POD_NAME and
INF_PROXY_NODE_NAME environment variables injected by Kubernetes Downward API
in cluster deployments. These are stamped on every tap message for downstream
enrichment. The values are absent in docker-compose and default to None, so
they're optional in the message.
"""

from __future__ import annotations

import importlib

import pytest


def test_build_tap_message_includes_pod_identity_from_env(monkeypatch):
    """Pod identity fields are read from env vars and included in tap message."""
    # Set env vars before importing modules (or reload them after)
    monkeypatch.setenv("INF_PROXY_POD_NAME", "model-serving-abc123")
    monkeypatch.setenv("INF_PROXY_NODE_NAME", "node-gpu-02")

    # Reload config and message_writer to pick up the new env vars
    from app import config as config_mod
    from app import message_writer as message_writer_mod

    importlib.reload(config_mod)
    importlib.reload(message_writer_mod)

    # Build message without pod_name/node_name arguments
    message = message_writer_mod.build_tap_message(
        endpoint="/v1/chat/completions",
        streamed=False,
        received_at="2024-01-01T00:00:00+00:00",
        completed_at="2024-01-01T00:00:01+00:00",
        metadata={"user_id": "user1"},
        model_name="Qwen/Qwen2.5-7B",
        sampling={"seed": 42},
        prompt_text="Hello",
        prompt_token_ids=[1, 2, 3],
        output_text="Hi there",
        output_token_ids=[4, 5, 6],
        output_logprobs=None,
        finish_reason="stop",
        tool_calls=False,
        response_id="resp-123",
    )

    # Verify pod identity in the tap section
    assert message["tap"]["pod_name"] == "model-serving-abc123"
    assert message["tap"]["node_name"] == "node-gpu-02"


def test_build_tap_message_pod_identity_when_only_pod_name_set(monkeypatch):
    """Pod identity fields are independent; one can be set without the other."""
    monkeypatch.setenv("INF_PROXY_POD_NAME", "model-serving-xyz")
    # Remove node name env var if set
    monkeypatch.delenv("INF_PROXY_NODE_NAME", raising=False)

    from app import config as config_mod
    from app import message_writer as message_writer_mod

    importlib.reload(config_mod)
    importlib.reload(message_writer_mod)

    message = message_writer_mod.build_tap_message(
        endpoint="/v1/chat/completions",
        streamed=False,
        received_at="2024-01-01T00:00:00+00:00",
        completed_at="2024-01-01T00:00:01+00:00",
        metadata={"user_id": "user1"},
        model_name="Qwen/Qwen2.5-7B",
        sampling={"seed": 42},
        prompt_text="Hello",
        prompt_token_ids=[1, 2, 3],
        output_text="Hi there",
        output_token_ids=[4, 5, 6],
        output_logprobs=None,
        finish_reason="stop",
        tool_calls=False,
        response_id="resp-123",
    )

    assert message["tap"]["pod_name"] == "model-serving-xyz"
    assert message["tap"]["node_name"] is None


def test_build_tap_message_pod_identity_when_env_unset(monkeypatch):
    """Pod identity fields are None when env vars not set (docker-compose, local)."""
    # Remove both env vars
    monkeypatch.delenv("INF_PROXY_POD_NAME", raising=False)
    monkeypatch.delenv("INF_PROXY_NODE_NAME", raising=False)

    from app import config as config_mod
    from app import message_writer as message_writer_mod

    importlib.reload(config_mod)
    importlib.reload(message_writer_mod)

    message = message_writer_mod.build_tap_message(
        endpoint="/v1/chat/completions",
        streamed=False,
        received_at="2024-01-01T00:00:00+00:00",
        completed_at="2024-01-01T00:00:01+00:00",
        metadata={"user_id": "user1"},
        model_name="Qwen/Qwen2.5-7B",
        sampling={"seed": 42},
        prompt_text="Hello",
        prompt_token_ids=[1, 2, 3],
        output_text="Hi there",
        output_token_ids=[4, 5, 6],
        output_logprobs=None,
        finish_reason="stop",
        tool_calls=False,
        response_id="resp-123",
    )

    assert message["tap"]["pod_name"] is None
    assert message["tap"]["node_name"] is None
