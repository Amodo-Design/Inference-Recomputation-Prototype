"""load_settings: the verifier vLLM image must be explicitly pinned.

difr's verifier re-implements the exact serving vLLM version's seeded
sampler; an unpinned image can silently diverge from the provers and turn
every verdict into `fail`. A missing/floating ORCH_VLLM_IMAGE therefore has
to be a startup error, not a silent `:latest` fallback (which is exactly the
incident that motivated this: the configmap declared v0.25.1 while stale
orchestrators spawned `:latest`).
"""

from __future__ import annotations

import pytest

from app.config import load_settings

REQUIRED_ENV = {
    "ORCH_VLLM_IMAGE": "vllm/vllm-openai:v0.25.1",
}


def _set_env(monkeypatch, **overrides):
    env = {**REQUIRED_ENV, **overrides}
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


@pytest.fixture(autouse=True)
def _pinned_vllm_image(monkeypatch):
    """Every test in this module gets ORCH_VLLM_IMAGE pinned by default so
    load_settings() doesn't fail on the unrelated pinned-image guard; tests
    can still override/unset other env vars afterward via monkeypatch."""
    _set_env(monkeypatch)


def _load():
    return load_settings()


def test_pinned_vllm_image_accepted(monkeypatch):
    _set_env(monkeypatch)
    assert load_settings().vllm_image == "vllm/vllm-openai:v0.25.1"


def test_digest_pinned_vllm_image_accepted(monkeypatch):
    digest = "vllm/vllm-openai@sha256:" + "a" * 64
    _set_env(monkeypatch, ORCH_VLLM_IMAGE=digest)
    assert load_settings().vllm_image == digest


def test_missing_vllm_image_is_fatal(monkeypatch):
    _set_env(monkeypatch, ORCH_VLLM_IMAGE=None)
    with pytest.raises(RuntimeError, match="ORCH_VLLM_IMAGE"):
        load_settings()


def test_latest_vllm_image_is_fatal(monkeypatch):
    _set_env(monkeypatch, ORCH_VLLM_IMAGE="vllm/vllm-openai:latest")
    with pytest.raises(RuntimeError, match="pinned"):
        load_settings()


def test_tagless_vllm_image_is_fatal(monkeypatch):
    _set_env(monkeypatch, ORCH_VLLM_IMAGE="vllm/vllm-openai")
    with pytest.raises(RuntimeError, match="pinned"):
        load_settings()


def test_model_cache_pvc_default_none(monkeypatch):
    _set_env(monkeypatch, ORCH_MODEL_CACHE_PVC=None)
    assert load_settings().model_cache_pvc is None


def test_model_cache_pvc_set(monkeypatch):
    _set_env(monkeypatch, ORCH_MODEL_CACHE_PVC="hf-model-cache")
    assert load_settings().model_cache_pvc == "hf-model-cache"


def test_model_cache_pvc_empty_is_none(monkeypatch):
    _set_env(monkeypatch, ORCH_MODEL_CACHE_PVC="")
    assert load_settings().model_cache_pvc is None


def test_placement_overrides_default_empty(monkeypatch):
    monkeypatch.delenv("ORCH_VLLM_PLACEMENT_OVERRIDES", raising=False)
    assert _load().vllm_placement_overrides == {}


def test_placement_overrides_parsed(monkeypatch):
    monkeypatch.setenv("ORCH_MODEL_CACHE_PVC", "hf-model-cache")
    monkeypatch.setenv(
        "ORCH_VLLM_PLACEMENT_OVERRIDES",
        '{"openai/gpt-oss-120b": {"node_selector": {"kubernetes.io/hostname": '
        '"gpu-node-3"}, "cache_pvc": "hf-model-cache-node-3"}}',
    )
    overrides = _load().vllm_placement_overrides
    assert overrides["openai/gpt-oss-120b"].node_selector == {
        "kubernetes.io/hostname": "gpu-node-3"
    }
    assert overrides["openai/gpt-oss-120b"].cache_pvc == "hf-model-cache-node-3"


def test_cached_override_requires_global_cache(monkeypatch):
    monkeypatch.delenv("ORCH_MODEL_CACHE_PVC", raising=False)
    monkeypatch.setenv(
        "ORCH_VLLM_PLACEMENT_OVERRIDES",
        '{"m": {"node_selector": {}, "cache_pvc": "some-pvc"}}',
    )
    with pytest.raises(RuntimeError):
        _load()


def test_vllm_memory_headroom_defaults(monkeypatch):
    _set_env(monkeypatch)
    settings = load_settings()
    assert settings.vllm_gpu_memory_utilization == 0.85
    assert settings.vllm_max_num_batched_tokens == 2048


def test_vllm_memory_headroom_env_override(monkeypatch):
    _set_env(
        monkeypatch,
        ORCH_VLLM_GPU_MEMORY_UTILIZATION="0.7",
        ORCH_VLLM_MAX_NUM_BATCHED_TOKENS="4096",
    )
    settings = load_settings()
    assert settings.vllm_gpu_memory_utilization == 0.7
    assert settings.vllm_max_num_batched_tokens == 4096


def test_runner_model_concurrency_default_and_env(monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.delenv("ORCH_RUNNER_MODEL_CONCURRENCY", raising=False)
    assert load_settings().runner_model_concurrency == 1

    monkeypatch.setenv("ORCH_RUNNER_MODEL_CONCURRENCY", "4")
    assert load_settings().runner_model_concurrency == 4


def test_runner_model_concurrency_rejects_below_one(monkeypatch):
    _set_env(monkeypatch)
    monkeypatch.setenv("ORCH_RUNNER_MODEL_CONCURRENCY", "0")
    with pytest.raises(RuntimeError):
        load_settings()
