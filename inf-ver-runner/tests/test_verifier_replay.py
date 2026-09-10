"""Replay-correctness plumbing in app.verifier.

The Gumbel replay in difr assumes the noise tensor has EXACTLY the width of
the serving vLLM's logits (the model config's padded vocab_size — 152064 for
Qwen2.5, not len(tokenizer) and not max(token_id)+1): each generated token
consumes one full row of the per-request generator's stream, so a wrong width
desynchronizes every row after the first. It likewise assumes the noise comes
from the same RNG algorithm as the prover's — CUDA Philox — so a silent CPU
fallback (MT19937) invalidates the replay entirely.

Unlike the rest of this suite these tests import app.verifier (torch +
transformers); skip cleanly where those aren't installed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from app.config import VerifierModelConfig  # noqa: E402
from app.verifier import CachedModelVerifier  # noqa: E402


def make_verifier(*, require_cuda: bool = False, **model_overrides) -> CachedModelVerifier:
    model = VerifierModelConfig(
        id="Qwen/Qwen2.5-7B-Instruct",
        display_name="qwen2.5-7b",
        **model_overrides,
    )
    return CachedModelVerifier(
        model_config=model,
        vllm_url="http://vllm:8000/v1",
        vllm_api_key=None,
        concurrency=1,
        margin_clip=10.0,
        require_cuda=require_cuda,
    )


def test_model_config_carries_vocab_size():
    cfg = VerifierModelConfig.from_dict({"id": "m", "vocab_size": 152064})
    assert cfg.vocab_size == 152064
    assert VerifierModelConfig.from_dict({"id": "m"}).vocab_size is None


def test_configured_vocab_size_wins():
    verifier = make_verifier(vocab_size=152064)
    assert verifier._resolve_vocab_size(observed_max_id=151664) == 152064


def test_observed_token_beyond_vocab_is_fatal():
    # A token id at/above the configured width means the width is wrong;
    # silently growing the tensor (the old max(token_id)+1 behavior) would
    # desync the replay instead of surfacing the misconfiguration.
    verifier = make_verifier(vocab_size=1000)
    with pytest.raises(ValueError, match="vocab"):
        verifier._resolve_vocab_size(observed_max_id=1000)


def test_unconfigured_vocab_size_comes_from_hf_config(monkeypatch):
    verifier = make_verifier()
    calls = []

    class FakeHFConfig:
        vocab_size = 152064

    def fake_from_pretrained(model_id, revision=None):
        calls.append((model_id, revision))
        return FakeHFConfig()

    monkeypatch.setattr(
        "app.verifier.AutoConfig",
        type("A", (), {"from_pretrained": staticmethod(fake_from_pretrained)}),
    )
    assert verifier._resolve_vocab_size(observed_max_id=10) == 152064
    # Cached: a second resolve must not refetch.
    assert verifier._resolve_vocab_size(observed_max_id=10) == 152064
    assert calls == [("Qwen/Qwen2.5-7B-Instruct", None)]


@pytest.mark.skipif(torch.cuda.is_available(), reason="exercises the no-CUDA failure path")
def test_require_cuda_fails_fast_without_gpu():
    with pytest.raises(RuntimeError, match="CUDA"):
        make_verifier(require_cuda=True)


@pytest.mark.skipif(torch.cuda.is_available(), reason="exercises the no-CUDA fallback path")
def test_cpu_fallback_must_be_explicit():
    verifier = make_verifier(require_cuda=False)
    assert verifier._device.type == "cpu"
