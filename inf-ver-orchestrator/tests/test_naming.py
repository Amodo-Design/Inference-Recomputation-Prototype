"""Slugging and derived resource names."""

from __future__ import annotations

from app.naming import job_name, slug, vllm_name, vllm_service_url


def test_slug_sanitizes_hf_names():
    assert slug("Qwen/Qwen2.5-7B-Instruct") == "qwen-qwen2-5-7b-instruct"
    assert slug("openai/gpt-oss-20b") == "openai-gpt-oss-20b"


def test_slug_collapses_runs_and_strips_edges():
    assert slug("__Weird//..Name--") == "weird-name"


def test_slug_truncates_to_budget():
    long_name = "org/" + "a" * 100
    s = slug(long_name)
    assert len(s) <= 40
    assert not s.endswith("-")


def test_resource_names_and_service_url():
    name = "Qwen/Qwen2.5-7B-Instruct"
    assert vllm_name(name) == "verify-qwen-qwen2-5-7b-instruct"
    assert job_name(name) == "inf-ver-runner-qwen-qwen2-5-7b-instruct"
    assert (
        vllm_service_url(name)
        == "http://verify-qwen-qwen2-5-7b-instruct-kserve-workload-svc:8000/v1"
    )


def test_seed_job_name():
    from app.naming import SEED_PREFIX, seed_job_name

    assert SEED_PREFIX == "seed-"
    assert seed_job_name("Qwen/Qwen2.5-7B-Instruct") == "seed-qwen-qwen2-5-7b-instruct"
