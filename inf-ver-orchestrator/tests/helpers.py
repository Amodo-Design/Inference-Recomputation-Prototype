"""Shared builders for orchestrator tests."""

from __future__ import annotations

import uuid
from typing import Any

from app.config import Settings
from app.ledger import PendingModel
from app.reconciler import ObservedJob


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = dict(
        ledger_api_url="http://ledger-api:8000",
        namespace="infver",
        poll_seconds=0.01,
        runner_image="harbor.infver.local/infver_images/inf-ver-runner:latest",
        vllm_image="vllm/vllm-openai:latest",
        vllm_max_logprobs=200,
        vllm_max_model_len=8192,
        vllm_node_selector={},
        job_backoff_limit=3,
        keep_failed_jobs=True,
    )
    values.update(overrides)
    return Settings(**values)


def make_model(
    *,
    model_id: str | None = None,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    threshold: float | None = 0.1,
    pending: int = 3,
    seed: int | None = 42,
) -> PendingModel:
    return PendingModel(
        model_id=model_id or str(uuid.uuid4()),
        model_name=model_name,
        verification_threshold=threshold,
        seed=seed,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        pending=pending,
    )


class FakeKube:
    """In-memory stand-in for KubeClient, driven by main.reconcile_once /
    main.apply_action tests. Records applied creates/deletes so tests can
    assert on what main.py sent through the client interface."""

    def __init__(
        self,
        *,
        jobs: dict[str, ObservedJob] | None = None,
        seed_jobs: dict[str, ObservedJob] | None = None,
        vllms: dict[str, str] | None = None,
    ) -> None:
        self.jobs = dict(jobs or {})
        self.seed_jobs = dict(seed_jobs or {})
        self.vllms = dict(vllms or {})
        self.created_jobs: list[dict[str, Any]] = []
        self.deleted_jobs: list[str] = []
        self.created_llmisvcs: list[dict[str, Any]] = []
        self.deleted_llmisvcs: list[str] = []

    def list_managed_jobs(self) -> dict[str, ObservedJob]:
        return dict(self.jobs)

    def list_managed_seed_jobs(self) -> dict[str, ObservedJob]:
        return dict(self.seed_jobs)

    def list_managed_llmisvcs(self) -> dict[str, str]:
        return dict(self.vllms)

    def create_job(self, manifest: dict[str, Any]) -> None:
        self.created_jobs.append(manifest)

    def delete_job(self, name: str) -> None:
        self.deleted_jobs.append(name)

    def create_llmisvc(self, manifest: dict[str, Any]) -> None:
        self.created_llmisvcs.append(manifest)

    def delete_llmisvc(self, name: str) -> None:
        self.deleted_llmisvcs.append(name)
