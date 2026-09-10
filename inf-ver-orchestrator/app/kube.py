"""Thin Kubernetes client for the orchestrator's managed resources.

Everything the reconciler decides is applied through this class; tests fake
it (see tests/test_main.py) so no cluster is needed. Only resources labeled
`app.kubernetes.io/managed-by: inf-ver-orchestrator` are ever listed or
deleted — the orchestrator cannot touch the serving stack.
"""

from __future__ import annotations

import logging
from typing import Any

from kubernetes import client, config

from .naming import JOB_PREFIX, SEED_PREFIX
from .reconciler import JobState, ObservedJob
from .templates import MANAGED_BY, MODEL_ID_LABEL

log = logging.getLogger(__name__)

LLMISVC_GROUP = "serving.kserve.io"
LLMISVC_VERSION = "v1alpha1"
LLMISVC_PLURAL = "llminferenceservices"

_SELECTOR = f"app.kubernetes.io/managed-by={MANAGED_BY}"


def _job_state(job: Any) -> JobState:
    status = job.status
    if status.succeeded:
        return JobState.SUCCEEDED
    for condition in status.conditions or []:
        if condition.type == "Failed" and condition.status == "True":
            return JobState.FAILED
    return JobState.ACTIVE


class KubeClient:
    def __init__(self, namespace: str) -> None:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        self._namespace = namespace
        self._batch = client.BatchV1Api()
        self._custom = client.CustomObjectsApi()

    def _list_jobs_with_prefix(self, prefix: str) -> dict[str, ObservedJob]:
        jobs = self._batch.list_namespaced_job(
            self._namespace, label_selector=_SELECTOR
        )
        observed: dict[str, ObservedJob] = {}
        for job in jobs.items:
            if not job.metadata.name.startswith(prefix):
                continue
            model_id = (job.metadata.labels or {}).get(MODEL_ID_LABEL)
            if model_id:
                observed[model_id] = ObservedJob(
                    name=job.metadata.name, state=_job_state(job)
                )
        return observed

    def list_managed_jobs(self) -> dict[str, ObservedJob]:
        """Managed runner Jobs keyed by ledger model_id."""
        return self._list_jobs_with_prefix(JOB_PREFIX)

    def list_managed_seed_jobs(self) -> dict[str, ObservedJob]:
        """Managed model-cache seed Jobs keyed by ledger model_id."""
        return self._list_jobs_with_prefix(SEED_PREFIX)

    def list_managed_llmisvcs(self) -> dict[str, str]:
        """Managed verifier LLMISvc names keyed by ledger model_id."""
        result = self._custom.list_namespaced_custom_object(
            LLMISVC_GROUP,
            LLMISVC_VERSION,
            self._namespace,
            LLMISVC_PLURAL,
            label_selector=_SELECTOR,
        )
        observed: dict[str, str] = {}
        for item in result.get("items", []):
            metadata = item.get("metadata", {})
            model_id = metadata.get("labels", {}).get(MODEL_ID_LABEL)
            if model_id:
                observed[model_id] = metadata["name"]
        return observed

    def create_job(self, manifest: dict[str, Any]) -> None:
        self._batch.create_namespaced_job(self._namespace, manifest)

    def delete_job(self, name: str) -> None:
        # Propagate so the runner pod goes with the Job.
        self._batch.delete_namespaced_job(
            name, self._namespace, propagation_policy="Background"
        )

    def create_llmisvc(self, manifest: dict[str, Any]) -> None:
        self._custom.create_namespaced_custom_object(
            LLMISVC_GROUP,
            LLMISVC_VERSION,
            self._namespace,
            LLMISVC_PLURAL,
            manifest,
        )

    def delete_llmisvc(self, name: str) -> None:
        self._custom.delete_namespaced_custom_object(
            LLMISVC_GROUP,
            LLMISVC_VERSION,
            self._namespace,
            LLMISVC_PLURAL,
            name,
        )
