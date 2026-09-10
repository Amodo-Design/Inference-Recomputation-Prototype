"""inf-ver-orchestrator entrypoint: reconcile verifier pairs forever.

Each cycle: read the ledger's pending-verification queue grouped by model,
observe the managed Jobs/LLMISvcs in the namespace, and apply the
reconciler's spawn/reap decisions. Errors in a cycle are logged and the next
cycle retries — the loop itself never exits.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .config import Settings, load_settings
from .kube import KubeClient
from .ledger import LedgerClient
from .reconciler import Action, decide
from .templates import build_job, build_llmisvc, build_seed_job

log = logging.getLogger(__name__)


def apply_action(action: Action, kube: KubeClient, settings: Settings) -> None:
    log.info(
        "%s model_id=%s name=%s model=%s",
        action.kind,
        action.model_id,
        action.name,
        action.model.model_name if action.model else None,
    )
    if action.kind == "create_vllm":
        kube.create_llmisvc(build_llmisvc(action.model, settings))
    elif action.kind == "create_job":
        kube.create_job(build_job(action.model, settings))
    elif action.kind == "delete_vllm":
        kube.delete_llmisvc(action.name)
    elif action.kind == "delete_job":
        kube.delete_job(action.name)
    elif action.kind == "create_seed":
        kube.create_job(build_seed_job(action.model, settings))
    elif action.kind == "delete_seed":
        kube.delete_job(action.name)


def reconcile_once(ledger: LedgerClient, kube: Any, settings: Settings) -> list[Action]:
    pending = ledger.pending_by_model()
    jobs = kube.list_managed_jobs()
    vllms = kube.list_managed_llmisvcs()
    seeds = kube.list_managed_seed_jobs()
    cache_exempt = frozenset(
        name
        for name, override in settings.vllm_placement_overrides.items()
        if override.cache_pvc is None
    )
    actions = decide(
        pending,
        jobs,
        vllms,
        seeds=seeds,
        model_cache=settings.model_cache_pvc is not None,
        cache_exempt_models=cache_exempt,
        keep_failed_jobs=settings.keep_failed_jobs,
    )
    for action in actions:
        try:
            apply_action(action, kube, settings)
        except Exception:
            # One bad resource must not stall the other models' pairs.
            log.exception(
                "Failed to apply %s for model_id=%s", action.kind, action.model_id
            )
    return actions


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = load_settings()
    ledger = LedgerClient(settings.ledger_api_url)
    kube = KubeClient(settings.namespace)
    log.info(
        "Orchestrator started namespace=%s poll=%ss runner_image=%s",
        settings.namespace,
        settings.poll_seconds,
        settings.runner_image,
    )
    while True:
        try:
            reconcile_once(ledger, kube, settings)
        except Exception:
            log.exception("Reconcile cycle failed, retrying next cycle")
        time.sleep(settings.poll_seconds)


if __name__ == "__main__":
    main()
