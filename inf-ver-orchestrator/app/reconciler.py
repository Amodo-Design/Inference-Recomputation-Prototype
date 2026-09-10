"""Pure spawn/reap decision logic — no k8s or HTTP calls in here.

Desired state: every model with pending events AND a non-NULL
verification_threshold gets a verifier pair (LLMISvc + runner Job). Models
with NULL thresholds are skipped — their events stay pending on purpose and
verify retroactively once a threshold is set.

Reaping: a runner exits 0 when its queue drains, so a succeeded Job whose
model has nothing pending means the pair is done — delete both to free the
GPU. A succeeded Job with NEW pending events is deleted alone (Jobs are
immutable-ish; the next cycle recreates it against the still-warm vLLM).
Failed Jobs are kept for inspection when configured, but their vLLM is
always reaped once the Job is no longer running.

When model caching is enabled, an eligible model's verifier pair is gated
behind a one-shot seed Job that warms the shared model cache: pair creation
is withheld until the seed SUCCEEDS, and the seed Job itself is created or
(when not kept for inspection) deleted as part of the same pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .ledger import PendingModel


class JobState(str, Enum):
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class ObservedJob:
    name: str
    state: JobState


@dataclass(frozen=True)
class Action:
    kind: str  # create_vllm | create_job | delete_vllm | delete_job | create_seed | delete_seed
    model_id: str
    # Resource name for deletes; creates carry the full PendingModel instead.
    name: str | None = None
    model: PendingModel | None = None


def decide(
    pending: list[PendingModel],
    jobs: dict[str, ObservedJob],
    vllms: dict[str, str],
    *,
    seeds: dict[str, ObservedJob] | None = None,
    model_cache: bool = False,
    cache_exempt_models: frozenset[str] = frozenset(),
    keep_failed_jobs: bool = True,
) -> list[Action]:
    """One reconcile pass: compare the ledger's pending queue against the
    observed managed resources (both keyed by ledger model_id) and emit the
    creations/deletions that move the cluster toward the desired state."""
    actions: list[Action] = []
    pending_by_id = {p.model_id: p for p in pending}

    for model_id in sorted(
        set(pending_by_id) | set(jobs) | set(vllms) | set(seeds or {})
    ):
        model = pending_by_id.get(model_id)
        job = jobs.get(model_id)
        vllm = vllms.get(model_id)
        eligible = model is not None and model.eligible
        model_actions: list[Action] = []

        if eligible:
            blocked = (
                job is not None
                and job.state is JobState.FAILED
                and keep_failed_jobs
            )
            if blocked:
                # The Job stays for inspection (delete it manually to retry),
                # but its idle vLLM still gives the GPU back.
                if vllm is not None:
                    model_actions.append(Action("delete_vllm", model_id, name=vllm))
            else:
                if vllm is None:
                    model_actions.append(Action("create_vllm", model_id, model=model))
                if job is None:
                    model_actions.append(Action("create_job", model_id, model=model))
                elif job.state is JobState.SUCCEEDED:
                    # Drained earlier, but new events arrived: delete so the
                    # next cycle recreates it against the still-warm vLLM.
                    model_actions.append(Action("delete_job", model_id, name=job.name))
                elif job.state is JobState.FAILED:
                    model_actions.append(Action("delete_job", model_id, name=job.name))
                # ACTIVE job: it is draining; leave the pair alone.

                if (
                    model_cache
                    and seeds is not None
                    and model.model_name not in cache_exempt_models
                ):
                    seed = seeds.get(model_id)
                    creates = [
                        a for a in model_actions if a.kind.startswith("create_")
                    ]
                    if creates and (seed is None or seed.state is not JobState.SUCCEEDED):
                        # Withhold pair creation until the model is cached.
                        model_actions = [
                            a
                            for a in model_actions
                            if not a.kind.startswith("create_")
                        ]
                        if seed is None:
                            model_actions.append(
                                Action("create_seed", model_id, model=model)
                            )
                        elif seed.state is JobState.FAILED and not keep_failed_jobs:
                            model_actions.append(
                                Action("delete_seed", model_id, name=seed.name)
                            )
                        # ACTIVE, or FAILED+kept: wait.
                elif model_cache and seeds is not None:
                    # Cache-exempt model (uncached placement override): creates
                    # pass through ungated; a stale seed Job is useless — reap
                    # it once it is not actively running.
                    seed = seeds.get(model_id)
                    if seed is not None and seed.state is not JobState.ACTIVE:
                        model_actions.append(
                            Action("delete_seed", model_id, name=seed.name)
                        )
        else:
            # Nothing to verify (drained, or threshold NULL): wind the pair
            # down. An ACTIVE runner is left to notice and exit on its own.
            if job is not None and job.state is JobState.SUCCEEDED:
                model_actions.append(Action("delete_job", model_id, name=job.name))
            if job is not None and job.state is JobState.FAILED and not keep_failed_jobs:
                model_actions.append(Action("delete_job", model_id, name=job.name))
            if vllm is not None and (job is None or job.state is not JobState.ACTIVE):
                model_actions.append(Action("delete_vllm", model_id, name=vllm))

            if model_cache and seeds is not None:
                seed = seeds.get(model_id)
                if seed is not None and (
                    seed.state is JobState.SUCCEEDED
                    or (seed.state is JobState.FAILED and not keep_failed_jobs)
                ):
                    model_actions.append(
                        Action("delete_seed", model_id, name=seed.name)
                    )

        actions.extend(model_actions)

    return actions
