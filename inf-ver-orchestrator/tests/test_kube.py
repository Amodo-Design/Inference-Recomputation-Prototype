"""KubeClient's prefix-partitioning against real (stubbed) list_namespaced_job
responses — no cluster involved, but exercising the actual _list_jobs_with_prefix
body rather than a FakeKube passthrough."""

from __future__ import annotations

from types import SimpleNamespace

from app.kube import KubeClient, _SELECTOR
from app.reconciler import JobState, ObservedJob
from app.templates import MANAGED_BY, MODEL_ID_LABEL


def _job(name: str, *, labels: dict[str, str] | None = None, succeeded: bool = True):
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, labels=labels),
        status=SimpleNamespace(succeeded=1 if succeeded else 0, conditions=[]),
    )


class _StubBatch:
    def __init__(self, items: list) -> None:
        self.items = items
        self.calls: list[dict] = []

    def list_namespaced_job(self, namespace: str, label_selector: str):
        self.calls.append({"namespace": namespace, "label_selector": label_selector})
        return SimpleNamespace(items=self.items)


def _make_client(items: list) -> tuple[KubeClient, _StubBatch]:
    kube = KubeClient.__new__(KubeClient)
    kube._namespace = "infver"
    stub = _StubBatch(items)
    kube._batch = stub
    return kube, stub


def test_list_managed_jobs_and_seed_jobs_partition_real_batch_response():
    items = [
        _job("inf-ver-runner-x", labels={MODEL_ID_LABEL: "m1"}),
        _job("seed-x", labels={MODEL_ID_LABEL: "m2"}),
        _job("some-other-legacy-job", labels={MODEL_ID_LABEL: "m3"}),
        _job("seed-nolabel", labels=None),
    ]
    kube, stub = _make_client(items)

    runner_jobs = kube.list_managed_jobs()
    seed_jobs = kube.list_managed_seed_jobs()

    assert runner_jobs == {"m1": ObservedJob("inf-ver-runner-x", JobState.SUCCEEDED)}
    assert seed_jobs == {"m2": ObservedJob("seed-x", JobState.SUCCEEDED)}

    # Legacy-named and unlabeled jobs must appear in neither view.
    assert "m3" not in runner_jobs and "m3" not in seed_jobs
    assert all(j.name != "seed-nolabel" for j in runner_jobs.values())
    assert all(j.name != "seed-nolabel" for j in seed_jobs.values())
    assert all(j.name != "some-other-legacy-job" for j in seed_jobs.values())

    # Both calls must use the managed-by label selector.
    assert stub.calls
    for call in stub.calls:
        assert call["label_selector"] == _SELECTOR
        assert call["label_selector"] == f"app.kubernetes.io/managed-by={MANAGED_BY}"
