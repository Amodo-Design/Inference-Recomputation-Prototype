"""Spawn/reap decision matrix (pure logic, no cluster)."""

from __future__ import annotations

from app.reconciler import Action, JobState, ObservedJob, decide

from helpers import make_model


def kinds(actions: list[Action]) -> list[str]:
    return [a.kind for a in actions]


def test_pending_model_with_threshold_spawns_pair():
    model = make_model()
    actions = decide([model], jobs={}, vllms={})
    assert kinds(actions) == ["create_vllm", "create_job"]
    assert all(a.model is model for a in actions)


def test_null_threshold_model_is_skipped():
    model = make_model(threshold=None)
    assert decide([model], jobs={}, vllms={}) == []


def test_running_pair_left_alone_while_pending():
    model = make_model()
    actions = decide(
        [model],
        jobs={model.model_id: ObservedJob("inf-ver-runner-x", JobState.ACTIVE)},
        vllms={model.model_id: "verify-x"},
    )
    assert actions == []


def test_partial_pair_completed():
    model = make_model()
    actions = decide(
        [model],
        jobs={},
        vllms={model.model_id: "verify-x"},
    )
    assert kinds(actions) == ["create_job"]


def test_succeeded_job_with_no_pending_reaps_pair():
    model_id = "m-1"
    actions = decide(
        [],
        jobs={model_id: ObservedJob("inf-ver-runner-x", JobState.SUCCEEDED)},
        vllms={model_id: "verify-x"},
    )
    assert kinds(actions) == ["delete_job", "delete_vllm"]
    assert actions[0].name == "inf-ver-runner-x"
    assert actions[1].name == "verify-x"


def test_threshold_cleared_mid_flight_reaps_when_job_finishes():
    # Events still pending but threshold now NULL: an ACTIVE runner exits 0 on
    # its own; once SUCCEEDED, the pair is torn down and events stay pending.
    model = make_model(threshold=None)
    active = decide(
        [model],
        jobs={model.model_id: ObservedJob("inf-ver-runner-x", JobState.ACTIVE)},
        vllms={model.model_id: "verify-x"},
    )
    assert active == []
    done = decide(
        [model],
        jobs={model.model_id: ObservedJob("inf-ver-runner-x", JobState.SUCCEEDED)},
        vllms={model.model_id: "verify-x"},
    )
    assert kinds(done) == ["delete_job", "delete_vllm"]


def test_succeeded_job_with_new_pending_is_recreated():
    # Runner drained and exited, then new events arrived: delete the stale
    # Job (recreated next cycle against the still-warm vLLM), keep the vLLM.
    model = make_model()
    actions = decide(
        [model],
        jobs={model.model_id: ObservedJob("inf-ver-runner-x", JobState.SUCCEEDED)},
        vllms={model.model_id: "verify-x"},
    )
    assert kinds(actions) == ["delete_job"]


def test_failed_job_kept_for_inspection_but_vllm_reaped():
    model = make_model()
    actions = decide(
        [model],
        jobs={model.model_id: ObservedJob("inf-ver-runner-x", JobState.FAILED)},
        vllms={model.model_id: "verify-x"},
        keep_failed_jobs=True,
    )
    assert kinds(actions) == ["delete_vllm"]


def test_failed_job_deleted_for_retry_when_not_kept():
    model = make_model()
    actions = decide(
        [model],
        jobs={model.model_id: ObservedJob("inf-ver-runner-x", JobState.FAILED)},
        vllms={model.model_id: "verify-x"},
        keep_failed_jobs=False,
    )
    assert kinds(actions) == ["delete_job"]


def test_orphan_vllm_reaped():
    actions = decide([], jobs={}, vllms={"m-1": "verify-x"})
    assert kinds(actions) == ["delete_vllm"]


def test_independent_models_reconciled_separately():
    spawning = make_model(model_id="a-spawn", model_name="Org/Model-A")
    paused = make_model(model_id="b-paused", threshold=None, model_name="Org/Model-B")
    finished_id = "c-finished"
    actions = decide(
        [spawning, paused],
        jobs={finished_id: ObservedJob("inf-ver-runner-c", JobState.SUCCEEDED)},
        vllms={finished_id: "verify-c"},
    )
    assert [(a.kind, a.model_id) for a in actions] == [
        ("create_vllm", "a-spawn"),
        ("create_job", "a-spawn"),
        ("delete_job", finished_id),
        ("delete_vllm", finished_id),
    ]


def _seed(state):
    return {"m1": ObservedJob(name="seed-m1", state=state)}


def test_cache_off_never_emits_seed_actions():
    actions = decide([make_model(model_id="m1")], {}, {}, seeds=None, model_cache=False)
    assert {a.kind for a in actions} == {"create_vllm", "create_job"}


def test_cache_on_gates_pair_behind_seed():
    # No seed yet: only create_seed.
    actions = decide([make_model(model_id="m1")], {}, {}, seeds={}, model_cache=True)
    assert [a.kind for a in actions] == ["create_seed"]
    # Seed running: wait.
    actions = decide(
        [make_model(model_id="m1")], {}, {}, seeds=_seed(JobState.ACTIVE), model_cache=True
    )
    assert actions == []
    # Seed done: pair as normal.
    actions = decide(
        [make_model(model_id="m1")], {}, {}, seeds=_seed(JobState.SUCCEEDED), model_cache=True
    )
    assert {a.kind for a in actions} == {"create_vllm", "create_job"}


def test_cache_on_failed_seed_blocks_and_respects_keep_flag():
    actions = decide(
        [make_model(model_id="m1")], {}, {}, seeds=_seed(JobState.FAILED),
        model_cache=True, keep_failed_jobs=True,
    )
    assert actions == []  # kept for inspection, pair withheld
    actions = decide(
        [make_model(model_id="m1")], {}, {}, seeds=_seed(JobState.FAILED),
        model_cache=True, keep_failed_jobs=False,
    )
    assert [(a.kind, a.name) for a in actions] == [("delete_seed", "seed-m1")]


def test_cache_on_deletes_pass_through_without_seed():
    # Runner drained (SUCCEEDED) with new pending events, no seed (TTL'd away):
    # the delete passes through; the recreate next cycle will re-gate.
    jobs = {"m1": ObservedJob(name="inf-ver-runner-m1", state=JobState.SUCCEEDED)}
    actions = decide(
        [make_model(model_id="m1")], jobs, {"m1": "verify-m1"}, seeds={}, model_cache=True
    )
    assert [(a.kind, a.name) for a in actions] == [("delete_job", "inf-ver-runner-m1")]


def test_ineligible_model_cleans_up_seed():
    actions = decide(
        [], {}, {}, seeds=_seed(JobState.SUCCEEDED), model_cache=True
    )
    assert [(a.kind, a.name) for a in actions] == [("delete_seed", "seed-m1")]


def test_model_cache_on_with_no_seeds_dict_never_gates():
    # model_cache=True but seeds=None (caller didn't observe any seeds, e.g.
    # caching not wired up yet): must be byte-identical to cache-off — the
    # binding contract is "model_cache=False OR seeds is None -> no seed
    # actions ever", not "model_cache=False AND seeds is None".
    actions = decide([make_model(model_id="m1")], {}, {}, seeds=None, model_cache=True)
    assert {a.kind for a in actions} == {"create_vllm", "create_job"}
    assert not any(a.kind.startswith("_seed") or "seed" in a.kind for a in actions)


def test_cache_on_blocked_branch_ignores_seed_state():
    # FAILED runner Job kept for inspection: the pair is already blocked for
    # reasons unrelated to caching. A SUCCEEDED seed must not change the
    # blocked branch's output — no creates, no seed actions, just the usual
    # vLLM reap.
    actions = decide(
        [make_model(model_id="m1")],
        jobs={"m1": ObservedJob(name="inf-ver-runner-m1", state=JobState.FAILED)},
        vllms={"m1": "verify-m1"},
        seeds=_seed(JobState.SUCCEEDED),
        model_cache=True,
        keep_failed_jobs=True,
    )
    assert [(a.kind, a.name) for a in actions] == [("delete_vllm", "verify-m1")]


def test_cache_exempt_model_skips_seed_gating():
    # Anchor note: brief's snippet used a `_pending`-style builder; this repo's
    # actual builder is `make_model` (tests/helpers.py), which sets
    # `model_name` explicitly rather than deriving it. Adapted accordingly.
    model = make_model(model_id="m1")
    actions = decide(
        [model], {}, {}, seeds={}, model_cache=True,
        cache_exempt_models=frozenset({model.model_name}),
    )
    assert {a.kind for a in actions} == {"create_vllm", "create_job"}


def test_cache_exempt_model_stale_seed_cleaned_up():
    model = make_model(model_id="m1")
    actions = decide(
        [model], {}, {}, seeds=_seed(JobState.SUCCEEDED), model_cache=True,
        cache_exempt_models=frozenset({model.model_name}),
    )
    assert ("delete_seed", "seed-m1") in [(a.kind, a.name) for a in actions]
    assert {a.kind for a in actions} >= {"create_vllm", "create_job"}


def test_cache_exempt_model_stale_active_seed_not_reaped():
    # An ACTIVE seed is still doing work; exempt-model creates still pass
    # through ungated, but the seed itself is left alone (only non-ACTIVE
    # seeds are reaped).
    model = make_model(model_id="m1")
    actions = decide(
        [model], {}, {}, seeds=_seed(JobState.ACTIVE), model_cache=True,
        cache_exempt_models=frozenset({model.model_name}),
    )
    assert {a.kind for a in actions} == {"create_vllm", "create_job"}


def test_non_exempt_model_still_gated():
    model = make_model(model_id="m1")
    actions = decide(
        [model], {}, {}, seeds={}, model_cache=True,
        cache_exempt_models=frozenset({"someone/else"}),
    )
    assert [a.kind for a in actions] == ["create_seed"]
