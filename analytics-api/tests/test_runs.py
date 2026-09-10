"""/runs/{run_id}/cohort and /runs/{run_id}/economics — first-class run
views over the ledger's event tables plus the gpu-enricher's window-activity
contract.

Cohort = inference events whose model matches one of the run's allocation
models and whose ts falls in [run.started_at, run.finished_at ?? now].
Economics adds prover-side pod-window GPU integrals (via gpu-enricher) and
verify-side per-event enrichment sums, both null-safe against a missing
enricher/enrichment data."""

from __future__ import annotations

import datetime

import httpx
import pytest

from app.main import app
from conftest import seed_enrichment_row, seed_event_pair, seed_run

UTC = datetime.timezone.utc


async def test_cohort_counts_by_window_and_model(client, session):
    await seed_run(
        session,
        "run-1",
        started_at=datetime.datetime(2026, 8, 3, 10, 0, 0, tzinfo=UTC),
        finished_at=datetime.datetime(2026, 8, 3, 10, 1, 0, tzinfo=UTC),
        settings={
            "models": [
                {"model": "m-a", "percent": 50, "concurrency": 1},
                {"model": "m-b", "percent": 50, "concurrency": 1},
            ]
        },
    )

    # m-a: 2 events inside the window, pass/fail mix, with token counts.
    await seed_event_pair(
        session,
        model_name="m-a",
        inf_ts=datetime.datetime(2026, 8, 3, 10, 0, 10, tzinfo=UTC),
        ver_ts=datetime.datetime(2026, 8, 3, 10, 0, 15, tzinfo=UTC),
        result="pass",
        verifier_detail={"prompt_token_count": 10, "output_token_count": 5},
    )
    await seed_event_pair(
        session,
        model_name="m-a",
        inf_ts=datetime.datetime(2026, 8, 3, 10, 0, 20, tzinfo=UTC),
        ver_ts=datetime.datetime(2026, 8, 3, 10, 0, 25, tzinfo=UTC),
        result="fail",
        verifier_detail={"prompt_token_count": 20, "output_token_count": 8},
    )
    # m-a: 1 event OUTSIDE the window — must not leak into the cohort.
    await seed_event_pair(
        session,
        model_name="m-a",
        inf_ts=datetime.datetime(2026, 8, 3, 9, 0, 0, tzinfo=UTC),
        ver_ts=datetime.datetime(2026, 8, 3, 9, 0, 5, tzinfo=UTC),
        result="pass",
        verifier_detail={"prompt_token_count": 999, "output_token_count": 999},
    )
    # m-b: 1 event inside the window.
    await seed_event_pair(
        session,
        model_name="m-b",
        inf_ts=datetime.datetime(2026, 8, 3, 10, 0, 30, tzinfo=UTC),
        ver_ts=datetime.datetime(2026, 8, 3, 10, 0, 35, tzinfo=UTC),
        result="pass",
        verifier_detail={"prompt_token_count": 7, "output_token_count": 3},
    )
    # Unrelated model (not in the run's allocation) inside the window.
    await seed_event_pair(
        session,
        model_name="m-unrelated",
        inf_ts=datetime.datetime(2026, 8, 3, 10, 0, 40, tzinfo=UTC),
        ver_ts=datetime.datetime(2026, 8, 3, 10, 0, 45, tzinfo=UTC),
        result="pass",
        verifier_detail={"prompt_token_count": 1, "output_token_count": 1},
    )

    r = await client.get("/runs/run-1/cohort")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run_id"] == "run-1"
    by_model = {m["model"]: m for m in body["models"]}
    assert by_model["m-a"]["event_count"] == 2
    assert by_model["m-a"]["verified_count"] == 2
    assert by_model["m-a"]["pass_count"] == 1 and by_model["m-a"]["fail_count"] == 1
    assert by_model["m-a"]["unverifiable_count"] == 0
    assert by_model["m-a"]["prompt_tokens"] == 30
    assert by_model["m-a"]["output_tokens"] == 13
    assert by_model["m-b"]["event_count"] == 1
    assert by_model["m-b"]["prompt_tokens"] == 7
    assert by_model["m-b"]["output_tokens"] == 3
    assert "m-unrelated" not in by_model


async def test_cohort_404_unknown_run(client):
    assert (await client.get("/runs/nope/cohort")).status_code == 404


async def test_economics_labels_methods_and_composes_enricher(client, session, monkeypatch):
    await seed_run(
        session,
        "run-1",
        started_at=datetime.datetime(2026, 8, 3, 12, 0, 0, tzinfo=UTC),
        finished_at=datetime.datetime(2026, 8, 3, 12, 1, 0, tzinfo=UTC),
        settings={"models": [{"model": "m-a", "percent": 100, "concurrency": 1}]},
    )

    _, ver_id_1 = await seed_event_pair(
        session,
        model_name="m-a",
        inf_ts=datetime.datetime(2026, 8, 3, 12, 0, 5, tzinfo=UTC),
        inf_started=datetime.datetime(2026, 8, 3, 12, 0, 3, tzinfo=UTC),
        ver_ts=datetime.datetime(2026, 8, 3, 12, 0, 10, tzinfo=UTC),
        ver_started=datetime.datetime(2026, 8, 3, 12, 0, 9, tzinfo=UTC),
        pod_name="prover-pod-1",
        ver_pod_name="verify-pod-1",
        result="pass",
        verifier_detail={
            "prompt_token_count": 100,
            "output_token_count": 50,
            "runner_concurrency": 3,
        },
    )
    _, ver_id_2 = await seed_event_pair(
        session,
        model_name="m-a",
        inf_ts=datetime.datetime(2026, 8, 3, 12, 0, 15, tzinfo=UTC),
        inf_started=datetime.datetime(2026, 8, 3, 12, 0, 13, tzinfo=UTC),
        ver_ts=datetime.datetime(2026, 8, 3, 12, 0, 20, tzinfo=UTC),
        ver_started=datetime.datetime(2026, 8, 3, 12, 0, 19, tzinfo=UTC),
        pod_name="prover-pod-1",
        ver_pod_name="verify-pod-1",
        result="pass",
        verifier_detail={
            "prompt_token_count": 50,
            "output_token_count": 25,
            "runner_concurrency": 3,
        },
    )

    # tensor_active_time_s is seeded but must now be IGNORED by verify busy
    # (it comes from the gpu-enricher pod-window call below instead).
    await seed_enrichment_row(
        session, ver_id_1, tensor_active_time_s=0.5, concurrent_events=2
    )
    await seed_enrichment_row(
        session, ver_id_2, tensor_active_time_s=0.25, concurrent_events=3
    )

    captured_calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_calls.append(dict(request.url.params))
        return httpx.Response(
            200,
            json={
                "busy_seconds": 2.0,
                "sample_count": 10,
                "sm_occupancy_mean": 0.5,
                "pipe_activity_s": {},
                "window_s": 60.0,
            },
        )

    monkeypatch.setattr(
        app.state, "enricher_transport", httpx.MockTransport(handler), raising=False
    )

    r = await client.get("/runs/run-1/economics")
    assert r.status_code == 200, r.text
    m = r.json()["models"][0]
    assert m["prover"]["method"] == "pod-window integral"
    assert m["prover"]["busy_seconds"] == 2.0
    assert m["prover"]["sample_count"] == 10
    assert m["prover"]["pods"] == ["prover-pod-1"]
    assert len(captured_calls) == 2
    # re.escape("prover-pod-1") escapes each "-" to a single backslash;
    # runs.py then doubles every backslash so the pattern survives PromQL's
    # string-literal layer intact (a lone "\-" is not a valid Go escape and
    # Prometheus 400s). The runtime string therefore carries a literal
    # double backslash before each hyphen — written here as "\\\\" per
    # hyphen since each source "\\\\" is two actual backslash characters.
    assert captured_calls[0]["pod_pattern"] == "^(prover\\\\-pod\\\\-1)$"

    assert m["verify"]["method"] == "pod-window integral"
    assert m["verify"]["busy_seconds"] == 2.0
    assert m["verify"]["sample_count"] == 10
    assert m["verify"]["pods"] == ["verify-pod-1"]
    assert captured_calls[1]["pod_pattern"] == "^(verify\\\\-pod\\\\-1)$"
    # drain window: [min(verification started_at), max(verification ts)].
    assert captured_calls[1]["from"] == "2026-08-03T12:00:09+00:00"
    assert captured_calls[1]["to"] == "2026-08-03T12:00:20+00:00"
    assert m["verify"]["prompt_plus_output_tokens"] == 225
    assert m["verify"]["busy_per_token"] == pytest.approx(2.0 / 225)
    assert m["verify"]["event_count"] == 2
    assert m["verify"]["window"] == {
        "from": "2026-08-03T12:00:09+00:00",
        "to": "2026-08-03T12:00:20+00:00",
    }
    # configured = max(runner_concurrency) = 3; observed comes from
    # concurrent_events (2, 3) — observed ≈ configured - 1 is expected,
    # since concurrent_events counts OTHER concurrently-running events.
    assert m["verify"]["concurrency"] == {
        "configured": 3,
        "observed_mean": 2.5,
        "observed_max": 3,
    }
    # prover side unchanged — still its own pattern/window, no concurrency
    # block (that's verify-only).
    assert "concurrency" not in m["prover"]


async def test_economics_null_sides_degrade_gracefully(client, session):
    await seed_run(
        session,
        "run-1",
        started_at=datetime.datetime(2026, 8, 3, 13, 0, 0, tzinfo=UTC),
        finished_at=datetime.datetime(2026, 8, 3, 13, 1, 0, tzinfo=UTC),
        settings={"models": [{"model": "m-a", "percent": 100, "concurrency": 1}]},
    )
    await seed_event_pair(
        session,
        model_name="m-a",
        inf_ts=datetime.datetime(2026, 8, 3, 13, 0, 5, tzinfo=UTC),
        ver_ts=datetime.datetime(2026, 8, 3, 13, 0, 10, tzinfo=UTC),
        result="pass",
        verifier_detail={"prompt_token_count": 10, "output_token_count": 5},
    )

    r = await client.get("/runs/run-1/economics")
    assert r.status_code == 200, r.text
    m = r.json()["models"][0]
    assert m["prover"]["method"] == "pod-window integral"
    assert m["prover"]["busy_seconds"] is None
    assert m["prover"]["sample_count"] is None
    assert m["prover"]["pods"] == []
    assert m["verify"]["method"] == "pod-window integral"
    assert m["verify"]["busy_seconds"] is None
    assert m["verify"]["busy_per_token"] is None
    assert m["verify"]["pods"] == []
    assert m["verify"]["concurrency"] == {
        "configured": None,
        "observed_mean": None,
        "observed_max": None,
    }


async def test_economics_verify_window_is_drain_window(client, session, monkeypatch):
    """The verify-side gpu-enricher call must use the drain window
    [min(verification started_at), max(verification ts)], not the run
    window — verification can run well past the run's own finished_at."""
    await seed_run(
        session,
        "run-1",
        started_at=datetime.datetime(2026, 8, 3, 15, 0, 0, tzinfo=UTC),
        finished_at=datetime.datetime(2026, 8, 3, 15, 1, 0, tzinfo=UTC),
        settings={"models": [{"model": "m-a", "percent": 100, "concurrency": 1}]},
    )
    await seed_event_pair(
        session,
        model_name="m-a",
        inf_ts=datetime.datetime(2026, 8, 3, 15, 0, 5, tzinfo=UTC),
        ver_ts=datetime.datetime(2026, 8, 3, 15, 0, 40, tzinfo=UTC),
        ver_started=datetime.datetime(2026, 8, 3, 15, 0, 12, tzinfo=UTC),
        ver_pod_name="verify-pod-1",
        result="pass",
        verifier_detail={"prompt_token_count": 10, "output_token_count": 5},
    )
    await seed_event_pair(
        session,
        model_name="m-a",
        inf_ts=datetime.datetime(2026, 8, 3, 15, 0, 20, tzinfo=UTC),
        ver_ts=datetime.datetime(2026, 8, 3, 15, 1, 30, tzinfo=UTC),
        ver_started=datetime.datetime(2026, 8, 3, 15, 0, 8, tzinfo=UTC),
        ver_pod_name="verify-pod-1",
        result="pass",
        verifier_detail={"prompt_token_count": 20, "output_token_count": 10},
    )

    captured_calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_calls.append(dict(request.url.params))
        return httpx.Response(200, json={"busy_seconds": 1.0, "sample_count": 4})

    monkeypatch.setattr(
        app.state, "enricher_transport", httpx.MockTransport(handler), raising=False
    )

    r = await client.get("/runs/run-1/economics")
    assert r.status_code == 200, r.text

    # No inference pod_name was seeded, so the prover side has no pods to
    # call the enricher for — the only call made is the verify-side one.
    # Its window = [min started_at, max ts] across the two verifications =
    # [15:00:08, 15:01:30], not the run window [15:00:00, 15:01:00].
    assert len(captured_calls) == 1
    assert captured_calls[0]["from"] == "2026-08-03T15:00:08+00:00"
    assert captured_calls[0]["to"] == "2026-08-03T15:01:30+00:00"


async def test_economics_garbage_200_from_enricher_degrades_to_null(client, session, monkeypatch):
    """A gpu-enricher that answers 200 with a non-JSON body (e.g. a
    misrouted request hitting a reverse proxy's error page) must not 5xx
    this route — it degrades exactly like a connect error or a 4xx would."""
    await seed_run(
        session,
        "run-1",
        started_at=datetime.datetime(2026, 8, 3, 14, 0, 0, tzinfo=UTC),
        finished_at=datetime.datetime(2026, 8, 3, 14, 1, 0, tzinfo=UTC),
        settings={"models": [{"model": "m-a", "percent": 100, "concurrency": 1}]},
    )
    await seed_event_pair(
        session,
        model_name="m-a",
        inf_ts=datetime.datetime(2026, 8, 3, 14, 0, 5, tzinfo=UTC),
        ver_ts=datetime.datetime(2026, 8, 3, 14, 0, 10, tzinfo=UTC),
        pod_name="prover-pod-1",
        result="pass",
        verifier_detail={"prompt_token_count": 10, "output_token_count": 5},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>", headers={"content-type": "text/html"})

    monkeypatch.setattr(
        app.state, "enricher_transport", httpx.MockTransport(handler), raising=False
    )

    r = await client.get("/runs/run-1/economics")
    assert r.status_code == 200, r.text
    m = r.json()["models"][0]
    assert m["prover"]["busy_seconds"] is None
    assert m["prover"]["sample_count"] is None


async def test_economics_409_before_start(client, session):
    await seed_run(
        session,
        "run-1",
        state="queued",
        started_at=None,
        finished_at=None,
        settings={"models": [{"model": "m-a", "percent": 100, "concurrency": 1}]},
    )
    r = await client.get("/runs/run-1/economics")
    assert r.status_code == 409
    assert r.json()["detail"] == "run has not started"
