"""/window-activity: live pod-window aggregates with guardrails."""

from __future__ import annotations

import datetime as dt

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

import app.main as main_mod
from app.prometheus import HEADLINE_METRIC, METRICS, OCCUPANCY_METRIC, PrometheusClient

FROM = dt.datetime(2026, 8, 3, 12, 0, 0, tzinfo=dt.timezone.utc)
TO = dt.datetime(2026, 8, 3, 12, 0, 10, tzinfo=dt.timezone.utc)


def prom_stub(values_by_metric: dict[str, list[float]]) -> PrometheusClient:
    def handler(request: httpx.Request) -> httpx.Response:
        metric = request.url.params["query"].split("{")[0]
        values = values_by_metric.get(metric, [])
        return httpx.Response(
            200,
            json={
                "data": {
                    "result": [
                        {"values": [[i * 0.2, str(v)] for i, v in enumerate(values)]}
                    ]
                    if values
                    else []
                }
            },
        )

    return PrometheusClient("http://prom", transport=httpx.MockTransport(handler))


@pytest.fixture
def client():
    return AsyncClient(transport=ASGITransport(app=main_mod.app), base_url="http://t")


async def test_window_activity_aggregates(client, monkeypatch):
    stub = prom_stub(
        {
            HEADLINE_METRIC: [0.5, 0.5, 1.0],          # integrate -> 0.4 busy-s
            OCCUPANCY_METRIC: [0.2, 0.4],              # mean -> 0.3
            "DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE": [1.0],  # 0.2 busy-s
        }
    )
    main_mod.app.state.prom = stub
    r = await client.get(
        "/window-activity",
        params={"pod_pattern": "qwen.*-kserve-.*", "from": FROM.isoformat(), "to": TO.isoformat()},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["busy_seconds"] == pytest.approx(0.4)
    assert body["sm_occupancy_mean"] == pytest.approx(0.3)
    assert body["pipe_activity_s"]["DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE"] == pytest.approx(0.2)
    assert body["sample_count"] == 3
    assert body["window_s"] == pytest.approx(10.0)


async def test_window_activity_selector_is_regex_pod_matcher(client):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params["query"])
        return httpx.Response(200, json={"data": {"result": []}})

    main_mod.app.state.prom = PrometheusClient("http://prom", transport=httpx.MockTransport(handler))
    r = await client.get(
        "/window-activity",
        params={"pod_pattern": "abc-.*", "from": FROM.isoformat(), "to": TO.isoformat()},
    )
    assert r.status_code == 200
    assert all(q.endswith('{pod=~"abc-.*"}') for q in seen)
    assert len(seen) == len(METRICS)


async def test_window_guardrails(client):
    main_mod.app.state.prom = prom_stub({})
    async def get(**params):
        return await client.get("/window-activity", params=params)

    base = {"pod_pattern": "x.*", "from": FROM.isoformat()}
    assert (await get(**base, to=FROM.isoformat())).status_code == 400            # to <= from
    assert (await get(**base, to=(FROM + dt.timedelta(hours=7)).isoformat())).status_code == 400  # > 6h
    assert (
        await get(pod_pattern="p" * 513, **{"from": FROM.isoformat()}, to=TO.isoformat())
    ).status_code == 400                                                          # pattern too long
    assert (await get(pod_pattern="", **{"from": FROM.isoformat()}, to=TO.isoformat())).status_code == 400


async def test_window_503_without_prometheus(client):
    if hasattr(main_mod.app.state, "prom"):
        del main_mod.app.state.prom
    r = await client.get(
        "/window-activity",
        params={"pod_pattern": "x.*", "from": FROM.isoformat(), "to": TO.isoformat()},
    )
    assert r.status_code == 503


async def test_window_activity_doubled_backslash_pattern_reaches_prometheus_verbatim(client):
    """analytics-api doubles re.escape's backslashes so the pattern survives
    PromQL's string-literal layer (a lone `\\-` is not a valid Go escape and
    Prometheus 400s). Prove the doubled-backslash form arrives at Prometheus
    exactly as sent, and the endpoint still 200s."""
    pattern = "^(prover\\\\-pod\\\\-1)$"
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params["query"])
        return httpx.Response(200, json={"data": {"result": []}})

    main_mod.app.state.prom = PrometheusClient("http://prom", transport=httpx.MockTransport(handler))
    r = await client.get(
        "/window-activity",
        params={"pod_pattern": pattern, "from": FROM.isoformat(), "to": TO.isoformat()},
    )
    assert r.status_code == 200, r.text
    assert all(q.endswith(f'{{pod=~"{pattern}"}}') for q in seen)
    assert len(seen) == len(METRICS)


async def test_window_activity_rejects_double_quote_in_pattern(client):
    main_mod.app.state.prom = prom_stub({})
    r = await client.get(
        "/window-activity",
        params={"pod_pattern": 'abc"}, or up{}=="', "from": FROM.isoformat(), "to": TO.isoformat()},
    )
    assert r.status_code == 400


async def test_window_activity_prometheus_400_becomes_502(client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad query")

    main_mod.app.state.prom = PrometheusClient("http://prom", transport=httpx.MockTransport(handler))
    r = await client.get(
        "/window-activity",
        params={"pod_pattern": "abc.*", "from": FROM.isoformat(), "to": TO.isoformat()},
    )
    assert r.status_code == 502
    assert "bad query" in r.json()["detail"]
