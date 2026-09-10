"""Prometheus range-query client: rectangle-rule integration, per-timestamp
summing across multi-GPU series, empty-window handling."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from app.prometheus import PrometheusClient, integrate, mean

T0 = datetime(2026, 7, 30, 0, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 7, 30, 0, 0, 2, tzinfo=timezone.utc)


def test_integrate_rectangle_rule():
    # 3 samples at 200ms of ratio 0.5 -> 0.3 busy-seconds.
    assert integrate([0.5, 0.5, 0.5]) == pytest.approx(0.3)


def test_integrate_empty_is_none():
    assert integrate([]) is None
    assert mean([]) is None


def _client(result: list[dict]) -> PrometheusClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/api/v1/query_range" in str(request.url)
        return httpx.Response(
            200, json={"status": "success", "data": {"result": result}}
        )

    return PrometheusClient(
        "http://prom:9090", transport=httpx.MockTransport(handler)
    )


async def test_single_series_values():
    client = _client(
        [{"values": [[0.0, "0.5"], [0.2, "0.7"]]}]
    )
    values = await client.range_values(
        "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", '{pod="p"}', T0, T1
    )
    assert values == pytest.approx([0.5, 0.7])


async def test_multi_gpu_series_summed_per_timestamp():
    client = _client(
        [
            {"values": [[0.0, "0.5"], [0.2, "0.5"]]},
            {"values": [[0.0, "0.25"], [0.2, "0.25"]]},
        ]
    )
    values = await client.range_values(
        "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", '{pod="p"}', T0, T1
    )
    assert values == pytest.approx([0.75, 0.75])


async def test_no_series_is_empty():
    client = _client([])
    assert await client.range_values("X", '{pod="p"}', T0, T1) == []


async def test_long_windows_are_chunked_under_the_point_cap():
    """A multi-hour window must split into <=MAX_CHUNK_SECONDS query_range
    calls (Prometheus caps at 11,000 points; 200ms over hours would 400),
    contiguous and boundary-safe."""
    import datetime as dt

    from app.prometheus import MAX_CHUNK_SECONDS, PrometheusClient

    calls: list[tuple[float, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        start = float(request.url.params["start"])
        end = float(request.url.params["end"])
        calls.append((start, end))
        # One sample at each chunk's start and end so boundary handling shows.
        return httpx.Response(
            200,
            json={"data": {"result": [{"values": [[start, "0.5"], [end, "0.5"]]}]}},
        )

    client = PrometheusClient("http://prom", transport=httpx.MockTransport(handler))
    t0 = dt.datetime(2026, 8, 5, 13, 0, 0, tzinfo=dt.timezone.utc)
    t1 = t0 + dt.timedelta(hours=2, minutes=30)  # 9000s -> 5 chunks of <=2000s
    values = await client.range_values("M", '{pod="p"}', t0, t1)

    assert len(calls) == 5
    # Chunks are contiguous, cover [t0, t1], and never exceed the cap.
    assert calls[0][0] == t0.timestamp() and calls[-1][1] == t1.timestamp()
    for (s, e), (s2, _) in zip(calls, calls[1:]):
        assert e == s2
        assert e - s <= MAX_CHUNK_SECONDS
    # Boundary timestamps shared by adjacent chunks are merged, not
    # double-counted: 5 chunks x 2 samples with 4 shared boundaries -> 6.
    assert len(values) == 6
    assert all(v == 0.5 for v in values)


async def test_short_windows_stay_single_query():
    import datetime as dt

    from app.prometheus import PrometheusClient

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params["query"])
        return httpx.Response(200, json={"data": {"result": []}})

    client = PrometheusClient("http://prom", transport=httpx.MockTransport(handler))
    t0 = dt.datetime(2026, 8, 5, 13, 0, 0, tzinfo=dt.timezone.utc)
    await client.range_values("M", '{pod="p"}', t0, t0 + dt.timedelta(seconds=30))
    assert len(calls) == 1
