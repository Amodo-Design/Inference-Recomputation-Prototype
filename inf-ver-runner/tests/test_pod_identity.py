"""EndpointSlice resolution: one ready backend -> identity; zero or multiple
-> None (never guess); missing ServiceAccount files -> None (not in-cluster)."""

from __future__ import annotations

import httpx
import pytest

from app.pod_identity import PodIdentity, PodIdentityResolver, service_name_from_url


def _resolver(tmp_path, slices: list[dict]) -> PodIdentityResolver:
    (tmp_path / "token").write_text("test-token")
    (tmp_path / "namespace").write_text("infver")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-token"
        assert "kubernetes.io%2Fservice-name%3Dverify-m" in str(request.url) or \
            "kubernetes.io/service-name=verify-m" in str(request.url)
        return httpx.Response(200, json={"items": slices})

    return PodIdentityResolver(
        vllm_url="http://verify-m:8000/v1",
        sa_dir=str(tmp_path),
        api_base="https://kubernetes.default.svc",
        transport=httpx.MockTransport(handler),
        ttl_seconds=0.0,  # no caching in tests
    )


def _slice(endpoints: list[dict]) -> dict:
    return {"endpoints": endpoints}


def _ep(pod: str, node: str | None = "gpu-node-1", ready: bool = True) -> dict:
    return {
        "conditions": {"ready": ready},
        "targetRef": {"kind": "Pod", "name": pod},
        **({"nodeName": node} if node else {}),
    }


def test_service_name_from_url():
    assert service_name_from_url("http://verify-m.infver.svc:8000/v1") == "verify-m"
    assert service_name_from_url("http://verify-m:8000/v1") == "verify-m"


@pytest.mark.asyncio
async def test_single_ready_backend(tmp_path):
    r = _resolver(tmp_path, [_slice([_ep("verify-m-pod-1")])])
    assert await r.get() == PodIdentity("verify-m-pod-1", "gpu-node-1")


@pytest.mark.asyncio
async def test_multiple_backends_returns_none(tmp_path):
    r = _resolver(tmp_path, [_slice([_ep("p1"), _ep("p2")])])
    assert await r.get() is None


@pytest.mark.asyncio
async def test_not_ready_backend_ignored(tmp_path):
    r = _resolver(tmp_path, [_slice([_ep("p1"), _ep("p2", ready=False)])])
    assert await r.get() == PodIdentity("p1", "gpu-node-1")


@pytest.mark.asyncio
async def test_outside_cluster_returns_none(tmp_path):
    r = PodIdentityResolver(
        vllm_url="http://vllm:8000/v1", sa_dir=str(tmp_path / "missing")
    )
    assert await r.get() is None
