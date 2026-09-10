"""Resolve the verifier vLLM Service's backend pod via EndpointSlices.

The runner sits outside its verifier pod (a plain HTTP client of the
verify-<slug> Service), and the client socket can never see the pod IP —
kube-proxy's DNAT is reversed on reply — so the K8s API is the only way to
learn which pod serves the traffic. EndpointSlices carry both the pod name
(targetRef.name) and its node (nodeName), so `endpointslices` get/list is
the only RBAC needed.

Resolution is TTL-cached: the Job and its verifier are created together per
drain cycle, so identity is normally stable for the Job's lifetime, but a
verifier pod restart mid-drain must not silently attribute events to a dead
pod. Zero or multiple ready backends resolve to None — never a guess.
Outside a cluster (no ServiceAccount files) every call returns None.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)

DEFAULT_SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"
DEFAULT_API_BASE = "https://kubernetes.default.svc"
DEFAULT_TTL_SECONDS = 30.0


@dataclass(frozen=True)
class PodIdentity:
    pod_name: str
    node_name: str | None


def service_name_from_url(vllm_url: str) -> str | None:
    """First DNS label of the URL's hostname == the K8s Service name."""
    hostname = urlparse(vllm_url).hostname
    return hostname.split(".")[0] if hostname else None


class PodIdentityResolver:
    def __init__(
        self,
        vllm_url: str,
        sa_dir: str = DEFAULT_SA_DIR,
        api_base: str = DEFAULT_API_BASE,
        transport: httpx.AsyncBaseTransport | None = None,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._service = service_name_from_url(vllm_url)
        self._sa_dir = Path(sa_dir)
        self._api_base = api_base
        self._transport = transport
        self._ttl = ttl_seconds
        self._cached_at: float | None = None
        self._cached: PodIdentity | None = None

    def _read_sa(self, name: str) -> str | None:
        try:
            return (self._sa_dir / name).read_text().strip()
        except OSError:
            return None

    async def get(self) -> PodIdentity | None:
        now = time.monotonic()
        if self._cached_at is not None and now - self._cached_at < self._ttl:
            return self._cached
        self._cached = await self._resolve()
        self._cached_at = now
        return self._cached

    async def _resolve(self) -> PodIdentity | None:
        token = self._read_sa("token")
        namespace = self._read_sa("namespace")
        if token is None or namespace is None or self._service is None:
            return None  # not running in a cluster
        ca = self._sa_dir / "ca.crt"
        url = (
            f"{self._api_base}/apis/discovery.k8s.io/v1/namespaces/{namespace}"
            f"/endpointslices"
        )
        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                verify=str(ca) if self._transport is None and ca.exists() else False,
                timeout=5.0,
            ) as client:
                resp = await client.get(
                    url,
                    params={
                        "labelSelector": f"kubernetes.io/service-name={self._service}"
                    },
                    headers={"Authorization": f"Bearer {token}"},
                )
                resp.raise_for_status()
                items = resp.json().get("items", [])
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("EndpointSlice lookup failed for %s: %s", self._service, exc)
            return None

        backends: dict[str, str | None] = {}
        for slice_ in items:
            for ep in slice_.get("endpoints") or []:
                if not (ep.get("conditions") or {}).get("ready"):
                    continue
                ref = ep.get("targetRef") or {}
                if ref.get("kind") != "Pod" or not ref.get("name"):
                    continue
                backends[ref["name"]] = ep.get("nodeName")

        if len(backends) != 1:
            log.warning(
                "Service %s has %s ready backends — recording no pod identity",
                self._service,
                len(backends),
            )
            return None
        pod_name, node_name = next(iter(backends.items()))
        return PodIdentity(pod_name=pod_name, node_name=node_name)
