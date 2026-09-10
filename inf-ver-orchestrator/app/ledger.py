"""Read-only ledger access: group pending verification events by model.

The orchestrator only ever reads `/inference-events/unverified/view` (the
payload-free feed, safe to page at size) — verdict writing is the runners'
job, and thresholds are set by inf-ver-ui.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)

PAGE_SIZE = 500


@dataclass(frozen=True)
class PendingModel:
    """One model's slice of the pending-verification queue."""

    model_id: str
    model_name: str
    verification_threshold: float | None
    seed: int | None
    temperature: float | None
    top_k: int | None
    top_p: float | None
    pending: int

    @property
    def eligible(self) -> bool:
        """Verification runs only for models with a threshold set; NULL means
        paused — events stay pending until a threshold appears in the UI."""
        return self.pending > 0 and self.verification_threshold is not None


class LedgerClient:
    def __init__(self, base_url: str, timeout_seconds: float = 10.0) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def pending_by_model(self) -> list[PendingModel]:
        """All pending (unverified) events, grouped per model."""
        groups: dict[str, dict[str, Any]] = {}
        offset = 0
        while True:
            resp = self._client.get(
                "/inference-events/unverified/view",
                params={"limit": PAGE_SIZE, "offset": offset},
            )
            resp.raise_for_status()
            page = resp.json()
            for item in page["items"]:
                model_id = item["model_id"]
                group = groups.get(model_id)
                if group is None:
                    sampling = item.get("sampling_config") or {}
                    groups[model_id] = {
                        "model_id": model_id,
                        "model_name": item["model_name"],
                        "verification_threshold": item.get("verification_threshold"),
                        "seed": sampling.get("seed"),
                        "temperature": sampling.get("temperature"),
                        "top_k": sampling.get("top_k"),
                        "top_p": sampling.get("top_p"),
                        "pending": 1,
                    }
                else:
                    group["pending"] += 1
            offset += len(page["items"])
            if offset >= page["total"] or not page["items"]:
                break
        return [PendingModel(**group) for group in groups.values()]
