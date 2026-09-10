"""inf-ver-runner entrypoint: verify one model's pending events, then exit.

No HTTP surface — this is a batch process (a k8s Job in the cluster, spawned
by inf-ver-orchestrator alongside a dedicated vLLM). Startup order:

1. Declare ourselves to the ledger (hostname + the verifier model we run),
   retrying while the ledger comes up — verification events need our
   hardware_id and verifier model_id.
2. Drain the pending queue for RUNNER_MODEL_ID via VerificationRunner.

Exit codes: 0 drained or verification paused (NULL threshold); nonzero fatal
(the Job's backoff policy owns retries).
"""

from __future__ import annotations

import asyncio
import logging
import sys

from .config import Settings, load_settings
from .ledger_client import LedgerClient
from .pod_identity import PodIdentityResolver
from .readiness import wait_for_vllm
from .runner import VerificationRunner
from .verifier import VerificationService

log = logging.getLogger(__name__)


async def _declare_with_retry(
    settings: Settings, ledger_client: LedgerClient
) -> dict | None:
    """Declare this runner's deployment; None if the ledger never came up."""
    for attempt in range(1, settings.max_declare_attempts + 1):
        try:
            return await ledger_client.declare(
                settings.runner_hostname, settings.model.id
            )
        except Exception as exc:
            log.warning(
                "Ledger declaration failed (%s/%s), retrying: %s",
                attempt,
                settings.max_declare_attempts,
                exc,
            )
            await asyncio.sleep(settings.poll_interval_seconds)
    return None


async def run(settings: Settings) -> int:
    ledger_client = LedgerClient(settings)
    try:
        verification_service = VerificationService(settings)
        await verification_service.startup()

        if not await wait_for_vllm(settings):
            return 1

        deployment = await _declare_with_retry(settings, ledger_client)
        if deployment is None:
            log.error("Ledger unreachable, giving up on declaration")
            return 1
        log.info(
            "Declared runner deployment hostname=%s model=%s hardware_id=%s model_id=%s",
            settings.runner_hostname,
            settings.model.id,
            deployment["hardware_id"],
            deployment["model_id"],
        )

        pod_identity_resolver = PodIdentityResolver(vllm_url=settings.vllm_url)
        runner = VerificationRunner(
            settings,
            ledger_client,
            verification_service,
            hardware_id=deployment["hardware_id"],
            verifier_model_id=deployment["model_id"],
            pod_identity_resolver=pod_identity_resolver,
        )
        try:
            return await runner.run_until_drained()
        finally:
            # However the drain ends (empty queue, paused, fatal), the
            # runner instance is going away: report the deployment closed.
            await ledger_client.close_deployment(settings.runner_hostname)
    finally:
        await ledger_client.aclose()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = load_settings()
    return asyncio.run(run(settings))


if __name__ == "__main__":
    sys.exit(main())
