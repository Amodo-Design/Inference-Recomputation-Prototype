"""Manifest builders for a model's verifier pair (LLMISvc + runner Job).

The LLMInferenceService deliberately has NO `router` section and NO
verify-tap sidecar: verifier traffic must never flow through the gateway/EPP
or a tap — its re-runs would be captured as prover inference events and
pollute the ledger. `spec.router` is optional in the CRD, so the controller
only creates the workload Deployment + Service.
"""

from __future__ import annotations

from typing import Any

from .config import Settings
from .ledger import PendingModel
from .naming import job_name, seed_job_name, vllm_name, vllm_service_url

MANAGED_BY = "inf-ver-orchestrator"
MODEL_ID_LABEL = "infver.io/model-id"


def _labels(model: PendingModel) -> dict[str, str]:
    return {
        "app.kubernetes.io/managed-by": MANAGED_BY,
        MODEL_ID_LABEL: model.model_id,
    }


def _placement(model: PendingModel, settings: Settings):
    return settings.vllm_placement_overrides.get(model.model_name)


def _effective_cache_pvc(model: PendingModel, settings: Settings) -> str | None:
    override = _placement(model, settings)
    if override is not None:
        # An override's cache decision is authoritative: its own node's PVC,
        # or None = uncached (hf://) — the global default-node PVC can never bind
        # on the override's node.
        return override.cache_pvc
    return settings.model_cache_pvc


def build_llmisvc(model: PendingModel, settings: Settings) -> dict[str, Any]:
    cache_pvc = _effective_cache_pvc(model, settings)
    vllm_args = (
        f"exec vllm serve /mnt/models"
        f" --served-model-name {model.model_name}"
        f" --port 8000"
        f" --disable-uvicorn-access-log"
        f" --max-model-len {settings.vllm_max_model_len}"
        f" --max-logprobs {settings.vllm_max_logprobs}"
        f" --gpu-memory-utilization {settings.vllm_gpu_memory_utilization}"
        f" --max-num-batched-tokens {settings.vllm_max_num_batched_tokens}"
        # difr's margin math expects pre-softmax logits (its reference runner
        # constructs vLLM with logprobs_mode="raw_logits"); the server default
        # is log-softmax logprobs.
        f" --logprobs-mode raw_logits"
    )
    spec: dict[str, Any] = {
        "model": {
            # Cached: serve from the node-local model cache (seeded by the
            # seed Job) instead of re-downloading from HF every spawn.
            "uri": (
                f"pvc://{cache_pvc}/{model.model_name}"
                if cache_pvc
                else f"hf://{model.model_name}"
            ),
            "name": model.model_name,
        },
        "replicas": 1,
        "template": {
            "containers": [
                {
                    "name": "main",
                    "image": settings.vllm_image,
                    # difr's replay is written against the V1 model runner's
                    # per-request Philox stream; Model Runner V2 (default for
                    # dense generate models since vLLM v0.22) samples with a
                    # stateless Triton kernel that cannot be replayed. Pin V1
                    # explicitly on every verifier vLLM.
                    "env": [
                        {"name": "VLLM_USE_V2_MODEL_RUNNER", "value": "0"},
                    ],
                    # The preset pod template sets runAsNonRoot, but the stock
                    # vLLM image runs as root — override or kubelet refuses it.
                    "securityContext": {"runAsNonRoot": False},
                    # Own the command: no tap in this pod, so vLLM binds the
                    # pod's serving port 8000 directly.
                    "command": ["/bin/bash", "-c", vllm_args],
                    "ports": [{"containerPort": 8000}],
                    "resources": {
                        "requests": {
                            "cpu": "4",
                            "memory": "24Gi",
                            "nvidia.com/gpu": "1",
                        },
                        "limits": {
                            "memory": "32Gi",
                            "nvidia.com/gpu": "1",
                        },
                    },
                }
            ],
        },
    }
    override = _placement(model, settings)
    node_selector = (
        override.node_selector if override is not None and override.node_selector
        else settings.vllm_node_selector
    )
    if node_selector:
        spec["template"]["nodeSelector"] = dict(node_selector)
    return {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "LLMInferenceService",
        "metadata": {
            "name": vllm_name(model.model_name),
            "namespace": settings.namespace,
            "labels": _labels(model),
        },
        "spec": spec,
    }


def _sampling_env(model: PendingModel) -> list[dict[str, str]]:
    env = []
    for env_name, value in (
        ("RUNNER_SEED", model.seed),
        ("RUNNER_TEMPERATURE", model.temperature),
        ("RUNNER_TOP_K", model.top_k),
        ("RUNNER_TOP_P", model.top_p),
    ):
        if value is not None:
            env.append({"name": env_name, "value": str(value)})
    return env


def build_job(model: PendingModel, settings: Settings) -> dict[str, Any]:
    env = [
        {"name": "RUNNER_MODEL_ID", "value": model.model_id},
        {"name": "RUNNER_MODEL_NAME", "value": model.model_name},
        {"name": "RUNNER_VLLM_URL", "value": vllm_service_url(model.model_name)},
        {"name": "LEDGER_API_URL", "value": settings.ledger_api_url},
        {"name": "RUNNER_HOSTNAME", "value": job_name(model.model_name)},
        {"name": "RUNNER_MAX_LOGPROBS", "value": str(settings.vllm_max_logprobs)},
        {
            "name": "RUNNER_VLLM_READY_TIMEOUT_SECONDS",
            "value": str(settings.runner_vllm_ready_timeout_seconds),
        },
        # One engine death takes ~2-3min to reload; the runner's default
        # 3x15s retry budget mass-poisons a drain. Give retries time to
        # outlive a reload instead.
        {"name": "RUNNER_MAX_ATTEMPTS", "value": "5"},
        {"name": "RUNNER_RETRY_BACKOFF_SECONDS", "value": "60"},
        # Concurrent verifications inside the runner (its verify semaphore).
        {"name": "RUNNER_MODEL_CONCURRENCY", "value": str(settings.runner_model_concurrency)},
    ] + _sampling_env(model)

    # The runner replays vLLM's per-request CUDA Philox exponential stream;
    # CPU MT19937 noise can never match it, so the Job needs a GPU on the
    # same pool as the verifier vLLM. runner_gpus=0 is the explicit CPU
    # escape hatch — the runner is then told not to hard-require CUDA.
    container: dict[str, Any] = {
        "name": "runner",
        "image": settings.runner_image,
        "imagePullPolicy": "Always",
        "env": env,
    }
    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "containers": [container],
    }
    if settings.runner_service_account:
        pod_spec["serviceAccountName"] = settings.runner_service_account
    if settings.runner_gpus > 0:
        gpus = str(settings.runner_gpus)
        container["resources"] = {
            "requests": {"cpu": "2", "memory": "4Gi", "nvidia.com/gpu": gpus},
            "limits": {"memory": "8Gi", "nvidia.com/gpu": gpus},
        }
        if settings.vllm_node_selector:
            pod_spec["nodeSelector"] = dict(settings.vllm_node_selector)
    else:
        env.append({"name": "RUNNER_REQUIRE_CUDA", "value": "false"})

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name(model.model_name),
            "namespace": settings.namespace,
            "labels": _labels(model),
        },
        "spec": {
            "backoffLimit": settings.job_backoff_limit,
            "template": {
                "metadata": {"labels": _labels(model)},
                "spec": pod_spec,
            },
        },
    }


def build_seed_job(model: PendingModel, settings: Settings) -> dict[str, Any]:
    """Idempotent model-download Job gating the verifier pair (only used when
    settings.model_cache_pvc is set). Cache hit = marker file exists = exits
    in seconds; miss = one huggingface-cli download (resumable). Runs the
    pinned vLLM image (already on the node, ships huggingface-cli), no GPU.
    The local PV's node affinity pins the pod to the cache's node."""
    model_dir = f"/cache/{model.model_name}"
    script = (
        "set -e\n"
        f'if [ -f "{model_dir}/.complete" ]; then echo "cache hit"; exit 0; fi\n'
        f'hf download "{model.model_name}" --local-dir "{model_dir}" --max-workers 4\n'
        f'touch "{model_dir}/.complete"\n'
    )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": seed_job_name(model.model_name),
            "namespace": settings.namespace,
            "labels": _labels(model),
        },
        "spec": {
            "ttlSecondsAfterFinished": 600,
            "backoffLimit": settings.job_backoff_limit,
            "template": {
                "metadata": {"labels": _labels(model)},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "seed",
                            "image": settings.vllm_image,
                            "command": ["/bin/bash", "-c", script],
                            # Keep HF's own cache/tmp on the volume so a large
                            # download can never fill the container filesystem.
                            "env": [
                                {
                                    "name": "HF_HOME",
                                    "value": "/cache/.hf",
                                    # xet must remain enabled: large-shard models like
                                    # gpt-oss-120b exceed huggingface_hub's non-xet
                                    # download limit and hard-fail without xet. 8Gi
                                    # memory limit is the OOM guard (validated:
                                    # 4Gi-OOM vs 8Gi-clean at 63GB with 4 workers).
                                },
                            ],
                            "resources": {
                                "requests": {"cpu": "1", "memory": "2Gi"},
                                "limits": {"memory": "8Gi"},
                            },
                            "volumeMounts": [{"name": "cache", "mountPath": "/cache"}],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "cache",
                            "persistentVolumeClaim": {
                                "claimName": _effective_cache_pvc(model, settings)
                            },
                        }
                    ],
                },
            },
        },
    }
