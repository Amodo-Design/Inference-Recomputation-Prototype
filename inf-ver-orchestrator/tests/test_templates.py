"""Manifest construction: router-less LLMISvc + runner Job env contract."""

from __future__ import annotations

from app.templates import MODEL_ID_LABEL, build_job, build_llmisvc

from helpers import make_model, make_settings


def env_dict(container: dict) -> dict[str, str]:
    return {e["name"]: e["value"] for e in container["env"]}


def test_llmisvc_has_no_router_and_no_tap():
    model = make_model()
    manifest = build_llmisvc(model, make_settings())
    assert "router" not in manifest["spec"]
    containers = manifest["spec"]["template"]["containers"]
    assert [c["name"] for c in containers] == ["main"]
    # No tap: vLLM owns the pod's serving port 8000 directly.
    assert containers[0]["ports"] == [{"containerPort": 8000}]
    assert "--port 8000" in containers[0]["command"][2]


def test_llmisvc_identity_and_model():
    model = make_model()
    manifest = build_llmisvc(model, make_settings())
    assert manifest["metadata"]["name"] == "verify-qwen-qwen2-5-7b-instruct"
    assert manifest["metadata"]["labels"][MODEL_ID_LABEL] == model.model_id
    assert manifest["spec"]["model"]["uri"] == "hf://Qwen/Qwen2.5-7B-Instruct"
    command = manifest["spec"]["template"]["containers"][0]["command"][2]
    assert "--served-model-name Qwen/Qwen2.5-7B-Instruct" in command
    assert "--max-logprobs 200" in command
    assert "runAsNonRoot" in str(manifest)  # stock vLLM image runs as root


def test_llmisvc_node_selector_optional():
    model = make_model()
    manifest = build_llmisvc(
        model, make_settings(vllm_node_selector={"kubernetes.io/hostname": "gpu-node-1"})
    )
    assert manifest["spec"]["template"]["nodeSelector"] == {
        "kubernetes.io/hostname": "gpu-node-1"
    }
    bare = build_llmisvc(model, make_settings())
    assert "nodeSelector" not in bare["spec"]["template"]


def test_job_env_contract():
    model = make_model()
    manifest = build_job(model, make_settings())
    assert manifest["metadata"]["name"] == "inf-ver-runner-qwen-qwen2-5-7b-instruct"
    assert manifest["metadata"]["labels"][MODEL_ID_LABEL] == model.model_id
    assert manifest["spec"]["template"]["metadata"]["labels"][MODEL_ID_LABEL] == model.model_id
    assert manifest["spec"]["backoffLimit"] == 3
    assert manifest["spec"]["template"]["spec"]["restartPolicy"] == "Never"

    env = env_dict(manifest["spec"]["template"]["spec"]["containers"][0])
    assert env["RUNNER_MODEL_ID"] == model.model_id
    assert env["RUNNER_MODEL_NAME"] == "Qwen/Qwen2.5-7B-Instruct"
    assert (
        env["RUNNER_VLLM_URL"]
        == "http://verify-qwen-qwen2-5-7b-instruct-kserve-workload-svc:8000/v1"
    )
    assert env["LEDGER_API_URL"] == "http://ledger-api:8000"
    assert env["RUNNER_SEED"] == "42"
    assert env["RUNNER_TEMPERATURE"] == "1.0"
    assert env["RUNNER_TOP_K"] == "50"
    assert env["RUNNER_TOP_P"] == "0.95"


def test_job_omits_unset_sampling_env():
    model = make_model(seed=None)
    env = env_dict(build_job(model, make_settings())["spec"]["template"]["spec"]["containers"][0])
    assert "RUNNER_SEED" not in env
    assert env["RUNNER_TEMPERATURE"] == "1.0"


def test_llmisvc_serves_raw_logits():
    # difr's Gumbel-margin math is written against pre-softmax logits
    # (token_difr_vllm.py constructs its LLM with logprobs_mode="raw_logits");
    # the stock server default is log-softmax logprobs.
    command = build_llmisvc(make_model(), make_settings())["spec"]["template"]["containers"][0][
        "command"
    ][2]
    assert "--logprobs-mode raw_logits" in command


def test_job_gets_gpu_for_rng_replay():
    # The runner replays vLLM's per-request CUDA Philox exponential stream;
    # a CPU-only runner draws MT19937 noise that can never match. It needs a
    # GPU and the same node pool as the verifier vLLM.
    selector = {"kubernetes.io/hostname": "gpu-node-1"}
    manifest = build_job(make_model(), make_settings(vllm_node_selector=selector))
    pod = manifest["spec"]["template"]["spec"]
    resources = pod["containers"][0]["resources"]
    assert resources["requests"]["nvidia.com/gpu"] == "1"
    assert resources["limits"]["nvidia.com/gpu"] == "1"
    assert pod["nodeSelector"] == selector


def test_job_gpu_can_be_disabled():
    # runner_gpus=0 is the explicit CPU escape hatch: no GPU resources, no
    # node selector, and the runner is told not to hard-require CUDA.
    manifest = build_job(make_model(), make_settings(runner_gpus=0))
    pod = manifest["spec"]["template"]["spec"]
    assert "resources" not in pod["containers"][0] or "nvidia.com/gpu" not in pod["containers"][
        0
    ].get("resources", {}).get("requests", {})
    assert "nodeSelector" not in pod
    env = env_dict(pod["containers"][0])
    assert env["RUNNER_REQUIRE_CUDA"] == "false"


def test_llmisvc_pins_v1_model_runner():
    # difr replays the V1 runner's per-request Philox stream; Model Runner V2
    # (vLLM default since v0.22 for dense generate models) is stateless
    # Triton noise and unreplayable.
    container = build_llmisvc(make_model(), make_settings())["spec"]["template"]["containers"][0]
    env = {e["name"]: e["value"] for e in container.get("env", [])}
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "0"


def test_job_sets_service_account_for_pod_identity():
    # Runner Jobs need endpointslices get/list to resolve their own pod
    # identity (pod name + node) via the K8s API; RBAC binds that to this
    # ServiceAccount.
    manifest = build_job(make_model(), make_settings())
    assert manifest["spec"]["template"]["spec"]["serviceAccountName"] == "inf-ver-runner"

    manifest = build_job(make_model(), make_settings(runner_service_account=""))
    assert "serviceAccountName" not in manifest["spec"]["template"]["spec"]


def test_job_passes_vllm_ready_timeout():
    # Ephemeral verifier cold-starts re-download the model every spawn; the
    # runner's built-in 900s readiness default is routinely exceeded for
    # large repos, failing the Job before vLLM ever comes up.
    env = env_dict(
        build_job(make_model(), make_settings(runner_vllm_ready_timeout_seconds=3600))[
            "spec"
        ]["template"]["spec"]["containers"][0]
    )
    assert env["RUNNER_VLLM_READY_TIMEOUT_SECONDS"] == "3600"


def test_build_seed_job_shape():
    from app.naming import seed_job_name
    from app.templates import build_seed_job

    model = make_model()
    settings = make_settings(model_cache_pvc="hf-model-cache")
    job = build_seed_job(model, settings)

    assert job["metadata"]["name"] == seed_job_name(model.model_name)
    assert job["metadata"]["labels"][MODEL_ID_LABEL] == model.model_id
    assert job["spec"]["ttlSecondsAfterFinished"] == 600
    assert job["spec"]["backoffLimit"] == settings.job_backoff_limit
    pod = job["spec"]["template"]["spec"]
    assert pod["restartPolicy"] == "Never"
    container = pod["containers"][0]
    assert container["image"] == settings.vllm_image
    assert "nvidia.com/gpu" not in container.get("resources", {}).get("requests", {})
    assert {"name": "cache", "mountPath": "/cache"} in container["volumeMounts"]
    assert pod["volumes"] == [
        {"name": "cache", "persistentVolumeClaim": {"claimName": "hf-model-cache"}}
    ]
    script = container["command"][-1]
    assert "/cache/Qwen/Qwen2.5-7B-Instruct/.complete" in script
    assert "hf download" in script
    assert "--max-workers 4" in script
    env = env_dict(container)
    assert "HF_HUB_DISABLE_XET" not in env
    assert env["HF_HOME"] == "/cache/.hf"
    assert container["resources"]["limits"]["memory"] == "8Gi"


def test_llmisvc_uri_switches_with_cache_setting():
    from app.templates import build_llmisvc

    cached = build_llmisvc(make_model(), make_settings(model_cache_pvc="hf-model-cache"))
    assert cached["spec"]["model"]["uri"] == "pvc://hf-model-cache/Qwen/Qwen2.5-7B-Instruct"
    plain = build_llmisvc(make_model(), make_settings(model_cache_pvc=None))
    assert plain["spec"]["model"]["uri"] == "hf://Qwen/Qwen2.5-7B-Instruct"


def _override_settings(**kw):
    from app.config import PlacementOverride

    return make_settings(
        model_cache_pvc="hf-model-cache",
        vllm_placement_overrides={
            "Qwen/Qwen2.5-7B-Instruct": PlacementOverride(
                node_selector={"kubernetes.io/hostname": "gpu-node-3"},
                **kw,
            )
        },
    )


def test_override_selector_and_cached_pvc_uri():
    settings = _override_settings(cache_pvc="hf-model-cache-node-3")
    manifest = build_llmisvc(make_model(), settings)
    assert manifest["spec"]["template"]["nodeSelector"] == {
        "kubernetes.io/hostname": "gpu-node-3"
    }
    assert (
        manifest["spec"]["model"]["uri"]
        == "pvc://hf-model-cache-node-3/Qwen/Qwen2.5-7B-Instruct"
    )


def test_override_without_cache_pvc_falls_back_to_hf():
    settings = _override_settings()  # cache_pvc=None
    manifest = build_llmisvc(make_model(), settings)
    assert manifest["spec"]["model"]["uri"] == "hf://Qwen/Qwen2.5-7B-Instruct"


def test_non_overridden_model_keeps_global_behaviour():
    from app.config import PlacementOverride

    settings = make_settings(
        model_cache_pvc="hf-model-cache",
        vllm_placement_overrides={
            "some/other-model": PlacementOverride(
                node_selector={"kubernetes.io/hostname": "gpu-node-3"},
                cache_pvc="hf-model-cache-node-3",
            )
        },
    )
    manifest = build_llmisvc(make_model(), settings)
    assert manifest["spec"]["model"]["uri"] == "pvc://hf-model-cache/Qwen/Qwen2.5-7B-Instruct"
    # Global selector (make_settings default) applies, not the override's.
    assert manifest["spec"]["template"].get("nodeSelector") != {
        "kubernetes.io/hostname": "gpu-node-3"
    }


def test_seed_job_mounts_override_pvc():
    from app.templates import build_seed_job

    settings = _override_settings(cache_pvc="hf-model-cache-node-3")
    job = build_seed_job(make_model(), settings)
    assert job["spec"]["template"]["spec"]["volumes"] == [
        {
            "name": "cache",
            "persistentVolumeClaim": {"claimName": "hf-model-cache-node-3"},
        }
    ]


def test_llmisvc_memory_headroom_flags():
    manifest = build_llmisvc(make_model(), make_settings())
    args = manifest["spec"]["template"]["containers"][0]["command"][-1]
    assert "--gpu-memory-utilization 0.85" in args
    assert "--max-num-batched-tokens 2048" in args


def test_runner_retry_budget_env():
    job = build_job(make_model(), make_settings())
    env = {e["name"]: e.get("value") for e in job["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["RUNNER_MAX_ATTEMPTS"] == "5"
    assert env["RUNNER_RETRY_BACKOFF_SECONDS"] == "60"


def test_runner_job_carries_model_concurrency():
    job = build_job(make_model(), make_settings(runner_model_concurrency=3))
    env = {e["name"]: e.get("value") for e in job["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["RUNNER_MODEL_CONCURRENCY"] == "3"
