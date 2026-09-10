# Kubernetes setup

GitOps configuration for the whole platform. `application-manifests/` holds
one ArgoCD `Application` per component; each pulls either an upstream Helm
chart or a Kustomize directory under `manifests/`. Applying the folder is the
entire bootstrap:

```sh
kubectl apply -f application-manifests/
```

| Application | Source | What it provides |
|---|---|---|
| `local-storage` | `manifests/local-storage` | StorageClass and static local PersistentVolumes for everything below |
| `kube-prometheus-stack` | Helm chart 87.16.1 | Prometheus and Grafana |
| `gpu-monitoring` | `manifests/gpu-monitoring` | DCGM ServiceMonitors, custom metric set, GPU fleet dashboard |
| `harbor` | Helm chart 1.19.1 | Container registry for the stack's images |
| `kubeflow` | `manifests/kubeflow-slim` | KServe, Istio, Knative, cert-manager, Trainer (no dashboard) |
| `gateway-api-crds`, `gie-crds` | upstream CRDs v1.5.1 / v1.4.0 | Prerequisites for KServe `LLMInferenceService` routing |
| `llm-serving` | `manifests/llm-serving` | Prover models as `LLMInferenceService`s with the verify-tap sidecar, behind one Gateway |
| `infver` | `manifests/infver` | The verification stack: Postgres, ledger API, message writer, orchestrator, analytics API, GPU enricher, prompt runner, Open WebUI, verification UI |

The configuration provides an example layout with three placeholder node names.
The sections below describe what you will need to decide or change to run it
on your own cluster.

## Cluster assumptions

The manifests expect these to exist already; none of them are installed here:

- ArgoCD in the `argocd` namespace.
- The NVIDIA GPU Operator in the `gpu-operator` namespace, including its DCGM
  exporter DaemonSet. This repo only adds ServiceMonitors that scrape it.
- MetalLB and ingress-nginx. Every web UI is a host-based Ingress on the
  ingress controller's LoadBalancer IP.
- Every node has NVIDIA GPUs. Prover models and verifier vLLMs are the only
  GPU consumers; everything else is CPU-only.
- Kubeflow's first sync takes several retries while CRDs and webhooks settle.
  Expect ten to twenty minutes.

## Placeholders to replace

Search the directory for these before the first sync:

- `<org>/<repo>` in every `application-manifests/*.yaml` `repoURL`. Point
  them at your fork.
- `gpu-node-1`, `gpu-node-2`, `gpu-node-3`. These are node names in PV node
  affinities (`manifests/local-storage/persistent-volumes.yaml`), the
  orchestrator's verifier placement settings (`manifests/infver/config.yaml`)
  and the storage helper script. Rename them to your nodes, or rebalance
  which workloads land where.
- `<ingress-lb-ip>` in this README's own DNS notes. Find the real value with
  `kubectl -n ingress-nginx get svc ingress-nginx-controller`.
- `<CHANGE_ME_...>` values: the Postgres password and Open WebUI secret in
  `manifests/infver/config.yaml`, the Harbor admin password in
  `application-manifests/harbor.yaml`.

## Storage

Persistence uses statically provisioned local volumes
(`kubernetes.io/no-provisioner`, `WaitForFirstConsumer`). Each PV is pinned
to one node by hostname and claimRef-reserved for one PVC, and nothing binds
until its backing directory exists under `/mnt/local-storage/` on that node.

`scripts/create-local-storage-dirs.sh` creates the directories. Run it as
root on each node; it detects the node from its hostname, or takes the node
name as an argument. If you add PVs, add the matching directory name to the
script.

The `hf-model-cache` volumes hold verifier model weights. The orchestrator
seeds them with a Job on first use and verifier pods mount them read-only, so
a model is downloaded once per node rather than once per verification cycle.
Eviction is manual: delete the model's directory under
`/mnt/local-storage/hf-model-cache/` while no verifier is running for it.

## Networking and DNS

The stack uses hostnames under `.infver.local`: `chat`, `verify`, `ledger`,
`writer`, `grafana`, `prometheus` and `harbor`. They are not in any DNS
server. Map them to `<ingress-lb-ip>` in `/etc/hosts` on every workstation
that needs the UIs, and map at least `harbor.infver.local` on every cluster
node so the container runtime can pull images. To use other hostnames, change
them in `application-manifests/*.yaml` and `manifests/infver/ingress.yaml`.

Harbor is served over plain HTTP, so each node's container runtime must trust
it as an insecure registry. For containerd:

```toml
# /etc/containerd/certs.d/harbor.infver.local/hosts.toml
server = "http://harbor.infver.local"

[host."http://harbor.infver.local"]
  capabilities = ["pull", "resolve", "push"]
  skip_verify = true
```

## Images

We ran a Harbor registry inside the cluster to hold the stack's images and
local replicas of upstream ones, which is why the `harbor` application and
the insecure-registry notes above exist. Any OCI registry works instead
(GitHub Container Registry, a cloud provider's registry, or a plain
`registry:2`): change the `image:` references in `manifests/infver/*.yaml`
and `ORCH_RUNNER_IMAGE` in `manifests/infver/config.yaml`, add an
`imagePullSecret` if the registry is private, and drop the `harbor`
application.

As configured, the `infver` manifests pull nine custom images from the Harbor
project `infver_images`, which is public so no pull secret is needed. Each
service directory at the repository root has a `Dockerfile`; build each one
for `linux/amd64` and push it under the name the manifests reference
(`ledger-api`, `message-writer`, `inf-proxy`, `inf-ver-orchestrator`,
`inf-ver-runner`, `inf-ver-ui`, `analytics-api`, `gpu-enricher`,
`prompt-runner`). The `ledger-api` image builds from `ledger/`, and
`inf-ver-runner` must be built with the repository root as its context
because it copies the vendored `difr/` package.

The runner image and the verifier vLLM image are also referenced from
`manifests/infver/config.yaml` (`ORCH_RUNNER_IMAGE`, `ORCH_VLLM_IMAGE`). The
runner's torch version must match the vLLM image's, because DiFR replays
vLLM's CUDA random stream; bump them together.

## Serving models

`manifests/llm-serving/` defines each prover as a KServe `LLMInferenceService`
running vLLM plus the `verify-tap` sidecar, all behind one Gateway API
endpoint (`llm-gateway`) with per-request routing by model name. Four models
are included: Qwen2.5 1.5B and 7B, and gpt-oss 20B and 120B. Choose GPUs
that support each model's quantization and have enough memory for its weights
and context length. Add node selectors for your GPU pools as needed; no
specific GPU product is selected by the supplied manifests. Resource requests
and limits are examples to tune for your hardware.

To add a model, copy one of the existing files, change the Hugging Face URI,
served name and `INF_PROXY_MODEL_HOSTNAME`, and keep the sidecar's
`INF_PROXY_DECLARE=true` and sampling variables so it declares itself to the
ledger at startup and its tapped events resolve. Multi-GPU serving within one pod works by setting
`--tensor-parallel-size` and the matching `nvidia.com/gpu` request.

## Verification placement and tunables

`manifests/infver/config.yaml` holds the orchestrator's settings. The ones
most likely to need changing:

- `ORCH_VLLM_NODE_SELECTOR`: which node pool verifier vLLMs run on.
- `ORCH_VLLM_PLACEMENT_OVERRIDES`: per-model exceptions, each with its own
  node selector and model-cache PVC. This mapping is empty by default. If a
  model needs a different GPU pool, provision a local PV and cache PVC on
  the target node and configure both the selector and PVC in its override.
  Replaying long outputs with `prompt_logprobs` can require more memory
  than serving the same model.
- `ORCH_VLLM_MAX_MODEL_LEN`: context length of the verifier vLLM, which bounds
  its KV-cache memory.

The ledger schema is applied from `manifests/infver/sql/` only when the
Postgres volume is first initialised. Schema changes on an existing volume
are a manual migration.

## GPU monitoring

The `gpu-monitoring` application scrapes the GPU Operator's DCGM exporter at
1 s for the fleet dashboard and at 200 ms, restricted to eight profiling
fields, for the GPU enricher's per-event windows. Both depend on two
ClusterPolicy patches that this repo cannot apply because the operator is not
managed here. Re-apply them if the operator resets its ClusterPolicy:

```sh
kubectl patch clusterpolicy cluster-policy --type merge \
  -p '{"spec":{"dcgmExporter":{"config":{"name":"dcgm-metrics-custom"}}}}'
kubectl patch clusterpolicy cluster-policy --type merge \
  -p '{"spec":{"dcgmExporter":{"env":[{"name":"DCGM_EXPORTER_INTERVAL","value":"200"}]}}}'
```

Profiling metrics (`DCGM_FI_PROF_*`) need datacenter-class GPUs. DCGM
multiplexes them over a limited number of hardware counters, so with many
fields enabled the per-precision series can look quantised. If so, drop
low-value fields from the `dcgm-metrics-custom` ConfigMap and the fast
ServiceMonitor's keep rule.

## Access and credentials

| Service | URL | Credentials |
|---|---|---|
| Open WebUI | http://chat.infver.local | none (`WEBUI_AUTH=false`) |
| Verification UI | http://verify.infver.local | none |
| Ledger API | http://ledger.infver.local | none; docs at `/docs` |
| Grafana | http://grafana.infver.local | `admin`, password in the `kube-prometheus-stack-grafana` Secret |
| Prometheus | http://prometheus.infver.local | none |
| Harbor | http://harbor.infver.local | `admin`, the password you set in `harbor.yaml` |
| ArgoCD | http://<ingress-lb-ip>/ | `admin`, password in `argocd-initial-admin-secret` |

## Security

This is a starting point for a private cluster, not a hardened deployment.
The UIs and APIs have no authentication, Harbor is HTTP-only, and the
LoadBalancer IP is assumed to be reachable only from the cluster's LAN. Add
TLS and authentication before exposing any of it more widely.
Ensure secret management is converted to a production grade process is serving to a wider audience.
