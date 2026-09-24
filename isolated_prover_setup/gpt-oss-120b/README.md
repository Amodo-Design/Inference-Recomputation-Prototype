# Prover served outside Kubernetes

A vLLM prover run by hand with Docker Compose, replacing the
`LLMInferenceService` in `kubernetes_setup/manifests/llm-serving/` for the
node that carries the hardware tap. **ArgoCD does not watch this directory.**

`gpt-oss-120b` is the example model; the same layout serves any model vLLM
supports. The compose file comments explain each setting and why it is
load-bearing.

## Why it is not in Kubernetes

The tapped node's value is that every frame it puts on the wire can be
enumerated. A CNI, VXLAN encapsulation, a kubelet and control-plane chatter
all add flows that are hard to account for, so the node leaves the cluster and
its traffic is curated down to inference plus a declared health check. See
[`frame-processor/README.md`](../../frame-processor/README.md).

## Running it

Needs the NVIDIA driver, Docker, the Container Toolkit, and the weights on
disk as a plain model directory (`config.json` + safetensors) — **not** a
HuggingFace hub cache, which would be re-downloaded.

```sh
cp .env.example .env          # first time only
docker compose up -d
docker compose logs -f        # a large model takes many minutes to load
```

Confirm it serves on the tapped interface and nowhere else:

```sh
ss -lntp | grep 8000          # must show the tapped address, NOT 0.0.0.0
```

A listener on `0.0.0.0` is a correctness bug here, not a convenience: it makes
the model reachable by a path the tap does not observe.

## What this does NOT do — read before trusting any verdict

In Kubernetes each prover runs a `verify-tap` sidecar (the `inf-proxy` image)
doing four jobs. Only the first survives the move:

| Sidecar job | Here |
|---|---|
| Copy traffic to `message-writer` | **frame-processor does it** |
| Pin seed/temperature/top_k/top_p per request | **nobody** |
| Declare the deployment to the ledger | **nobody** |
| Stamp pod/node name for GPU attribution | **nobody** — emitted as null |

In order of how quietly they fail:

1. **Sampling is not pinned.** The verifier needs all four sampling values
   from the declared model row, or events resolve as `unverifiable /
   sampling_config_missing`. Bare vLLM honours whatever the client sends, so
   every client must send the declared values on every request — otherwise
   served and declared configuration drift apart and the verdicts mean
   nothing. This is the first thing to suspect behind unexplained replay
   mismatches.
2. **The deployment is not declared**, so events resolve to no model. Declare
   it once, with a hostname matching `FRAME_PROCESSOR_HOSTNAME_MAP` in
   `kubernetes_setup/manifests/infver/frame-processor.yaml`:

   ```sh
   curl -X POST http://<ledger-host>/model-deployments/declare \
     -H 'content-type: application/json' \
     -d '{"hostname":"<hostname>","model_name":"<model>",
          "seed":42,"temperature":1.0,"top_k":200,"top_p":0.95}'
   ```
3. **GPU enrichment is unavailable.** `gpu-enricher` selects on `pod_name`
   falling back to `node_name`, both null here, and a node outside the cluster
   has no DCGM exporter.

The alternative, if client-side sampling discipline proves too fragile, is
running `inf-proxy` in front of vLLM (vLLM on `--port 8001`, proxy on 8000).
That restores jobs 2 and 3 at the cost of putting software back on the
observed node — exactly the trade the physical tap was meant to remove.

## Verifier pairing

The verifier runs in-cluster, spawned by `inf-ver-orchestrator`
(`ORCH_VLLM_PLACEMENT_OVERRIDES` in
`kubernetes_setup/manifests/infver/config.yaml`). Two things must stay in step
with this file: the **vLLM image tag** must equal `ORCH_VLLM_IMAGE`, and both
sides must stay **TP=1** — a mismatch changes reduction order, and difr's
replay has never been validated across one.
