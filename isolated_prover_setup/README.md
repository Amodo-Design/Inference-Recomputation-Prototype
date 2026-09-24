# Isolated prover setup

The parts of the stack that deliberately do **not** run in Kubernetes. They
belong on the node the hardware tap observes, and are deployed by hand —
**ArgoCD does not watch this directory.**

## Why anything lives outside the cluster

`frame-processor` makes a strong claim about the link it taps: that every
frame crossing it can be classified, and that anything unexplained becomes a
recorded finding. That claim is only tractable if the link carries very little
besides inference.

A pod on a cluster node cannot offer that. A CNI, VXLAN encapsulation, kubelet
traffic and control-plane chatter all put frames on the wire continuously, and
most of them are neither inference nor anything the tap can meaningfully
whitelist. So the prover leaves the cluster: it runs under Docker Compose on a
node whose traffic profile is curated down to inference plus one declared
health check, and the tapped link becomes quiet enough that *silence itself is
evidence* — the property the health check here exists to exploit.

The verifier, the ledger and everything else stay in Kubernetes. Only the
observed node moves.

## What is here

| Directory | What it is |
|---|---|
| [`gpt-oss-120b/`](gpt-oss-120b/) | The prover: vLLM under Docker Compose, bound to the tapped interface. `gpt-oss-120b` is the example model; the layout serves any model vLLM supports. |
| [`tapped-link-health/`](tapped-link-health/) | The sender half of the tapped-link health check. Its responder runs in-cluster on the capture host. |
| [`systemd/`](systemd/) | Template units that start both projects at boot, stop them cleanly, and repair them if they drift. |

Each directory has its own README and a `.env.example` to copy. The compose
files carry the reasoning for individual settings.

## Read this before trusting a verdict

Moving a prover out of Kubernetes loses the `verify-tap` sidecar, and with it
sampling-parameter pinning, deployment declaration and GPU attribution.
`frame-processor` replaces only the traffic capture. What that costs, and what
you must do by hand instead, is in
[`gpt-oss-120b/README.md`](gpt-oss-120b/README.md) — the section headed *What
this does NOT do*.
