# Tapped-link health check (sender)

The half of the health check that runs on the tapped node: it sends one fixed
datagram to one address on the tapped link every 30 seconds, and does nothing
else. Its answering half is a Kubernetes Deployment on the capture host,
`kubernetes_setup/manifests/infver/tapped-link-health.yaml`.

**ArgoCD does not watch this directory.** Deployed by hand, like the prover
alongside it. The compose file comments cover the container settings and why
each is load-bearing.

## What it is for

frame-processor validates frames that arrive, so every one of its rules — the
class whitelist, the address rules, the pins — is satisfied by nothing
arriving at all. Because the tapped node's traffic is curated down to almost
nothing, a link with no inference on it produces no frames, and "the prover is
idle" becomes indistinguishable from "the capture has gone blind": a
`complete` window with zero frames in it, while inference the tap never saw
goes unrecorded.

This closes that. One datagram each way on a declared cadence, with
frame-processor told to expect both legs
(`FRAME_PROCESSOR_EXPECTED_BEATS`), so absence becomes a finding and the
window carrying it is incomplete rather than clean.

It is *answered* rather than merely sent because a passive tap sits inline:
the monitor port for a direction carries what that end put on the wire, so a
datagram sent into a link severed downstream of the tap still reaches the tap
and still reads healthy. A reply cannot be produced that way.

Cost is a few hundred KB a day. Raising `--interval` lowers it proportionally;
one capture window is the point past which a blind capture is noticed a window
late rather than within a beat.

## Deploying it

No image is transferred and no registry contacted: the compose file borrows an
image already on the node purely for its Python interpreter, and the program
is a single stdlib-only file mounted in. It is copied at deploy time rather
than vendored here, so the bytes on the wire and the digest frame-processor
expects stay one fact.

```sh
scp frame-processor/tools/tapped_link_health.py \
    <tapped-node>:/opt/infver/tapped-link-health/

# on the node
cd /opt/infver/tapped-link-health
cp .env.example .env           # first time only
docker compose up -d
docker compose logs -f         # "sent=120 answered=120"
```

**Start the responder first**, or requests get an ICMP port-unreachable back —
correct, and a finding since `icmp` is not whitelisted, but noise you do not
need while setting up.

That `scp` is the whole inbound transfer, and it still lands on a port the
whitelist does not expect — so the window containing it is tainted, as is the
window containing your `ssh` session. Do it before the beats are declared, so
the two effects stay separable.

For reboots and drift, install the units in [`../systemd/`](../systemd/).

## Checking it

`answered` lagging `sent` means replies are not coming back: the responder
pod, or the return path. On the capture host both legs should appear on the
monitor ports, one direction each:

```sh
tcpdump -i <iface-in>  -c 3 -nn 'udp port 9999'   # REQ, from the tapped node
tcpdump -i <iface-out> -c 3 -nn 'udp port 9999'   # ACK, from the capture host
```

If one port shows both, `FRAME_PROCESSOR_IFACE_DIRECTION` does not match the
cabling. If neither shows anything, it is not crossing the tap — check that
`--bind` is the address on the tapped interface.
