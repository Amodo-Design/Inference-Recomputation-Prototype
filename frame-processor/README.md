# frame-processor

Reconstructs LLM inference traffic from raw network frames and emits the same
tap message `inf-proxy` posts to the Message writer (see
`docs/tap-message-contract.md`, `tap.source = "frame-processor"`).

This is the software half of the physical network tap: a hardware tap mirrors
every frame on the tapped link, and this service turns those frames back into
paired HTTP request/response exchanges, picks out the LLM traffic, and taps it
to the ledger. Until the hardware exists, it runs on the tapped node itself
(`hostNetwork` + `NET_RAW`), where an `AF_PACKET` socket sees the same bytes
the tap will eventually mirror. The `verify-tap` (`inf-proxy`) sidecars stay
live in parallel, so every inference is recorded twice — `tap.source`
distinguishes the two, and diffing them is how this service is validated.
Duplicate verification events are a known non-goal for now.

The design and results are written up in
**[Fitting a Network TAP to our Inference Verification Prototype](https://amododesign.com/notes/2026-09-15-network-tap-inference-verification/)**.

## Pipeline

```text
one TPACKET_V3 ring receiver process per monitor interface
  → single-pass userspace Ethernet/VLAN/[VXLAN]/IPv4/TCP decode
  → symmetric inner-flow hash
  → stateful flow worker processes
      → TCP reassembly → HTTP/SSE assembly → LLM filter/message build
  → parent message queue → persistent Message-writer HTTP threads
```

The pcap replay path runs the same decoder and assembly stages in one
process. Live capture deliberately uses processes: raw decoding,
reassembly, h11 and JSON are CPU work, so Python threads would remain
serialised by the GIL. Only the final network I/O uses threads.

Assembly is **incremental**: each exchange is emitted the moment its response
ends, while its connection stays open. That matters because the clients here
hold pooled keep-alive connections open across many requests — parsing per
connection instead would hold every finished inference hostage to the idle
timeout, delaying verification behind a connection that may never close.

## Capturing a hardware tap: two monitor ports, one pipeline

A passive tap on a full-duplex link cannot present one combined feed. The link
carries two physically separate signal paths, and a tap can only split a signal
that exists, so it gives one monitor output per direction — two ports on the
capture NIC, each carrying half of every conversation.

Live capture therefore starts one ring reader for each interface:

```sh
FRAME_PROCESSOR_MODE=live FRAME_PROCESSOR_IFACE=<monitor-iface-a>,<monitor-iface-b> python3 -m app.main
```

Both must be given. One direction alone yields nothing at all rather than
half: an exchange is emitted when its *response* ends, so the request-only port
never completes one and the response-only port has no request to pair with.

Which port carries which direction is a property of how the cables were run,
changes if anyone re-plugs them, and is not something to configure. Each
reader hashes the unordered **inner** endpoint pair to the same flow worker,
so the two directions converge on exactly one `ConnectionTable` without
locks. Client/server is taken from the handshake (`ConnectionTable._roles`):
a SYN without ACK is a client opening, a SYN-ACK is a server answering. A
capture that joined mid-connection has no handshake to read and still falls
back to first-seen.

Two things the tap needs that a `cni0` capture did not:

- **Promiscuous mode**, set on the socket rather than the device so the kernel
  reverts it on exit. Mirrored frames are addressed between two other
  machines, and the NIC discards them in hardware otherwise. A clean `tcpdump`
  is not evidence the service can see anything — `tcpdump` enables promiscuous
  mode itself for the length of its run.
- **Kernel timestamps**, read directly from each `tpacket3_hdr`. Stamping in
  Python skews by direction once there are several sockets with independent
  queues, and these timestamps become the ledger's `ts`.

Drop counters are read from `PACKET_STATISTICS` every
`FRAME_PROCESSOR_STATS_INTERVAL` seconds and logged when non-zero, because capture
loss does not fail anything — it silently reconstructs incomplete inferences.

## Where the tap watches, and why that means decapsulating

`cni0` sees pod traffic before the CNI touches it. A hardware tap on a
physical link sees it after: pod traffic crossing a node boundary is
VXLAN-encapsulated, so the wire carries UDP between *node* addresses with the
inference request sealed inside. Decoding therefore sees through one tunnel
layer and hands the inner segment downstream, which is what lets both
vantage points produce the same tap message — verified by replaying one
synthetic conversation bare and tunnelled and requiring the results to be
identical (`tests/test_capture.py`).

Note that placement cannot substitute for this. The only traffic a tap on a
node's uplink can see is traffic crossing a node boundary, and pod traffic
crossing a node boundary is exactly what the CNI encapsulates: pod placement
decides *which hop* the tap sees, the CNI decides *whether it is
encapsulated*. Moving pods around changes the first, never the second.

Filtering is content-based (what the traffic *is*), not address-based: an
exchange is LLM traffic if it is a POST to an OpenAI-compatible completions
path with a JSON body carrying `model`. That survives pod IP churn and CNI
encapsulation and needs no packet marking. If it ever proves insufficient, the
fallback identifier is a header stamped by the `verify-tap` sidecar (already
in the request path) — not a change to vLLM.

## Development against pcap fixtures

The pipeline is source-agnostic; develop and test it offline:

```sh
# record a fixture on the tapped node while driving one chat request
tcpdump -i <iface> -w chat.pcap 'tcp'

# replay it through the pipeline (prints tap messages, does not POST)
FRAME_PROCESSOR_MODE=pcap FRAME_PROCESSOR_PCAP_PATH=chat.pcap python3 -m app.main
```

No captures are committed, because they are specific to the network they were
recorded on. Three cases cover the hard parts of the pipeline:

- a single non-streamed chat completion;
- a streamed completion with a long output (thousands of SSE chunks, the case
  that breaks naive parsing);
- a chat request recorded while other node traffic (image pulls, metrics
  scrapes) is flowing, where the filter must yield exactly one message.

Save the tap message(s) you expect next to each capture as
`<name>.expected.json`. Where you can, also keep the event the `verify-tap`
sidecar recorded for the same request. That event is the ground truth the
frame tap is validated against.

The unit tests build their frames in code and need no captures. Run them from
this directory with `python3 -m pytest`.

A wire capture replays with `FRAME_PROCESSOR_DECAP_PORTS` doing the work; setting it
blank reproduces the pre-decapsulation behaviour, which is a useful A/B on the
same file. Beware that a lossy capture reads as a *successful* tap: a replay of
`wire-vxlan.pcap` recovers four requests with correct prompts and empty
responses, because ~25% of the response segments are missing from the capture
itself and every affected connection stays open, so nothing warns. Check for
empty `output_text` with a `completed_at` equal to the end of the capture —
that combination means capture loss, not a short inference.

## Configuration (env)

| Variable | Default | Meaning |
|---|---|---|
| `FRAME_PROCESSOR_MODE` | `pcap` | `pcap` (replay a file) or `live` (AF_PACKET capture, Linux) |
| `FRAME_PROCESSOR_PCAP_PATH` | — | pcap file to replay (pcap mode) |
| `FRAME_PROCESSOR_IFACE` | auto | capture interfaces (live mode), comma-separated; empty = default-route interface |
| `FRAME_PROCESSOR_RING_BLOCK_SIZE` | `1048576` | bytes per packet-ring block |
| `FRAME_PROCESSOR_RING_BLOCK_COUNT` | `128` | blocks per interface (128 MiB/interface with the defaults) |
| `FRAME_PROCESSOR_RING_FRAME_SIZE` | `2048` | TPACKET frame geometry, bytes |
| `FRAME_PROCESSOR_RING_RETIRE_TIMEOUT_MS` | `10` | maximum time before the kernel retires a partially filled block |
| `FRAME_PROCESSOR_FLOW_WORKERS` | `4` | stateful TCP/HTTP worker processes; both directions of one inner flow use one worker |
| `FRAME_PROCESSOR_SEGMENT_QUEUE_SIZE` | `4096` | decoded TCP segments buffered per flow worker |
| `FRAME_PROCESSOR_MESSAGE_QUEUE_SIZE` | `256` | completed messages buffered from flow workers to the parent |
| `FRAME_PROCESSOR_FAST_CLASSIFY` | `true` | raw-header userspace precheck; definitely non-TCP frames skip dpkt, ambiguous layouts fail open |
| `FRAME_PROCESSOR_STATS_INTERVAL` | `30` | seconds between kernel drop-counter reports |
| `FRAME_PROCESSOR_PORTS` | all | comma-separated TCP ports; coarse pre-filter before reassembly. Applied *after* decapsulation, so these are inner ports |
| `FRAME_PROCESSOR_EXCLUDE_PORTS` | none | comma-separated TCP ports never reassembled; beats `FRAME_PROCESSOR_PORTS`. Also inner ports |
| `FRAME_PROCESSOR_DECAP_PORTS` | `4789,8472` | UDP ports carrying VXLAN; blank disables decapsulation |
| `FRAME_PROCESSOR_EMIT_ENABLED` | `false` | `true` POSTs tap messages; `false` prints them to stdout |
| `FRAME_PROCESSOR_MESSAGE_WRITER_URL` | `http://message-writer:8100` | tap destination |
| `FRAME_PROCESSOR_HOSTNAME_MAP` | empty | `model=hostname` pairs, e.g. `openai/gpt-oss-120b=kserve-gpt-oss-120b` — the ledger resolution key per model |
| `FRAME_PROCESSOR_IDLE_TIMEOUT` | `120` | seconds before a silent connection is flushed |
| `FRAME_PROCESSOR_IDLE_SWEEP_INTERVAL` | `1` | capture-time seconds between full active-connection expiry scans |
| `FRAME_PROCESSOR_MAX_STRANDED_BYTES` | `2097152` | bytes one direction may hold behind a missing segment before it is abandoned |
| `FRAME_PROCESSOR_EXCLUDE_HOST_PREFIXES` | `verify-` | request `Host` prefixes never tapped; blank disables |
| `FRAME_PROCESSOR_EMIT_WORKERS` | `8` | Message-writer worker threads, each with a persistent HTTP session |
| `FRAME_PROCESSOR_EMIT_QUEUE_SIZE` | `256` | completed tap messages buffered without blocking capture; overflow is counted and logged |
| `FRAME_PROCESSOR_EMIT_DRAIN_SECONDS` | `5` | how long shutdown waits for the queue; bounded so an unreachable Message writer cannot outlast the pod's grace period |
| `FRAME_PROCESSOR_ACCOUNT_ALL` | `true` | classify and count every frame on the link, not only the TCP the pipeline reconstructs (`app/accounting.py`) |
| `FRAME_PROCESSOR_IFACE_DIRECTION` | none | `iface=in\|out` pairs naming which way each tap monitor port faces, e.g. `<monitor-iface-a>=in,<monitor-iface-b>=out`. Interfaces not listed use the kernel's packet type, which is right only for a capture on this host's own end of the link |
| `FRAME_PROCESSOR_PEER_MAC` | unset | the tapped node's NIC. Setting it turns the link whitelist on (see below) |
| `FRAME_PROCESSOR_LOCAL_MAC` | sysfs | this host's address on the tapped link. Read from `/sys/class/net/<iface>/address` when unset and the capture is on this host's own end; on a tap it must be given, because a monitor port's address is not this host's address on the link |
| `FRAME_PROCESSOR_PEER_IP` / `FRAME_PROCESSOR_LOCAL_IP` | unset | the two addresses on the link; every IPv4 and ARP frame must be between exactly these |
| `FRAME_PROCESSOR_PEER_PORTS` | any | TCP ports the tapped node may serve — its source port going out, the destination port coming in |
| `FRAME_PROCESSOR_ALLOW_OUT` / `FRAME_PROCESSOR_ALLOW_IN` | nothing | frame classes the tapped node may emit / may be sent. Empty means nothing is allowed |
| `FRAME_PROCESSOR_EXPECTED_PINS` | none | `mac/class=digest` pairs: the payload a constant flow must carry, declared rather than learned |
| `FRAME_PROCESSOR_PIN_CLASSES` | none | classes to pin that `UNPINNED_CLASSES` exempts by default. `arp` is the case this exists for |
| `FRAME_PROCESSOR_PIN_WHOLE_FRAME` | none | classes pinned from byte zero rather than past the headers, so the declaration is the exact byte sequence |
| `FRAME_PROCESSOR_EXPECTED_BEATS` | none | `mac/class=seconds` pairs: a flow declared to appear on a fixed cadence. Missing beats are a finding, and so are extra ones |
| `FRAME_PROCESSOR_BEAT_TOLERANCE` | `5` | seconds a beat may land either side of its cadence before it is judged |
| `FRAME_PROCESSOR_LEDGER_URL` | unset | ledger-api base URL. Set it and every closed capture window is posted as a row (`POST /capture-windows`); unset, windows are logged only |
| `FRAME_PROCESSOR_CAPTURE_HOST` | hostname | recorded on each window as where the capture ran |
| `FRAME_PROCESSOR_TAPPED_HOSTNAME` | sole `HOSTNAME_MAP` target | the tapped node as its deployment declared it, so windows join to inference events on `hardware_id` |
| `FRAME_PROCESSOR_WINDOW_SECONDS` | `300` | window length; `0` turns windowing off |
| `FRAME_PROCESSOR_WINDOW_GRACE_SECONDS` | `5` | how long past a window's end a receiver waits before closing it |
| `FRAME_PROCESSOR_WINDOW_FLUSH_SECONDS` | `15` | how long past a window's end the parent waits before writing its row, so flow workers' exchange findings can still land in it |
| `FRAME_PROCESSOR_WINDOW_MAX_GROUPS` | `64` | finding groups kept per window; the excess is counted in `finding_groups_overflow` |
| `FRAME_PROCESSOR_SAMPLE_FRAME_BYTES` | `2048` | bytes of each sampled offending frame carried on a finding |

## Beats: the check a flow fails by not arriving

Every other rule here judges a frame that turned up. That leaves one failure
unreadable, and it is the failure this service has actually had. Once LLDP was
disabled on both ends, the tapped link carries nothing but ARP and the
inference itself — so no frames is exactly as consistent with a prover nobody
is asking for an inference as with a tap that has gone blind. The frame counts
agree, the whitelist is satisfied, every pin holds, and the window row comes
out `complete`. A blind capture certifies clean coverage of a link it cannot
see.

An earlier version of this section argued the opposite, on the strength of a
15.5-hour idle measurement in which the link still carried LLDP, BPDUs and
beacons from both ends. That floor is gone. Zero frames is now a legitimate
state, so silence cannot be an alarm on its own and something has to be
*declared* to be there.

A beat is that declaration: this `(sender, class)` appears every N seconds, so
its absence is evidence rather than the lack of it. The datagram is not sent
from here — the tap is passive, and so is this. It is emitted on a timer by
the tapped node and answered by this one (`tools/tapped_link_health.py`):

```text
tapped node  ── REQ, 31 bytes, every 30s ──▶  capture host  (out: <monitor-iface-b>)
tapped node  ◀───────── ACK, 31 bytes ────────  capture host  (in:  <monitor-iface-a>)
```

Three properties that shape it:

- **It proves delivery, not transmission.** A passive tap sits inline, so the
  monitor port for one direction carries what that end *put on the wire*. A
  beacon into a link severed downstream of the tap still arrives here and
  still reads healthy. A reply cannot: it exists only if the request landed.
- **Each leg is a separate declaration on a separate monitor port**, so a
  missing leg names which port went dark rather than only saying "blind". A
  request that keeps arriving while the answer stops localises the fault to
  the far end rather than to the tap.
- **It is its own flow.** ARP would have worked — on a /30 with two hosts the
  request is byte-identical every time, and the peer's kernel answers it with
  no daemon at all — but it is the data path's own resolution mechanism, and
  a health check that is also load-bearing for addressing is two jobs in one flow.
  A dedicated port is one flow doing one thing, which is also what makes its
  cadence a clean statement rather than something the neighbour table gets a
  vote in.

### Pinning the exact byte sequence

`udp-<port>` is not in `UNPINNED_CLASSES`, so the check is pinned by default
— but only from the end of the headers, and for this flow that is not enough.
Of the 73 bytes, the classification accounts for the ethertype, IP protocol
and destination port, the whitelist accounts for both MACs and both IP
addresses, and the pin covers the 31-byte payload. That leaves **sixteen bytes
constrained by nothing at all**:

```text
[15:16] DSCP/ECN     [16:18] total length   [18:20] identification
[20:22] flags/frag   [22:23] TTL            [24:26] IP checksum
[34:36] UDP sport    [38:40] UDP length     [40:42] UDP checksum
```

Twice every thirty seconds, that is tens of kilobytes a day of residual
capacity in the one flow on this link that could have had none — and the whole
premise here is that a prover able to hide bytes in traffic nobody accounted
for is not constrained by a tap.

`FRAME_PROCESSOR_PIN_WHOLE_FRAME=udp-9999` moves the pin to byte zero, so the
declaration becomes the exact byte sequence and a frame differing by one bit
of TTL is a `pin-mismatch`.

Two fields are excluded from it, because the sender does not choose them:

- **IPv4 identification.** Assigned by the sending kernel. Setting DF on an
  unconnected socket is documented to zero it and in practice does not, so
  it moves on every datagram; pinning it would
  raise a mismatch per frame and say nothing. It is the one field a
  whole-frame pin leaves free: 16 bits a beat, ~11.5 KB/day.
- **The IPv4 header checksum.** A function of the rest of the header. It is
  *verified arithmetically* rather than pinned, which is strictly stronger —
  the arithmetic holds for a frame nobody has seen before, where a pin can
  only compare against one that has. Only for whole-frame classes: a capture
  taken on a sending host sees checksums before the NIC computes them, so
  applying it to every IPv4 frame would raise a finding per frame on a replay
  of such a file.

Everything else stays pinned: 69 of the 73 bytes, against 31 for a payload
pin and 57 unchecked before any of this. It is opt-in per class because it is only sound
where the sender makes every header field deterministic, which is what the
check's socket options are for: an explicit bind fixes the source port,
`IP_TTL` and `IP_TOS` stop the digest moving with a host's defaults, and
`IP_MTU_DISCOVER=IP_PMTUDISC_DO` sets DF — on Linux an *unconnected* socket
sending a DF datagram gets identification 0 rather than a per-destination
counter, which is the one field that otherwise varies by design. The tool uses
`sendto` throughout and never `connect` for exactly that reason: a connected
socket takes an incrementing per-socket counter instead, DF or not.

Whether that actually holds is a property of the sending kernel, not something
this end can assert — so take the digest from a capture, never from a guess.

`FRAME_PROCESSOR_PIN_CLASSES` is unrelated to this flow and remains available for
tightening the traffic the data path emits anyway: `UNPINNED_CLASSES` exempts
ARP because "a router resolving several neighbours emits a different target
each time", which is true on a segment and false on a /30.

The check is two-sided. A missing beat is `beat-missed`; a frame arriving
inside the cadence is `beat-unscheduled`, because "ARP exclusively every 30s"
is a claim about both, and an unexpected frame's timing is a channel. The
schedule only ever moves forward and both the frame path and the timer path
advance it through one helper, so an absence noticed when a window closes
cannot be charged for again when traffic resumes.

A beat is a property of the link, but an accountant is per capture source,
and a tap's monitor port carries one direction only. A declared beat whose
sender cannot appear on an interface is therefore not armed there — arming
both legs on both ports reports the leg arriving on the *other* fibre as
missing, once per interval, forever. Each port watches its own leg and the
parent merges both into one window row, the same split the frame counts
already use. A capture with no declared direction — this host's own end of a
link, or a replay — sees both directions on one source and arms everything.
The startup line says which beats each interface armed, because a beat nobody
armed and a beat nobody declared look identical afterwards.

Where it is evaluated matters as much as what it checks: on each frame of a
declared flow, on the statistics tick so it reaches the log within a beat of
failing, and in `roll()` — the wall-clock path — because a beacon that has
stopped produces no frame to notice it by. Findings are filed on the window
the beat was *due* in, not the window that was open when the absence was
spotted, so a long silence taints every window it spans rather than one.

### Setting it up

`tools/tapped_link_health.py` is stdlib-only and runs one process at each end — as a
service, not a timer. A `systemd` timer defaults to `AccuracySec=1min`, which
would swamp a 30-second cadence on its own; the tool schedules against a
monotonic deadline instead, so it does not drift by however long each send
took.

```sh
# on the prover (the tapped node)
tapped_link_health.py send --peer 192.0.2.1 --bind 192.0.2.2 --interval 30

# on the capture host
tapped_link_health.py respond --bind 192.0.2.1
```

Both ends bind the same port, and that is not cosmetic: an unnamed UDP
datagram is named for whichever of its ports is not ephemeral, so a reply
addressed back to a low source port would be classified under *that* port —
a fresh class every beat, none of them the declared one, and the beat
silently never seen. `tests/test_beats.py` pins this down.

**A UDP-only flow would otherwise make the kernel emit ARP forever**, which is
the non-obvious part of choosing UDP. A cached MAC goes stale about every 30
seconds, and before re-asking the kernel waits ~5s for evidence the peer is
still receiving. TCP supplies that from its acknowledgements; UDP has none to
give. So the entry is never refreshed for free and the kernel probes — on both
ends, ~9,900 ARP frames and ~590 KB a day against the defaults, more than the
health check itself puts on the link, and in a class that is not pinned.

`tools/tapped_link_health.py` sets `MSG_CONFIRM`, the flag for exactly this
shape of protocol: an application that gets its own replies back knows what
the kernel cannot work out, and says so. It is set only when a reply has come
back since the last send, so a peer that has gone is never vouched for.
Expect one ARP exchange per end at startup and effectively none after. The
flag is Linux-only and never reaches the wire — it changes what the local
kernel does about its own cache, not a byte of the frame, so pins and digests
are unaffected. `tests/test_link_health.py` covers it.

Then capture it before declaring anything, because two of the three figures
below can only be measured:

```sh
tcpdump -i <monitor-iface-a> -w health.pcap 'udp port 9999'   # and again on <monitor-iface-b>
FRAME_PROCESSOR_PIN_WHOLE_FRAME=udp-9999 python3 tools/account.py health.pcap
```

The pins table gives the frame digests, and its `distinct` column is the check
that the headers really are deterministic on these hosts: anything other than
`1` per sender means a field is still moving and a whole-frame pin is not yet
safe to declare. The beats table gives the cadence actually kept, which is
what `FRAME_PROCESSOR_BEAT_TOLERANCE` should cover rather than the interval the
config asserts.

```
FRAME_PROCESSOR_ALLOW_OUT=arp,ipv4-tcp,udp-9999
FRAME_PROCESSOR_ALLOW_IN=arp,ipv4-tcp,udp-9999
FRAME_PROCESSOR_PIN_WHOLE_FRAME=udp-9999
FRAME_PROCESSOR_EXPECTED_PINS=02:00:00:00:00:01/udp-9999=<REQ frame digest>,02:00:00:00:00:02/udp-9999=<ACK frame digest>
FRAME_PROCESSOR_EXPECTED_BEATS=02:00:00:00:00:01/udp-9999=30,02:00:00:00:00:02/udp-9999=30
FRAME_PROCESSOR_BEAT_TOLERANCE=5
```

Without `FRAME_PROCESSOR_PIN_WHOLE_FRAME` the digests are the payload's, and
`tapped_link_health.py --digest` prints those without needing a capture
(`56caec88fc977c52` / `05b1b8b40446faee`). That is the weaker of the two
declarations, and the sixteen bytes above stay unchecked.

## Capture windows: the account as a record

Everything above ends in a log line unless it is written somewhere, so the
account of the link is cut into wall-clock-aligned windows (five minutes by
default) and every closed window is posted to the ledger as one row, findings
included. The two tables and the view behind that row are drawn in the
[capture-side ERD](../README.md#capture-side-erd), kept to one side of the core
ledger schema. The write path:

```text
receiver (per interface) ─ FrameAccountant cuts WindowReports on the wall clock ─┐
flow worker (per shard)  ─ ExchangeFinding for an exchange that should not be there ─┤ status queue
                                                                                      ▼
parent ─ WindowMerger: one row per window for the whole link ─ LedgerReporter thread ─ POST /capture-windows
```

Three properties the row is built to have:

- **It exists when nothing happened.** A window with zero frames is still a
  row, so a window that is *missing* means the tap was blind for it — the
  failure this service has actually had, and one a findings-only record could
  never show. The ledger's `GET /capture-windows/latest` is the thing to alert
  on: a newest window older than a few window lengths means the tap is down.
  That covers a tap which stopped writing. It does not, on its own, cover one
  that keeps writing while seeing nothing: a zero-frame window is clean by
  every rule that judges frames, so a blind capture would post clean rows and
  the inferences crossing at the time would read as `complete`. A declared
  beat (above) is what closes that — with one, an empty window carries a
  `beat-missed` finding and is `INCOMPLETE`, so the absence reaches the same
  read-time capture status as any other fault.
- **It is about the link, not a port.** A two-port tap sees each direction on
  a different interface; the parent sums both into one row and marks it
  incomplete if either interface failed to report.
- **It says when it was cut short.** A process that starts or stops
  mid-window files a `capture-gap` finding on that window, so a restart reads
  as a tainted window rather than a clean seam.

Findings on a row are grouped by (kind, class, direction, sender), each with a
count, first/last time and up to three samples carrying the first bytes of the
frame. Groups per window are capped: past the cap the window is still tainted
and the excess is counted, so a prover minting a class per UDP port cannot
grow a row without bound. Exchange-level findings from the flow workers
(`unexpected-exchange`, `non-http-stream`) are filed on the window their
timestamp falls in under the class `http-exchange`.

Each inference event then has a capture status derived at read time
(`GET /inference-events/{id}/capture` on the ledger): `complete`, `tainted`,
`partial` or `uncovered`. It is deliberately not written onto the
verification verdict — the replay stays true when the link was noisy — and
the headline number the whole scheme exists to produce is passes with
complete capture.

A stored capture can be filed the same way:

```sh
FRAME_PROCESSOR_PEER_MAC=… FRAME_PROCESSOR_TAPPED_HOSTNAME=kserve-gpt-oss-120b \
python3 tools/account.py link-all.pcap --ledger http://ledger-api:8000
```

## The link whitelist

Accounting says what every frame *is*; the whitelist (`app/policy.py`) says
whether it is allowed to be on this link, going the way it was going. It is
written for a point-to-point link with one tapped node on the far end, which
is what the tapped node's isolated /30 is. Through the optical tap:

```sh
FRAME_PROCESSOR_IFACE=<monitor-iface-a>,<monitor-iface-b>
FRAME_PROCESSOR_IFACE_DIRECTION=<monitor-iface-a>=in,<monitor-iface-b>=out
FRAME_PROCESSOR_PEER_MAC=02:00:00:00:00:01   FRAME_PROCESSOR_LOCAL_MAC=02:00:00:00:00:02
FRAME_PROCESSOR_PEER_IP=192.0.2.2         FRAME_PROCESSOR_LOCAL_IP=192.0.2.1
FRAME_PROCESSOR_PEER_PORTS=8000
FRAME_PROCESSOR_ALLOW_OUT=lldp,arp,ipv4-tcp  FRAME_PROCESSOR_ALLOW_IN=lldp,arp,ipv4-tcp
FRAME_PROCESSOR_EXPECTED_PINS=02:00:00:00:00:01/lldp=<frame digest>,02:00:00:00:00:02/lldp=<frame digest>
```

Directions are named from the tapped node's side: `out` is what it emits,
`in` is what it is sent. Where the direction comes from depends on where the
capture sits, and getting this wrong is loud rather than subtle — every frame
in one direction fails the other direction's list:

- **On a tap**, each monitor port carries one direction, and which port
  carries which is a property of how the fibres were run. Declare it with
  `FRAME_PROCESSOR_IFACE_DIRECTION`. A monitor port never transmits, so the
  kernel's packet type would call everything on it "received" and both
  directions would read as `out`.
- **On this host's own end of the link** (a capture without a tap, on the
  interface that terminates the link), interfaces with no declared direction
  use the kernel's packet type: `PACKET_OUTGOING` means this host
  sent it, anything else arrived from the wire. The peer cannot move a frame
  to the other list by writing a different source MAC — a frame that arrived
  is an `out` frame whatever it claims, and the claim is then checked.
- **A pcap replay** has neither and falls back to the source MAC, which is
  where the `unknown` direction comes from.

Each departure from the whitelist is its own finding kind, so a report says
what was wrong rather than that something was:

| finding | meaning |
|---|---|
| `class-not-allowed` | a frame class that may not travel this direction (any UDP from the tapped node, LLDP towards it, IPv6 anywhere) |
| `mac-unknown` | a third MAC on a two-node link, or a frame whose direction and source MAC disagree |
| `ip-outside-link` | an IPv4 packet not between exactly the two link addresses |
| `tcp-port-unexpected` | a TCP segment on a port the tapped node was not declared to serve — in particular, the tapped node *dialling out* |
| `arp-address-unexpected` | ARP about any address but the two on the link |
| `pin-mismatch` | a pinned flow carrying something other than its declared payload |
| `pin-undeclared` | a constant flow the whitelist did not mention; it is pinned to what it carries, but it should have been declared |

Try a rule against a stored window before enforcing it on the wire; the same
environment drives both:

```sh
FRAME_PROCESSOR_PEER_MAC=02:00:00:00:00:03 FRAME_PROCESSOR_PEER_IP=192.0.2.2 FRAME_PROCESSOR_LOCAL_IP=192.0.2.1 \
FRAME_PROCESSOR_PEER_PORTS=8000 FRAME_PROCESSOR_ALLOW_OUT=lldp,arp,ipv4-tcp FRAME_PROCESSOR_ALLOW_IN=arp,ipv4-tcp \
FRAME_PROCESSOR_EXPECTED_PINS=02:00:00:00:00:03/lldp=<frame digest> \
python3 tools/account.py link-all.pcap      # exit 0 only when complete with no findings
```

`FRAME_PROCESSOR_EXCLUDE_PORTS` is the safer of the two port filters, and usually the
one to reach for. Both keep noise out of the reassembler, but they fail in
opposite directions: an allowlist that goes stale drops inference *silently* —
a model served on a port nobody added never reaches the reassembler and the tap
runs on with nothing to show — whereas a denylist can only ever discard what it
was explicitly named. It costs throughput when it is wrong, not coverage.

It exists because the reassembler's load is not necessarily inference at all.
When a database shares the tapped link, verification pulls stored logits across
it in bulk — easily outweighing the inference traffic — with every frame
decoded, reassembled, and handed to an HTTP parser that could never make sense
of it. `FRAME_PROCESSOR_EXCLUDE_PORTS=5432` removes that
without putting an inference at risk. Better still is to keep the database off
the tapped link, which removes the traffic rather than filtering it — but that
is a placement decision, and this is an env var.

Both port filters remain entirely in userspace. On the ring path the common
IPv4/VXLAN/TCP decoder applies them before copying a payload or dispatching a
segment, so they save allocation, IPC, reassembly and parsing while every
mirrored frame still enters the AF_PACKET ring.

`FRAME_PROCESSOR_FAST_CLASSIFY` selects a bounds-checked single-pass raw decoder for
the common IPv4 path. It rejects outer UDP/VXLAN carrying inner non-TCP before
allocation and decodes candidate TCP without constructing dpkt object graphs.
Truncated, IPv6 and unfamiliar layouts fail open into the established decoder,
so the fast path cannot silently narrow coverage. Disable it for an A/B against
the same pcap.

The live ring amortises a wake-up across a whole block and inspects UDP noise
in-place; only candidate TCP is copied before a block is returned to the
kernel. Nothing allocates a Python `bytes`/`Frame` object for a noise packet.
Ring drops, queue drops,
worker errors and high-water marks are reported independently so a bottleneck
can be located rather than inferred from missing inferences.

Message-writer requests never run on the capture thread. Completed messages enter
a bounded queue and `FRAME_PROCESSOR_EMIT_WORKERS` persistent HTTP sessions drain it.
If that queue fills, the new message is dropped and an error/counter makes the
loss explicit; blocking would instead stop packet reads and corrupt unrelated
connections. The queue also warns once, at half depth, while there is still
headroom to raise `FRAME_PROCESSOR_EMIT_WORKERS` or `FRAME_PROCESSOR_EMIT_QUEUE_SIZE` —
a drop only reports loss that has already happened. Normal shutdown drains
accepted messages before the workers exit.

Each delivered inference logs one INFO line naming its session, so a running
tap can be watched by tailing it.

`FRAME_PROCESSOR_HOSTNAME_MAP` matters: `tap.hostname` is how the Message writer
resolves the event to a declared model deployment. Unmapped models fall back
to `kserve-<model basename>`.

`FRAME_PROCESSOR_IDLE_TIMEOUT` does not affect tap latency — exchanges are emitted
as they complete. It only salvages an exchange the capture cut in half and
releases state for connections that vanished without a close, so longer is
safer: flushing early truncates an inference that was merely slow, and the
bytes that follow start a stream the parser cannot resynchronise.

`FRAME_PROCESSOR_DECAP_PORTS` defaults cover flannel (8472, the port Linux shipped
before the standard settled) and the IANA assignment (4789). Confirm what the
cluster actually uses with `ip -d link show flannel.1` and set this explicitly
in the deployment rather than relying on the default: a mismatch is silent —
the tap runs normally and simply never tags anything — so the configured value
is logged at startup to make it diagnosable from the first line. Geneve is a
different header rather than another port, so 6081 does not belong here until
`_vxlan_payload` knows the difference.

`FRAME_PROCESSOR_EXCLUDE_HOST_PREFIXES` keeps the verifier's own replays out of the
ledger. frame-processor exists to find inference traffic *for* the verifier,
so recording the verifier's replays would file them as fresh events against the
production deployment — and every event it files is another event
needing verification. The verifier pair lands on whichever node has the GPU,
frequently the tapped one, so the exclusion matches the `Host` the runner
dials (`verify-<model-slug>-kserve-workload-svc`) rather than an address.

## Capacity testing and sizing

Test packet rate, not only payload bandwidth. With iperf3, `-b 8M -P 8
--bidir -l 64` is 128 Mbps of UDP payload but **250,000 datagrams/second**
across both directions; encapsulation makes the captured wire byte rate much
higher. A single missing TCP frame can poison every later inference on one
pooled connection, so a small drop percentage is not a useful pass condition.

Start with about six CPU cores available to the pod (two receiver processes
plus four flow workers) and no tight CPU limit. The default rings reserve 128
MiB per interface; allow roughly 4 GiB for two rings, process overhead,
reassembly, completion messages and transient logprob payloads, then tune from
observed peaks.

For each load step, require all of the following before increasing it:

- zero packet-ring/kernel drops, truncations and malformed blocks;
- zero segment-queue, message-queue and emitter-queue drops;
- no receiver/worker failures or forced shutdowns; and
- every prompt accounted for, with no capture-caused unverifiable records.

A useful ladder is ten minutes at 250 kpps, then 500 kpps, while separately
running the 800-prompt/32-concurrent inference workload. Keep the UDP-only run
as an acquisition benchmark and the combined run as the end-to-end acceptance
test.

## Build & push

Built from the **repo root** (it copies `inf-proxy/app/capture.py`, the pure
prompt/token-id/sampling extractors, so both taps render these identically):

```sh
docker build -t <registry-host>/infver_images/frame-processor:latest -f frame-processor/Dockerfile .
docker push <registry-host>/infver_images/frame-processor:latest
```

The tag must match the `image:` reference in
`kubernetes_setup/manifests/infver/frame-processor.yaml` (and
`tapped-link-health.yaml`, which runs the same image), so substitute the same
registry in both.

## Placeholders

Angle-bracket values describe one physical link and one deployment; nothing
here has a sensible default for them, and the build, the manifests or the
process itself will fail until they are replaced.

| Placeholder | Where | What to put there |
|---|---|---|
| `<registry-host>` | the build command above, `Dockerfile` header, `kubernetes_setup/manifests/infver/frame-processor.yaml`, `tapped-link-health.yaml` | The registry the cluster pulls from. `docker push` fails on the literal value. |
| `<monitor-iface-a>`, `<monitor-iface-b>` | `FRAME_PROCESSOR_IFACE` and `FRAME_PROCESSOR_IFACE_DIRECTION` in `frame-processor.yaml` and the examples in this file | The capture host's two monitor-port interfaces (`ip -br link`). Which is `in` and which is `out` depends on how the tap's fibres were run; `tools/account.py --live <iface>` on each shows which direction it carries. |
| `<tapped-node-mac>`, `<capture-host-mac>` | `FRAME_PROCESSOR_PEER_MAC`, `FRAME_PROCESSOR_LOCAL_MAC` and the `EXPECTED_PINS` / `EXPECTED_BEATS` keys in `frame-processor.yaml` | The two ends of the tapped link. Setting the peer MAC is what turns the whitelist on. |
| `<tapped-node-ip>`, `<capture-host-ip>` | `FRAME_PROCESSOR_PEER_IP`, `FRAME_PROCESSOR_LOCAL_IP` in `frame-processor.yaml`; `PEER_ADDR` / `BIND_ADDR` in `isolated_prover_setup/*/.env` | The link's two addresses. The same pair must appear on both sides or the health check never completes a beat. |
| `<REQ frame digest>`, `<ACK frame digest>` | `FRAME_PROCESSOR_EXPECTED_PINS` in `frame-processor.yaml` | Read off a capture with `tools/account.py` (see [Pinning the exact byte sequence](#pinning-the-exact-byte-sequence)). |

Concrete addresses in this file — `192.0.2.1` / `192.0.2.2` and the
`02:00:00:00:00:0x` MACs — are documentation-range examples standing in for
the same values, not defaults.

## Known simplifications (scaffold)

- TCP reassembly drops exact-duplicate segments but does not trim partially
  overlapping retransmissions.
- HTTP/1.1 only (the tapped hop is plaintext HTTP/1.1; HTTP/2 would need a new
  parser stage, not a tweak).
- One tunnel layer, VXLAN only. Fragmented IPv4 is dropped and counted rather
  than reassembled — a later fragment has no transport header and a first
  fragment has a truncated payload, so neither can be decoded from one frame.
  On this path fragments mean the overlay MTU is wrong, not that the traffic
  is unusual.
- A dropped frame still costs the connection it was on. Nothing can refill the
  hole — the endpoints never learn an observer missed a packet, so there is no
  retransmission to wait for — and the parser cannot restart mid-message, so
  every later inference on that connection is lost with it. Past
  `FRAME_PROCESSOR_MAX_STRANDED_BYTES` the direction is abandoned and logged, which
  bounds the memory and makes the loss visible; it does not win the connection
  back. Loss below that threshold is still only reported when a connection
  *completes*, so a stalled-but-open one stays quiet until the idle flush.
  Complete exchanges emitted before the gap remain valid. Requests queued
  behind it can never be paired with a response and are filed anyway, with
  whatever was captured, because an inference the tap saw and never filed
  cannot be told apart from one that never happened. They arrive as events
  with a prompt and an empty response, which `inf-ver-runner` rejects on a
  pre-check as `unverifiable / tokenization_mismatch` instead of replaying —
  so they can never be reported as a *failed* verification. That error code
  is shared with genuine tokenizer problems, so the kernel drop counters for
  the same window are what identify capture loss as the cause.
- No in-kernel packet filter. Every cross-node pod frame still crosses the
  AF_PACKET socket boundary, by design. The mapped userspace reader walks and
  rejects irrelevant packets without allocating a Python frame object.
- Streamed responses are merged from SSE chunks; token IDs / logprobs are
  extracted when present in the stream, else recorded as `null` (the contract
  allows this).
- One exchange at a time per connection (no pipelining) — matches real client
  behaviour on this path.
