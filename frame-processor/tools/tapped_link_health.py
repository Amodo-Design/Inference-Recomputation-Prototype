#!/usr/bin/env python3
"""Tapped-link health: a fixed datagram, both ways, on a fixed cadence.

Not part of frame-processor. That service is passive and this is not — it is the thing
being observed, and it lives here only because the bytes it puts on the wire
are the bytes the accountant is told to expect (`FRAME_PROCESSOR_EXPECTED_PINS`),
and those two must not be able to drift apart. Nothing under `app/` imports
it — the criterion for living here, shared with `account.py` alongside — and
unlike that one it imports nothing from `app/` either: stdlib only, sharing a
payload and nothing else.

Why it exists: once LLDP was disabled at both ends, the tapped link carries
nothing but ARP and the inference itself, so a link with no inference on it
produces no frames — and no frames is exactly as consistent with an idle
prover as with a capture that has gone blind. Everything frame-processor checks is
a rule about a frame that arrived, and all of them are satisfied by nothing
arriving. A declared health check closes the gap: something is
asserted to cross this link every N seconds, so absence becomes evidence.

    prover  ── REQ every N seconds ──▶  capture host
    prover  ◀────────── ACK ──────────  capture host

The reply is the half that makes it a proof of *delivery*. A passive tap sits
inline, so the monitor port for one direction carries what that end put on the
wire: a beacon sent into a link severed downstream of the tap still reaches
the tap and still reads healthy. An answer cannot be faked that way — it
exists only if the request arrived.

Two constraints the payload has to meet, both of them consequences of how the
accountant judges it:

- **Byte-identical every time.** `udp-<port>` is a pinned class, so every
  datagram is compared against the first one seen. No timestamp, no sequence
  number, no counter — which is why this sends a constant and liveness is
  read from arrival times alone.
- **Both ends bound to the same port.** An unnamed UDP datagram is named for
  whichever of its ports is not ephemeral, so a reply addressed to a low
  source port would be classified under *that* port instead and mint a class
  per beat. Symmetric ports make both legs `udp-<port>`, always.

Run one of each:

    tapped_link_health.py send    --peer 192.0.2.1 --bind 192.0.2.2  # prover
    tapped_link_health.py respond --bind 192.0.2.1                    # capture host
"""

from __future__ import annotations

import argparse
import logging
import signal
import socket
import sys
import time

# The wire contract. Changing either of these changes the digest the tap is
# configured to expect, so both ends and FRAME_PROCESSOR_EXPECTED_PINS move together.
# Equal length each way, and past the 18 bytes at which a frame reaches the
# 60-byte Ethernet minimum: a shorter one would be padded, and padding is
# separately required to be zero.
REQUEST = b"INFVER-TAPPED-LINK-HEALTH/1 REQ\n"
REPLY = b"INFVER-TAPPED-LINK-HEALTH/1 ACK\n"

DEFAULT_PORT = 9999
DEFAULT_INTERVAL = 30.0

# Tells the kernel our packets are getting through, so it does not have to ask
# the wire.
#
# Without it this health check GENERATES ARP, continuously and on both ends. A
# cached MAC address goes stale about every 30 seconds; before re-asking, the
# kernel waits ~5s for evidence the peer is still receiving. TCP supplies that
# evidence from its acknowledgements — UDP has none to give, so the kernel has
# nothing to refresh the entry with and sends an ARP request instead. Measured
# against the kernel defaults that is ~9,900 ARP frames and ~590 KB a day,
# more than this check itself puts on the link.
#
# MSG_CONFIRM is the escape hatch for exactly this shape of protocol: an
# application that gets its own replies back knows what the kernel cannot work
# out, and says so. Set only when a reply has actually come back since the
# last send — asserting it unconditionally would keep the entry fresh against
# a peer that had gone away, which is a claim we would not have earned.
#
# Linux only, and it never reaches the wire: it changes what the local kernel
# does about its own cache, not a single byte of the frame. Pins and digests
# are unaffected. Elsewhere the getattr yields 0, which is "no flags" — the
# tool still runs, it just lets the kernel probe as before.
MSG_CONFIRM = getattr(socket, "MSG_CONFIRM", 0)

# Set explicitly rather than inherited, so the frame digest does not move when
# a host's net.ipv4.ip_default_ttl does.
TTL = 64
TOS = 0

log = logging.getLogger("tapped-link-health")
_stop = False


def _on_signal(_signum, _frame) -> None:
    global _stop
    _stop = True


def _socket(bind_ip: str, port: int) -> socket.socket:
    """A socket whose datagrams are byte-identical frame to frame.

    The payload being constant is not enough. Pinned to the payload alone, a
    66-byte beacon still leaves sixteen header bytes — DSCP, total length,
    identification, flags, TTL, both checksums, source port and UDP length —
    that nothing checks, which is capacity in the one flow on this link that
    could have had none. Each option below removes one source of variance so
    the whole frame can be pinned instead:

    - ``SO_REUSEADDR`` and an explicit ``bind`` fix the source port. It is
      otherwise ephemeral and rotates, and it is also what keeps both legs in
      one class: an unnamed datagram is named for whichever of its ports is
      not ephemeral, so a reply to a low source port would be classified
      under *that* port instead.
    - ``IP_MTU_DISCOVER=IP_PMTUDISC_DO`` sets DF. On Linux an unconnected
      socket sending a DF datagram gets identification 0 rather than a
      per-destination counter, which is the one field that genuinely varies.
      This is why the tool uses ``sendto`` throughout and never ``connect``:
      a connected socket takes the incrementing per-socket counter instead,
      DF or not. At 66 bytes the fragmentation DF implies cannot arise.
    - ``IP_TTL`` and ``IP_TOS`` are set rather than inherited, so the digest
      does not change when a host's ``ip_default_ttl`` does.

    None of this is asserted to have worked: whether identification really is
    zero is a property of the sending kernel. Take the digest from a capture
    (`python3 tools/account.py`), not from this docstring.

    What is deliberately NOT here is anything that changes the frame's timing
    or provokes other traffic. MSG_CONFIRM above is the second half of that:
    left alone, a UDP-only flow makes the kernel emit ARP forever.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    for level, option, value in (
        (socket.IPPROTO_IP, "IP_MTU_DISCOVER", "IP_PMTUDISC_DO"),
        (socket.IPPROTO_IP, "IP_TTL", TTL),
        (socket.IPPROTO_IP, "IP_TOS", TOS),
    ):
        name = getattr(socket, option, None)
        if name is None:  # not Linux; the tool still runs, less deterministically
            log.warning("%s unavailable here; frame headers may vary", option)
            continue
        setting = value if isinstance(value, int) else getattr(socket, value, None)
        if setting is None:
            log.warning("%s unavailable here; frame headers may vary", value)
            continue
        try:
            sock.setsockopt(level, name, setting)
        except OSError as exc:
            log.warning("could not set %s: %s; frame headers may vary", option, exc)
    sock.bind((bind_ip, port))
    return sock


def send(peer: str, port: int, bind_ip: str, interval: float) -> int:
    """Emit a request every ``interval`` seconds and drain the answers.

    Scheduled against a monotonic deadline rather than by sleeping the
    interval, so the cadence does not walk forward by however long each send
    took. A machine suspended past several deadlines resumes on the next one
    instead of firing a burst to catch up — the tap has already recorded that
    silence, and a burst would be recorded as unscheduled frames on top of it.
    """
    sock = _socket(bind_ip, port)
    sock.settimeout(0.5)
    next_at = time.monotonic()
    sent = received = 0
    # Whether a reply has come back since the last request went out. This is
    # the evidence MSG_CONFIRM asserts, so it is tracked rather than assumed:
    # the first request has nothing behind it and goes unflagged, which costs
    # one ARP exchange at startup and nothing after.
    answered_since_send = False
    while not _stop:
        now = time.monotonic()
        if now < next_at:
            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            if data == REPLY:
                received += 1
                answered_since_send = True
            continue
        sock.sendto(REQUEST, MSG_CONFIRM if answered_since_send else 0, (peer, port))
        answered_since_send = False
        sent += 1
        next_at += interval
        if next_at < now:
            next_at = now + interval
        if sent % 120 == 0:
            log.info("sent=%d answered=%d", sent, received)
    log.info("stopping: sent=%d answered=%d", sent, received)
    sock.close()
    return 0


def respond(bind_ip: str, port: int) -> int:
    """Answer every request with the fixed reply, and nothing else with it.

    Only ``REQUEST`` is answered: a responder that echoed whatever arrived
    would be a way to put chosen bytes on the link, which is the opposite of
    what a whitelisted flow is for.
    """
    sock = _socket(bind_ip, port)
    sock.settimeout(0.5)
    answered = ignored = 0
    while not _stop:
        try:
            data, sender = sock.recvfrom(2048)
        except socket.timeout:
            continue
        if data != REQUEST:
            ignored += 1
            continue
        # Confirmed on the strength of the request that just arrived. Weaker
        # evidence than the sender's — a request proves the peer can reach us,
        # not that our replies reach it — but on a point-to-point link with
        # two fixed addresses it cannot mislead: if the peer were gone no
        # request would arrive, so there would be no reply to flag. Liveness
        # here is the tap's job, not the neighbour cache's.
        sock.sendto(REPLY, MSG_CONFIRM, sender)
        answered += 1
        if answered % 120 == 0:
            log.info("answered=%d ignored=%d", answered, ignored)
    log.info("stopping: answered=%d ignored=%d", answered, ignored)
    sock.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", nargs="?", choices=("send", "respond"))
    parser.add_argument("--peer", help="the other end's address (send mode)")
    parser.add_argument("--bind", default="0.0.0.0", help="local address to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument(
        "--digest",
        action="store_true",
        help="print the PAYLOAD digests and exit. Enough for a payload pin; "
        "a whole-frame pin (FRAME_PROCESSOR_PIN_WHOLE_FRAME) covers headers this "
        "end never sees, so take that digest from `python3 tools/account.py` "
        "over a real capture instead",
    )
    args = parser.parse_args(argv)

    if args.digest:
        import hashlib

        for name, payload in (("REQ", REQUEST), ("ACK", REPLY)):
            print(f"{name} {hashlib.blake2b(payload, digest_size=8).hexdigest()}")
        return 0

    if args.mode is None:
        parser.error("a mode is required unless --digest is given")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    if args.mode == "send":
        if not args.peer:
            parser.error("send mode needs --peer")
        if args.interval <= 0:
            parser.error("--interval must be positive")
        return send(args.peer, args.port, args.bind, args.interval)
    return respond(args.bind, args.port)


if __name__ == "__main__":
    sys.exit(main())
