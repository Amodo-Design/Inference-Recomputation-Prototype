"""HTTP/1.1 parsing over reassembled byte streams.

Turns a connection's two byte streams into paired request/response exchanges,
incrementally: `ExchangeAssembler` is fed bytes as reassembly makes them
contiguous and yields each exchange the moment its response ends, so an
inference is tapped when it finishes rather than when its connection does.
Handles the shapes this path actually produces: JSON POST requests
(Content-Length), JSON responses (Content-Length), and streamed completions
(Transfer-Encoding: chunked carrying text/event-stream SSE frames).

Framing is delegated to h11 — the sans-IO HTTP/1.1 state machine underneath
urllib3 and httpx — so chunked decoding, header folding, and the
Content-Length/Transfer-Encoding precedence rules come from a parser that has
been attacked far harder than anything written here would be.

h11 wants to co-drive a live connection, which a passive tap does not have: we
observe both halves and answer neither. So each direction gets its own
Connection with the other role synthesised. The client's bytes go to a SERVER
connection, which is handed a throwaway 204 so it will start the next cycle;
the server's bytes go to a CLIENT connection primed with the method we already
parsed, because h11 frames a response against its request (HEAD, 204 and 304
carry no body, and guessing wrong would swallow the next response). Both
synthesised messages are discarded — h11 is sans-IO, `send()` returns bytes
rather than transmitting them, and nothing here owns a writable socket.

Truncated captures are normal here — a connection can be flushed on the idle
timeout mid-response. h11 raises on an incomplete body, so we keep whatever
decoded before the error and mark the message `truncated`; that is what lets a
partial SSE stream still yield the tokens it did carry.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import h11

log = logging.getLogger("frameprocessor.http")

# Synthesised request used to drive the response parser. The target is not
# part of response framing, so a fixed one avoids re-validating a captured
# path; the method is replayed from the real request because it is.
_PRIMER_HEADERS = [("host", "frame-processor"), ("content-length", "0")]

# Response bytes h11 may hold while no request method has arrived to frame them.
#
# Generous on purpose. The legitimate wait is a scheduling artifact between two
# independent reader processes and resolves in milliseconds, so a backlog
# anywhere near this means the request direction is gone rather than late. Past
# it the response is framed as a POST reply, which consumes the bytes instead of
# hoarding them for the life of a connection that may never close.
_MAX_UNPRIMED_RESPONSE_BYTES = 2 * 1024 * 1024


@dataclass
class HttpMessage:
    headers: dict[str, str] = field(default_factory=dict)  # keys lower-cased
    body: bytes = b""
    # request-only
    method: str | None = None
    path: str | None = None
    # response-only
    status: int | None = None
    # The capture ended before the body did; `body` is what survived.
    truncated: bool = False

    @property
    def is_sse(self) -> bool:
        return "text/event-stream" in self.headers.get("content-type", "")

    def json(self) -> dict[str, Any] | None:
        try:
            parsed = json.loads(self.body)
        except (ValueError, UnicodeDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None


@dataclass
class Exchange:
    """One request and the response that answered it.

    Timestamps are per-exchange, taken from the segments that carried the
    request's first byte and the response's last: on a pooled connection
    carrying many inferences, connection-level timing would stamp them all
    identically.
    """

    request: HttpMessage
    response: HttpMessage | None
    received_at: float | None = None
    completed_at: float | None = None


def parse_exchanges(client_data: bytes, server_data: bytes) -> list[Exchange]:
    """Batch form: both directions in full, exchanges out.

    Nothing in the pipeline takes this path — pcap replay and live capture
    both feed the assembler as bytes arrive. It exists as the oracle the
    assembler is checked against: the same bytes fragmented any which way must
    reconstruct to exactly this.
    """
    assembler = ExchangeAssembler()
    exchanges = assembler.feed_client(client_data)
    exchanges += assembler.feed_server(server_data)
    exchanges += assembler.finish()
    return exchanges


def _headers_to_dict(headers: Any) -> dict[str, str]:
    """h11 hands back a list of (name, value) byte pairs; last value wins."""
    return {
        name.decode("latin-1").lower(): value.decode("latin-1") for name, value in headers
    }


class ExchangeAssembler:
    """Incremental HTTP/1.1 exchange assembly for one connection.

    Bytes are fed in as TCP reassembly makes them contiguous, and an exchange
    is returned the moment its response ends — not when the connection does.
    On the pooled keep-alive connections this path uses, that is the whole
    difference between tapping an inference as it finishes and waiting out the
    idle timeout for a connection that may stay open for hours.

    One assembler per connection, and it holds the h11 state machines across
    exchanges, so a connection carrying twenty inferences parses as twenty
    cycles rather than one buffer re-parsed twenty times.

    A direction that cannot be parsed at all — the usual cause is a capture
    that joined mid-message, where h11 has no way to resynchronise — is marked
    dead, and its bytes are dropped on arrival from then on instead of
    accumulating for the life of the connection.
    """

    def __init__(self) -> None:
        self._req_conn = h11.Connection(our_role=h11.SERVER)
        self._resp_conn = h11.Connection(our_role=h11.CLIENT)

        self._req_open: HttpMessage | None = None
        self._req_body = bytearray()
        self._req_started_at: float | None = None

        self._resp_open: HttpMessage | None = None
        self._resp_body = bytearray()

        # Requests parsed and awaiting the response that answers them; HTTP/1.1
        # without pipelining answers in order, so this pairs positionally.
        self._waiting: deque[tuple[HttpMessage, float | None]] = deque()
        # Methods in arrival order: h11 frames a response against its request
        # (HEAD, 204 and 304 carry no body), so each response cycle is primed
        # with the method its request actually used.
        self._methods: deque[str] = deque()
        self._primed = False
        self._server_pending = False
        self._server_pending_ts: float | None = None
        # Response bytes accepted since the last successful prime — see
        # _no_method_is_coming.
        self._unprimed_bytes = 0

        self._req_dead = False
        self._resp_dead = False
        self.unparseable = False  # a direction died with no message recovered
        # Set by the pipeline once it has raised the finding for the flag
        # above, so a connection that keeps feeding bytes is one finding.
        self.unparseable_reported = False

    # --- feeding ---------------------------------------------------------

    def feed_client(self, data: bytes, ts: float | None = None) -> list[Exchange]:
        """Client→server bytes in; exchanges completed by them out."""
        if data and not self._req_dead:
            self._req_conn.receive_data(data)
            self._pump_requests(ts)
        # A newly parsed request can unblock a response that arrived first.
        response_ts = self._server_pending_ts if self._server_pending else ts
        return self._pump_responses(response_ts)

    def feed_server(self, data: bytes, ts: float | None = None) -> list[Exchange]:
        """Server→client bytes in; exchanges completed by them out."""
        if data and not self._resp_dead:
            self._resp_conn.receive_data(data)
            self._server_pending = True
            self._server_pending_ts = ts
            if not self._primed:
                self._unprimed_bytes += len(data)
        return self._pump_responses(ts)

    def finish(self, ts: float | None = None) -> list[Exchange]:
        """End of capture for this connection: salvage whatever is half-read.

        A tap has no guarantee of seeing a clean close, so a response cut off
        by the idle flush still yields the tokens it did carry, and a request
        that was never answered is still a real observation.
        """
        done: list[Exchange] = []
        if not self._resp_dead:
            self._resp_conn.receive_data(b"")  # EOF, not a graceful close
            self._server_pending = True
            done.extend(self._pump_responses(ts))

        if self._resp_open is not None:
            exchange = self._pair(self._close_open_response(), ts)
            if exchange is not None:
                done.append(exchange)

        while self._waiting:
            request, started = self._waiting.popleft()
            done.append(
                Exchange(request=request, response=None, received_at=started, completed_at=ts)
            )

        if self._req_open is not None:
            self._req_open.body = bytes(self._req_body)
            self._req_open.truncated = True
            done.append(
                Exchange(
                    request=self._req_open,
                    response=None,
                    received_at=self._req_started_at,
                    completed_at=ts,
                )
            )
            self._req_open = None
        return done

    # --- request direction ----------------------------------------------

    def _pump_requests(self, ts: float | None) -> None:
        while not self._req_dead:
            try:
                event = self._req_conn.next_event()
            except h11.RemoteProtocolError as exc:
                self._die_requests(exc)
                return

            if event is h11.NEED_DATA or event is h11.PAUSED:
                return
            if isinstance(event, h11.ConnectionClosed):
                return

            if isinstance(event, h11.Request):
                self._req_open = HttpMessage(
                    headers=_headers_to_dict(event.headers),
                    method=event.method.decode("latin-1"),
                    path=event.target.decode("latin-1"),
                )
                self._req_body = bytearray()
                self._req_started_at = ts
                self._methods.append(self._req_open.method)
            elif isinstance(event, h11.Data):
                self._req_body += event.data
            elif isinstance(event, h11.EndOfMessage):
                if self._req_open is not None:
                    self._req_open.body = bytes(self._req_body)
                    self._waiting.append((self._req_open, self._req_started_at))
                    self._req_open = None
                self._next_request_cycle()

    def _next_request_cycle(self) -> None:
        # h11 will not begin another cycle until our own side has answered. A
        # tap never answers, so synthesise a bodiless response and drop the
        # bytes it produces — nothing is ever written to the wire.
        try:
            self._req_conn.send(h11.Response(status_code=204, headers=[]))
            self._req_conn.send(h11.EndOfMessage())
            self._req_conn.start_next_cycle()
        except h11.ProtocolError:
            self._req_dead = True  # e.g. the request asked to close

    def _die_requests(self, exc: Exception) -> None:
        self._req_dead = True
        if self._req_open is not None:
            self._req_open.body = bytes(self._req_body)
            self._req_open.truncated = True
            self._waiting.append((self._req_open, self._req_started_at))
            self._req_open = None
            log.debug("Truncated request (%s)", exc)
        else:
            self.unparseable = True
            # Nothing was recovered, so these bytes never parsed as HTTP at
            # all — distinct from the truncated case above, which is a normal
            # capture artifact on a tap. On a link whose only permitted TCP is
            # inference, a stream that is not HTTP is the smuggling case, and
            # it used to leave no trace anywhere: the flag set here was read
            # by nothing and the message logged at debug, invisible at the
            # deployed level.
            log.warning("finding: non-http-stream (request direction) — %s", exc)

    # --- response direction ----------------------------------------------

    def _pump_responses(self, ts: float | None) -> list[Exchange]:
        done: list[Exchange] = []
        while not self._resp_dead:
            if not self._primed and not self._prime():
                break
            try:
                event = self._resp_conn.next_event()
            except h11.RemoteProtocolError as exc:
                self._die_responses(exc, ts, done)
                break

            if event is h11.NEED_DATA or event is h11.PAUSED:
                self._server_pending = False  # h11 consumed everything fed
                self._server_pending_ts = None
                break
            if isinstance(event, h11.ConnectionClosed):
                break
            if isinstance(event, h11.InformationalResponse):
                continue  # 1xx; the real response follows

            if isinstance(event, h11.Response):
                self._resp_open = HttpMessage(
                    headers=_headers_to_dict(event.headers), status=event.status_code
                )
                self._resp_body = bytearray()
            elif isinstance(event, h11.Data):
                self._resp_body += event.data
            elif isinstance(event, h11.EndOfMessage):
                if self._resp_open is not None:
                    self._resp_open.body = bytes(self._resp_body)
                    exchange = self._pair(self._resp_open, ts)
                    self._resp_open = None
                    if exchange is not None:
                        done.append(exchange)
                if not self._next_response_cycle():
                    break
        return done

    def _prime(self) -> bool:
        """Ready the response parser for one cycle. False = nothing to do yet."""
        if self._methods and self._waiting:
            method = self._methods.popleft()
        elif self._no_method_is_coming():
            # Capture joined mid-connection, or the request direction died.
            # POST is the shape of everything on this path that carries a body.
            method = "POST"
        else:
            # The two physical directions are read by independent processes.
            # A complete response can therefore reach this worker before the
            # request that caused it, even though TCP ordering within each
            # direction is intact. Keep h11's received bytes buffered until
            # the complete request and its real method arrive. Waiting only
            # for the method is insufficient: h11 emits Request after the
            # headers, before its body reaches EndOfMessage and enters
            # ``_waiting``; consuming the response in that interval would
            # still make _pair() drop it.
            return False

        try:
            self._resp_conn.send(h11.Request(method=method, target="/", headers=_PRIMER_HEADERS))
            self._resp_conn.send(h11.EndOfMessage())
        except h11.ProtocolError as exc:
            log.debug("Cannot prime the response parser: %s", exc)
            self._resp_dead = True
            return False
        self._primed = True
        self._unprimed_bytes = 0
        return True

    def _no_method_is_coming(self) -> bool:
        """True when no further request method can arrive on this connection.

        Waiting for a method still in flight is correct; waiting for one that
        will never arrive is not. h11 retains every byte it has not been able
        to frame, and on the pooled keep-alive connections this path uses
        nothing reclaims that until the connection closes — which may be never.

        ``_req_dead`` is the definite signal. The byte threshold catches the
        case this layer is never told about: reassembly abandoning the request
        direction upstream, after which client bytes simply stop arriving.
        """
        if self._req_dead:
            return True
        return self._unprimed_bytes > _MAX_UNPRIMED_RESPONSE_BYTES

    def _next_response_cycle(self) -> bool:
        self._primed = False
        try:
            self._resp_conn.start_next_cycle()
            return True
        except h11.ProtocolError:
            self._resp_dead = True
            return False

    def _die_responses(self, exc: Exception, ts: float | None, done: list[Exchange]) -> None:
        self._resp_dead = True
        if self._resp_open is not None:
            exchange = self._pair(self._close_open_response(), ts)
            if exchange is not None:
                done.append(exchange)
            log.debug("Truncated response (%s)", exc)
        else:
            self.unparseable = True
            log.warning("finding: non-http-stream (response direction) — %s", exc)

    def _close_open_response(self) -> HttpMessage:
        response = self._resp_open
        assert response is not None
        response.body = bytes(self._resp_body)
        response.truncated = True
        self._resp_open = None
        return response

    def _pair(self, response: HttpMessage, ts: float | None) -> Exchange | None:
        if not self._waiting:
            log.debug("Response with no captured request on this connection; dropped")
            return None
        request, started = self._waiting.popleft()
        return Exchange(
            request=request, response=response, received_at=started, completed_at=ts
        )


def parse_sse_events(body: bytes) -> list[dict[str, Any]]:
    """JSON `data:` events from an SSE body, `[DONE]` sentinel excluded."""
    events: list[dict[str, Any]] = []
    for block in body.replace(b"\r\n", b"\n").split(b"\n\n"):
        data_lines = [
            line[len(b"data:"):].strip()
            for line in block.split(b"\n")
            if line.startswith(b"data:")
        ]
        if not data_lines:
            continue
        payload = b"\n".join(data_lines)
        if payload == b"[DONE]":
            continue
        try:
            event = json.loads(payload)
        except ValueError:
            log.debug("Skipping undecodable SSE event: %r", payload[:120])
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def merge_stream_events(events: list[dict[str, Any]], object_kind: str) -> dict[str, Any]:
    """Fold OpenAI streaming chunks into one response-shaped dict.

    Text deltas are concatenated; scalar fields (id, model, finish_reason,
    usage) take the last non-null value; token IDs / logprobs are concatenated
    across chunks when the backend streams them (vLLM does when asked).

    `object_kind` follows the endpoint ("chat" | "text"), because the two
    OpenAI APIs stream different shapes: chat carries its text in
    `choices[0].delta.content`, legacy completions in `choices[0].text`, and
    their logprobs objects differ likewise. The merged dict is rebuilt in the
    kind it was given, so a caller reads it exactly as it reads a non-streamed
    body of the same kind. Required rather than defaulted: assuming chat is
    how the text shape came to be silently dropped in the first place.
    """
    merged: dict[str, Any] = {}
    text_parts: list[str] = []
    token_ids: list[int] = []
    logprobs: list[Any] = []
    finish_reason: str | None = None

    for event in events:
        for key in ("id", "model", "usage"):
            if event.get(key) is not None:
                merged[key] = event[key]
        for key in ("prompt_token_ids", "input_token_ids"):
            if isinstance(event.get(key), list):
                merged[key] = event[key]

        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or choice.get("message") or {}
        if object_kind == "text":
            if isinstance(choice.get("text"), str):
                text_parts.append(choice["text"])
        elif isinstance(delta, dict) and isinstance(delta.get("content"), str):
            text_parts.append(delta["content"])
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        for key in ("token_ids", "output_token_ids"):
            if isinstance(choice.get(key), list):
                token_ids.extend(choice[key])
            elif isinstance(delta, dict) and isinstance(delta.get(key), list):
                token_ids.extend(delta[key])
        # Chat streams per-token entries under `content`; legacy completions
        # stream a whole logprobs object per chunk (parallel `tokens` /
        # `token_logprobs` arrays, no `content`), which is collected object by
        # object rather than dropped.
        chunk_logprobs = choice.get("logprobs")
        if isinstance(chunk_logprobs, dict) and isinstance(chunk_logprobs.get("content"), list):
            logprobs.extend(chunk_logprobs["content"])
        elif chunk_logprobs:
            logprobs.append(chunk_logprobs)

    # An empty join is no text at all, not an empty generation: a stream that
    # carried none is the capture-loss signature, and it must read the same
    # here as it does on the non-streamed path.
    text = "".join(text_parts) or None
    merged_choice: dict[str, Any] = {"finish_reason": finish_reason}
    if object_kind == "text":
        merged_choice["text"] = text
    else:
        merged_choice["message"] = {"role": "assistant", "content": text}
    if token_ids:
        merged_choice["token_ids"] = token_ids
    if logprobs:
        merged_choice["logprobs"] = logprobs if object_kind == "text" else {"content": logprobs}
    merged["choices"] = [merged_choice]
    return merged
