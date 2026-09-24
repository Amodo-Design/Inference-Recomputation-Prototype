"""Build and send tap messages (docs/tap-message-contract.md).

Same contract inf-proxy emits, with the differences a passive observer
forces, stated honestly rather than faked:

- tap.source            = "frame-processor" (this is also what keeps the two
                          observers' events distinguishable while both run in
                          parallel; the sidecar sends "inf-proxy");
- tap.received_at/completed_at come from packet timestamps, not server clocks;
- tap.upstream_base_url = null — a frame tap has no upstream, it observed
                          <dst-ip>:<port> (recorded in tap.observed_on);
- tap.pod_name/node_name = null — the wire shows addresses, not pod identity;
- sampling.reporting_flags_added are all false/null — this tap is passive and
  injected nothing (during parallel running the verify-tap sidecar upstream
  is what injects reporting flags, and the token IDs it forced the model to
  return are visible on the wire and captured here).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, cast

import requests

from app.config import (
    FRAME_PROCESSOR_EMIT_ENABLED,
    FRAME_PROCESSOR_EMIT_DRAIN_SECONDS,
    FRAME_PROCESSOR_EMIT_QUEUE_SIZE,
    FRAME_PROCESSOR_EMIT_WORKERS,
    FRAME_PROCESSOR_HOSTNAME_MAP,
    FRAME_PROCESSOR_MESSAGE_WRITER_PATH,
    FRAME_PROCESSOR_MESSAGE_WRITER_TIMEOUT,
    FRAME_PROCESSOR_MESSAGE_WRITER_URL,
    FRAME_PROCESSOR_VERSION,
)
from app.extract import (
    detect_constrained_decoding,
    extract_prompt_output_token_ids,
    extract_prompt_text,
    extract_request_prompt_token_ids,
    sampling_config,
)
from app.filter import LlmClassification
from app.http_stream import Exchange, merge_stream_events, parse_sse_events

log = logging.getLogger("frameprocessor.emitter")


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def resolve_hostname(model_name: str) -> str:
    """The ledger resolution key for a model (see config.FRAME_PROCESSOR_HOSTNAME_MAP)."""
    if model_name in FRAME_PROCESSOR_HOSTNAME_MAP:
        return FRAME_PROCESSOR_HOSTNAME_MAP[model_name]
    fallback = f"kserve-{model_name.rsplit('/', 1)[-1].lower()}"
    log.warning("Model %r not in FRAME_PROCESSOR_HOSTNAME_MAP; falling back to %r", model_name, fallback)
    return fallback


def _object_kind(endpoint: str) -> str:
    """Which response shape the endpoint returns, as inf-proxy names it.

    Chat Completions puts the generation in `choices[0].message.content` and
    its logprobs under `content`; legacy Completions puts it in
    `choices[0].text` with the older parallel-array logprobs. Reading one
    shape out of the other yields an event whose text and logprobs are null
    while its token IDs are intact — which is indistinguishable from a
    complete capture and passes the verifier's TOKEN_CAPTURE_INCOMPLETE
    pre-check, so it has to be the endpoint that decides, not a guess.
    """
    return "text" if endpoint == "/v1/completions" else "chat"


def _output_text(choice: dict[str, Any], object_kind: str) -> str | None:
    if object_kind == "text":
        text = choice.get("text")
    else:
        text = (choice.get("message") or {}).get("content")
    return text if isinstance(text, str) else None


def _output_logprobs(choice: dict[str, Any]) -> Any:
    """Logprobs exactly as inf-proxy records them.

    The chat `content` list when that is the shape, else the provider's own
    object untouched — legacy completions returns parallel `tokens` /
    `token_logprobs` arrays with no `content` key, and reaching for `content`
    there silently discards the logprobs the verifier needs.
    """
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict) and isinstance(logprobs.get("content"), list):
        return logprobs["content"]
    return logprobs or None


def build_tap_message(
    exchange: Exchange,
    classification: LlmClassification,
    *,
    received_at: float | None,
    completed_at: float | None,
    observed_on: str | None = None,
) -> dict[str, Any]:
    request_body = classification.request_body
    response = exchange.response
    object_kind = _object_kind(classification.endpoint)

    streamed = bool(response is not None and response.is_sse)
    if response is None:
        response_body: dict[str, Any] = {}
    elif streamed:
        response_body = merge_stream_events(parse_sse_events(response.body), object_kind)
    else:
        response_body = response.json() or {}

    prompt_ids, output_ids = extract_prompt_output_token_ids({"response": response_body})
    # A replay client submits the prompt as token IDs and the response never
    # echoes them back, so the request is the only place they exist.
    if prompt_ids is None:
        prompt_ids = extract_request_prompt_token_ids(request_body)
    choices = response_body.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    usage = response_body.get("usage") or {}

    headers = exchange.request.headers

    return {
        "tap": {
            "source": "frame-processor",
            "proxy_version": FRAME_PROCESSOR_VERSION,
            "hostname": resolve_hostname(classification.model_name),
            "upstream_base_url": None,
            "observed_on": observed_on,
            "endpoint": classification.endpoint,
            "streamed": streamed,
            "received_at": _iso(received_at),
            "completed_at": _iso(completed_at),
            "pod_name": None,
            "node_name": None,
        },
        "session": {
            "session_id": headers.get("x-openwebui-chat-id") or response_body.get("id"),
            "user_id": headers.get("x-openwebui-user-id"),
            "chat_id": headers.get("x-openwebui-chat-id"),
            "message_id": headers.get("x-openwebui-message-id"),
            "response_id": response_body.get("id"),
        },
        "model": {
            "name": response_body.get("model") or classification.model_name,
            "revision": None,
            "tokenizer_revision": None,
        },
        "sampling": {
            **sampling_config(request_body),
            # Passive: this tap injects nothing. Any reporting flags on the
            # wire were added upstream (verify-tap) and their effects are
            # captured in the token-id/logprob fields below.
            "reporting_flags_added": {
                "return_token_ids": False,
                "logprobs": False,
                "top_logprobs": None,
            },
        },
        "request": {
            "prompt_text": extract_prompt_text(request_body),
            "prompt_token_ids": prompt_ids,
        },
        "response": {
            "output_text": _output_text(choice, object_kind),
            "output_token_ids": output_ids,
            "output_logprobs": _output_logprobs(choice),
            "finish_reason": choice.get("finish_reason"),
            "tool_calls": bool(message.get("tool_calls")),
            "usage_completion_tokens": usage.get("completion_tokens"),
            "constrained_decoding": detect_constrained_decoding(request_body),
        },
    }


# Stands in for an HTTP status when emitting is disabled and the message went
# to stdout instead. Nothing was accepted by a writer, but nothing was lost.
PRINTED = 0


def emit(
    message: dict[str, Any],
    *,
    session: requests.Session | None = None,
    enabled: bool = FRAME_PROCESSOR_EMIT_ENABLED,
) -> int | None:
    """POST to the Message writer, or print when emitting is disabled.

    Returns the writer's status code when it took the message, ``PRINTED``
    when emitting is off, and None when the record was lost — so a caller can
    report the outcome alongside whatever else it knows. Test with ``is not
    None``: ``PRINTED`` is deliberately falsy and success is not a boolean.

    Best-effort, like inf-proxy: the tap is off-path, so a failure can only
    ever lose a record, never affect a client.  ``EmitterPool`` supplies a
    persistent per-worker Session; the optional argument keeps this primitive
    directly usable by tests and development callers.
    """
    if not enabled:
        print(json.dumps(message, indent=2))
        return PRINTED
    url = f"{FRAME_PROCESSOR_MESSAGE_WRITER_URL}{FRAME_PROCESSOR_MESSAGE_WRITER_PATH}"
    owned_session = session is None
    client = session or requests.Session()
    try:
        resp = client.post(url, json=message, timeout=FRAME_PROCESSOR_MESSAGE_WRITER_TIMEOUT)
        if resp.status_code >= 400:
            log.error("Message writer returned %s: %s", resp.status_code, resp.text[:500])
            return None
        # Acceptance is announced by EmitterPool, which holds the running
        # count that goes on the line. Both callers of emit() are in this file.
        return resp.status_code
    except requests.RequestException as exc:
        log.error("Failed to send tap message to %s: %s", url, exc)
        return None
    finally:
        if owned_session:
            client.close()


@dataclass(frozen=True)
class EmitterStats:
    submitted: int
    delivered: int
    failed: int
    dropped: int
    high_water: int
    queued: int


_STOP = object()


class EmitterPool:
    """Nonblocking handoff from capture to persistent HTTP worker threads.

    Message-writer latency must never stop AF_PACKET reads: at inference
    concurrency 32 the old synchronous loop spent several seconds posting one
    completion at a time.  A bounded queue protects memory; overflow is made
    explicit in counters/logs rather than silently reintroducing backpressure.
    """

    def __init__(
        self,
        *,
        enabled: bool = FRAME_PROCESSOR_EMIT_ENABLED,
        workers: int = FRAME_PROCESSOR_EMIT_WORKERS,
        queue_size: int = FRAME_PROCESSOR_EMIT_QUEUE_SIZE,
        drain_seconds: float = FRAME_PROCESSOR_EMIT_DRAIN_SECONDS,
        session_factory: Callable[[], requests.Session] = requests.Session,
    ) -> None:
        self.enabled = enabled
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max(1, queue_size))
        self._session_factory = session_factory
        self._drain_seconds = drain_seconds
        self._threads: list[threading.Thread] = []
        self._stats_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._submitted = 0
        self._delivered = 0
        self._failed = 0
        self._dropped = 0
        self._high_water = 0
        self._closed = False
        # Depth at which the queue is reported as falling behind. Half is early
        # enough to act on and late enough not to fire on an ordinary burst;
        # the alternative signal, a drop, only arrives once records are lost.
        self._backpressure_at = max(1, self._queue.maxsize // 2)
        self._backpressure_logged = False

        if enabled:
            for index in range(max(1, workers)):
                thread = threading.Thread(
                    target=self._worker,
                    name=f"frame-processor-emitter-{index + 1}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)

    def _increment(self, field: str) -> int:
        with self._stats_lock:
            value = getattr(self, field) + 1
            setattr(self, field, value)
            return value

    def submit(self, message: dict[str, Any]) -> bool:
        """Hand off one message without ever waiting for queue capacity."""
        with self._close_lock:
            if self._closed:
                raise RuntimeError("EmitterPool is closed")

            if not self.enabled:
                self._increment("_submitted")
                if emit(message, enabled=False) is not None:
                    self._increment("_delivered")
                    return True
                self._increment("_failed")
                return False

            try:
                self._queue.put_nowait(message)
            except queue.Full:
                dropped = self._increment("_dropped")
                # Do not turn an overloaded queue into a logging bottleneck.
                if dropped == 1 or dropped % 100 == 0:
                    log.error(
                        "Emitter queue full; dropped %d tap message(s) total", dropped
                    )
                return False
            self._increment("_submitted")
            depth = self._queue.qsize()
            with self._stats_lock:
                self._high_water = max(self._high_water, depth)
                backlogged = depth >= self._backpressure_at and not self._backpressure_logged
                if backlogged:
                    self._backpressure_logged = True
            if backlogged:
                # Outside the lock: this must not serialise submitters. Once
                # per pool, because the condition persists and repeating it
                # would be the logging flood the depth is warning about.
                log.warning(
                    "Emitter queue %d/%d deep; the Message writer is not keeping "
                    "up. Raise FRAME_PROCESSOR_EMIT_WORKERS or FRAME_PROCESSOR_EMIT_QUEUE_SIZE "
                    "before tap messages start being dropped.",
                    depth,
                    self._queue.maxsize,
                )
            return True

    def _worker(self) -> None:
        """Drain the queue through one persistent session until _STOP.

        The session is opened *inside* the try, and a failure to open one does
        not end the thread. A worker that exits before its first task_done()
        leaves close() waiting on completions nobody will ever record, so the
        thread that cannot emit still drains and counts failures instead.
        """
        session: requests.Session | None = None
        try:
            try:
                session = self._session_factory()
            except Exception:  # noqa: BLE001 — draining still has to happen
                log.exception(
                    "Emitter worker could not open an HTTP session; it will drain "
                    "the queue and count every message as failed"
                )
            while True:
                item = self._queue.get()
                try:
                    if item is _STOP:
                        return
                    if session is None:
                        self._increment("_failed")
                        continue
                    message = cast(dict[str, Any], item)
                    try:
                        status = emit(message, session=session, enabled=True)
                    except Exception:  # noqa: BLE001 — keep the worker alive
                        log.exception("Unexpected error while emitting a tap message")
                        status = None
                    if status is None:
                        self._increment("_failed")
                        continue
                    # One line per tapped inference, as it lands, carrying the
                    # running total so a tail shows both the event and how many
                    # preceded it. This ran at DEBUG while the emitter shared a
                    # process with packet decoding, where per-message INFO
                    # contended with it on the logging lock; capture now runs in
                    # its own processes and the parent only drains a queue.
                    log.info(
                        "Tapped inference sent (session=%s status=%s) [%d delivered]",
                        message.get("session", {}).get("session_id"),
                        status,
                        self._increment("_delivered"),
                    )
                finally:
                    self._queue.task_done()
        finally:
            if session is not None:
                session.close()

    @property
    def stats(self) -> EmitterStats:
        with self._stats_lock:
            return EmitterStats(
                submitted=self._submitted,
                delivered=self._delivered,
                failed=self._failed,
                dropped=self._dropped,
                high_water=self._high_water,
                queued=self._queue.qsize(),
            )

    def close(self) -> EmitterStats:
        """Drain accepted work under a deadline, stop workers, return counters.

        The drain is bounded rather than unconditional. Queue.join() has no
        timeout, so an unreachable Message writer — or a worker that died
        without recording its completions — would hold shutdown open past any
        pod grace period, and SIGKILL then discards the backlog regardless. A
        deadline turns that into a logged loss instead of a hang.
        """
        with self._close_lock:
            if self._closed:
                return self.stats
            self._closed = True

        if self.enabled:
            self._drain(self._drain_seconds)
            for _ in self._threads:
                self._queue.put(_STOP)
            for thread in self._threads:
                thread.join(timeout=self._drain_seconds)
        return self.stats

    def _drain(self, timeout: float) -> None:
        """Wait up to `timeout` for accepted messages to finish emitting."""
        if timeout <= 0:
            return
        # Queue.join() cannot take a deadline, so it is waited on through a
        # daemon thread that the process can outlive.
        joiner = threading.Thread(
            target=self._queue.join, name="frame-processor-emitter-drain", daemon=True
        )
        joiner.start()
        joiner.join(timeout=timeout)
        if joiner.is_alive():
            log.error(
                "Emitter did not drain within %.0fs; abandoning %d queued tap "
                "message(s)",
                timeout,
                self._queue.qsize(),
            )
