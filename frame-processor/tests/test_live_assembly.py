"""Live assembly: an inference is tapped when it ends, not when its
connection does.

The clients on this path hold pooled keep-alive connections open across many
requests, so anything that waits for the close — or for the idle flush that
stands in for one — delays every verification behind it. These tests pin the
properties that batch-per-connection parsing could not provide.
"""

from __future__ import annotations

import json

from app import config
from app.capture import TcpSegment
from app.http_stream import ExchangeAssembler, parse_exchanges
from app.main import process_segments
from app.reassembly import ConnectionTable
from tests.helpers import (
    CLIENT,
    SERVER,
    conversation,
    http_request,
    http_response_json,
    http_response_sse,
    keepalive,
)

BODY = json.dumps({"model": "openai/gpt-oss-120b", "messages": []}).encode()
REQUEST = http_request("/v1/chat/completions", BODY)


def _response(n: int) -> bytes:
    return http_response_json(
        json.dumps({"id": f"cmpl-{n}", "model": "openai/gpt-oss-120b", "choices": []}).encode()
    )


def test_exchange_emits_while_the_connection_is_still_open():
    """The point of the whole change: no FIN, no flush, still emitted."""
    segments = keepalive([(REQUEST, _response(1)), (REQUEST, _response(2))])
    consumed: list[TcpSegment] = []

    def source():
        for seg in segments:
            consumed.append(seg)
            yield seg

    stream = process_segments(source())
    first = next(stream)

    assert first["tap"]["endpoint"] == "/v1/chat/completions"
    assert first["session"]["response_id"] == "cmpl-1"
    # Emitted mid-capture: segments were still unread when it came out, so it
    # cannot have come from the end-of-source drain.
    assert len(consumed) < len(segments)


def test_every_exchange_on_one_connection_is_tapped():
    segments = keepalive([(REQUEST, _response(n)) for n in range(1, 4)])
    messages = list(process_segments(iter(segments)))

    assert [m["session"]["response_id"] for m in messages] == ["cmpl-1", "cmpl-2", "cmpl-3"]


def test_complete_response_direction_may_arrive_before_request_direction():
    """Independent tap readers preserve per-direction order, not cross-order."""
    response = http_response_json(
        json.dumps(
            {
                "id": "cmpl-server-first",
                "model": "openai/gpt-oss-120b",
                "choices": [{"message": {"content": "hi"}}],
            }
        ).encode()
    )
    segments = conversation(REQUEST, response)
    server_first = [seg for seg in segments if seg.src == SERVER[0]] + [
        seg for seg in segments if seg.src == CLIENT[0]
    ]

    messages = list(process_segments(iter(server_first)))

    assert len(messages) == 1
    assert messages[0]["session"]["response_id"] == "cmpl-server-first"
    assert messages[0]["response"]["output_text"] == "hi"


def test_buffered_response_waits_for_request_body_and_keeps_server_timestamp():
    """Request headers name the method before the request is pairable."""
    assembler = ExchangeAssembler()
    response = _response(1)
    header_end = REQUEST.index(b"\r\n\r\n") + 4

    assert assembler.feed_server(response, ts=5.0) == []
    # h11 has emitted the Request event here, but the body/EOM is incomplete.
    assert assembler.feed_client(REQUEST[: header_end + 2], ts=1.0) == []
    done = assembler.feed_client(REQUEST[header_end + 2 :], ts=2.0)

    assert len(done) == 1
    assert done[0].response is not None
    assert done[0].response.json()["id"] == "cmpl-1"
    assert done[0].received_at == 1.0
    assert done[0].completed_at == 5.0


def test_each_exchange_carries_its_own_timestamps():
    """Connection-level timing stamped every inference on a pooled connection
    identically; tap.completed_at becomes the ledger's `ts`."""
    segments = keepalive([(REQUEST, _response(1)), (REQUEST, _response(2))], gap=5.0)
    first, second = list(process_segments(iter(segments)))

    assert first["tap"]["received_at"] != second["tap"]["received_at"]
    assert first["tap"]["received_at"] < first["tap"]["completed_at"]
    # The second inference starts only after the first has finished.
    assert first["tap"]["completed_at"] < second["tap"]["received_at"]


def test_streamed_response_emits_on_its_terminator():
    """An SSE completion ends at the chunked terminator, with the connection
    still open — that is the moment the tokens are all present."""
    chunks = [
        json.dumps({"id": "s1", "choices": [{"delta": {"content": "Hel"}}]}).encode(),
        json.dumps(
            {"id": "s1", "choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]}
        ).encode(),
    ]
    # A second exchange follows on the same connection, so the stream's own
    # terminator — not the end of the capture — is what releases the first.
    segments = keepalive([(REQUEST, http_response_sse(chunks)), (REQUEST, _response(2))])
    consumed: list[TcpSegment] = []

    def source():
        for seg in segments:
            consumed.append(seg)
            yield seg

    message = next(process_segments(source()))
    assert message["tap"]["streamed"] is True
    assert message["response"]["output_text"] == "Hello"
    assert len(consumed) < len(segments)


def test_idle_flush_salvages_a_half_captured_response():
    """A response cut off mid-stream still yields the tokens it did carry."""
    truncated = http_response_sse([json.dumps({"choices": [{"delta": {"content": "part"}}]}).encode()])
    truncated = truncated[: len(truncated) - 12]  # capture stopped mid-stream
    segments = keepalive([(REQUEST, truncated)])

    # Traffic on an unrelated connection, long after: this is what drives the
    # idle flush in a live capture.
    later = conversation(b"", b"", ts0=segments[-1].ts + 10_000)
    messages = list(process_segments(iter(segments + later)))

    assert len(messages) == 1
    assert messages[0]["response"]["output_text"] == "part"


def test_a_capture_gap_still_files_what_it_captured():
    """A gap poisons the rest of a pooled TCP direction, and is filed anyway.

    Complete exchanges from before the gap are unaffected. Requests queued
    behind it can never be paired with their responses, and are emitted with
    an empty response rather than dropped: an inference the tap saw and never
    filed is indistinguishable from one that never happened, whereas a filed
    one is visible in the ledger as something nothing could verify.

    The asserted shape is the point. A stranded event carries the request's
    identity and nothing from the response — no response id, no output — and
    that absence is all a consumer has to key on, because nothing in the
    tap-message contract says "capture was incomplete".
    """
    responses = [_response(n) for n in range(1, 4)]
    segments = keepalive(
        [(REQUEST, response) for response in responses],
        mtu=64,
    )
    second_response_seq = 5001 + len(responses[0])
    lossy = [
        seg
        for seg in segments
        if not (
            (seg.src, seg.sport) == SERVER
            and seg.seq == second_response_seq
        )
    ]
    assert len(lossy) == len(segments) - 1

    messages = list(process_segments(iter(lossy)))

    # The exchange before the gap paired normally; the two behind it are filed
    # from the request alone.
    assert len(messages) == 3
    complete, *stranded = messages
    assert complete["session"]["response_id"] == "cmpl-1"

    for message in stranded:
        # The request survived — this is a real inference, correctly attributed.
        assert message["model"]["name"] == "openai/gpt-oss-120b"
        # Nothing of the response did, which is the only tell it is incomplete.
        assert message["session"]["response_id"] is None
        assert message["response"]["output_text"] is None
        assert message["response"]["output_token_ids"] is None


def test_idle_cleanup_is_timed_instead_of_scanning_after_every_segment(monkeypatch):
    """A burst shorter than the sweep cadence performs no full-table scan."""
    segments = conversation(REQUEST, _response(1), mtu=1)
    calls = 0
    original = ConnectionTable.idle_flush

    def counted(self, now, idle_timeout):
        nonlocal calls
        calls += 1
        return original(self, now, idle_timeout)

    monkeypatch.setattr(config, "FRAME_PROCESSOR_IDLE_SWEEP_INTERVAL", 1)
    monkeypatch.setattr(ConnectionTable, "idle_flush", counted)

    assert len(list(process_segments(iter(segments)))) == 1
    assert calls == 0


def test_nothing_is_released_until_the_response_completes():
    """Emission is at end of response, and no earlier: verification replays
    the recorded output tokens, so a record without them is unverifiable
    (inf-ver-runner returns `unverifiable` when either token list is empty).
    """
    sse = http_response_sse(
        [
            json.dumps({"id": "c1", "prompt_token_ids": [1, 2, 3]}).encode(),
            json.dumps(
                {"id": "c1", "choices": [{"delta": {"content": "hi"}, "token_ids": [10],
                                          "finish_reason": "stop"}]}
            ).encode(),
        ]
    )
    terminator = b"0\r\n\r\n"  # the chunked end marker, i.e. end of generation
    assert sse.endswith(terminator)

    assembler = ExchangeAssembler()
    assert assembler.feed_client(REQUEST, ts=1.0) == []  # request alone: nothing
    assert assembler.feed_server(sse[: -len(terminator)], ts=2.0) == []  # mid-stream: nothing

    done = assembler.feed_server(terminator, ts=3.0)
    assert len(done) == 1
    assert done[0].response is not None
    assert done[0].completed_at == 3.0


def test_incremental_parsing_matches_batch_at_every_fragmentation():
    """Segment boundaries fall wherever the network puts them — mid-header,
    mid-chunk-size, mid-body. Feeding a byte at a time must reconstruct
    exactly what parsing the whole stream at once does.
    """
    sse = http_response_sse(
        [
            json.dumps({"id": "s1", "choices": [{"delta": {"content": "one"}}]}).encode(),
            json.dumps(
                {"id": "s1", "choices": [{"delta": {"content": "two"}, "token_ids": [7, 8]}]}
            ).encode(),
        ]
    )
    client = REQUEST + REQUEST
    server = _response(1) + sse

    expected = parse_exchanges(client, server)
    assert len(expected) == 2

    for chunk in (1, 3, 64, 5000):  # 1 = the harshest possible fragmentation
        assembler = ExchangeAssembler()
        got = []
        for i in range(0, len(client), chunk):
            got += assembler.feed_client(client[i : i + chunk])
        for i in range(0, len(server), chunk):
            got += assembler.feed_server(server[i : i + chunk])
        got += assembler.finish()

        assert len(got) == len(expected), chunk
        for actual, want in zip(got, expected):
            assert actual.request.path == want.request.path, chunk
            assert actual.request.body == want.request.body, chunk
            assert actual.response.status == want.response.status, chunk
            assert actual.response.body == want.response.body, chunk


def test_assembler_drops_bytes_once_a_direction_is_unparseable():
    """A stream joined mid-message can never resynchronise; holding its bytes
    would mean buffering a busy connection forever."""
    assembler = ExchangeAssembler()
    assert assembler.feed_client(b"\x00\x01 not a request at all\r\n\r\n") == []
    assert assembler.unparseable

    # Later, well-formed traffic on the same dead direction stays dropped
    # rather than accumulating.
    assert assembler.feed_client(REQUEST) == []
    assert assembler.finish() == []
