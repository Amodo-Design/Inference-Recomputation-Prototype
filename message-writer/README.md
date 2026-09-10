# message-writer

Ingests **tap messages** from `inf-proxy` (the network-tap replacement) and turns each into
an `inference_event` in the ledger.

```
inf-proxy ──tap message──▶ message-writer ──▶ Ledger Access Layer
                                 │  resolve (hostname, ts) -> model_id + hardware_id
                                 │  hash token-ids / logprobs
                                 └▶ POST /inference-events
```

## What it does

For each `POST /inferences` (tap message; schema in
[`app/schemas.py`](app/schemas.py)):

1. **Resolve** — `GET {LEDGER_API_URL}/model-deployments/resolve?hostname=&ts=&model_name=`
   using `tap.hostname` + `tap.completed_at` + `model.name`. Yields `model_id` +
   `hardware_id`. A `model_name_matches: false` is logged (integrity flag) but not fatal.
2. **Hash** — the input side (`prompt_token_ids`, else `prompt_text`) and output side
   (`output_token_ids` + `output_logprobs`) are serialised to canonical JSON; `hash_*` are
   SHA-256 over those bytes. With `MESSAGE_WRITER_STORE_RAW_LOGITS=true` the raw bytes are
   also stored in `input_raw_logits` / `output_raw_logits`. All BYTEA sent as base64.
3. **Write** — `POST {LEDGER_API_URL}/inference-events` with a fresh `id`,
   `session_id` (chat id / response id), `ts`, resolved ids, hashes, and text
   representations.

Failure handling: no active deployment → `422`; ledger unreachable → `502`. (The proxy
sends the tap best-effort, so these never reach the end user.)

## Endpoints

- `POST /inferences` — ingest a tap message → writes an `inference_event`, returns the new
  `inference_event_id` + resolved ids.
- `GET /health`

## Configuration

See [`.env.example`](.env.example): `LEDGER_API_URL`, `MESSAGE_WRITER_APP_PORT` (8100),
`MESSAGE_WRITER_STORE_RAW_LOGITS`.

## Run

Standalone:

```bash
cd message-writer
docker build -t message-writer .
docker run --rm -p 8100:8100 -e LEDGER_API_URL=http://your-ledger:8000 message-writer
```

## Tests

```bash
cd message-writer
uv run pytest
```

No Docker or ledger needed — the ledger boundary (`resolve` / `create_inference_event`) is
mocked and the transform is pure. Covers the tap→event encoding + hash integrity
(`test_transform.py`), the `POST /inferences` branches (`test_ingest.py`), and the ledger
HTTP client contract (`test_ledger_client.py`).

## Layout

```
app/
  main.py          # FastAPI: POST /inferences -> resolve -> write
  ledger_client.py # async httpx client (resolve + create_inference_event)
  transform.py     # tap message -> inference_event payload (hashing, mapping)
  schemas.py       # lenient tap-message models
  config.py        # env-driven settings
```

## Known limitations

Deliberate simplifications in the current prototype — not yet addressed:

- **All ledger errors map to `502`.** Any non-2xx from the ledger (including a `409`
  duplicate or `422` bad-payload, which are really data problems) surfaces as
  `502 Bad Gateway` via `raise_for_status()`. A finer mapping would pass client-error
  statuses through instead.
- **No idempotency.** `build_inference_event` mints a fresh `uuid4` per call, so
  re-delivering the same tap message writes a **duplicate** `inference_event`. Since the
  proxy taps best-effort (and may retry), a real deployment should derive the event `id`
  from a stable key (e.g. the tap `response_id` / session) so retries de-duplicate.
