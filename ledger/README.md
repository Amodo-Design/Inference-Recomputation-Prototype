# Verification Ledger

A FastAPI **Read + Create** access layer over the verification-ledger Postgres schema,
plus a Docker Compose stack that runs Postgres (auto-initialized from `sql/`) and the API.

## Layout

```
ledger/
  sql/001_verification_ledger.sql   # schema (auto-loaded by Postgres on first start)
  app/                              # FastAPI + SQLAlchemy 2.0 async (psycopg3) app
    config.py                       # env-driven settings
    database.py                     # async engine + session dependency
    models.py                       # ORM models mapping the schema
    schemas.py                      # Pydantic request/response models
    routers/                        # one router per table (POST + GET list + GET by id)
    main.py                         # app wiring + /health
  Dockerfile
  docker-compose.yml
  requirements.txt
  .env.example
```

## Run

```bash
cd ledger
docker compose up --build
```

- API + interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Postgres: `localhost:5432` (user/db default to `ledger`; password from `.env`)

Copy `.env.example` to `.env` and set `POSTGRES_PASSWORD` (required) / connection string.

## API

Each table exposes three endpoints:

| Resource              | Path                    |
|-----------------------|-------------------------|
| Hardware owners       | `/hardware-owners`      |
| Hardware              | `/hardware`             |
| Models                | `/models`               |
| Inference events      | `/inference-events`     |
| Verification events   | `/verification-events`  |

- `POST /<resource>` — insert a row (returns the created row, `201`).
- `GET /<resource>` — list rows (`limit`, `offset` query params).
- `GET /<resource>/{id}` — fetch one row (`404` if absent).

`UUID` primary/foreign keys are supplied by the client (the schema has no DB defaults).
`BYTEA` columns (hashes, raw logits, margins) are sent and returned as **base64 strings**.

### Verifier polling & verification views

Beyond the generic CRUD:

- `GET /inference-events/unverified?limit=` — inference events with **no**
  verification_event row yet, **newest first**. Each item is `{event, model}` — the
  joined model row supplies the sampling config + model name the verifier needs.
  This is what inf-ver-runner drains (scoped to its model via `model_id=`).
- `GET /verification-events/view?limit=&offset=&result=&model=&search=` — paged, flat
  verification_event + inference_event + model join for the results UI
  (`{items, total, limit, offset}`).
- `GET /verification-events/stats?model=&search=` — pass/fail/unverifiable counts +
  averages, accepting the same `model` / `search` filters as the view.
- `POST /verification-events/replay?result=&model=&search=` /
  `POST /verification-events/{id}/replay` — **re-verify**. Deletes only the
  verification event(s), so the underlying inference events look unverified again and a
  runner picks them back up. The original verdicts are not kept. The bulk form takes the
  view's filters and returns `{replayed: n}`.
- `DELETE /verification-events?result=&model=&search=` /
  `DELETE /verification-events/{id}` — **permanent delete**. Removes the verification
  event(s) *and* their inference events, so nothing is re-queued. An inference event is
  kept only if another verification event still references it. The bulk form takes the
  view's filters and returns `{deleted: n}`; with no filters it deletes everything.

### Per-model operator settings

Two mutable settings live on the `model` row but are deliberately excluded from the
`model_id` identity hash, so changing them never creates a new model:

- `PATCH /models/{model_id}/verification-threshold` with `{"verification_threshold": x}` —
  the pass mark the runner applies to that model. `null` pauses verification: the
  orchestrator spawns no runner and pending events wait, verifying retroactively once a
  threshold is set.
- `PATCH /models/{model_id}/delta-max` with `{"delta_max": x}` — the per-token margin cap
  (difr's delta max). `null` means the runner's built-in default.

`verification_event` carries the verifier's verdict (`result`, `result_detail`, `error_code`,
`verification_threshold` — the pass threshold in force for that run, set dynamically
rather than stored on the model config) and links to the verifier's declared model via `verifier_model_id`
(FK → `model`), plus a `verifier_detail` JSONB payload (token comparison, output
texts, metric summary, verifier latency). The verifier registers itself at startup
via `POST /model-deployments/declare` — the hardware it runs on and the model it
launched with — and every event it writes references those ids. Note the init script only
runs on an **empty** data directory — reset the volume after schema changes
(`docker compose down && rm -rf data`).

### Model deployments — declare & resolve

A tapped inference event only knows a model *name* + timestamp + destination *hostname*.
`inference_event.model_id` must point at a specific declared config, so:

- `POST /model-deployments/declare` — a model process declares at startup, sending only
  natural data (`hostname`, `model_name`, params). The **ledger owns all ids**: `model_id`
  is derived from the config (identical config → same row, so duplicates are deduped
  automatically), `hardware_id` is matched by `hostname`, `deployment_id` is generated. It
  upserts hardware + model config, closes the host's active deployment, and opens a new
  one. The same config on many hosts shares one `model_id`; each host runs one model at a
  time. This is the canonical spin-up write path.
- `GET /model-deployments/resolve?hostname=&ts=&model_name=` — resolves `hostname` +
  timestamp to the active deployment, returning `model_id` + `hardware_id`
  (+ `model_name_matches` cross-check). Use these to fill an inference event's FKs.
- `GET /model-deployments` / `GET /model-deployments/{id}` — list / fetch.

### Example

```bash
# Model declares it started on host gpu-node-1 (no ids sent — the ledger derives them).
# Returns the derived model_id + generated hardware_id/deployment_id.
curl -X POST localhost:8000/model-deployments/declare -H 'content-type: application/json' -d '{
  "hostname": "gpu-node-1",
  "model_name": "Qwen/Qwen3-8B",
  "temperature": 0.7,
  "seed": 42,
  "decoding_algorithm": "gumbel_max",
  "started_at": "2026-07-08T12:00:00Z"
}'

# Resolve a tapped inference (hostname + timestamp) -> model_id + hardware_id
curl "localhost:8000/model-deployments/resolve?hostname=gpu-node-1&ts=2026-07-08T12:30:00Z&model_name=Qwen/Qwen3-8B"

# Create an inference event using the resolved ids (hashes are base64-encoded bytes)
curl -X POST localhost:8000/inference-events -H 'content-type: application/json' -d '{
  "id": "33333333-3333-3333-3333-333333333333",
  "session_id": "chat-abc",
  "ts": "2026-07-08T12:30:00Z",
  "model_id": "<model_id from resolve>",
  "hardware_id": "<hardware_id from resolve>",
  "hash_input_raw_logits": "aGVsbG8=",
  "hash_output_raw_logits": "d29ybGQ="
}'
```

## Tests

Run with [uv](https://docs.astral.sh/uv/) (provisions the venv from `pyproject.toml`):

```bash
cd ledger
uv run pytest
```

Tests spin up an ephemeral Postgres via **testcontainers**, load every `sql/*.sql`
file in order, and exercise the API in-process — so **Docker must be
running** (the first run pulls `postgres:16`). They cover inference-event storage, the
BYTEA base64 round-trip (incl. opaque binary), and a full tap-message → ledger round-trip.

## Notes

- The SQL file is the source of truth for DDL; the ORM models are hand-aligned to it and
  do not create tables. The init script runs only on an **empty** data directory.
- Postgres data is bind-mounted to `ledger/data/` (gitignored). To reset the database,
  stop the stack and delete it: `docker compose down && rm -rf data`.
