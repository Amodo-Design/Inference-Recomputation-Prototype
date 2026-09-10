# inf-proxy — inference tap

A **network-tap replacement**. It sits in front of an OpenAI-compatible model,
forwards inference traffic (client → model → client) while enforcing the declared
sampling configuration, and copies each completed inference to the **Message writer**
for the verification ledger.

```
client ──▶ nginx :8000 ──▶ FastAPI :9000 ──▶ upstream model ──▶ client   (forwarded as-is)
                                          └─▶ tap message ──▶ Message writer  (fire-and-forget)
```

- Intercepts `/v1/chat/completions` and `/v1/completions`; all other routes
  (`/v1/models`, metadata) go straight to the model via nginx.
- Injects **reporting** flags so the model returns data for the ledger:
  `return_token_ids`, and (optionally) `logprobs` + `top_logprobs`.
- **Pins sampling parameters.** Each of `INF_PROXY_SEED`, `INF_PROXY_TEMPERATURE`,
  `INF_PROXY_TOP_K` and `INF_PROXY_TOP_P` that is set overrides the client's value on
  every request, so actual generation matches what was declared to the ledger. Any
  left unset passes through from the client unchanged.
- The response is never modified, delayed, or blocked. The tap send is best-effort and
  runs in the background: if the Message writer is down, the client is unaffected.

## What it emits

For each inference it POSTs a **tap message** to
`${INF_PROXY_MESSAGE_WRITER_URL}${INF_PROXY_MESSAGE_WRITER_PATH}`. The schema and the
downstream mapping to an `inference_event` are defined in
[`../message-writer/app/schemas.py`](../message-writer/app/schemas.py).

The message's `tap.hostname` is the ledger **resolution key** — derived from the upstream
URL (or `INF_PROXY_MODEL_HOSTNAME`). It **must** match the `hostname` a model declares at
startup, or the Message writer's `/model-deployments/resolve` call will 404.

## Configuration

See [`.env.example`](.env.example). Key vars: `INF_PROXY_MODEL_BASE_URL` (upstream),
`INF_PROXY_MODEL_HOSTNAME` (resolution key override), `INF_PROXY_MESSAGE_WRITER_URL`,
`INF_PROXY_LOGPROBS` / `INF_PROXY_TOP_LOGPROBS`.

## Run

```bash
cd inf-proxy
docker build -t inf-proxy .
docker run --rm -p 8000:8000 \
  -e INF_PROXY_MODEL_BASE_URL=http://your-model:8000/v1 \
  -e INF_PROXY_MESSAGE_WRITER_URL=http://your-message-writer:8100 \
  inf-proxy
```

## Layout

```
app/
  main.py            # FastAPI app: forward + background tap
  streaming.py       # SSE relay; accumulates token ids/logprobs/text, fires the tap
  message_writer.py  # tap-message builder + best-effort sender (owns the aiohttp session)
  capture.py         # pure extractors (token ids, prompt text, client sampling config)
  payload.py         # injects reporting-only flags (return_token_ids, logprobs)
  config.py          # env-driven settings
nginx.conf.template  # nginx fronts the app; non-tap routes go straight to the model
entrypoint.sh        # renders nginx config, starts uvicorn + nginx
```
