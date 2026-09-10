"""Message-writer configuration, read from the environment."""

from __future__ import annotations

import os


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


# Ledger Access Layer this writer resolves against and writes to.
LEDGER_API_URL = os.getenv("LEDGER_API_URL", "http://ledger-api:8000").rstrip("/")
LEDGER_TIMEOUT = _as_int(os.getenv("MESSAGE_WRITER_LEDGER_TIMEOUT"), 30)

APP_PORT = _as_int(os.getenv("MESSAGE_WRITER_APP_PORT"), 8100)

# Store the raw token-id / logprob payloads in the inference_event BYTEA columns
# (in addition to their hashes). Off => only the hashes are written.
STORE_RAW_LOGITS = _as_bool(os.getenv("MESSAGE_WRITER_STORE_RAW_LOGITS"), True)
