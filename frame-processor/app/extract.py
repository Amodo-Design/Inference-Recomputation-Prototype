"""Shared payload extractors, reused from inf-proxy.

`inf-proxy/app/capture.py` is pure stdlib (prompt-text rendering, token-id
extraction, client sampling config). Both taps MUST render these identically
or comparing their events is meaningless, so frame-processor loads that module
rather than duplicating it — the same cross-service sharing pattern
inf-ver-runner uses for difr's gumbel_verify.

Loaded by file path (not package import) because both services name their
package `app`. The Dockerfile copies the module to /app/inf_proxy_capture.py;
local dev and tests find it through the repo layout.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_CANDIDATES = (
    Path(__file__).resolve().parent.parent / "inf_proxy_capture.py",  # container
    Path(__file__).resolve().parents[2] / "inf-proxy" / "app" / "capture.py",  # repo
)


def _load():
    for path in _CANDIDATES:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("inf_proxy_capture", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["inf_proxy_capture"] = module
            spec.loader.exec_module(module)
            return module
    raise ImportError(
        "inf-proxy capture module not found; looked in: "
        + ", ".join(str(p) for p in _CANDIDATES)
    )


_capture = _load()

extract_prompt_text = _capture.extract_prompt_text
extract_prompt_output_token_ids = _capture.extract_prompt_output_token_ids
sampling_config = _capture.sampling_config


def _is_int_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, int) for item in value)


def extract_request_prompt_token_ids(payload: dict[str, Any]) -> list[int] | None:
    """Prompt token IDs the CLIENT submitted, when it submitted IDs not text.

    Local to frame-processor rather than shared, because only a passive tap can need
    it. inf-proxy sets `return_token_ids` on every request it forwards, so the
    response it reads always carries the IDs; frame-processor injects nothing and
    reads only what was already on the wire, so when the response has none the
    request is the only place they exist. A shared copy would be dead code in
    inf-proxy — and touching that module rebuilds the in-path sidecar image.

    The completions API accepts `prompt` as a token-ID array (or a batch of
    them), which is how a replay client reproduces an inference exactly. Text
    prompts return None: detokenizing is not a tap's job. Batches take the
    first prompt, since everything downstream is one prompt per event.
    """
    ids = payload.get("prompt_token_ids")
    if _is_int_list(ids) and ids:
        return ids

    prompt = payload.get("prompt")
    if _is_int_list(prompt) and prompt:
        return prompt
    # A batch: [[id, ...], ...]. An empty first prompt is no prompt at all.
    if isinstance(prompt, list) and prompt and _is_int_list(prompt[0]) and prompt[0]:
        return prompt[0]
    return None


def detect_constrained_decoding(payload: dict[str, Any]) -> bool:
    """True when the request constrains generation beyond the declared
    sampling contract - a grammar/mask the verifier cannot replay.

    Forced tool choice: ``tool_choice`` naming a function (dict) or
    ``"required"`` (``"auto"``/``"none"`` leave sampling unconstrained).
    Structured output: ``response_format`` of any type except plain text.

    A verbatim copy of `inf-proxy/app/payload.detect_constrained_decoding`,
    not a load like the extractors above: that module imports inf-proxy's
    config, so it cannot be read by a service that has none. The flag goes
    into the hashed output payload (`message-writer/app/transform.py`), so a
    tap that reads it differently files a different ledger event for the same
    inference — the drift these two taps exist to detect would be coming from
    the taps themselves. Keep the two in lockstep; `tests/test_extract.py`
    pins the cases inf-proxy's `test_capture_integrity.py` pins.
    """
    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, dict) or tool_choice == "required":
        return True
    response_format = payload.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") not in (None, "text"):
        return True
    return False
