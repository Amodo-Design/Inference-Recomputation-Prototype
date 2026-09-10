"""Capture helpers.

Pure extractors for pulling token IDs, prompt text, and the client's sampling
config out of OpenAI-compatible request payloads and model responses. Used to
assemble the tap message. (Extractors retained from the original verifier
client; the verifier HTTP call has been removed — the proxy is now a tap.)
"""

from __future__ import annotations

from typing import Any


TOKEN_ID_KEYS = {
    "prompt_token_ids",
    "input_token_ids",
    "output_token_ids",
    "completion_token_ids",
    "generated_token_ids",
    "token_ids",
}


def _is_int_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, int) for item in value)


def _find_token_ids(data: Any, preferred_keys: tuple[str, ...]) -> list[int] | None:
    if isinstance(data, dict):
        for key in preferred_keys:
            value = data.get(key)
            if _is_int_list(value):
                return value
        for key, value in data.items():
            if key in TOKEN_ID_KEYS and _is_int_list(value):
                return value
        for value in data.values():
            found = _find_token_ids(value, preferred_keys)
            if found:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_token_ids(item, preferred_keys)
            if found:
                return found
    return None


def extract_prompt_output_token_ids(
    completed_event: dict[str, Any],
) -> tuple[list[int] | None, list[int] | None]:
    """Extract authoritative token IDs only if vLLM returned them.

    Prompt IDs are read from the top level; output IDs are scoped to the first
    choice so a top-level `prompt_token_ids` can't be mistaken for the output.
    """
    response = completed_event.get("response", completed_event)

    prompt_ids = _find_token_ids(response, ("prompt_token_ids", "input_token_ids"))

    output_ids: list[int] | None = None
    choices = response.get("choices") if isinstance(response, dict) else None
    if isinstance(choices, list) and choices:
        output_ids = _find_token_ids(
            choices[0],
            ("output_token_ids", "completion_token_ids", "generated_token_ids", "token_ids"),
        )
    if output_ids is None:
        output_ids = _find_token_ids(
            response, ("output_token_ids", "completion_token_ids", "generated_token_ids")
        )
    return prompt_ids, output_ids


def _content_text(content: Any) -> str:
    """Flatten an OpenAI-style message/input content value to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") not in (None, "text", "input_text", "output_text"):
                    parts.append(f"[{item.get('type')}]")
        return "\n".join(part for part in parts if part)
    return ""


def extract_prompt_text(payload: dict[str, Any]) -> str | None:
    """Render the submitted prompt as plain text for audit display only."""
    sections: list[str] = []

    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        sections.append(f"system: {instructions.strip()}")

    # Chat Completions payloads carry `messages`; legacy completions carry `prompt`.
    turns = payload.get("messages")
    if not isinstance(turns, list):
        turns = payload.get("input")
    if not isinstance(turns, list) and isinstance(payload.get("prompt"), str):
        turns = payload.get("prompt")

    if isinstance(turns, str):
        sections.append(turns)
    elif isinstance(turns, list):
        for turn in turns:
            if isinstance(turn, str):
                sections.append(turn)
                continue
            if not isinstance(turn, dict):
                continue
            if turn.get("type") and turn.get("type") != "message":
                continue
            text = _content_text(turn.get("content"))
            if text:
                sections.append(f"{turn.get('role', 'user')}: {text}")

    prompt_text = "\n\n".join(sections).strip()
    return prompt_text or None


def sampling_config(payload: dict[str, Any]) -> dict[str, Any]:
    """The sampling params the CLIENT actually sent (None when unset). The tap
    does not inject defaults — this reflects the real request."""
    return {
        "seed": payload.get("seed"),
        "temperature": payload.get("temperature"),
        "top_k": payload.get("top_k"),
        "top_p": payload.get("top_p"),
        "max_output_tokens": payload.get("max_output_tokens") or payload.get("max_tokens"),
    }
