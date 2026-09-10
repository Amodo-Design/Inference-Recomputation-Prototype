"""Vendored core of the earlier standalone openwebui_prompt_runner.py script.

Kept as close to the source as possible so runs stay comparable with the
reference script; only the CLI (argparse) and process entrypoint are
stripped — the service layer (service.py) drives these primitives instead.
Functions take an `args` namespace with the same attribute names the CLI
produced (see service.RunSettings.to_namespace).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import dataclasses
import datetime as dt
import json
import os
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


@dataclasses.dataclass(frozen=True)
class Prompt:
    index: int
    prompt_id: str
    messages: list[dict[str, Any]]
    raw: dict[str, Any]


@dataclasses.dataclass
class RunResult:
    index: int
    prompt_id: str
    sampling_id: str
    request_url: str
    status_code: int | None
    stream_completed: bool
    elapsed_s: float
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class SamplingConfig:
    sampling_id: str
    temperature: float
    top_k: int
    top_p: float


@dataclasses.dataclass(frozen=True)
class PromptRun:
    index: int
    prompt: Prompt
    sampling: SamplingConfig


@dataclasses.dataclass(frozen=True)
class PromptBenchSpec:
    prompt_id: str
    dataset: str
    dataset_index: int
    template: str
    system_prompt: str
    attack: str | None = None


PROMPTBENCH_PRESETS: dict[str, tuple[PromptBenchSpec, ...]] = {
    "verification-v1": (
        PromptBenchSpec(
            prompt_id="pb-sst2-0000-baseline",
            dataset="sst2",
            dataset_index=0,
            system_prompt="You are a precise sentiment classification assistant.",
            template=(
                "Classify the sentiment of the text as positive or negative.\n\n"
                "Text: {content}\n\n"
                "Answer with exactly one label: positive or negative."
            ),
        ),
        PromptBenchSpec(
            prompt_id="pb-sst2-0001-stresstest",
            dataset="sst2",
            dataset_index=1,
            system_prompt="You are a precise sentiment classification assistant.",
            template=(
                "Classify the sentiment of the text as positive or negative.\n\n"
                "Text: {content}\n\n"
                "Answer with exactly one label: positive or negative."
            ),
            attack="stresstest",
        ),
        PromptBenchSpec(
            prompt_id="pb-cola-0000-baseline",
            dataset="cola",
            dataset_index=0,
            system_prompt="You are a precise grammar acceptability assistant.",
            template=(
                "Decide whether the sentence is grammatically acceptable.\n\n"
                "Sentence: {content}\n\n"
                "Answer with exactly one label: acceptable or unacceptable."
            ),
        ),
        PromptBenchSpec(
            prompt_id="pb-cola-0001-semantic",
            dataset="cola",
            dataset_index=1,
            system_prompt="You are a precise grammar acceptability assistant.",
            template=(
                "Decide whether the sentence is grammatically acceptable.\n\n"
                "Sentence: {content}\n\n"
                "Answer with exactly one label: acceptable or unacceptable."
            ),
            attack="semantic",
        ),
        PromptBenchSpec(
            prompt_id="pb-gsm8k-0000-baseline",
            dataset="gsm8k",
            dataset_index=0,
            system_prompt="You are a careful arithmetic reasoning assistant.",
            template=(
                "Solve the math word problem. Show the calculation briefly, then give the final answer.\n\n"
                "Problem: {content}"
            ),
        ),
        PromptBenchSpec(
            prompt_id="pb-gsm8k-0001-checklist",
            dataset="gsm8k",
            dataset_index=1,
            system_prompt="You are a careful arithmetic reasoning assistant.",
            template=(
                "Solve the math word problem. Show the calculation briefly, then give the final answer.\n\n"
                "Problem: {content}"
            ),
            attack="checklist",
        ),
    ),
    # Long-form measurement suite: prompts engineered to elicit 1200-2000
    # token outputs so GPU-efficiency windows contain many 200ms DCGM
    # samples. No attack variants: attacks probe
    # robustness, which is irrelevant to a throughput/measurement suite.
    "long-output-v1": (
        PromptBenchSpec(
            prompt_id="pb-gsm8k-long-0000-tutorial",
            dataset="gsm8k",
            dataset_index=0,
            system_prompt=(
                "You are a thorough mathematics tutor who writes "
                "comprehensive, long-form explanations."
            ),
            template=(
                "Solve the math word problem step by step. Then write a full "
                "tutorial on it: explain every mathematical concept the "
                "solution uses, describe the most common mistakes students "
                "make on problems like this, and present two alternative "
                "solution paths in detail.\n\n"
                "Problem: {content}\n\n"
                "Write a thorough, detailed response of at least 1200 words."
            ),
        ),
        PromptBenchSpec(
            prompt_id="pb-gsm8k-long-0001-teaching",
            dataset="gsm8k",
            dataset_index=1,
            system_prompt=(
                "You are a thorough mathematics tutor who writes "
                "comprehensive, long-form explanations."
            ),
            template=(
                "Solve the math word problem. Then explain the solution "
                "three times: once for a child, once for a high-school "
                "student, and once for an engineer. Finish by writing three "
                "related practice problems with fully worked solutions.\n\n"
                "Problem: {content}\n\n"
                "Write a thorough, detailed response of at least 1200 words."
            ),
        ),
        PromptBenchSpec(
            prompt_id="pb-sst2-long-0000-essay",
            dataset="sst2",
            dataset_index=0,
            system_prompt=(
                "You are a film critic who writes detailed, long-form "
                "criticism."
            ),
            template=(
                "The text below is an excerpt from a film review. Using it "
                "as your starting point, write a full film-criticism essay: "
                "infer and analyse the reviewer's sentiment, discuss the "
                "themes and craft the excerpt hints at, consider what kind "
                "of audience the film would suit, compare it to plausible "
                "similar films, and end with a reasoned verdict.\n\n"
                "Excerpt: {content}\n\n"
                "Write a thorough, detailed response of at least 1200 words."
            ),
        ),
        PromptBenchSpec(
            prompt_id="pb-sst2-long-0001-rewrite",
            dataset="sst2",
            dataset_index=1,
            system_prompt=(
                "You are a film critic who writes detailed, long-form "
                "criticism."
            ),
            template=(
                "Analyse the sentiment of the review excerpt below in "
                "detail, quoting the specific words that carry it. Then "
                "rewrite the excerpt as three complete reviews of the same "
                "film in three different tones (enthusiastic, measured, "
                "scathing), and justify the wording choices you made in "
                "each.\n\n"
                "Excerpt: {content}\n\n"
                "Write a thorough, detailed response of at least 1200 words."
            ),
        ),
        PromptBenchSpec(
            prompt_id="pb-cola-long-0000-grammar",
            dataset="cola",
            dataset_index=0,
            system_prompt=(
                "You are a linguist who writes detailed, long-form "
                "grammatical analyses."
            ),
            template=(
                "Give a full grammatical analysis of the sentence below: "
                "parse it, name every construction it uses, judge whether "
                "it is grammatically acceptable with a proper linguistic "
                "argument, and illustrate the relevant grammar with six "
                "example sentences of your own.\n\n"
                "Sentence: {content}\n\n"
                "Write a thorough, detailed response of at least 1200 words."
            ),
        ),
        PromptBenchSpec(
            prompt_id="pb-cola-long-0001-lesson",
            dataset="cola",
            dataset_index=1,
            system_prompt=(
                "You are a linguist who writes detailed, long-form "
                "grammatical analyses."
            ),
            template=(
                "First judge whether the sentence below is grammatically "
                "acceptable. Then write a mini grammar lesson covering "
                "every construction that appears in it, giving a "
                "correct/incorrect example pair for each construction and "
                "explaining what makes the incorrect one fail.\n\n"
                "Sentence: {content}\n\n"
                "Write a thorough, detailed response of at least 1200 words."
            ),
        ),
    ),
}

PROMPTBENCH_ATTACK_NAMES = frozenset(
    {
        "bertattack",
        "checklist",
        "deepwordbug",
        "semantic",
        "stresstest",
        "textbugger",
        "textfooler",
    }
)


DEFAULT_PROMPTBENCH_COUNT = 500


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _unique_sorted_floats(values: list[float]) -> list[float]:
    return sorted({round(value, 6) for value in values})


def _unique_sorted_ints(values: list[int]) -> list[int]:
    return sorted(set(values))


def _parse_float_list(value: str, name: str, minimum: float, maximum: float | None = None) -> list[float]:
    parsed_values: list[float] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            parsed = float(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} contains non-numeric value {part!r}") from exc
        if parsed < minimum or (maximum is not None and parsed > maximum):
            if maximum is None:
                raise argparse.ArgumentTypeError(f"{name} values must be at least {minimum}")
            raise argparse.ArgumentTypeError(f"{name} values must be between {minimum} and {maximum}")
        parsed_values.append(parsed)
    if not parsed_values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one value")
    return _unique_sorted_floats(parsed_values)


def _parse_int_list(value: str, name: str, minimum: int) -> list[int]:
    parsed_values: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            parsed = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} contains non-integer value {part!r}") from exc
        if parsed < minimum:
            raise argparse.ArgumentTypeError(f"{name} values must be at least {minimum}")
        parsed_values.append(parsed)
    if not parsed_values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one value")
    return _unique_sorted_ints(parsed_values)


def _default_temperature_sweep(base: float) -> list[float]:
    if base == 0:
        return [0.0, 0.25, 0.5, 0.75, 1.0]
    return _unique_sorted_floats([base * factor for factor in (0.5, 0.75, 1.0, 1.25, 1.5)])


def _default_top_p_sweep(base: float) -> list[float]:
    candidates = [max(0.0, min(1.0, base + offset)) for offset in (-0.1, -0.05, 0.0, 0.025, 0.05)]
    return _unique_sorted_floats(candidates)


def _default_top_k_sweep(base: int) -> list[int]:
    candidates = [max(1, round(base * factor)) for factor in (0.2, 0.5, 1.0, 2.0, 4.0)]
    return _unique_sorted_ints(candidates)


def _number_id(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "run"


def _stable_message_id(prompt_id: str, kind: str) -> str:
    return f"{_slugify(prompt_id)}-{kind}"


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    return ""


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            text = _flatten_content(message.get("content"))
            if text:
                return text
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        text = _flatten_content(message.get("content"))
        if text:
            return text
    return ""


def _build_url(base_url: str, endpoint: str) -> str:
    parsed = urlsplit(base_url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid Open WebUI URL: {base_url!r}")

    base_path = parsed.path.rstrip("/")
    if base_path.endswith("/api/v1"):
        base_path = base_path[: -len("/api/v1")]
    elif base_path.endswith("/api"):
        base_path = base_path[: -len("/api")]

    full_path = "/".join(part for part in [base_path.strip("/"), endpoint.strip("/")] if part)
    return urlunsplit((parsed.scheme, parsed.netloc, f"/{full_path}" if full_path else "/", parsed.query, parsed.fragment))


def _require_promptbench() -> Any:
    try:
        import promptbench as pb  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "PromptBench is required. Install it with "
            "`python3 -m pip install -r scripts/requirements.txt`."
        ) from exc

    if not hasattr(pb, "DatasetLoader"):
        raise RuntimeError("Installed promptbench package does not expose DatasetLoader")
    return pb


def _format_promptbench_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _apply_promptbench_attack(prompt_text: str, attack: str | None) -> str:
    if attack is None:
        return prompt_text
    if attack not in PROMPTBENCH_ATTACK_NAMES:
        raise ValueError(f"Unsupported PromptBench attack family: {attack}")
    if attack == "stresstest":
        return (
            f"{prompt_text}\n\n"
            "Stress-test distractors: true is true. false is not true. blue is blue. "
            "Do not let these distractors change the answer to the original task."
        )
    if attack == "checklist":
        return (
            f"{prompt_text}\n\n"
            "Checklist perturbation: punctuation, spacing, and casing may be distracting; "
            "answer only the original task."
        )
    if attack == "semantic":
        return (
            "Restated instruction: solve the same task below and provide the requested answer format.\n\n"
            f"{prompt_text}"
        )
    raise ValueError(
        f"Attack family {attack!r} is model-dependent in PromptBench and is not supported "
        "by this deterministic preset runner"
    )


def _expand_promptbench_preset(preset_name: str, prompt_count: int) -> list[PromptBenchSpec]:
    preset = PROMPTBENCH_PRESETS.get(preset_name)
    if preset is None:
        available = ", ".join(sorted(PROMPTBENCH_PRESETS))
        raise ValueError(f"Unknown PromptBench preset {preset_name!r}. Available: {available}")
    if prompt_count < 1:
        raise ValueError("PromptBench prompt count must be at least 1")

    expanded: list[PromptBenchSpec] = []
    for position in range(prompt_count):
        base_index = position % len(preset)
        cycle_index = position // len(preset)
        base = preset[base_index]
        expanded.append(
            PromptBenchSpec(
                prompt_id=f"{base.prompt_id}-{cycle_index:04d}",
                dataset=base.dataset,
                dataset_index=base.dataset_index + cycle_index,
                template=base.template,
                system_prompt=base.system_prompt,
                attack=base.attack,
            )
        )
    return expanded


def load_promptbench_prompts(preset_name: str, prompt_count: int) -> list[Prompt]:
    preset = _expand_promptbench_preset(preset_name, prompt_count)
    pb = _require_promptbench()
    datasets: dict[str, Any] = {}
    prompts: list[Prompt] = []
    seen_ids: set[str] = set()

    for spec in preset:
        if spec.prompt_id in seen_ids:
            raise ValueError(f"Duplicate PromptBench prompt id {spec.prompt_id!r}")
        if spec.dataset not in datasets:
            try:
                datasets[spec.dataset] = pb.DatasetLoader.load_dataset(spec.dataset)
            except Exception as exc:
                raise RuntimeError(f"PromptBench failed to load dataset {spec.dataset!r}: {exc}") from exc

        dataset = datasets[spec.dataset]
        try:
            record = dataset[spec.dataset_index]
        except Exception as exc:
            raise RuntimeError(
                f"PromptBench dataset {spec.dataset!r} has no item at index {spec.dataset_index}"
            ) from exc
        if not isinstance(record, dict):
            raise RuntimeError(
                f"PromptBench dataset {spec.dataset!r} item {spec.dataset_index} must be an object"
            )

        formatted_record = {key: _format_promptbench_value(value) for key, value in record.items()}
        try:
            user_prompt = spec.template.format(**formatted_record)
        except KeyError as exc:
            raise RuntimeError(
                f"PromptBench preset {preset_name!r} references missing field {exc!s} "
                f"from {spec.dataset}[{spec.dataset_index}]"
            ) from exc

        attacked_prompt = _apply_promptbench_attack(user_prompt, spec.attack)
        messages = [
            {"role": "system", "content": spec.system_prompt},
            {"role": "user", "content": attacked_prompt},
        ]
        prompts.append(
            Prompt(
                index=len(prompts),
                prompt_id=spec.prompt_id,
                messages=messages,
                raw={
                    "id": spec.prompt_id,
                    "source": "promptbench",
                    "preset": preset_name,
                    "dataset": spec.dataset,
                    "dataset_index": spec.dataset_index,
                    "attack": spec.attack,
                },
            )
        )
        seen_ids.add(spec.prompt_id)

    if not prompts:
        raise ValueError(f"PromptBench preset {preset_name!r} produced no prompts")
    return prompts


def _response_text(response) -> str:
    try:
        return response.read().decode("utf-8", "replace")
    except Exception:
        return ""


def _parse_http_error(error: HTTPError) -> tuple[int, str]:
    body = _response_text(error).strip()
    message = body or error.reason or f"HTTP {error.code}"
    return error.code, message


def _post_json(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
    stream: bool,
) -> tuple[int | None, bool, str]:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    request = Request(url, data=body, headers=headers, method="POST")

    try:
        with urlopen(request, timeout=timeout) as response:
            status_code = int(response.status)
            content_type = response.headers.get("Content-Type", "")
            if stream and "text/event-stream" in content_type:
                completed = False
                while True:
                    raw_line = response.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    if line == "data: [DONE]":
                        completed = True
                        break
                    if line.startswith("data:"):
                        event_data = line[5:].strip()
                        if not event_data or event_data == "[DONE]":
                            completed = event_data == "[DONE]"
                            if completed:
                                break
                            continue
                        try:
                            json.loads(event_data)
                        except json.JSONDecodeError as exc:
                            return status_code, False, f"Malformed SSE payload: {exc.msg}"
                    elif line.startswith("{") and line.endswith("}"):
                        try:
                            json.loads(line)
                        except json.JSONDecodeError as exc:
                            return status_code, False, f"Malformed SSE payload: {exc.msg}"
                return status_code, completed, ""

            response_body = response.read().decode("utf-8", "replace")
            return status_code, True, response_body
    except HTTPError as error:
        status_code, message = _parse_http_error(error)
        return status_code, False, message
    except (TimeoutError, URLError) as error:
        return None, False, str(error)


def _fetch_json(url: str, api_key: str, timeout: float) -> tuple[int | None, Any, str | None]:
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.status), json.load(response), None
    except HTTPError as error:
        body = _response_text(error).strip()
        if not body:
            return error.code, {}, error.reason or f"HTTP {error.code}"
        try:
            return error.code, json.loads(body), body
        except json.JSONDecodeError:
            return error.code, {}, body
    except (TimeoutError, URLError) as error:
        return None, {}, str(error)
    except json.JSONDecodeError as error:
        return None, {}, f"Invalid JSON response from {url}: {error.msg}"


def _model_ids_from_response(payload: Any) -> list[str]:
    if isinstance(payload, list):
        ids = []
        for item in payload:
            if isinstance(item, dict):
                model_id = item.get("id")
                if isinstance(model_id, str):
                    ids.append(model_id)
        return ids
    if isinstance(payload, dict):
        items = payload.get("data") or payload.get("items")
        if isinstance(items, list):
            ids = []
            for item in items:
                if isinstance(item, dict):
                    model_id = item.get("id")
                    if isinstance(model_id, str):
                        ids.append(model_id)
            return ids
    return []


def _build_payload(
    prompt: Prompt,
    sampling: SamplingConfig,
    model: str,
    run_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    message_prefix = f"{prompt.prompt_id}-{sampling.sampling_id}"
    user_message_id = _stable_message_id(message_prefix, "user")
    assistant_message_id = _stable_message_id(message_prefix, "assistant")
    user_text = _last_user_text(prompt.messages)

    payload = {
        "model": model,
        "messages": copy.deepcopy(prompt.messages),
        "stream": True,
        "seed": args.seed,
        "temperature": sampling.temperature,
        "top_k": sampling.top_k,
        "top_p": sampling.top_p,
        "max_tokens": args.max_tokens,
        # Use a local chat id so the backend has a stable filterable key without
        # forcing a persisted chat record.
        "chat_id": f"local:{run_id}",
        "id": assistant_message_id,
        "user_message": {
            "id": user_message_id,
            "role": "user",
            "content": user_text,
        },
    }
    return payload


def _run_single_prompt(
    prompt_run: PromptRun,
    args: argparse.Namespace,
    run_id: str,
    api_key: str,
    base_url: str,
    timeout: float,
) -> RunResult:
    prompt = prompt_run.prompt
    sampling = prompt_run.sampling
    request_payload = _build_payload(prompt, sampling, args.model, run_id, args)
    endpoints = ["/api/chat/completions", "/api/v1/chat/completions"]
    started = time.perf_counter()
    last_error: str | None = None

    for endpoint in endpoints:
        request_url = _build_url(base_url, endpoint)
        status_code, completed, message = _post_json(
            request_url,
            request_payload,
            api_key=api_key,
            timeout=timeout,
            stream=True,
        )
        if status_code == 404 and endpoint != endpoints[-1]:
            last_error = f"HTTP 404 from {request_url}"
            continue

        elapsed_s = time.perf_counter() - started
        if status_code is None or status_code >= 400:
            return RunResult(
                index=prompt_run.index,
                prompt_id=prompt.prompt_id,
                sampling_id=sampling.sampling_id,
                request_url=request_url,
                status_code=status_code,
                stream_completed=False,
                elapsed_s=elapsed_s,
                error=message,
            )

        if not completed:
            return RunResult(
                index=prompt_run.index,
                prompt_id=prompt.prompt_id,
                sampling_id=sampling.sampling_id,
                request_url=request_url,
                status_code=status_code,
                stream_completed=False,
                elapsed_s=elapsed_s,
                error=message or "stream ended before [DONE]",
            )

        return RunResult(
            index=prompt_run.index,
            prompt_id=prompt.prompt_id,
            sampling_id=sampling.sampling_id,
            request_url=request_url,
            status_code=status_code,
            stream_completed=True,
            elapsed_s=elapsed_s,
        )

    elapsed_s = time.perf_counter() - started
    return RunResult(
        index=prompt_run.index,
        prompt_id=prompt.prompt_id,
        sampling_id=sampling.sampling_id,
        request_url=_build_url(base_url, endpoints[-1]),
        status_code=404,
        stream_completed=False,
        elapsed_s=elapsed_s,
        error=last_error or "Open WebUI endpoint not found",
    )


def _preflight(args: argparse.Namespace, prompts: list[Prompt]) -> None:
    if not prompts:
        raise RuntimeError("No prompts loaded")
    models_url = _build_url(args.openwebui_url, "/api/models")
    status_code, payload, error = _fetch_json(models_url, args.api_key, args.timeout)
    if status_code == 404:
        models_url = _build_url(args.openwebui_url, "/api/v1/models")
        status_code, payload, error = _fetch_json(models_url, args.api_key, args.timeout)

    if status_code is None:
        raise RuntimeError(f"Unable to reach Open WebUI model list: {error}")
    if status_code >= 400:
        detail = f": {error}" if error else ""
        raise RuntimeError(f"Unable to read Open WebUI model list: HTTP {status_code}{detail}")

    model_ids = set(_model_ids_from_response(payload))
    if args.model not in model_ids:
        available = ", ".join(sorted(model_ids)) or "<empty>"
        raise RuntimeError(f"Model {args.model!r} not found in Open WebUI model list. Available: {available}")


def _load_prompt_suite(args: argparse.Namespace) -> tuple[list[Prompt], str]:
    prompts = load_promptbench_prompts(args.promptbench_preset, args.prompt_count)
    return prompts, f"PromptBench preset {args.promptbench_preset} ({len(prompts)} prompts)"


def _sampling_configs(args: argparse.Namespace) -> list[SamplingConfig]:
    sweep_requested = args.sampling_sweep or args.temperature_values or args.top_p_values or args.top_k_values
    if not sweep_requested:
        return [
            SamplingConfig(
                sampling_id=f"t{_number_id(args.temperature)}-p{_number_id(args.top_p)}-k{args.top_k}",
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        ]

    try:
        temperature_values = (
            _parse_float_list(args.temperature_values, "--temperature-values", 0.0)
            if args.temperature_values
            else _default_temperature_sweep(args.temperature)
        )
        top_p_values = (
            _parse_float_list(args.top_p_values, "--top-p-values", 0.0, 1.0)
            if args.top_p_values
            else _default_top_p_sweep(args.top_p)
        )
        top_k_values = (
            _parse_int_list(args.top_k_values, "--top-k-values", 1)
            if args.top_k_values
            else _default_top_k_sweep(args.top_k)
        )
    except argparse.ArgumentTypeError as exc:
        raise ValueError(str(exc)) from exc

    configs: list[SamplingConfig] = []
    seen_ids: set[str] = set()
    for temperature in temperature_values:
        for top_p in top_p_values:
            for top_k in top_k_values:
                sampling_id = f"t{_number_id(temperature)}-p{_number_id(top_p)}-k{top_k}"
                if sampling_id in seen_ids:
                    continue
                configs.append(
                    SamplingConfig(
                        sampling_id=sampling_id,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                )
                seen_ids.add(sampling_id)
    if not configs:
        raise ValueError("Sampling sweep produced no configurations")
    return configs


def _prompt_runs(prompts: list[Prompt], sampling_configs: list[SamplingConfig]) -> list[PromptRun]:
    runs: list[PromptRun] = []
    for sampling in sampling_configs:
        for prompt in prompts:
            runs.append(PromptRun(index=len(runs), prompt=prompt, sampling=sampling))
    return runs


def _select_prompt_runs(prompt_runs: list[PromptRun], start_prompt: int) -> tuple[list[PromptRun], str]:
    if start_prompt > len(prompt_runs):
        raise ValueError(
            f"--start-prompt {start_prompt} is beyond the expanded request count "
            f"({len(prompt_runs)})"
        )

    selected_runs = prompt_runs[start_prompt - 1 :]
    if start_prompt == 1:
        return selected_runs, f"{len(prompt_runs)} total requests"
    return selected_runs, f"requests {start_prompt}-{len(prompt_runs)} of {len(prompt_runs)}"


