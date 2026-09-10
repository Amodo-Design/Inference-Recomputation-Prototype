"""Run lifecycle + API behaviour with stubbed prompt suite and HTTP calls."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

import app.core as core
import app.service as service


def make_prompts(n: int) -> list[core.Prompt]:
    return [
        core.Prompt(index=i, prompt_id=f"p-{i}", messages=[], raw={}) for i in range(n)
    ]


def stub_suite(monkeypatch, n_prompts: int = 3):
    monkeypatch.setattr(
        core, "_load_prompt_suite", lambda args: (make_prompts(n_prompts), "stub suite")
    )
    monkeypatch.setattr(service, "_resolve_api_key", lambda: "stub-token")


def stub_single_prompt(monkeypatch, fail_ids: set[str] | None = None, delay: float = 0.0):
    fail_ids = fail_ids or set()

    def fake_run(prompt_run, args, run_id, api_key, base_url, timeout):
        if delay:
            time.sleep(delay)
        failed = prompt_run.prompt.prompt_id in fail_ids
        return core.RunResult(
            index=prompt_run.index,
            prompt_id=prompt_run.prompt.prompt_id,
            sampling_id=prompt_run.sampling.sampling_id,
            request_url="http://stub/api/chat/completions",
            status_code=500 if failed else 200,
            stream_completed=not failed,
            elapsed_s=0.01,
            error="boom" if failed else None,
        )

    monkeypatch.setattr(core, "_run_single_prompt", fake_run)


def wait_for(run: service.Run, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while run.state in ("pending", "running"):
        if time.monotonic() > deadline:
            raise AssertionError(f"run stuck in state {run.state}")
        time.sleep(0.01)


def make_settings(**overrides) -> service.RunSettings:
    values = {
        "models": [{"model": "Qwen/Qwen2.5-7B-Instruct", "percent": 100, "concurrency": 1}],
        "skip_preflight": True,
    }
    values.update(overrides)
    return service.RunSettings(**values)


def test_run_completes_and_reports_results(monkeypatch):
    stub_suite(monkeypatch, 3)
    stub_single_prompt(monkeypatch)
    run = service.Run("test-run", make_settings())

    service.execute_run(run)

    assert run.state == "completed"
    assert run.total_requests == 3
    assert run.completed == 3
    assert run.failed == 0
    assert [r["status"] for r in run.results] == ["ok", "ok", "ok"]
    assert run.finished_at is not None
    assert all(r["model"] == "Qwen/Qwen2.5-7B-Instruct" for r in run.results)


def test_failure_stops_run_unless_continue_on_error(monkeypatch):
    stub_suite(monkeypatch, 3)
    stub_single_prompt(monkeypatch, fail_ids={"p-0"})

    run = service.Run("test-run", make_settings())
    service.execute_run(run)
    assert run.state == "completed"
    assert run.failed >= 1
    # Sequential run stops at the first failure.
    assert run.completed == 1

    run = service.Run("test-run", make_settings(continue_on_error=True))
    service.execute_run(run)
    assert run.state == "completed"
    assert run.completed == 3
    assert run.failed == 1


def test_multi_model_run_splits_prompts_and_labels_results(monkeypatch):
    stub_suite(monkeypatch, 4)
    seen_models: list[str] = []

    def fake_run(prompt_run, args, run_id, api_key, base_url, timeout):
        seen_models.append(args.model)
        return core.RunResult(
            index=prompt_run.index,
            prompt_id=prompt_run.prompt.prompt_id,
            sampling_id=prompt_run.sampling.sampling_id,
            request_url="http://stub/api/chat/completions",
            status_code=200,
            stream_completed=True,
            elapsed_s=0.01,
        )

    monkeypatch.setattr(core, "_run_single_prompt", fake_run)
    settings = make_settings(models=[
        {"model": "m-a", "percent": 50, "concurrency": 2},
        {"model": "m-b", "percent": 50, "concurrency": 1},
    ])
    run = service.Run("test-run", settings)
    service.execute_run(run)

    assert run.state == "completed"
    assert run.total_requests == 4 and run.completed == 4 and run.failed == 0
    assert sorted(seen_models) == ["m-a", "m-a", "m-b", "m-b"]
    by_model = {r["model"] for r in run.results}
    assert by_model == {"m-a", "m-b"}
    # request_index values are globally unique across the model slices.
    assert len({r["request_index"] for r in run.results}) == 4


def test_on_result_callback_receives_each_row(monkeypatch):
    stub_suite(monkeypatch, 3)
    stub_single_prompt(monkeypatch)
    rows: list[dict] = []
    run = service.Run("test-run", make_settings())
    service.execute_run(run, on_result=rows.append)
    assert len(rows) == 3
    assert {"request_index", "prompt_id", "sampling_id", "model",
            "status_code", "stream_completed", "elapsed_s", "error"} <= set(rows[0])


def test_preflight_checks_every_model(monkeypatch):
    stub_suite(monkeypatch, 2)
    stub_single_prompt(monkeypatch)
    checked: list[str] = []
    monkeypatch.setattr(core, "_preflight", lambda args, prompts: checked.append(args.model))
    settings = make_settings(
        models=[
            {"model": "m-a", "percent": 50, "concurrency": 1},
            {"model": "m-b", "percent": 50, "concurrency": 1},
        ],
        skip_preflight=False,
    )
    run = service.Run("test-run", settings)
    service.execute_run(run)
    assert sorted(checked) == ["m-a", "m-b"]


def test_setup_error_marks_run_error(monkeypatch):
    def broken(args):
        raise RuntimeError("promptbench unavailable")

    monkeypatch.setattr(core, "_load_prompt_suite", broken)
    monkeypatch.setattr(service, "_resolve_api_key", lambda: "stub-token")
    run = service.Run("test-run", make_settings())
    service.execute_run(run)
    assert run.state == "error"
    assert "promptbench unavailable" in (run.error or "")


def test_settings_namespace_matches_cli_contract():
    ns = make_settings().to_namespace(model="Qwen/Qwen2.5-7B-Instruct", concurrency=3)
    for attr in (
        "openwebui_url", "api_key", "model", "promptbench_preset", "prompt_count",
        "start_prompt", "concurrency", "seed", "temperature", "top_k", "top_p",
        "sampling_sweep", "temperature_values", "top_p_values", "top_k_values",
        "max_tokens", "timeout", "continue_on_error", "skip_preflight",
    ):
        assert hasattr(ns, attr), attr
    assert ns.model == "Qwen/Qwen2.5-7B-Instruct" and ns.concurrency == 3
    assert not hasattr(ns, "models")  # the vendored core never sees the list
    assert ns.seed == 42 and ns.temperature == 1.0
    assert ns.top_k == 50 and ns.top_p == 0.95
    assert ns.sampling_sweep is False


def test_legacy_single_model_fields_are_rejected():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        service.RunSettings(model="m", concurrency=1)  # dropped form
    with pytest.raises(ValidationError):
        make_settings(model="m")  # extra=forbid rejects the old field


def test_allocation_validation():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):  # empty list
        make_settings(models=[])
    with pytest.raises(ValidationError):  # percents must sum to 100
        make_settings(models=[{"model": "a", "percent": 60, "concurrency": 1}])
    with pytest.raises(ValidationError):  # duplicates rejected
        make_settings(models=[
            {"model": "a", "percent": 50, "concurrency": 1},
            {"model": "a", "percent": 50, "concurrency": 1},
        ])
    with pytest.raises(ValidationError):  # concurrency cap 32
        make_settings(models=[{"model": "a", "percent": 100, "concurrency": 33}])


def test_validation_rejected_before_persistence_check():
    # Pydantic validation runs before the store check: bad payloads are 422
    # even when persistence is down.
    client = TestClient(service.app)
    assert client.post("/runs", json={}).status_code == 422
    assert client.post("/runs", json={"models": [], "skip_preflight": True}).status_code == 422
    assert (
        client.post("/runs", json={"model": "m", "skip_preflight": True}).status_code == 422
    )  # legacy form rejected
    assert (
        client.post(
            "/runs",
            json={"models": [{"model": "m", "percent": 100, "concurrency": 1}], "seed": 7},
        ).status_code
        == 422
    )


def test_resolve_api_key_prefers_configured_key(monkeypatch):
    monkeypatch.setattr(service, "OPEN_WEBUI_API_KEY", "configured")
    assert service._resolve_api_key() == "configured"


def test_resolve_api_key_self_signs_in_when_blank(monkeypatch):
    monkeypatch.setattr(service, "OPEN_WEBUI_API_KEY", "")

    class FakeResponse:
        def read(self):
            import json as j
            return j.dumps({"token": "jwt-123"}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        return FakeResponse()

    monkeypatch.setattr(service.urllib.request, "urlopen", fake_urlopen)
    assert service._resolve_api_key() == "jwt-123"
    assert seen["url"].endswith("/api/v1/auths/signin")


def test_run_reports_error_when_signin_fails(monkeypatch):
    monkeypatch.setattr(service, "OPEN_WEBUI_API_KEY", "")

    def boom(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(service.urllib.request, "urlopen", boom)
    run = service.Run("test-run", make_settings())
    service.execute_run(run)
    assert run.state == "error"
    assert "connection refused" in (run.error or "")
