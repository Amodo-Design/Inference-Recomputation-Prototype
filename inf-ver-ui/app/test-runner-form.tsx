"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { AllocationRow, rowsAreValid } from "./run-label";

/** Launch form for the in-cluster prompt runner. The suite is split across
 * one or more models by percentage (largest-remainder, exact totals) with a
 * per-model concurrency. Sampling stays pinned by the verify-taps. Submitting
 * QUEUES the run — the prompt-runner executes one at a time. */
export default function TestRunnerForm({ models }: { models: string[] }) {
  const router = useRouter();
  const [rows, setRows] = useState<AllocationRow[]>([
    { model: models[0] ?? "", percent: 100, concurrency: 1 }
  ]);
  const [note, setNote] = useState<string | null>(null);
  const [isError, setIsError] = useState(false);
  const [isPending, startTransition] = useTransition();
  const [submitting, setSubmitting] = useState(false);

  const totalPercent = rows.reduce((sum, row) => sum + row.percent, 0);
  const duplicates = new Set(rows.map((r) => r.model)).size !== rows.length;
  const rowsValid = rowsAreValid(rows) && totalPercent === 100 && !duplicates;

  function setRow(index: number, patch: Partial<AllocationRow>) {
    setRows((prev) =>
      prev.map((row, i) => {
        if (i !== index) return row;
        const next = { ...row, ...patch };
        // A slider can never push the TOTAL above 100%: clamp this row's
        // share to whatever the other rows leave available (min 1).
        if (patch.percent !== undefined) {
          const othersTotal = prev.reduce(
            (sum, other, j) => (j === index ? sum : sum + other.percent),
            0
          );
          next.percent = Math.max(1, Math.min(next.percent, 100 - othersTotal));
        }
        return next;
      })
    );
  }

  function addRow() {
    const unused = models.find((m) => !rows.some((r) => r.model === m));
    setRows((prev) => {
      if (prev.length === 0) {
        return [{ model: unused ?? models[0] ?? "", percent: 100, concurrency: 1 }];
      }
      // Seed the new row from the largest existing row: take half (floor)
      // of its percent and deduct it from that row, so add-then-queue works
      // without the operator having to touch any sliders. A largest row of
      // 1% can't be split further — seed 1 without deducting (the validity
      // gate still requires the total to sum to 100, so this case leaves
      // the form invalid until the operator adjusts a slider).
      let largestIndex = 0;
      for (let i = 1; i < prev.length; i++) {
        if (prev[i].percent > prev[largestIndex].percent) largestIndex = i;
      }
      const largest = prev[largestIndex];
      const share = Math.floor(largest.percent / 2);
      const next = prev.map((row, i) =>
        i === largestIndex && share >= 1 ? { ...row, percent: row.percent - share } : row
      );
      const seeded = share >= 1 ? share : 1;
      return [...next, { model: unused ?? models[0] ?? "", percent: seeded, concurrency: 1 }];
    });
  }

  function removeRow(index: number) {
    setRows((prev) => prev.filter((_, i) => i !== index));
  }

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setNote(null);
    setIsError(false);
    const form = new FormData(event.currentTarget);
    const num = (name: string) => Number(form.get(name));
    const payload = {
      models: rows,
      promptbench_preset: String(form.get("promptbench_preset") ?? "verification-v1"),
      prompt_count: num("prompt_count"),
      start_prompt: num("start_prompt"),
      max_tokens: num("max_tokens"),
      timeout: num("timeout"),
      continue_on_error: form.get("continue_on_error") === "on",
      skip_preflight: form.get("skip_preflight") === "on"
    };

    setSubmitting(true);
    try {
      const response = await fetch("/api/test-runner/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const body = await response.json();
      if (!response.ok) {
        setIsError(true);
        setNote(`Queueing failed (${response.status}): ${JSON.stringify(body.detail ?? body)}`);
        return;
      }
      setNote(`Run ${body.id} queued.`);
      startTransition(() => router.refresh());
    } finally {
      setSubmitting(false);
    }
  }

  const busy = submitting || isPending;

  return (
    <form className="test-runner-form" onSubmit={submit} aria-label="Queue test run">
      <div className="allocation-rows">
        {rows.map((row, index) => (
          <div className="filters allocation-row" key={index}>
            <div className="filter-field">
              <label htmlFor={`tr-model-${index}`}>Model</label>
              <select
                id={`tr-model-${index}`}
                value={row.model}
                onChange={(e) => setRow(index, { model: e.target.value })}
                required
              >
                {models.length === 0 ? <option value="">no active provers</option> : null}
                {models.map((name) => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
            </div>
            <div className="filter-field filter-grow">
              <label htmlFor={`tr-percent-${index}`}>Share: {row.percent}%</label>
              <input
                id={`tr-percent-${index}`}
                type="range"
                min={1}
                max={100}
                value={row.percent}
                onChange={(e) => setRow(index, { percent: Number(e.target.value) })}
              />
            </div>
            <div className="filter-field">
              <label htmlFor={`tr-conc-${index}`}>Concurrency</label>
              <input
                id={`tr-conc-${index}`}
                type="number"
                min={1}
                max={32}
                value={row.concurrency}
                onChange={(e) => setRow(index, { concurrency: Number(e.target.value) })}
              />
            </div>
            {rows.length > 1 ? (
              <div className="filter-actions">
                <button type="button" className="delete-button" onClick={() => removeRow(index)} aria-label={`Remove model row ${index + 1}`}>
                  Remove
                </button>
              </div>
            ) : null}
          </div>
        ))}
        <div className="allocation-total">
          <button type="button" className="delete-button action-button-neutral" onClick={addRow} disabled={rows.length >= models.length}>
            Add model
          </button>
          <span className={totalPercent === 100 ? "status status-pass" : "status status-fail"}>
            Total {totalPercent}%
          </span>
          {duplicates ? <span className="status status-fail">duplicate models</span> : null}
        </div>
      </div>

      <div className="filters">
        <div className="filter-field">
          <label htmlFor="tr-preset">Preset</label>
          <select id="tr-preset" name="promptbench_preset" defaultValue="verification-v1">
            <option value="verification-v1">verification-v1</option>
            <option value="long-output-v1">long-output-v1</option>
          </select>
        </div>
        <div className="filter-field">
          <label htmlFor="tr-count">Prompts</label>
          <input id="tr-count" name="prompt_count" type="number" min="1" defaultValue={20} />
        </div>
        <div className="filter-field">
          <label htmlFor="tr-start">Start at</label>
          <input id="tr-start" name="start_prompt" type="number" min="1" defaultValue={1} />
        </div>
        <div className="filter-field">
          <label htmlFor="tr-max">Max tokens</label>
          <input id="tr-max" name="max_tokens" type="number" min="1" defaultValue={512} />
        </div>
        <div className="filter-field">
          <label htmlFor="tr-timeout">Timeout (s)</label>
          <input id="tr-timeout" name="timeout" type="number" min="1" defaultValue={120} />
        </div>
        <div className="filter-field">
          <label className="filter-check">
            <input name="continue_on_error" type="checkbox" /> Continue on error
          </label>
          <label className="filter-check">
            <input name="skip_preflight" type="checkbox" /> Skip preflight
          </label>
        </div>
        <div className="filter-actions">
          <button type="submit" disabled={busy || !rowsValid || models.length === 0}>
            {busy ? "Queueing" : "Queue run"}
          </button>
        </div>
      </div>
      {note ? (
        <p className="target-note" role="status">
          {note}
          {isError ? " ⚠" : ""}
        </p>
      ) : null}
    </form>
  );
}
