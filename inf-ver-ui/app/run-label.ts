export type AllocationRow = { model: string; percent: number; concurrency: number };

/** True when every row has a model, a percent in [1, 100], and a
 * concurrency in [1, 32] — the same per-row bounds the backend's
 * ModelAllocation enforces (models field, percent/concurrency ge/le). Does
 * NOT check the total===100 or no-duplicates rules; callers combine this
 * with those checks. */
export function rowsAreValid(rows: AllocationRow[]): boolean {
  return rows.every(
    (r) =>
      Boolean(r.model) &&
      Number.isFinite(r.percent) &&
      r.percent >= 1 &&
      r.percent <= 100 &&
      Number.isFinite(r.concurrency) &&
      r.concurrency >= 1 &&
      r.concurrency <= 32
  );
}

/** Renders a run's per-model allocation as a compact summary, e.g.
 * "modelA 60%×2, modelB 40%×1". Shared by the run history/queue tables and
 * the live Active Run panel so both describe multi-model runs the same
 * way. */
export function allocationLabel(settings: { models?: AllocationRow[] } & Record<string, unknown>): string {
  const models = settings.models;
  if (!Array.isArray(models) || models.length === 0) return "-";
  return models
    .map((m) => `${m.model} ${m.percent}%×${m.concurrency}`)
    .join(", ");
}
