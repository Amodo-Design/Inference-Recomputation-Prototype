// Shared helpers for the server-rendered dashboard pages.

export type SearchParams = Record<string, string | string[] | undefined>;

// The ledger is the system of record for verification: results and the
// per-model verification thresholds all live there. Analysis-tab reads come
// from analytics-api instead (see ANALYTICS_BASE below) — the ledger is no
// longer the UI's single backend.
export const LEDGER_BASE = process.env.LEDGER_API_URL ?? "http://ledger-api:8000";

// Analysis-tab rows and run economics/cohort views come from the analytics
// service, split out of the ledger so the ledger stays a pure write path.
export const ANALYTICS_BASE = process.env.ANALYTICS_API_URL ?? "http://analytics-api:8400";

export async function getJson<T>(base: string, path: string): Promise<T> {
  const response = await fetch(`${base}${path}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function firstValue(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

export function parsePage(value: string | string[] | undefined): number {
  const page = Number.parseInt(firstValue(value) ?? "1", 10);
  return Number.isFinite(page) && page > 0 ? page : 1;
}

export const PAGE_SIZE_OPTIONS = [10, 25, 50, 100] as const;
export const DEFAULT_PAGE_SIZE = 10;

// Falls back to the default for anything not in PAGE_SIZE_OPTIONS, so a
// hand-edited or stale URL can't request an arbitrarily large page.
export function parsePageSize(value: string | string[] | undefined): number {
  const size = Number.parseInt(firstValue(value) ?? "", 10);
  return (PAGE_SIZE_OPTIONS as readonly number[]).includes(size) ? size : DEFAULT_PAGE_SIZE;
}

export function formatPercent(value: number | null): string {
  if (value === null || !Number.isFinite(value)) {
    return "-";
  }
  return `${(value * 100).toFixed(2)}%`;
}

export function formatNumber(value: number | null, fractionDigits = 0): string {
  if (value === null || !Number.isFinite(value)) {
    return "-";
  }
  return value.toLocaleString(undefined, {
    maximumFractionDigits: fractionDigits,
    minimumFractionDigits: fractionDigits
  });
}

export function formatDate(value: string): string {
  return new Intl.DateTimeFormat("en-GB", {
    dateStyle: "medium",
    timeStyle: "medium"
  }).format(new Date(value));
}

// Ledger row shapes shared across tabs.
export type ModelRead = {
  model_id: string;
  model_name: string;
  temperature: number | null;
  top_k: number | null;
  top_p: number | null;
  seed: number | null;
  decoding_algorithm: string | null;
  verification_threshold: number | null;
  delta_max: number | null;
};
