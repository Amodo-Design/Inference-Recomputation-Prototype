import { NextResponse } from "next/server";
import { forwardedQuery } from "../query";

const LEDGER_BASE = process.env.LEDGER_API_URL ?? "http://ledger-api:8000";
const PAGE_LIMIT = 1000;

type VerificationEventView = {
  id: string;
  inference_event_id: string;
  ts: string;
  result: string | null;
  result_detail: string | null;
  error_code: string | null;
  verification_threshold: number | null;
  verifier_model_id: string;
  verifier_model_name: string | null;
  model_name: string;
  sampling_config: Record<string, unknown>;
  exact_match_level_pct: number | null;
  mean_logit_difference: number | null;
  std_dev_logit_difference: number | null;
  difr_margins: number[];
  session_id: string;
  input_text_representation: string | null;
  output_text_representation: string | null;
  verifier_detail: Record<string, unknown> | null;
};

type VerificationEventPage = {
  items: VerificationEventView[];
  total: number;
  limit: number;
  offset: number;
};

// CSV mirrors the dashboard table: same columns, same formatting.
const EXPORT_HEADERS = [
  "Result",
  "Error Code",
  "Created",
  "Inference Event",
  "Prover Model",
  "Verifier Model",
  "Exact Match Percentage",
  "Logit Difference Threshold",
  "Mean Logit Difference",
  "DiFR Margins",
  "Prompt Tokens",
  "Output Tokens",
  // Latency hidden for now; it is recorded in verifier_detail.
  // "Latency (ms)",
  "Detail",
  "User Prompt",
  "Prover Output",
  "Verifier Output",
  "Output Token Comparison",
  "Database Row"
];

function formatPercent(value: number | null): string {
  if (value === null || !Number.isFinite(value)) {
    return "-";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function formatNumber(value: number | null, digits = 4): string {
  if (value === null || !Number.isFinite(value)) {
    return "-";
  }
  return value.toFixed(digits);
}

function exportRow(event: VerificationEventView): (string | number)[] {
  const detail = event.verifier_detail ?? {};
  return [
    event.result ?? "-",
    event.error_code ?? "-",
    event.ts,
    event.inference_event_id,
    event.model_name,
    event.verifier_model_name ?? "-",
    formatPercent(event.exact_match_level_pct),
    formatNumber(event.verification_threshold),
    formatNumber(event.mean_logit_difference),
    JSON.stringify(event.difr_margins),
    (detail.prompt_token_count as number | undefined) ?? "-",
    (detail.output_token_count as number | undefined) ?? "-",
    // (detail.latency_ms as number | undefined) ?? "-",
    event.result_detail ?? "-",
    event.input_text_representation ?? "-",
    (detail.prover_output_text as string | undefined) ??
      event.output_text_representation ??
      "-",
    (detail.verifier_output_text as string | undefined) ?? "-",
    JSON.stringify(detail.output_token_comparison ?? []),
    JSON.stringify(event, null, 2)
  ];
}

function csvCell(value: string | number): string {
  const text = String(value);
  if (/[",\n\r]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

async function fetchAllEvents(filterQuery: string): Promise<VerificationEventView[]> {
  const events: VerificationEventView[] = [];
  let offset = 0;
  for (;;) {
    const response = await fetch(
      `${LEDGER_BASE}/verification-events/view?limit=${PAGE_LIMIT}&offset=${offset}${filterQuery}`,
      { cache: "no-store" }
    );
    if (!response.ok) {
      throw new Error(`ledger returned ${response.status}`);
    }
    const page = (await response.json()) as VerificationEventPage;
    events.push(...page.items);
    if (page.items.length < PAGE_LIMIT) {
      return events;
    }
    offset += PAGE_LIMIT;
  }
}

export async function GET(request: Request) {
  const q = forwardedQuery(request.url);
  const filterQuery = q ? `&${q.slice(1)}` : "";
  let events: VerificationEventView[];
  try {
    events = await fetchAllEvents(filterQuery);
  } catch (error) {
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "export failed" },
      { status: 502 }
    );
  }
  const lines = [
    EXPORT_HEADERS.map(csvCell).join(","),
    ...events.map((event) => exportRow(event).map(csvCell).join(","))
  ];
  return new NextResponse(lines.join("\r\n"), {
    status: 200,
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": 'attachment; filename="verification_events.csv"'
    }
  });
}
