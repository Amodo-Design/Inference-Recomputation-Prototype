import { NextResponse } from "next/server";

// Per-model margin cap (difr "delta max") lives in the LEDGER's model row,
// mutable and outside the model's identity hash — same pattern as the
// verification threshold.
const LEDGER_BASE = process.env.LEDGER_API_URL ?? "http://ledger-api:8000";

export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ modelId: string }> }
) {
  const { modelId } = await params;
  const payload = await request.json();
  const response = await fetch(
    `${LEDGER_BASE}/models/${encodeURIComponent(modelId)}/delta-max`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }
  );
  const body = await response.json();
  return NextResponse.json(body, { status: response.status });
}
