import { NextResponse } from "next/server";

// Per-model verification threshold lives in the LEDGER's model row (mutable,
// outside the model's identity hash). inf-ver-ui no longer talks to any
// verifier process — the ledger is its single backend.
const LEDGER_BASE = process.env.LEDGER_API_URL ?? "http://ledger-api:8000";

export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ modelId: string }> }
) {
  const { modelId } = await params;
  const payload = await request.json();
  const response = await fetch(
    `${LEDGER_BASE}/models/${encodeURIComponent(modelId)}/verification-threshold`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }
  );
  const body = await response.json();
  return NextResponse.json(body, { status: response.status });
}
