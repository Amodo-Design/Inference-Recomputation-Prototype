import { NextResponse } from "next/server";

const LEDGER_BASE = process.env.LEDGER_API_URL ?? "http://ledger-api:8000";

export async function POST(
  _request: Request,
  { params }: { params: Promise<{ requestId: string }> }
) {
  const { requestId } = await params;
  const response = await fetch(
    `${LEDGER_BASE}/verification-events/${encodeURIComponent(requestId)}/replay`,
    { method: "POST" }
  );
  if (response.status === 204) {
    return new NextResponse(null, { status: 204 });
  }
  const body = await response.json().catch(() => ({}));
  return NextResponse.json(body, { status: response.status });
}
