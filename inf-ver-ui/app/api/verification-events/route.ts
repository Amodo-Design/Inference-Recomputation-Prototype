import { NextResponse } from "next/server";

import { forwardedQuery } from "./query";

const LEDGER_BASE = process.env.LEDGER_API_URL ?? "http://ledger-api:8000";

export async function DELETE(request: Request) {
  const response = await fetch(
    `${LEDGER_BASE}/verification-events${forwardedQuery(request.url)}`,
    { method: "DELETE" }
  );
  const body = await response.json().catch(() => ({}));
  return NextResponse.json(body, { status: response.status });
}
