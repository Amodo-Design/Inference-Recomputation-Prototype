import { NextResponse } from "next/server";

const PROMPT_RUNNER_BASE =
  process.env.PROMPT_RUNNER_URL ?? "http://prompt-runner:8200";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ runId: string }> }
) {
  const { runId } = await params;
  const response = await fetch(
    `${PROMPT_RUNNER_BASE}/runs/${encodeURIComponent(runId)}`,
    { cache: "no-store" }
  );
  const body = await response.json();
  return NextResponse.json(body, { status: response.status });
}

export async function DELETE(
  _request: Request,
  { params }: { params: Promise<{ runId: string }> }
) {
  const { runId } = await params;
  const response = await fetch(
    `${PROMPT_RUNNER_BASE}/runs/${encodeURIComponent(runId)}`,
    { method: "DELETE" }
  );
  const body = await response.json();
  return NextResponse.json(body, { status: response.status });
}
