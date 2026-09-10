import { NextResponse } from "next/server";

// The in-cluster prompt-runner service (deterministic PromptBench suite
// driven through Open WebUI, so responses land as tapped inference events).
const PROMPT_RUNNER_BASE =
  process.env.PROMPT_RUNNER_URL ?? "http://prompt-runner:8200";

export async function GET() {
  const response = await fetch(`${PROMPT_RUNNER_BASE}/runs`, {
    cache: "no-store"
  });
  const body = await response.json();
  return NextResponse.json(body, { status: response.status });
}

export async function POST(request: Request) {
  const payload = await request.json();
  const response = await fetch(`${PROMPT_RUNNER_BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const body = await response.json();
  return NextResponse.json(body, { status: response.status });
}

export async function DELETE() {
  // Deletes ALL terminal runs (the durable history); queued/running are
  // untouched by the prompt-runner.
  const response = await fetch(`${PROMPT_RUNNER_BASE}/runs`, { method: "DELETE" });
  const body = await response.json();
  return NextResponse.json(body, { status: response.status });
}
