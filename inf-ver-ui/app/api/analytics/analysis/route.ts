import { NextResponse } from "next/server";
import { forwardedAnalysisQuery } from "../analysis-query";

const ANALYTICS_BASE = process.env.ANALYTICS_API_URL ?? "http://analytics-api:8400";

export async function GET(request: Request) {
  const q = forwardedAnalysisQuery(request.url);
  let response: Response;
  try {
    response = await fetch(`${ANALYTICS_BASE}/analysis${q}`, {
      cache: "no-store"
    });
  } catch (error) {
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : "analysis fetch failed" },
      { status: 502 }
    );
  }
  const body = await response.json();
  return NextResponse.json(body, { status: response.status });
}
