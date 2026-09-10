// Pass the Analysis tab's (larger) filter set straight through to
// analytics-api, including the run-window inference timestamp filters.
export function forwardedAnalysisQuery(url: string): string {
  const incoming = new URL(url).searchParams;
  const params = new URLSearchParams();
  for (const key of [
    "result",
    "model",
    "prover_model",
    "verifier_model",
    "search",
    "from_ts",
    "to_ts",
    "inference_from_ts",
    "inference_to_ts",
    "limit"
  ] as const) {
    const value = incoming.get(key);
    if (value) params.set(key, value);
  }
  const query = params.toString();
  return query ? `?${query}` : "";
}
