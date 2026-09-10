// Pass the dashboard's filter params straight through to the ledger.
export function forwardedQuery(url: string): string {
  const incoming = new URL(url).searchParams;
  const params = new URLSearchParams();
  for (const key of ["result", "model", "search"] as const) {
    const value = incoming.get(key);
    if (value) params.set(key, value);
  }
  const query = params.toString();
  return query ? `?${query}` : "";
}
