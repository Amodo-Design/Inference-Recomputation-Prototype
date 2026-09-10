"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";

/** Deletes one TERMINAL run (its result rows cascade server-side). The
 * ledger's verification events from the run always survive — only the run
 * window (and so its analysis-tab picker entry) is lost. */
export default function DeleteRunButton({ runId }: { runId: string }) {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function remove() {
    setBusy(true);
    setError(null);
    try {
      const response = await fetch(`/api/test-runner/runs/${encodeURIComponent(runId)}`, {
        method: "DELETE"
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setError(`Delete failed (${response.status}): ${JSON.stringify(body.detail ?? body)}`);
        return;
      }
      startTransition(() => router.refresh());
    } catch (err) {
      setError(err instanceof Error ? err.message : "delete request failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <button type="button" className="delete-button" onClick={remove} disabled={busy || isPending}>
        {busy ? "Deleting" : "Delete"}
      </button>
      {error ? <div className="muted">{error}</div> : null}
    </>
  );
}
