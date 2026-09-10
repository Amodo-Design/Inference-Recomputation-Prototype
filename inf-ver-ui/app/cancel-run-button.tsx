"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";

export default function CancelRunButton({ runId }: { runId: string }) {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function cancel() {
    setBusy(true);
    setError(null);
    try {
      const response = await fetch(`/api/test-runner/runs/${encodeURIComponent(runId)}`, {
        method: "DELETE"
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setError(`Cancel failed (${response.status}): ${JSON.stringify(body.detail ?? body)}`);
        return;
      }
      startTransition(() => router.refresh());
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <button type="button" className="delete-button" onClick={cancel} disabled={busy || isPending}>
        {busy ? "Cancelling" : "Cancel"}
      </button>
      {error ? <div className="muted">{error}</div> : null}
    </>
  );
}
