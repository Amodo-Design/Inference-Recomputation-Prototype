"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import { useRouter } from "next/navigation";

const ARM_TIMEOUT_MS = 5000;

/** Two-click "Delete all" for the run history (terminal runs only — the
 * queue and the running run are untouched server-side). First click arms;
 * the confirm state names the count and warns that run windows (analysis
 * run-picker entries) are lost forever; it disarms after a timeout or on
 * blur. No browser dialogs — they block the automation stack. */
export default function DeleteAllRunsButton({ count }: { count: number }) {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();
  const [armed, setArmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const disarmTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (disarmTimer.current) clearTimeout(disarmTimer.current);
    };
  }, []);

  function arm() {
    setArmed(true);
    setError(null);
    disarmTimer.current = setTimeout(() => setArmed(false), ARM_TIMEOUT_MS);
  }

  async function confirm() {
    if (disarmTimer.current) clearTimeout(disarmTimer.current);
    setBusy(true);
    setError(null);
    try {
      const response = await fetch("/api/test-runner/runs", { method: "DELETE" });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setError(`Delete all failed (${response.status}): ${JSON.stringify(body.detail ?? body)}`);
        return;
      }
      startTransition(() => router.refresh());
    } catch (err) {
      setError(err instanceof Error ? err.message : "delete request failed");
    } finally {
      setBusy(false);
      setArmed(false);
    }
  }

  if (count === 0) return null;

  return (
    <span className="delete-all-runs">
      {/* One element for both states: React reuses the DOM node across the
          swap, so the arming click leaves the button focused and onBlur
          can't fire before the confirm click. The server deletes ALL
          terminal runs — the visible list is fetch-capped, so the copy says
          "all history" rather than a possibly-understated count. */}
      {armed ? (
        <button type="button" className="delete-button delete-button-danger" onClick={confirm} onBlur={() => setArmed(false)} disabled={busy || isPending}>
          {busy ? "Deleting" : "Confirm: delete ALL run history (picker windows lost forever)"}
        </button>
      ) : (
        <button type="button" className="delete-button" onClick={arm} disabled={busy || isPending}>
          Delete all
        </button>
      )}
      {error ? <div className="muted">{error}</div> : null}
    </span>
  );
}
