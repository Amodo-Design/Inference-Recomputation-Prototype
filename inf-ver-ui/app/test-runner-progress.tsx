"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { allocationLabel, AllocationRow } from "./run-label";

type RunDetail = {
  id: string;
  state: string;
  total_requests: number;
  completed: number;
  failed: number;
  error: string | null;
  log: string[];
  settings: { models?: AllocationRow[] } & Record<string, unknown>;
};

/** Live progress for the active run: polls the run detail while it is
 * running and refreshes the page once it finishes (so the history table and
 * the Verification tab's pending counts catch up). */
export default function TestRunnerProgress({ runId }: { runId: string }) {
  const router = useRouter();
  const [detail, setDetail] = useState<RunDetail | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    async function poll() {
      try {
        const response = await fetch(`/api/test-runner/runs/${encodeURIComponent(runId)}`);
        if (response.ok) {
          const body = (await response.json()) as RunDetail;
          if (cancelled) return;
          setDetail(body);
          if (body.state !== "running") {
            router.refresh();
            return; // finished — stop polling
          }
        }
      } catch {
        // transient; keep polling
      }
      if (!cancelled) timer = setTimeout(poll, 3000);
    }

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [runId, router]);

  if (!detail) {
    return <div className="empty">Loading run {runId}…</div>;
  }

  const progress =
    detail.total_requests > 0
      ? `${detail.completed}/${detail.total_requests}`
      : "preparing (prompt suite / preflight)";

  return (
    <div className="test-run-progress">
      <p>
        Run <span className="mono">{detail.id}</span> against{" "}
        <strong>{allocationLabel(detail.settings)}</strong> —{" "}
        <strong>{detail.state}</strong>, {progress} requests
        {detail.failed > 0 ? `, ${detail.failed} failed` : ""}.
      </p>
      {detail.error ? <p className="target-note">⚠ {detail.error}</p> : null}
      <details open>
        <summary>Log ({detail.log.length} lines)</summary>
        <pre className="wrapped-text">{detail.log.slice(-15).join("\n")}</pre>
      </details>
    </div>
  );
}
