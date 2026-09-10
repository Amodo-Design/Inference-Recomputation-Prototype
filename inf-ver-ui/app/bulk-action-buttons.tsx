"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";

type BulkFilters = { result: string; model: string; q: string };

function bulkQuery(filters: BulkFilters): string {
  const params = new URLSearchParams();
  if (filters.result) params.set("result", filters.result);
  if (filters.model) params.set("model", filters.model);
  if (filters.q) params.set("search", filters.q);
  const query = params.toString();
  return query ? `?${query}` : "";
}

export function TrashIcon() {
  return (
    <svg
      aria-hidden="true"
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 6h18" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  );
}

function BulkButton({
  total,
  filters,
  idleLabel,
  retryLabel,
  confirmPrompt,
  pendingLabel,
  icon,
  request
}: {
  total: number;
  filters: BulkFilters;
  idleLabel: string;
  retryLabel: string;
  confirmPrompt: (count: string) => string;
  pendingLabel: string;
  icon?: React.ReactNode;
  request: (query: string) => Promise<Response>;
}) {
  const router = useRouter();
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState(false);
  const [isPending, startTransition] = useTransition();

  async function run() {
    setError(false);
    const response = await request(bulkQuery(filters));
    if (!response.ok) {
      setError(true);
      setConfirming(false);
      return;
    }
    setConfirming(false);
    startTransition(() => router.refresh());
  }

  if (!confirming) {
    return (
      <button
        type="button"
        className="delete-button toolbar-button"
        onClick={() => setConfirming(true)}
        disabled={total === 0}
        title={error ? `${idleLabel} failed, try again` : confirmPrompt(String(total))}
      >
        {icon}
        {error ? retryLabel : idleLabel}
      </button>
    );
  }

  return (
    <span className="delete-confirm-inline">
      <span className="delete-warning">{confirmPrompt(total.toLocaleString())}</span>
      <button
        type="button"
        className="delete-button delete-button-danger toolbar-button"
        onClick={run}
        disabled={isPending}
      >
        {isPending ? pendingLabel : "Confirm"}
      </button>
      <button
        type="button"
        className="delete-button toolbar-button"
        onClick={() => setConfirming(false)}
        disabled={isPending}
      >
        Cancel
      </button>
    </span>
  );
}

export default function BulkActionButtons({
  total,
  filters
}: {
  total: number;
  filters: BulkFilters;
}) {
  return (
    <>
      <BulkButton
        total={total}
        filters={filters}
        idleLabel="Replay shown"
        retryLabel="Retry replay"
        confirmPrompt={(count) => `Re-verify all ${count} shown events?`}
        pendingLabel="Replaying"
        request={(query) =>
          fetch(`/api/verification-events/replay${query}`, { method: "POST" })
        }
      />
      <BulkButton
        total={total}
        filters={filters}
        idleLabel="Delete shown"
        retryLabel="Retry delete"
        confirmPrompt={(count) => `Permanently delete all ${count} shown events?`}
        pendingLabel="Deleting"
        icon={<TrashIcon />}
        request={(query) =>
          fetch(`/api/verification-events${query}`, { method: "DELETE" })
        }
      />
    </>
  );
}
