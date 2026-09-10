"use client";

import { useTransition } from "react";
import { useRouter } from "next/navigation";

export default function RefreshButton() {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();

  return (
    <button
      type="button"
      className="refresh-button"
      onClick={() => startTransition(() => router.refresh())}
      disabled={isPending}
      title="Reload data from the ledger"
    >
      {isPending ? "Refreshing…" : "Refresh"}
    </button>
  );
}
