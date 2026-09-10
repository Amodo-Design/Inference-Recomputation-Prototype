"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";

export default function ReplayEventButton({ id }: { id: string }) {
  const router = useRouter();
  const [error, setError] = useState(false);
  const [busy, setBusy] = useState(false);
  const [isPending, startTransition] = useTransition();

  async function replay() {
    setError(false);
    setBusy(true);
    const response = await fetch(
      `/api/verification-events/${encodeURIComponent(id)}/replay`,
      { method: "POST" }
    );
    setBusy(false);
    // 404 = someone else already deleted/replayed it; refreshing shows that.
    if (!response.ok && response.status !== 404) {
      setError(true);
      return;
    }
    startTransition(() => router.refresh());
  }

  return (
    <button
      type="button"
      className="delete-button action-button-neutral"
      onClick={replay}
      disabled={busy || isPending}
      title={
        error
          ? "Replay failed, try again"
          : "Delete this result and re-verify the event"
      }
    >
      {busy || isPending ? "Replaying" : error ? "Retry replay" : "Replay"}
    </button>
  );
}
