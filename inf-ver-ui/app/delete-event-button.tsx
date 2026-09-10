"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { TrashIcon } from "./bulk-action-buttons";

export default function DeleteEventButton({ id }: { id: string }) {
  const router = useRouter();
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState(false);
  const [isPending, startTransition] = useTransition();

  async function remove() {
    setError(false);
    const response = await fetch(
      `/api/verification-events/${encodeURIComponent(id)}`,
      { method: "DELETE" }
    );
    if (!response.ok) {
      setError(true);
      setConfirming(false);
      return;
    }
    startTransition(() => router.refresh());
  }

  if (!confirming) {
    return (
      <button
        type="button"
        className="delete-button toolbar-button"
        onClick={() => setConfirming(true)}
        title={error ? "Delete failed, try again" : "Permanently delete this record"}
        aria-label="Delete this record permanently"
      >
        <TrashIcon />
        {error ? "Retry" : null}
      </button>
    );
  }

  return (
    <span className="delete-confirm">
      <button
        type="button"
        className="delete-button delete-button-danger"
        onClick={remove}
        disabled={isPending}
      >
        {isPending ? "Deleting" : "Confirm"}
      </button>
      <button
        type="button"
        className="delete-button"
        onClick={() => setConfirming(false)}
        disabled={isPending}
      >
        Cancel
      </button>
    </span>
  );
}
