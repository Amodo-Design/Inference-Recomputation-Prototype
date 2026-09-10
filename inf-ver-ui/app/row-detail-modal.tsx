"use client";

import { useRef } from "react";

// "Show detail" replaces the old Database Row <details> column: the full
// row rendered as prettified JSON inside a native <dialog>.
export default function RowDetailModal({ value }: { value: unknown }) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  return (
    <>
      <button
        type="button"
        className="delete-button action-button-neutral"
        onClick={() => dialogRef.current?.showModal()}
      >
        Show detail
      </button>
      <dialog
        ref={dialogRef}
        className="row-detail-dialog"
        aria-label="Row detail"
      >
        <div className="row-detail-header">
          <strong>Row detail</strong>
          <button
            type="button"
            className="delete-button action-button-neutral"
            onClick={() => dialogRef.current?.close()}
          >
            Close
          </button>
        </div>
        <pre>{JSON.stringify(value, null, 2)}</pre>
      </dialog>
    </>
  );
}
