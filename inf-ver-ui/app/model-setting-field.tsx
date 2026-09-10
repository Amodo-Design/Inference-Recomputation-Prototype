"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import type { ModelRead } from "./lib";

/** Inline editor for one nullable numeric model setting (verification
 * threshold or delta max). Both live on the ledger's model row — mutable,
 * outside the identity hash. Blank = NULL; what NULL means is per setting
 * (threshold: verification paused; delta max: runner default). */
export default function ModelSettingField({
  model,
  field,
  endpoint,
  placeholder,
  nullNote,
  setNote
}: {
  model: ModelRead;
  field: "verification_threshold" | "delta_max";
  /** /api path segment after the model id, e.g. "threshold" | "delta-max" */
  endpoint: string;
  placeholder: string;
  nullNote: string;
  setNote: string;
}) {
  const router = useRouter();
  const current = model[field];
  const initial = current === null ? "" : current.toFixed(4);
  const [value, setValue] = useState(initial);
  const [note, setNoteText] = useState<string | null>(null);
  const [isError, setIsError] = useState(false);
  const [isPending, startTransition] = useTransition();

  async function save(nextValue: number | null) {
    setNoteText(null);
    setIsError(false);
    const response = await fetch(
      `/api/models/${encodeURIComponent(model.model_id)}/${endpoint}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [field]: nextValue })
      }
    );
    if (!response.ok) {
      setIsError(true);
      setNoteText(`Update failed (${response.status}).`);
      return;
    }
    const body = (await response.json()) as ModelRead;
    const saved = body[field];
    if (saved === null) {
      setValue("");
      setNoteText(nullNote);
    } else {
      setValue(saved.toFixed(4));
      setNoteText(`${setNote} ${saved.toFixed(4)}`);
    }
    startTransition(() => router.refresh());
  }

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (value.trim() === "") {
      await save(null);
      return;
    }
    const parsed = Number.parseFloat(value);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      setIsError(true);
      setNoteText("Enter a positive value, or leave blank to clear.");
      return;
    }
    await save(parsed);
  }

  return (
    <form className="target-control setting-field" onSubmit={submit}>
      <div className="target-input">
        <input
          id={`${field}-${model.model_id}`}
          aria-label={`${field} for ${model.model_name}`}
          type="number"
          inputMode="decimal"
          min="0"
          step="0.0001"
          placeholder={placeholder}
          value={value}
          onChange={(event) => setValue(event.target.value)}
        />
        <button type="submit" disabled={isPending}>
          {isPending ? "Saving" : "Apply"}
        </button>
      </div>
      {note ? (
        <p className="target-note">
          {note}
          {isError ? " ⚠" : ""}
        </p>
      ) : null}
    </form>
  );
}
