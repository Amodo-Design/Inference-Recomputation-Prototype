"use client";

import { useEffect, useRef, useState } from "react";

type OutputTokenComparison = {
  index: number;
  prover_token_id: number;
  verifier_token_id: number;
  prover_text: string;
  verifier_text: string;
  exact_match: boolean;
  margin: number | null;
};

function tokenText(value: string): string {
  return value.length > 0 ? value : " ";
}

function logitDifferenceParts(value: number | null): {
  whole: string;
  fraction: string;
} | null {
  if (value === null || !Number.isFinite(value)) {
    return null;
  }

  const [whole, fraction] = value.toFixed(4).split(".");
  return { whole, fraction };
}

function verifierPreviewTokens(
  tokens: OutputTokenComparison[],
  hoveredMismatchIndex: number | null
): OutputTokenComparison[] {
  if (hoveredMismatchIndex === null) {
    return tokens;
  }

  const changedTokenOffset = tokens.findIndex((token) => token.index === hoveredMismatchIndex);
  const changedToken = tokens[changedTokenOffset];
  if (!changedToken || changedToken.exact_match) {
    return tokens;
  }

  return tokens.slice(0, changedTokenOffset + 1);
}

function tokenOffset(tokens: OutputTokenComparison[], tokenIndex: number | null): number {
  if (tokenIndex === null) {
    return -1;
  }

  return tokens.findIndex((token) => token.index === tokenIndex);
}

function nextMismatchIndex(tokens: OutputTokenComparison[], startOffset: number): number | null {
  const mismatch = tokens.slice(startOffset).find((token) => !token.exact_match);
  return mismatch?.index ?? null;
}

// The expanded view below renders one interactive span per output token
// (prover row + verifier row), each with hover/focus handlers. For a table
// page of 25 rows with long completions that is many thousands of hydrated
// DOM nodes — expensive on every refresh even when nobody looks at the
// diff. Rendered only once a row's <details> is actually opened, and
// unmounted again on close, so a page of collapsed rows stays cheap.
function ExpandedDiff({ comparison }: { comparison: OutputTokenComparison[] }) {
  const [hoveredMismatchIndex, setHoveredMismatchIndex] = useState<number | null>(null);
  const verifierRowRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (hoveredMismatchIndex === null || verifierRowRef.current === null) {
      return;
    }

    const verifierRow = verifierRowRef.current;
    verifierRow.scrollTop = verifierRow.scrollHeight;
    verifierRow.scrollLeft = verifierRow.scrollWidth;
  }, [hoveredMismatchIndex]);

  const mismatchTokens = comparison.filter((token) => !token.exact_match);
  const mismatches = mismatchTokens.length;
  const verifierTokens = verifierPreviewTokens(
    comparison,
    hoveredMismatchIndex
  );
  const isVerifierPreview = hoveredMismatchIndex !== null;
  const hoveredMismatchOffset = tokenOffset(comparison, hoveredMismatchIndex);
  const selectedMismatch = mismatchTokens.find((token) => token.index === hoveredMismatchIndex) ?? null;
  const selectedMismatchOrdinal = selectedMismatch
    ? mismatchTokens.findIndex((token) => token.index === selectedMismatch.index) + 1
    : null;
  const selectedLogitDifference = logitDifferenceParts(selectedMismatch?.margin ?? null);

  return (
    <div
      className="token-diff"
      aria-label="Output token comparison"
      onMouseLeave={() => setHoveredMismatchIndex(null)}
    >
      <div className="token-row-label">Prover</div>
      <div className="token-row-frame">
        <pre className="token-row">
          {comparison.map((token, offset) => {
            const isAfterHoveredMismatch =
              hoveredMismatchOffset >= 0 && offset > hoveredMismatchOffset;
            const selectedTokenMismatch = token.index === hoveredMismatchIndex;
            const tokenStateClass = isVerifierPreview
              ? selectedTokenMismatch
                ? "token-diff-prover"
                : "token-match"
              : token.exact_match
                ? "token-match"
                : "token-diff-prover";

            return (
              <span
                className={`token-piece token-piece-hoverable ${tokenStateClass} ${
                  isAfterHoveredMismatch ? "token-piece-muted" : ""
                }`}
                key={`prover-${token.index}`}
                tabIndex={0}
                onBlur={() => setHoveredMismatchIndex(null)}
                onFocus={() => setHoveredMismatchIndex(nextMismatchIndex(comparison, offset))}
                onMouseEnter={() => setHoveredMismatchIndex(nextMismatchIndex(comparison, offset))}
              >
                {tokenText(token.prover_text)}
              </span>
            );
          })}
        </pre>
        <div className="token-selection-metrics" aria-live="polite">
          <div>
            <span>Changed</span>
            <strong>
              {selectedMismatchOrdinal === null
                ? `- / ${mismatches.toLocaleString()}`
                : `${selectedMismatchOrdinal.toLocaleString()} / ${mismatches.toLocaleString()}`}
            </strong>
          </div>
          <div>
            <span>Logit Difference</span>
            {selectedLogitDifference ? (
              <strong className="token-logit-value">
                <span>{selectedLogitDifference.whole}</span>
                <span>.</span>
                <span>{selectedLogitDifference.fraction}</span>
              </strong>
            ) : (
              <strong className="token-logit-placeholder">-</strong>
            )}
          </div>
        </div>
      </div>
      <div className="token-row-label">
        {isVerifierPreview ? "Verifier token preview" : "Verifier"}
      </div>
      <pre className="token-row" ref={verifierRowRef}>
        {verifierTokens.map((token) => {
          const isChangedToken = token.index === hoveredMismatchIndex;
          const text = isVerifierPreview && !isChangedToken ? token.prover_text : token.verifier_text;

          return (
            <span
              className={`token-piece ${
                isVerifierPreview
                  ? isChangedToken
                    ? "token-diff-verifier"
                    : "token-match"
                  : token.exact_match
                    ? "token-match"
                    : "token-diff-verifier"
              }`}
              key={`verifier-${token.index}`}
            >
              {tokenText(text)}
            </span>
          );
        })}
      </pre>
    </div>
  );
}

export default function OutputDiffDetails({ comparison }: { comparison: OutputTokenComparison[] }) {
  const [isOpen, setIsOpen] = useState(false);

  if (comparison.length === 0) {
    return <span className="muted">-</span>;
  }

  const mismatches = comparison.reduce((count, token) => count + (token.exact_match ? 0 : 1), 0);

  return (
    <details
      className="diff-details"
      open={isOpen}
      onToggle={(event) => setIsOpen(event.currentTarget.open)}
    >
      <summary>
        {mismatches.toLocaleString()} / {comparison.length.toLocaleString()} different
      </summary>
      {isOpen ? <ExpandedDiff comparison={comparison} /> : null}
    </details>
  );
}
