"use client";

import { useId, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { BreakdownBars } from "./ui";
import type { Track, TransitionBreakdown } from "@/lib/types";

function tone(score: number) {
  if (score >= 0.8) return "border-good/40 text-good";
  if (score >= 0.6) return "border-warn/40 text-warn";
  return "border-bad/40 text-bad";
}

/** "8A → 9A · 124 → 126 BPM · 0.92" with a hover/focus tooltip of the breakdown. */
export function TransitionChip({ from, to, t }: { from: Track; to: Track; t: TransitionBreakdown }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <div className="relative flex justify-center py-1">
      <button
        type="button"
        aria-describedby={open ? id : undefined}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className={`inline-flex items-center gap-1.5 rounded-full border bg-surface px-3 py-1 text-[11px] font-semibold tabular-nums transition hover:bg-surface-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-hi ${tone(t.score)}`}
      >
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" aria-hidden="true"><path d="M12 4v16m0 0-6-6m6 6 6-6" strokeLinecap="round" strokeLinejoin="round" /></svg>
        <span className="text-text-2">{from.camelot ?? "?"} → {to.camelot ?? "?"}</span>
        <span className="text-muted">·</span>
        <span className="text-text-2">{Math.round(from.bpm)} → {Math.round(to.bpm)} BPM</span>
        <span className="text-muted">·</span>
        <span>{t.score.toFixed(2)}</span>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            id={id}
            role="tooltip"
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.12 }}
            className="absolute top-full z-30 mt-1 w-64 rounded-xl border border-border bg-surface-2 p-3 shadow-2xl shadow-black/60"
          >
            <p className="mb-2 text-xs font-bold text-text">Transition score {t.score.toFixed(2)}</p>
            <BreakdownBars t={t} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
