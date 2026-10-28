"use client";

import { camelotColor } from "@/lib/camelot";
import type { Track, TransitionBreakdown } from "@/lib/types";

export function CamelotPill({ track, size = "sm" }: {
  track: Pick<Track, "camelot" | "camelot_num" | "camelot_letter">;
  size?: "sm" | "xs";
}) {
  const label = track.camelot ?? "—";
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border font-bold tabular-nums ${
        size === "xs" ? "px-1.5 py-0 text-[10px]" : "px-2 py-0.5 text-xs"
      }`}
      style={{
        color: camelotColor(track.camelot_num, track.camelot_letter),
        borderColor: camelotColor(track.camelot_num, track.camelot_letter, 0.45),
        background: camelotColor(track.camelot_num, track.camelot_letter, 0.12),
      }}
      title={track.camelot ? `Camelot ${track.camelot} (${track.camelot_letter === "B" ? "major" : "minor"})` : "Key unknown"}
    >
      {label}
    </span>
  );
}

export function EnergyBar({ value, target, width = "w-20" }: { value: number; target?: number; width?: string }) {
  return (
    <div className={`relative h-1.5 ${width} rounded-full bg-surface-3`} role="meter" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100} aria-label="Energy">
      <div className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-accent-lo to-accent-hi" style={{ width: `${value * 100}%` }} />
      {target != null && (
        <div className="absolute -top-1 h-3.5 w-0.5 rounded bg-amber" style={{ left: `calc(${target * 100}% - 1px)` }} title={`Target ${(target * 100).toFixed(0)}`} />
      )}
    </div>
  );
}

export function ScoreChip({ score, label = "match" }: { score: number; label?: string }) {
  const pct = Math.round(score * 100);
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-accent/15 px-2 py-0.5 text-xs font-bold text-accent-hi tabular-nums">
      {pct}% <span className="font-medium text-text-2">{label}</span>
    </span>
  );
}

const COMPONENT_LABELS: [keyof TransitionBreakdown, string][] = [
  ["harmonic", "Harmonic (key)"],
  ["tempo", "Tempo"],
  ["energy_smoothness", "Energy flow"],
  ["timbre", "Timbre"],
  ["loudness", "Loudness"],
  ["era", "Era"],
];

export function BreakdownBars({ t }: { t: TransitionBreakdown }) {
  return (
    <dl className="grid grid-cols-[auto_1fr_auto] items-center gap-x-2 gap-y-1.5 text-xs">
      {COMPONENT_LABELS.map(([k, label]) => {
        const v = t[k] as number;
        return (
          <div key={k} className="contents">
            <dt className="text-text-2">{label}</dt>
            <dd className="h-1.5 rounded-full bg-surface-3">
              <div className="h-full rounded-full bg-accent-hi" style={{ width: `${Math.max(2, v * 100)}%` }} />
            </dd>
            <dd className="text-right font-semibold tabular-nums text-text">{v.toFixed(2)}</dd>
          </div>
        );
      })}
      <div className="contents">
        <dt className="text-text-2">Energy Δ</dt>
        <dd />
        <dd className="text-right font-semibold tabular-nums text-text">
          {t.energy_delta >= 0 ? "+" : ""}{t.energy_delta.toFixed(2)}
        </dd>
      </div>
    </dl>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton rounded-md ${className}`} aria-hidden="true" />;
}

export function PlayButton({ onClick, label }: { onClick: () => void; label: string }) {
  return (
    <button
      type="button"
      onClick={(e) => { e.stopPropagation(); onClick(); }}
      aria-label={label}
      className="flex h-10 w-10 items-center justify-center rounded-full bg-accent text-white shadow-lg shadow-black/40 transition hover:scale-105 hover:bg-accent-hi focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-hi"
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 4.5v15l13-7.5z" /></svg>
    </button>
  );
}
