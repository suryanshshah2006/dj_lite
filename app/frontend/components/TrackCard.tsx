"use client";

import { motion } from "framer-motion";
import { Cover } from "./Cover";
import { CamelotPill, PlayButton, ScoreChip } from "./ui";
import { usePlayer } from "./PlayerContext";
import type { Track } from "@/lib/types";

export function TrackCard({ track, onSelect, selected = false, score, index = 0, footer }: {
  track: Track;
  onSelect?: (t: Track) => void;
  selected?: boolean;
  score?: number;
  index?: number;
  footer?: React.ReactNode;
}) {
  const { play } = usePlayer();
  const select = () => onSelect?.(track);
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, delay: Math.min(index, 12) * 0.03 }}
      role={onSelect ? "button" : undefined}
      tabIndex={onSelect ? 0 : undefined}
      aria-pressed={onSelect ? selected : undefined}
      aria-label={`${track.name} by ${track.artists.join(", ")}${onSelect ? " — show what plays next" : ""}`}
      onClick={select}
      onKeyDown={(e) => {
        if (onSelect && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); select(); }
      }}
      className={`group relative rounded-xl p-3 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-hi ${
        onSelect ? "cursor-pointer" : ""
      } ${selected ? "bg-surface-3 ring-1 ring-accent" : "bg-surface hover:bg-surface-2"}`}
    >
      <div className="relative">
        <Cover track={track} className="shadow-lg shadow-black/40" />
        {score != null && (
          <div className="absolute left-2 top-2"><ScoreChip score={score} /></div>
        )}
        <div className="absolute bottom-2 right-2 translate-y-2 opacity-0 transition-all duration-200 group-hover:translate-y-0 group-hover:opacity-100 group-focus-within:translate-y-0 group-focus-within:opacity-100">
          <PlayButton onClick={() => play(track)} label={`Play ${track.name}`} />
        </div>
      </div>
      <p className="mt-3 truncate text-sm font-bold text-text" title={track.name}>{track.name}</p>
      <p className="truncate text-xs text-text-2" title={track.artists.join(", ")}>{track.artists.join(", ")}</p>
      <div className="mt-2 flex items-center gap-2 text-xs text-muted">
        <CamelotPill track={track} size="xs" />
        <span className="tabular-nums">{Math.round(track.bpm)} BPM</span>
        {track.year && <span className="tabular-nums">· {track.year}</span>}
      </div>
      {footer}
    </motion.div>
  );
}

export function TrackCardSkeleton() {
  return (
    <div className="rounded-xl bg-surface p-3" aria-hidden="true">
      <div className="skeleton aspect-square rounded-lg" />
      <div className="skeleton mt-3 h-3.5 w-3/4 rounded" />
      <div className="skeleton mt-2 h-3 w-1/2 rounded" />
      <div className="skeleton mt-3 h-3 w-1/3 rounded" />
    </div>
  );
}
