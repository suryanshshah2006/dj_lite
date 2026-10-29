"use client";

import { AnimatePresence, motion } from "framer-motion";
import { usePlayer } from "./PlayerContext";
import { Cover } from "./Cover";
import { CamelotPill } from "./ui";
import { COVERS_ENABLED } from "@/lib/api";

export function PlayerBar() {
  const { current, stop } = usePlayer();
  return (
    <div className="fixed inset-x-0 bottom-0 z-40 border-t border-border bg-surface/95 backdrop-blur">
      <div className="mx-auto flex h-[92px] max-w-[1600px] items-center gap-4 px-4">
        <AnimatePresence mode="wait">
          {current ? (
            <motion.div
              key={current.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              transition={{ duration: 0.2 }}
              className="flex w-full items-center gap-4"
            >
              <div className="hidden w-56 shrink-0 items-center gap-3 lg:flex">
                <Cover track={current} className="w-14" rounded="rounded-md" />
                <div className="min-w-0">
                  <p className="truncate text-sm font-bold">{current.name}</p>
                  <p className="truncate text-xs text-text-2">{current.artists.join(", ")}</p>
                  <div className="mt-1 flex items-center gap-2 text-[11px] text-muted">
                    <CamelotPill track={current} size="xs" />
                    <span className="tabular-nums">{Math.round(current.bpm)} BPM</span>
                  </div>
                </div>
              </div>
              {COVERS_ENABLED ? (
                <iframe
                  title={`Spotify player: ${current.name}`}
                  src={`https://open.spotify.com/embed/track/${current.id}?utm_source=generator&theme=0`}
                  className="h-[80px] min-w-0 flex-1 rounded-xl"
                  allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
                  loading="lazy"
                />
              ) : (
                // the embed shows album artwork, so it is off together with covers
                <p className="flex h-[80px] min-w-0 flex-1 items-center rounded-xl border border-dashed border-border px-4 text-sm text-muted">
                  Playback preview is disabled in this build (NEXT_PUBLIC_COVERS=off).
                </p>
              )}
              <button
                type="button"
                onClick={stop}
                aria-label="Close player"
                className="rounded-full p-2 text-text-2 transition hover:bg-surface-3 hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-hi"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18" strokeLinecap="round" /></svg>
              </button>
            </motion.div>
          ) : (
            <motion.p
              key="idle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-3 text-sm text-muted"
            >
              <span className="flex h-9 w-9 items-center justify-center rounded-full border border-border">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 4.5v15l13-7.5z" /></svg>
              </span>
              Nothing playing — hit play on any track to preview it here.
            </motion.p>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
