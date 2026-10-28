"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import Link from "next/link";
import { api } from "@/lib/api";
import type { NextResponse, Track } from "@/lib/types";
import { TrackCard, TrackCardSkeleton } from "@/components/TrackCard";
import { BreakdownBars } from "@/components/ui";

export default function HomePage() {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<Track[] | null>(null);
  const [searching, setSearching] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [seed, setSeed] = useState<Track | null>(null);
  const [next, setNext] = useState<NextResponse | null>(null);
  const [loadingNext, setLoadingNext] = useState(false);
  const nextRef = useRef<HTMLElement>(null);

  // Debounced search
  useEffect(() => {
    setSearching(true);
    const h = setTimeout(() => {
      api.search(q)
        .then((r) => { setResults(r); setError(null); })
        .catch((e: Error) => setError(e.message))
        .finally(() => setSearching(false));
    }, 250);
    return () => clearTimeout(h);
  }, [q]);

  const choose = (t: Track) => {
    setSeed(t);
    setNext(null);
    setLoadingNext(true);
    api.next({ seed_id: t.id, n: 10, alpha: 0.5 })
      .then(setNext)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoadingNext(false));
    setTimeout(() => nextRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  };

  return (
    <div className="mx-auto max-w-[1400px]">
      <section className="relative overflow-hidden rounded-2xl border border-border bg-gradient-to-br from-accent-lo/40 via-surface to-surface px-6 py-10 sm:px-10 sm:py-14">
        <div className="pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-accent/25 blur-3xl" />
        <p className="text-sm font-semibold uppercase tracking-widest text-accent-hi">Find your next track</p>
        <h1 className="mt-2 max-w-2xl text-3xl font-extrabold tracking-tight sm:text-5xl">
          Search a song. See what mixes into it.
        </h1>
        <p className="mt-3 max-w-xl text-text-2">
          DJ-Lite ranks follow-ups by key compatibility, tempo, energy flow and artist style.
        </p>
        <label className="mt-7 flex max-w-2xl items-center gap-3 rounded-full border border-border bg-bg/80 px-5 py-4 shadow-xl shadow-black/30 focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/40">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-muted" aria-hidden="true">
            <circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" strokeLinecap="round" />
          </svg>
          <span className="sr-only">Search songs or artists</span>
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Songs or artists — try “Daft Punk”, “Everlong”, “Fast Car”"
            className="w-full bg-transparent text-lg text-text placeholder:text-muted focus:outline-none"
            autoFocus
          />
          {searching && <span className="h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-accent border-t-transparent" aria-label="Searching" />}
        </label>
      </section>

      {error && (
        <p role="alert" className="mt-6 rounded-lg border border-bad/40 bg-bad/10 px-4 py-3 text-sm text-bad">{error}</p>
      )}

      <section className="mt-10" aria-labelledby="results-h">
        <div className="mb-4 flex items-baseline justify-between">
          <h2 id="results-h" className="text-xl font-bold">{q.trim() ? `Results for “${q.trim()}”` : "Featured tracks"}</h2>
          {results && <span className="text-sm text-muted">{results.length} tracks</span>}
        </div>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
          {results === null
            ? Array.from({ length: 12 }, (_, i) => <TrackCardSkeleton key={i} />)
            : results.map((t, i) => (
                <TrackCard key={t.id} track={t} index={i} onSelect={choose} selected={seed?.id === t.id} />
              ))}
        </div>
        {results?.length === 0 && !searching && (
          <p className="mt-6 text-text-2">No tracks match “{q}”. Try an artist name instead.</p>
        )}
      </section>

      <AnimatePresence>
        {seed && (
          <motion.section
            ref={nextRef}
            key={seed.id}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-12 scroll-mt-6"
            aria-labelledby="next-h"
          >
            <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="text-sm font-semibold uppercase tracking-widest text-accent-hi">Next up</p>
                <h2 id="next-h" className="text-2xl font-extrabold">After “{seed.name}”</h2>
                <p className="text-sm text-text-2">Balanced between style match and smooth transitions (α = 0.5). Hover a card for the breakdown.</p>
              </div>
              <Link href={`/mix?seed=${seed.id}`} className="rounded-full bg-accent px-5 py-2.5 text-sm font-bold text-white transition hover:bg-accent-hi focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-hi">
                Build a mix from this →
              </Link>
            </div>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
              {loadingNext || !next
                ? Array.from({ length: 10 }, (_, i) => <TrackCardSkeleton key={i} />)
                : next.items.map((it, i) => (
                    <div key={it.track.id} className="group/next relative">
                      <TrackCard track={it.track} index={i} score={it.final} onSelect={choose} />
                      <div className="pointer-events-none absolute inset-x-2 top-[52%] z-20 rounded-xl border border-border bg-surface-2/95 p-3 opacity-0 shadow-2xl shadow-black/50 backdrop-blur transition-opacity duration-150 group-hover/next:opacity-100">
                        <p className="mb-2 text-xs font-semibold text-text">
                          Relevance {it.relevance.score.toFixed(2)} · Transition {it.transition.score.toFixed(2)}
                        </p>
                        <BreakdownBars t={it.transition} />
                      </div>
                    </div>
                  ))}
            </div>
          </motion.section>
        )}
      </AnimatePresence>
    </div>
  );
}
