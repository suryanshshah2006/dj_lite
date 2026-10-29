"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { api } from "@/lib/api";
import { ARC_PRESETS } from "@/lib/arcs";
import { formatDuration } from "@/lib/camelot";
import type { MixResponse, Track } from "@/lib/types";
import { SeedPicker } from "@/components/SeedPicker";
import { ArcEditor } from "@/components/ArcEditor";
import { ArcChart } from "@/components/ArcChart";
import { TransitionChip } from "@/components/TransitionChip";
import { Cover } from "@/components/Cover";
import { CamelotPill, EnergyBar, PlayButton, Skeleton } from "@/components/ui";
import { usePlayer } from "@/components/PlayerContext";

// α presets from the v2 evaluation's trade-off curve (knee between 0.5 and 0.75).
const BALANCE_PRESETS = [
  { label: "DJ mode", alpha: 0.5, note: "Seamless mixing: keys and tempos line up" },
  { label: "Discovery mode", alpha: 0.75, note: "Closer to the seed's style, rougher transitions" },
];

export default function MixPage() {
  return (
    <Suspense fallback={null}>
      <MixBuilder />
    </Suspense>
  );
}

function MixBuilder() {
  const params = useSearchParams();
  const [seed, setSeed] = useState<Track | null>(null);
  const [preset, setPreset] = useState(ARC_PRESETS[0].id);
  const [points, setPoints] = useState(ARC_PRESETS[0].points);
  const [alpha, setAlpha] = useState(0.5);
  const [length, setLength] = useState(12);
  const [mix, setMix] = useState<MixResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { play } = usePlayer();

  useEffect(() => {
    const id = params.get("seed");
    if (id) api.track(id).then(setSeed).catch(() => undefined);
  }, [params]);

  const choosePreset = (id: string) => {
    setPreset(id);
    setPoints(ARC_PRESETS.find((p) => p.id === id)!.points);
  };

  const generate = () => {
    if (!seed) return;
    setLoading(true);
    setError(null);
    api.mix({ seed_ids: [seed.id], arc: points, alpha, length })
      .then(setMix)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  };

  const totalMs = mix?.steps.reduce((a, s) => a + s.track.duration_ms, 0) ?? 0;
  const meanTransition = mix && mix.steps.length > 1
    ? mix.steps.slice(1).reduce((a, s) => a + (s.transition?.score ?? 0), 0) / (mix.steps.length - 1)
    : null;

  return (
    <div className="mx-auto max-w-[1400px]">
      <header>
        <p className="text-sm font-semibold uppercase tracking-widest text-accent-hi">Mix Builder</p>
        <h1 className="mt-1 text-3xl font-extrabold tracking-tight sm:text-4xl">Shape a set, not just a playlist</h1>
        <p className="mt-2 max-w-2xl text-text-2">Pick a seed, draw the energy you want over time, and choose how much to favour style over seamless transitions.</p>
      </header>

      <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
        {/* Controls */}
        <section className="flex flex-col gap-6 rounded-2xl border border-border bg-surface p-5" aria-label="Mix settings">
          <Field label="1. Seed song">
            <SeedPicker value={seed} onChange={setSeed} />
          </Field>

          <Field label="2. Energy arc" hint="Pick a preset or drag the points">
            <div className="mb-3 flex flex-wrap gap-2" role="radiogroup" aria-label="Arc presets">
              {ARC_PRESETS.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  role="radio"
                  aria-checked={preset === p.id}
                  onClick={() => choosePreset(p.id)}
                  className={`rounded-full border px-3 py-1.5 text-xs font-semibold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-hi ${
                    preset === p.id ? "border-accent bg-accent/20 text-text" : "border-border text-text-2 hover:border-surface-3 hover:text-text"
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
            <ArcEditor points={points} onChange={(p) => { setPoints(p); setPreset("custom"); }} />
          </Field>

          <Field label="3. Balance" hint="Preset or fine-tune with the slider">
            <div className="mb-3 grid grid-cols-2 gap-2" role="radiogroup" aria-label="Balance presets">
              {BALANCE_PRESETS.map((p) => (
                <button
                  key={p.label}
                  type="button"
                  role="radio"
                  aria-checked={alpha === p.alpha}
                  onClick={() => setAlpha(p.alpha)}
                  className={`rounded-xl border px-3 py-2 text-left transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-hi ${
                    alpha === p.alpha ? "border-accent bg-accent/20" : "border-border hover:border-surface-3"
                  }`}
                >
                  <span className="block text-xs font-bold text-text">{p.label} <span className="font-medium text-muted">α {p.alpha}</span></span>
                  <span className="mt-0.5 block text-[11px] leading-snug text-text-2">{p.note}</span>
                </button>
              ))}
            </div>
            <div className="flex justify-between text-xs font-semibold text-text-2">
              <span>Smooth transitions</span><span>Style match</span>
            </div>
            <input
              type="range" min={0} max={1} step={0.05} value={alpha}
              onChange={(e) => setAlpha(Number(e.target.value))}
              className="dj-range mt-2 w-full"
              aria-label="Style match versus smooth transitions"
              aria-valuetext={`alpha ${alpha.toFixed(2)}`}
            />
            <p className="mt-1.5 text-xs text-muted">α = {alpha.toFixed(2)} · {Math.round(alpha * 100)}% style match, {Math.round((1 - alpha) * 100)}% transition quality</p>
          </Field>

          <Field label="4. Length">
            <div className="flex items-center gap-3">
              <input
                type="range" min={4} max={30} step={1} value={length}
                onChange={(e) => setLength(Number(e.target.value))}
                className="dj-range w-full" aria-label="Playlist length"
              />
              <span className="w-16 text-right text-sm font-bold tabular-nums">{length} tracks</span>
            </div>
          </Field>

          <button
            type="button"
            onClick={generate}
            disabled={!seed || loading}
            className="flex items-center justify-center gap-2 rounded-full bg-accent px-6 py-3.5 text-sm font-extrabold text-white shadow-lg shadow-accent/25 transition hover:bg-accent-hi disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-hi"
          >
            {loading && <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />}
            {loading ? "Building your mix…" : seed ? "Generate mix" : "Choose a seed song first"}
          </button>
          {error && <p role="alert" className="rounded-lg border border-bad/40 bg-bad/10 px-3 py-2 text-sm text-bad">{error}</p>}
        </section>

        {/* Results */}
        <section aria-label="Generated mix" className="min-w-0">
          {!mix && !loading && (
            <div className="flex h-full min-h-80 flex-col items-center justify-center rounded-2xl border border-dashed border-border p-10 text-center">
              <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="var(--surface-3)" strokeWidth="1.5" aria-hidden="true"><path d="M3 17c3 0 4-10 7-10s3 8 6 8 3-5 5-5" strokeLinecap="round" /></svg>
              <p className="mt-4 font-bold">Your mix will appear here</p>
              <p className="mt-1 max-w-sm text-sm text-text-2">Each step picks the best next track for the arc, scored on relevance to your seed and how smoothly it mixes in.</p>
            </div>
          )}

          {loading && (
            <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
              <div className="space-y-3">{Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-16 w-full rounded-xl" />)}</div>
              <Skeleton className="h-64 rounded-2xl" />
            </div>
          )}

          {mix && !loading && (
            <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
              <div className="min-w-0">
                <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-text-2">
                  <span><b className="text-text">{mix.steps.length}</b> tracks</span>
                  <span>{formatDuration(totalMs)}</span>
                  {meanTransition != null && <span>avg transition <b className="tabular-nums text-text">{meanTransition.toFixed(2)}</b></span>}
                  <span className="rounded-full bg-surface-2 px-2 py-0.5 font-mono text-[11px] text-muted">
                    generator: {mix.generator}{mix.beam_width ? ` (B=${mix.beam_width})` : ""}
                    {mix.runtime_ms != null && ` · ${mix.runtime_ms < 1000 ? `${Math.round(mix.runtime_ms)} ms` : `${(mix.runtime_ms / 1000).toFixed(2)} s`}`}
                  </span>
                </div>
                <ol className="flex flex-col">
                  <AnimatePresence>
                    {mix.steps.map((s, i) => (
                      <motion.li
                        key={s.track.id}
                        initial={{ opacity: 0, x: -8 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.035 }}
                      >
                        {i > 0 && s.transition && (
                          <TransitionChip from={mix.steps[i - 1].track} to={s.track} t={s.transition} />
                        )}
                        <div className="group flex items-center gap-3 rounded-xl bg-surface p-2.5 pr-4 transition-colors hover:bg-surface-2">
                          <span className="w-6 text-right text-sm font-bold tabular-nums text-muted">{i + 1}</span>
                          <div className="relative w-12 shrink-0">
                            <Cover track={s.track} rounded="rounded-md" />
                            <div className="absolute inset-0 flex items-center justify-center opacity-0 transition group-hover:opacity-100 group-focus-within:opacity-100 [&_button]:h-8 [&_button]:w-8">
                              <PlayButton onClick={() => play(s.track)} label={`Play ${s.track.name}`} />
                            </div>
                          </div>
                          <div className="min-w-0 flex-1">
                            <p className="truncate text-sm font-bold">{s.track.name}</p>
                            <p className="truncate text-xs text-text-2">{s.track.artists.join(", ")}</p>
                          </div>
                          <span className="hidden w-16 text-right text-xs tabular-nums text-text-2 sm:block">{Math.round(s.track.bpm)} BPM</span>
                          <CamelotPill track={s.track} />
                          <div className="hidden w-24 flex-col gap-1 md:flex" title={`Energy ${s.track.energy.toFixed(2)} (target ${s.target_energy.toFixed(2)})`}>
                            <EnergyBar value={s.track.energy} target={s.target_energy} width="w-24" />
                            <span className="text-[10px] tabular-nums text-muted">energy {s.track.energy.toFixed(2)}</span>
                          </div>
                        </div>
                      </motion.li>
                    ))}
                  </AnimatePresence>
                </ol>
              </div>
              <aside className="h-fit rounded-2xl border border-border bg-surface p-4 xl:sticky xl:top-6">
                <ArcChart mix={mix} />
                <p className="mt-3 text-xs leading-relaxed text-muted">
                  Dashed amber = the arc you drew (resampled to {mix.steps.length} steps). Violet = energy of the chosen tracks. The amber tick on each energy bar marks its target.
                </p>
              </aside>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="text-sm font-bold">{label}</h2>
        {hint && <span className="text-xs text-muted">{hint}</span>}
      </div>
      {children}
    </div>
  );
}
