"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { api } from "@/lib/api";
import type { EvalRow, EvalSummary } from "@/lib/types";
import { Skeleton } from "@/components/ui";

const STEPS = [
  {
    title: "Clean the catalogue",
    body: "We start from 1.2M songs with Spotify audio features. Broken rows (no tempo, tiny or hour-long tracks, spoken word) are removed and the same song released on many albums is merged into one — about 1.09M tracks remain.",
  },
  {
    title: "Translate keys to the Camelot wheel",
    body: "DJs mix by key using the Camelot wheel: 12 numbers × major (B) or minor (A). Neighbours on the wheel (8A → 9A) or relative keys (8A ↔ 8B) blend without clashing.",
  },
  {
    title: "Score every transition",
    body: "For any pair A → B we rate harmonic fit, tempo match (half/double time allowed), how much the energy jumps, timbre, loudness and era — combined into one 0–1 transition score you can inspect.",
  },
  {
    title: "Learn each artist's style",
    body: "Artists who share albums (compilations, features, soundtracks) tend to share a scene. From those links we learn a 64-number style fingerprint per artist — no genre labels needed.",
  },
  {
    title: "Retrieve candidates fast",
    body: "Audio features plus style fingerprints go into a vector index, so the closest 500 candidates to a song come back in milliseconds and are re-ranked exactly.",
  },
  {
    title: "Follow the energy arc",
    body: "The mix generator scores every step on relevance to your seeds (α), transition quality (1 − α) and how close the track sits to your energy curve. Instead of grabbing the best next track one at a time, a beam search keeps the 20 most promising partial mixes and picks the best whole set, from a pool of 2,000 candidates.",
  },
];

export default function HowPage() {
  const [ev, setEv] = useState<EvalSummary | null>(null);
  useEffect(() => { api.evalSummary().then(setEv).catch(() => setEv({ available: false })); }, []);

  return (
    <div className="mx-auto max-w-[1100px]">
      <p className="text-sm font-semibold uppercase tracking-widest text-accent-hi">How it works</p>
      <h1 className="mt-1 text-3xl font-extrabold tracking-tight sm:text-4xl">From audio features to a DJ-style set</h1>
      <p className="mt-2 max-w-2xl text-text-2">No raw audio is analysed — everything is built on Spotify&apos;s precomputed audio features.</p>

      <ol className="mt-10 grid gap-4 sm:grid-cols-2">
        {STEPS.map((s, i) => (
          <motion.li
            key={s.title}
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.3, delay: (i % 2) * 0.05 }}
            className="relative rounded-2xl border border-border bg-surface p-5"
          >
            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-accent/20 text-sm font-extrabold text-accent-hi">{i + 1}</span>
            <h2 className="mt-3 text-lg font-bold">{s.title}</h2>
            <p className="mt-1.5 text-sm leading-relaxed text-text-2">{s.body}</p>
          </motion.li>
        ))}
      </ol>

      <section className="mt-14" aria-labelledby="eval-h">
        <h2 id="eval-h" className="text-2xl font-extrabold">How well does it work?</h2>
        <p className="mt-1 max-w-2xl text-sm text-text-2">
          We check recommendations against genre labels from a separate dataset (never used by the model) and measure transition quality directly.
        </p>
        <div className="mt-6">
          {ev === null ? (
            <Skeleton className="h-48 w-full rounded-2xl" />
          ) : ev.available && ev.rows?.length ? (
            <EvalTable rows={ev.rows} source={ev.source} figure={api.figureUrl(ev.figure_url)} />
          ) : (
            <div className="rounded-2xl border border-dashed border-border p-8 text-center">
              <p className="font-bold">Evaluation results will appear here</p>
              <p className="mt-1 text-sm text-text-2">The evaluation runs are still in progress. Once they finish, this section shows the results table and the style-vs-transition trade-off chart.</p>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

function pct(v: number | null) {
  return v == null ? "—" : v.toFixed(3);
}

function EvalTable({ rows, source, figure }: { rows: EvalRow[]; source?: string; figure: string | null }) {
  const pools = [...new Set(rows.map((r) => r.pool))];
  return (
    <div className="space-y-8">
      {pools.map((pool) => (
        <div key={pool} className="overflow-x-auto rounded-2xl border border-border">
          {/* no ligatures: Manrope would render the "(c)" method label as © */}
          <table className="w-full min-w-[640px] text-sm [font-variant-ligatures:none]">
            <caption className="bg-surface px-4 py-3 text-left text-sm font-bold">
              {pool === "labeled" ? "Primary: candidates restricted to genre-labeled tracks" : "Secondary: full catalogue"}
            </caption>
            <thead className="bg-surface-2 text-xs uppercase tracking-wide text-text-2">
              <tr>
                <th scope="col" className="px-4 py-2.5 text-left">Method</th>
                <th scope="col" className="px-4 py-2.5 text-right">Genre agreement</th>
                <th scope="col" className="px-4 py-2.5 text-right">Transition</th>
                <th scope="col" className="px-4 py-2.5 text-right">|ΔBPM|</th>
                <th scope="col" className="px-4 py-2.5 text-right">Diversity</th>
              </tr>
            </thead>
            <tbody>
              {rows.filter((r) => r.pool === pool).map((r) => (
                <tr key={r.label} className="border-t border-border">
                  <th scope="row" className="px-4 py-2.5 text-left font-semibold">{r.label}</th>
                  <td className="px-4 py-2.5 text-right tabular-nums">
                    {pct(r.genre_agree)}
                    {r.genre_agree_lo != null && <span className="ml-1 text-xs text-muted">[{r.genre_agree_lo.toFixed(3)}, {r.genre_agree_hi?.toFixed(3)}]</span>}
                  </td>
                  <td className="px-4 py-2.5 text-right tabular-nums">{r.trans.toFixed(3)}</td>
                  <td className="px-4 py-2.5 text-right tabular-nums">{r.dbpm.toFixed(2)}</td>
                  <td className="px-4 py-2.5 text-right tabular-nums">{r.ild.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      {source && <p className="text-xs text-muted">Source: {source}</p>}
      {figure && (
        <figure className="rounded-2xl border border-border bg-surface p-4">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={figure} alt="Genre agreement versus transition score for each value of alpha, with the baseline methods marked" className="w-full rounded-lg bg-white" />
          <figcaption className="mt-2 text-xs text-muted">Each blue point is one α setting: higher α favours style match, lower α favours smooth transitions.</figcaption>
        </figure>
      )}
    </div>
  );
}
