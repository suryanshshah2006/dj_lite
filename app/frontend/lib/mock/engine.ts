// Mock API: mirrors the backend's scoring on 30 real tracks so the UI can be built
// without loading the 1.09M-track dataset. Constants copy configs/transitions.yaml
// (v0.1) and configs/retrieval.yaml. Style similarity is faked with a hand-assigned
// "scene" per track (the real backend uses artist style embeddings).

import tracksJson from "./tracks.json";
import { resampleArc } from "../arcs";
import type {
  MixRequest, MixResponse, MixStep, NextRequest, NextResponse, RelevanceBreakdown, Track,
  TransitionBreakdown,
} from "../types";

export const MOCK_TRACKS = tracksJson as Track[];

const W = { harmonic: 0.3, tempo: 0.3, energy_smoothness: 0.15, timbre: 0.15, loudness: 0.05, era: 0.05 };
const HARM = { same: 1.0, adjacent: 0.9, relative: 0.9, diagonal: 0.6, two_step: 0.3, other: 0.0, null_key: 0.5 };
const REL = { audio: 0.35, style: 0.5, era: 0.15 };
const ARC_WEIGHT = 0.5;

const SCENE: Record<string, string> = {
  "The Weeknd": "pop", "Dua Lipa": "pop", "Ed Sheeran": "pop", "Billie Eilish": "pop",
  "Luis Fonsi": "latin", Avicii: "edm", Disclosure: "house", deadmau5: "edm", M83: "indie",
  "Major Lazer": "edm", "Martin Garrix": "edm", Zedd: "edm", "Clean Bandit": "house",
  "The Killers": "rock", Nirvana: "rock", "Foo Fighters": "rock", "The White Stripes": "rock",
  "Arctic Monkeys": "indie", "Guns N' Roses": "rock", Queen: "rock", TOTO: "rock",
  "Kendrick Lamar": "hiphop", "The Notorious B.I.G.": "hiphop", "Travis Scott": "hiphop",
  "Tracy Chapman": "acoustic", "Vance Joy": "acoustic", "Bon Iver": "acoustic",
  "The Dave Brubeck Quartet": "jazz", "Arijit Singh": "bollywood",
};
const NEAR: Record<string, string[]> = {
  edm: ["house", "pop"], house: ["edm", "pop"], pop: ["edm", "house", "latin", "indie"],
  rock: ["indie"], indie: ["rock", "pop", "acoustic"], acoustic: ["indie", "jazz"],
  hiphop: ["pop"], latin: ["pop"], jazz: ["acoustic"], bollywood: ["pop"],
};

const clip = (x: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, x));
const ramp = (x: number, full: number, zero: number) => clip((zero - x) / (zero - full));

function harmonic(a: Track, b: Track): number {
  if (a.camelot_num == null || b.camelot_num == null) return HARM.null_key;
  const diff = Math.abs(a.camelot_num - b.camelot_num) % 12;
  const dist = Math.min(diff, 12 - diff);
  const sameLetter = a.camelot_letter === b.camelot_letter;
  if (sameLetter && dist === 0) return HARM.same;
  if (sameLetter && dist === 1) return HARM.adjacent;
  if (!sameLetter && dist === 0) return HARM.relative;
  if (!sameLetter && dist === 1) return HARM.diagonal;
  if (sameLetter && dist === 2) return HARM.two_step;
  return HARM.other;
}

function tempo(a: Track, b: Track): number {
  const k = Math.round(Math.log2(b.bpm / a.bpm));
  const b2 = b.bpm / 2 ** k;
  const pct = Math.abs(a.bpm - b2) / Math.min(a.bpm, b2);
  return Math.exp(-((pct / 0.04) ** 2)) * (k !== 0 ? 0.9 : 1);
}

const TIMBRE = ["danceability", "acousticness", "instrumentalness", "speechiness", "valence"] as const;
const AUDIO = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"] as const;
const MEAN: Record<string, number> = {
  danceability: 0.49, energy: 0.51, speechiness: 0.08, acousticness: 0.44,
  instrumentalness: 0.25, liveness: 0.2, valence: 0.45,
};

function cosine(a: Track, b: Track, cols: readonly string[]): number {
  let dot = 0, na = 0, nb = 0;
  for (const c of cols) {
    const x = (a as unknown as Record<string, number>)[c] - MEAN[c];
    const y = (b as unknown as Record<string, number>)[c] - MEAN[c];
    dot += x * y; na += x * x; nb += y * y;
  }
  return na > 0 && nb > 0 ? dot / Math.sqrt(na * nb) : 0;
}

function era(a: Track, b: Track): number {
  if (a.year == null || b.year == null) return 1;
  const gap = Math.abs(a.year - b.year);
  return 1 - 0.5 * clip((gap - 5) / 25);
}

export function transition(a: Track, b: Track): TransitionBreakdown {
  const parts = {
    harmonic: harmonic(a, b),
    tempo: tempo(a, b),
    energy_smoothness: clip(1 - Math.abs(b.energy - a.energy)),
    timbre: clip((cosine(a, b, TIMBRE) + 1) / 2),
    loudness: ramp(Math.abs(b.loudness - a.loudness), 3, 10),
    era: era(a, b),
  };
  const score = (Object.keys(W) as (keyof typeof W)[]).reduce((s, k) => s + W[k] * parts[k], 0);
  return { score, ...parts, energy_delta: b.energy - a.energy };
}

function style(a: Track, b: Track): number {
  const sa = SCENE[a.artists[0]] ?? "pop";
  const sb = SCENE[b.artists[0]] ?? "pop";
  if (sa === sb) return 0.92;
  return NEAR[sa]?.includes(sb) ? 0.7 : 0.38;
}

export function relevance(seed: Track, c: Track): RelevanceBreakdown {
  const audio = (cosine(seed, c, AUDIO) + 1) / 2;
  const e = era(seed, c);
  const s = style(seed, c);
  return { score: REL.audio * audio + REL.style * s + REL.era * e, audio, style: s, era: e };
}

const sharesArtist = (a: Track, b: Track) => a.artists.some((x) => b.artists.includes(x));

const delay = <T,>(v: T, ms = 220) => new Promise<T>((r) => setTimeout(() => r(v), ms));

export const mockApi = {
  async search(q: string): Promise<Track[]> {
    const s = q.trim().toLowerCase();
    if (!s) return delay(MOCK_TRACKS.slice(0, 20), 120);
    const hits = MOCK_TRACKS.filter(
      (t) => t.name.toLowerCase().includes(s) || t.artists.some((a) => a.toLowerCase().includes(s)),
    );
    return delay(hits.slice(0, 20), 160);
  },

  async track(id: string): Promise<Track> {
    const t = MOCK_TRACKS.find((x) => x.id === id);
    if (!t) throw new Error(`track ${id} not found`);
    return delay(t, 80);
  },

  async next({ seed_id, n = 10, alpha = 0.5 }: NextRequest): Promise<NextResponse> {
    const seed = await this.track(seed_id);
    const items = MOCK_TRACKS.filter((c) => c.id !== seed.id && !sharesArtist(c, seed))
      .map((c) => {
        const rel = relevance(seed, c);
        const tr = transition(seed, c);
        return { track: c, final: alpha * rel.score + (1 - alpha) * tr.score, relevance: rel, transition: tr };
      })
      .sort((x, y) => y.final - x.final);
    // one track per artist
    const used = new Set<string>();
    const out = items.filter((it) => {
      if (it.track.artists.some((a) => used.has(a))) return false;
      it.track.artists.forEach((a) => used.add(a));
      return true;
    });
    return delay({ seed, alpha, items: out.slice(0, n) });
  },

  async mix({ seed_ids, arc, alpha = 0.5, length = 10 }: MixRequest): Promise<MixResponse> {
    const started = performance.now();
    const seeds = await Promise.all(seed_ids.map((id) => this.track(id)));
    const target = resampleArc(arc, length);
    const first = seeds[0];
    const steps: MixStep[] = [{
      track: first, target_energy: target[0], relevance: 1, transition: null,
      arc_fit: 1 - Math.abs(first.energy - target[0]), objective: 0,
    }];
    const usedArtists = new Set(first.artists);
    let current = first;
    for (let t = 1; t < length; t++) {
      let best: MixStep | null = null;
      for (const c of MOCK_TRACKS) {
        if (steps.some((s) => s.track.id === c.id) || c.artists.some((a) => usedArtists.has(a))) continue;
        const rel = seeds.reduce((s, sd) => s + relevance(sd, c).score, 0) / seeds.length;
        const tr = transition(current, c);
        const fit = 1 - Math.abs(c.energy - target[t]);
        const objective = alpha * rel + (1 - alpha) * tr.score + ARC_WEIGHT * fit;
        if (!best || objective > best.objective) {
          best = { track: c, target_energy: target[t], relevance: rel, transition: tr, arc_fit: fit, objective };
        }
      }
      if (!best) break; // mock pool exhausted
      steps.push(best);
      best.track.artists.forEach((a) => usedArtists.add(a));
      current = best.track;
    }
    // The mock only implements greedy (the backend defaults to beam_v1).
    return delay({ generator: "greedy_v0 (mock)", alpha, gamma: ARC_WEIGHT, beam_width: null,
                   pool_size: MOCK_TRACKS.length, runtime_ms: performance.now() - started,
                   arc: target, steps }, 380);
  },
};
