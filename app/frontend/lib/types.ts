// Shared API contract between the frontend and the FastAPI backend (app/backend).
// The mock API (lib/mock/engine.ts) returns exactly these shapes.

export interface Track {
  id: string;
  name: string;
  artists: string[];
  album: string;
  year: number | null;
  duration_ms: number;
  bpm: number;
  tempo_folded: number;
  key: number;
  mode: number;
  camelot: string | null; // e.g. "8A"; null when no key was detected
  camelot_num: number | null;
  camelot_letter: "A" | "B" | null;
  energy: number;
  valence: number;
  danceability: number;
  acousticness: number;
  instrumentalness: number;
  speechiness: number;
  liveness: number;
  loudness: number;
  has_style: boolean;
}

/** transition_score_v0.1 and its components, all in [0, 1] except energy_delta. */
export interface TransitionBreakdown {
  score: number;
  harmonic: number;
  tempo: number;
  energy_smoothness: number;
  timbre: number;
  loudness: number;
  era: number;
  energy_delta: number; // signed, B - A
}

/** Relevance to the seed; style is null when either side has no style vector. */
export interface RelevanceBreakdown {
  score: number;
  audio: number;
  style: number | null;
  era: number;
}

export interface NextItem {
  track: Track;
  final: number; // alpha * relevance + (1 - alpha) * transition
  relevance: RelevanceBreakdown;
  transition: TransitionBreakdown;
}

export interface NextResponse {
  seed: Track;
  alpha: number;
  items: NextItem[];
}

export interface NextRequest {
  seed_id: string;
  n?: number;
  alpha?: number;
}

export interface MixRequest {
  seed_ids: string[];
  arc: number[]; // target energies (control points; resampled to `length`)
  alpha?: number;
  length?: number;
  generator?: "beam_v1" | "greedy_v0"; // backend default: beam_v1
}

export interface MixStep {
  track: Track;
  target_energy: number;
  relevance: number;
  /** Transition from the previous track into this one (null for the first track). */
  transition: TransitionBreakdown | null;
  arc_fit: number;
  objective: number;
}

export interface MixResponse {
  generator: string; // "beam_v1" | "greedy_v0" ("... (mock)" in mock mode)
  alpha: number;
  gamma?: number; // arc-fit weight
  beam_width?: number | null; // beam_v1 only
  pool_size?: number;
  runtime_ms?: number; // server-side generation time
  arc: number[]; // resampled target curve, one value per step
  steps: MixStep[];
}

export interface EvalRow {
  pool: string;
  label: string;
  genre_agree: number | null;
  genre_agree_lo: number | null;
  genre_agree_hi: number | null;
  genre_macro: number | null;
  trans: number;
  trans_lo: number;
  trans_hi: number;
  dbpm: number;
  ild: number;
  artist_rep: number;
}

export interface EvalSummary {
  available: boolean;
  source?: string; // which report file the rows came from
  rows?: EvalRow[];
  figure_url?: string | null;
}
