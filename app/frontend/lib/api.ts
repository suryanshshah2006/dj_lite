// API client. By default it calls the FastAPI backend at NEXT_PUBLIC_API_BASE.
// NEXT_PUBLIC_API_MODE=mock switches to the in-browser mock engine (30 tracks) for offline demos.

import { mockApi } from "./mock/engine";
import type { EvalSummary, MixRequest, MixResponse, NextRequest, NextResponse, Track } from "./types";

export const API_MODE = process.env.NEXT_PUBLIC_API_MODE === "mock" ? "mock" : "live";
/** NEXT_PUBLIC_COVERS=off: never fetch or show Spotify artwork or the playback embed;
 *  covers are the generated gradient tiles (public-safe screenshots and demos). */
export const COVERS_ENABLED = process.env.NEXT_PUBLIC_COVERS !== "off";
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${detail ? `: ${detail}` : ""}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  /** Empty query -> featured tracks. */
  search(q: string): Promise<Track[]> {
    if (API_MODE === "mock") return mockApi.search(q);
    return q.trim() ? http(`/api/search?q=${encodeURIComponent(q.trim())}`) : http("/api/featured");
  },
  track(id: string): Promise<Track> {
    return API_MODE === "mock" ? mockApi.track(id) : http(`/api/track/${encodeURIComponent(id)}`);
  },
  next(req: NextRequest): Promise<NextResponse> {
    return API_MODE === "mock" ? mockApi.next(req) : http("/api/next", { method: "POST", body: JSON.stringify(req) });
  },
  mix(req: MixRequest): Promise<MixResponse> {
    return API_MODE === "mock" ? mockApi.mix(req) : http("/api/mix", { method: "POST", body: JSON.stringify(req) });
  },
  evalSummary(): Promise<EvalSummary> {
    return API_MODE === "mock" ? Promise.resolve({ available: false }) : http("/api/eval");
  },
  /** Cover URL or null. Mock mode uses a Next.js route that proxies oEmbed. */
  async cover(id: string): Promise<string | null> {
    const url = API_MODE === "mock"
      ? `/api/mock/cover/${encodeURIComponent(id)}`
      : `${API_BASE}/api/cover/${encodeURIComponent(id)}`;
    const res = await fetch(url);
    if (!res.ok) return null;
    const body = (await res.json()) as { url: string | null };
    return body.url;
  },
  figureUrl(path: string | null | undefined): string | null {
    if (!path) return null;
    return path.startsWith("http") ? path : `${API_BASE}${path}`;
  },
};
