// API client. NEXT_PUBLIC_API_MODE=mock (default) uses the in-browser mock engine;
// NEXT_PUBLIC_API_MODE=live calls the FastAPI backend at NEXT_PUBLIC_API_BASE.

import { mockApi } from "./mock/engine";
import type { EvalSummary, MixRequest, MixResponse, NextRequest, NextResponse, Track } from "./types";

export const API_MODE = process.env.NEXT_PUBLIC_API_MODE === "live" ? "live" : "mock";
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
  search(q: string): Promise<Track[]> {
    return API_MODE === "mock" ? mockApi.search(q) : http(`/api/search?q=${encodeURIComponent(q)}`);
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
