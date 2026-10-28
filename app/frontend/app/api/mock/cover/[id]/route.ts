// Mock-mode stand-in for the backend's GET /api/cover/{id}: fetches Spotify's public
// oEmbed thumbnail, caches results (including failures) in memory, and throttles itself
// to ~5 requests/second. The real backend caches in SQLite.

import { NextResponse } from "next/server";

const cache = new Map<string, string | null>();
const MIN_INTERVAL_MS = 200; // ~5 req/s
let nextSlot = 0;

async function throttle() {
  const now = Date.now();
  const wait = Math.max(0, nextSlot - now);
  nextSlot = Math.max(now, nextSlot) + MIN_INTERVAL_MS;
  if (wait) await new Promise((r) => setTimeout(r, wait));
}

export async function GET(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  if (!/^[A-Za-z0-9]{22}$/.test(id)) {
    return NextResponse.json({ url: null, error: "invalid id" }, { status: 400 });
  }
  if (cache.has(id)) return NextResponse.json({ url: cache.get(id), cached: true });

  await throttle();
  let url: string | null = null;
  try {
    const res = await fetch(
      `https://open.spotify.com/oembed?url=https://open.spotify.com/track/${id}`,
      { headers: { "User-Agent": "DJ-Lite (local dev)" }, cache: "no-store" },
    );
    if (res.ok) {
      const body = (await res.json()) as { thumbnail_url?: string };
      url = body.thumbnail_url ?? null;
    }
  } catch {
    url = null;
  }
  cache.set(id, url); // failures cached too
  return NextResponse.json({ url, cached: false });
}
