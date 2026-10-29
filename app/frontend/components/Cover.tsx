"use client";

import { useEffect, useState } from "react";
import { api, COVERS_ENABLED } from "@/lib/api";
import { coverGradient, initials } from "@/lib/camelot";
import type { Track } from "@/lib/types";

// Session-level cache + a small client-side queue so only displayed covers are
// requested, a few at a time. The server side throttles to ~5 req/s and caches.
const urlCache = new Map<string, string | null>();
const inflight = new Map<string, Promise<string | null>>();
let active = 0;
const queue: (() => void)[] = [];
const MAX_CONCURRENT = 4;

function schedule<T>(fn: () => Promise<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const run = () => {
      active++;
      fn().then(resolve, reject).finally(() => {
        active--;
        queue.shift()?.();
      });
    };
    if (active < MAX_CONCURRENT) run();
    else queue.push(run);
  });
}

function loadCover(id: string): Promise<string | null> {
  if (urlCache.has(id)) return Promise.resolve(urlCache.get(id)!);
  if (!inflight.has(id)) {
    inflight.set(
      id,
      schedule(() => api.cover(id))
        .catch(() => null)
        .then((u) => {
          urlCache.set(id, u);
          inflight.delete(id);
          return u;
        }),
    );
  }
  return inflight.get(id)!;
}

export function Cover({ track, className = "", rounded = "rounded-lg" }: {
  track: Pick<Track, "id" | "name" | "camelot_num" | "camelot_letter">;
  className?: string;
  rounded?: string;
}) {
  // url: undefined = not known yet, null = no cover. `loaded` is keyed by URL so an image
  // that finished loading (e.g. from the browser cache, before this effect ran) stays shown.
  const [url, setUrl] = useState<string | null | undefined>(
    COVERS_ENABLED ? urlCache.get(track.id) : null);
  const [loaded, setLoaded] = useState<string | null>(null);
  const status = url === null ? "fallback" : url && loaded === url ? "image" : "loading";

  useEffect(() => {
    let alive = true;
    if (!COVERS_ENABLED) return;  // gradient tiles only; no cover requests at all
    if (urlCache.has(track.id)) {
      setUrl(urlCache.get(track.id)!);
      return;
    }
    setUrl(undefined);
    loadCover(track.id).then((u) => { if (alive) setUrl(u); });
    return () => { alive = false; };
  }, [track.id]);

  return (
    // size container: the initials' cqw font size scales with the tile, not the viewport
    <div className={`relative aspect-square overflow-hidden [container-type:inline-size] ${rounded} ${className}`}>
      {status !== "image" && (
        status === "fallback" ? (
          <div
            className="absolute inset-0 flex items-center justify-center"
            style={{ background: coverGradient(track.camelot_num, track.camelot_letter) }}
            aria-label={`${track.name} (no cover art)`}
            role="img"
          >
            <span className="select-none text-[clamp(14px,30cqw,44px)] font-extrabold tracking-tight text-white/90 drop-shadow">
              {initials(track.name)}
            </span>
          </div>
        ) : (
          <div className="skeleton absolute inset-0" aria-hidden="true" />
        )
      )}
      {url && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={url}
          alt={`Cover art for ${track.name}`}
          loading="lazy"
          onLoad={() => setLoaded(url)}
          onError={() => { urlCache.set(track.id, null); setUrl(null); }}
          className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-300 ${status === "image" ? "opacity-100" : "opacity-0"}`}
        />
      )}
    </div>
  );
}
