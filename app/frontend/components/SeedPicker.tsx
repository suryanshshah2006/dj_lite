"use client";

import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { Track } from "@/lib/types";
import { Cover } from "./Cover";
import { CamelotPill } from "./ui";

/** Search-as-you-type combobox for choosing the seed song. */
export function SeedPicker({ value, onChange }: { value: Track | null; onChange: (t: Track | null) => void }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [hits, setHits] = useState<Track[]>([]);
  const [active, setActive] = useState(0);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const h = setTimeout(() => api.search(q).then((r) => { setHits(r.slice(0, 8)); setActive(0); }).catch(() => setHits([])), 200);
    return () => clearTimeout(h);
  }, [q, open]);

  useEffect(() => {
    const close = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const pick = (t: Track) => { onChange(t); setOpen(false); setQ(""); };

  if (value && !open) {
    return (
      <div className="flex items-center gap-3 rounded-xl border border-border bg-surface-2 p-2.5">
        <Cover track={value} className="w-12 shrink-0" rounded="rounded-md" />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-bold">{value.name}</p>
          <p className="truncate text-xs text-text-2">{value.artists.join(", ")}</p>
          <div className="mt-1 flex items-center gap-2 text-[11px] text-muted">
            <CamelotPill track={value} size="xs" /><span className="tabular-nums">{Math.round(value.bpm)} BPM · energy {value.energy.toFixed(2)}</span>
          </div>
        </div>
        <button type="button" onClick={() => setOpen(true)} className="rounded-full px-3 py-1.5 text-xs font-semibold text-accent-hi hover:bg-surface-3 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-hi">
          Change
        </button>
      </div>
    );
  }

  return (
    <div ref={box} className="relative">
      <input
        role="combobox"
        aria-expanded={open}
        aria-controls="seed-list"
        aria-autocomplete="list"
        value={q}
        autoFocus={open}
        onFocus={() => setOpen(true)}
        onChange={(e) => { setQ(e.target.value); setOpen(true); }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => Math.min(hits.length - 1, a + 1)); }
          if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(0, a - 1)); }
          if (e.key === "Enter" && hits[active]) { e.preventDefault(); pick(hits[active]); }
          if (e.key === "Escape") setOpen(false);
        }}
        placeholder="Search for a seed song…"
        className="w-full rounded-xl border border-border bg-surface-2 px-4 py-3 text-sm placeholder:text-muted focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/40"
      />
      {open && hits.length > 0 && (
        <ul id="seed-list" role="listbox" className="absolute z-30 mt-2 max-h-80 w-full overflow-auto rounded-xl border border-border bg-surface-2 p-1 shadow-2xl shadow-black/60">
          {hits.map((t, i) => (
            <li
              key={t.id}
              role="option"
              aria-selected={i === active}
              onMouseEnter={() => setActive(i)}
              onMouseDown={(e) => { e.preventDefault(); pick(t); }}
              className={`flex cursor-pointer items-center gap-3 rounded-lg p-2 ${i === active ? "bg-surface-3" : ""}`}
            >
              <Cover track={t} className="w-10 shrink-0" rounded="rounded" />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-semibold">{t.name}</p>
                <p className="truncate text-xs text-text-2">{t.artists.join(", ")}</p>
              </div>
              <CamelotPill track={t} size="xs" />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
