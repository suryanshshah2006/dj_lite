"use client";

import { useRef, useState } from "react";

const W = 560, H = 200, PAD_X = 24, PAD_Y = 18;

/** Energy-curve editor: drag control points vertically (keyboard: focus + arrows). */
export function ArcEditor({ points, onChange }: { points: number[]; onChange: (p: number[]) => void }) {
  const svg = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<number | null>(null);

  const x = (i: number) => PAD_X + (i / (points.length - 1)) * (W - 2 * PAD_X);
  const y = (v: number) => PAD_Y + (1 - v) * (H - 2 * PAD_Y);

  const setFromEvent = (i: number, clientY: number) => {
    const rect = svg.current!.getBoundingClientRect();
    const sy = ((clientY - rect.top) / rect.height) * H;
    const v = Math.min(1, Math.max(0, 1 - (sy - PAD_Y) / (H - 2 * PAD_Y)));
    const next = points.slice();
    next[i] = Math.round(v * 100) / 100;
    onChange(next);
  };

  // Smooth path through the points (Catmull-Rom → Bézier)
  const pts = points.map((v, i) => [x(i), y(v)] as const);
  let d = `M ${pts[0][0]} ${pts[0][1]}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(pts.length - 1, i + 2)];
    const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
    const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
    d += ` C ${c1[0]} ${c1[1]}, ${c2[0]} ${c2[1]}, ${p2[0]} ${p2[1]}`;
  }
  const area = `${d} L ${pts[pts.length - 1][0]} ${H - PAD_Y} L ${pts[0][0]} ${H - PAD_Y} Z`;

  return (
    <svg
      ref={svg}
      viewBox={`0 0 ${W} ${H}`}
      className="w-full touch-none select-none rounded-xl border border-border bg-bg"
      onPointerMove={(e) => drag !== null && setFromEvent(drag, e.clientY)}
      onPointerUp={() => setDrag(null)}
      onPointerLeave={() => setDrag(null)}
      role="group"
      aria-label="Energy curve editor. Focus a point and use the up and down arrow keys."
    >
      <defs>
        <linearGradient id="arc-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="var(--accent)" stopOpacity="0.45" />
          <stop offset="1" stopColor="var(--accent)" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={PAD_X} x2={W - PAD_X} y1={y(g)} y2={y(g)} stroke="var(--border)" strokeWidth="1" />
          <text x={4} y={y(g) + 3} fontSize="9" fill="var(--muted)">{Math.round(g * 100)}</text>
        </g>
      ))}
      <path d={area} fill="url(#arc-fill)" />
      <path d={d} fill="none" stroke="var(--accent-hi)" strokeWidth="2.5" strokeLinecap="round" />
      {points.map((v, i) => (
        <g key={i}>
          <circle cx={x(i)} cy={y(v)} r="16" fill="transparent" className="cursor-ns-resize"
            onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); setDrag(i); setFromEvent(i, e.clientY); }}
          />
          <circle
            cx={x(i)} cy={y(v)} r={drag === i ? 8 : 6.5}
            fill="var(--text)" stroke="var(--accent)" strokeWidth="3"
            tabIndex={0}
            role="slider"
            aria-label={`Point ${i + 1} energy`}
            aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(v * 100)}
            className="cursor-ns-resize outline-none transition-[r] focus-visible:stroke-amber"
            onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); setDrag(i); }}
            onKeyDown={(e) => {
              const step = e.shiftKey ? 0.1 : 0.02;
              if (e.key === "ArrowUp" || e.key === "ArrowDown") {
                e.preventDefault();
                const next = points.slice();
                next[i] = Math.min(1, Math.max(0, Math.round((v + (e.key === "ArrowUp" ? step : -step)) * 100) / 100));
                onChange(next);
              }
            }}
          />
        </g>
      ))}
    </svg>
  );
}
