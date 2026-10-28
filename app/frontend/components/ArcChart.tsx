"use client";

import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { MixResponse } from "@/lib/types";

/** Target arc vs achieved energy per playlist position. */
export function ArcChart({ mix }: { mix: MixResponse }) {
  const data = mix.steps.map((s, i) => ({
    pos: i + 1,
    target: Math.round(s.target_energy * 100) / 100,
    achieved: Math.round(s.track.energy * 100) / 100,
    name: s.track.name,
  }));
  const mae = mix.steps.reduce((a, s) => a + Math.abs(s.track.energy - s.target_energy), 0) / mix.steps.length;

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-sm font-bold">Energy arc</h3>
        <span className="text-xs text-text-2">mean |gap| <b className="tabular-nums text-text">{mae.toFixed(2)}</b></span>
      </div>
      <div className="h-56 w-full">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -18 }}>
            <CartesianGrid stroke="var(--border)" vertical={false} />
            <XAxis dataKey="pos" tick={{ fill: "var(--muted)", fontSize: 11 }} axisLine={{ stroke: "var(--border)" }} tickLine={false} />
            <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tick={{ fill: "var(--muted)", fontSize: 11 }} axisLine={false} tickLine={false} />
            <Tooltip
              cursor={{ stroke: "var(--surface-3)", strokeWidth: 2 }}
              contentStyle={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12 }}
              labelStyle={{ color: "var(--text)", fontWeight: 700 }}
              itemStyle={{ color: "var(--text-2)" }}
              labelFormatter={(pos: number) => `#${pos} · ${data[pos - 1]?.name ?? ""}`}
            />
            <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-2)" }} iconType="plainline" />
            <Line type="monotone" dataKey="target" name="Target" stroke="var(--amber)" strokeWidth={2} strokeDasharray="5 4" dot={false} activeDot={{ r: 4 }} />
            <Line type="monotone" dataKey="achieved" name="Achieved" stroke="var(--accent-hi)" strokeWidth={2} dot={{ r: 3.5, fill: "var(--accent-hi)", stroke: "var(--surface)", strokeWidth: 2 }} activeDot={{ r: 6 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
