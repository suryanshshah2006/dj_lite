// Energy-arc presets (control points in [0, 1], evenly spaced in time).

export interface ArcPreset {
  id: string;
  label: string;
  points: number[];
}

export const ARC_PRESETS: ArcPreset[] = [
  { id: "warmup-peak-cooldown", label: "Warm-up → Peak → Cool-down", points: [0.35, 0.5, 0.68, 0.85, 0.9, 0.7, 0.45] },
  { id: "steady", label: "Steady Groove", points: [0.7, 0.72, 0.7, 0.72, 0.7, 0.72, 0.7] },
  { id: "slow-build", label: "Slow Build", points: [0.25, 0.35, 0.45, 0.55, 0.68, 0.8, 0.92] },
  { id: "sunset", label: "Sunset", points: [0.75, 0.68, 0.6, 0.5, 0.42, 0.33, 0.25] },
];

/** Linear interpolation of control points onto `length` evenly spaced steps. */
export function resampleArc(points: number[], length: number): number[] {
  if (length <= 1) return [points[0] ?? 0.5];
  const out: number[] = [];
  for (let i = 0; i < length; i++) {
    const x = (i / (length - 1)) * (points.length - 1);
    const lo = Math.floor(x);
    const hi = Math.min(points.length - 1, lo + 1);
    const t = x - lo;
    out.push(points[lo] * (1 - t) + points[hi] * t);
  }
  return out;
}
