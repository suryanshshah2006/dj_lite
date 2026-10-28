// Camelot-key colours: one hue per wheel position (30° apart), minor (A) darker
// than major (B). Used for pills and for generated cover tiles.

export function camelotHue(num: number | null): number {
  if (num == null) return 260;
  return ((num - 1) * 30 + 200) % 360;
}

export function camelotColor(num: number | null, letter: string | null, alpha = 1): string {
  const h = camelotHue(num);
  const l = letter === "B" ? 62 : 52;
  const s = num == null ? 8 : 70;
  return `hsla(${h}, ${s}%, ${l}%, ${alpha})`;
}

export function coverGradient(num: number | null, letter: string | null): string {
  const h = camelotHue(num);
  const s = num == null ? 10 : 65;
  const l1 = letter === "B" ? 48 : 38;
  return `linear-gradient(135deg, hsl(${h}, ${s}%, ${l1}%) 0%, hsl(${(h + 40) % 360}, ${s}%, ${l1 - 22}%) 100%)`;
}

export function initials(name: string): string {
  const words = name
    .replace(/\(.*?\)|\[.*?\]/g, "")
    .split(/\s+/)
    .filter((w) => /[\p{L}\p{N}]/u.test(w));
  return (words.slice(0, 2).map((w) => [...w.replace(/[^\p{L}\p{N}]/gu, "")][0] ?? "").join("") || "♪").toUpperCase();
}

export function formatDuration(ms: number): string {
  const s = Math.round(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
