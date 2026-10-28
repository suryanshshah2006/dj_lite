"""Sanity check for features + transition scorer v0.

Writes reports/transition_sanity.txt and two figures in reports/figures/:
camelot_distribution.png and transition_score_hist.png.

Usage:
    python src/djlite/sanity_transitions.py
"""

from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from djlite.transitions import COMPONENTS, score_candidates, transition_score_v0  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Reference data-viz palette (light mode).
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

SHORT = {"harmonic": "harm", "tempo": "tempo", "energy_smoothness": "energy",
         "timbre": "timbre", "loudness": "loud", "era": "era"}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_camelot(counts: pd.Series, path: Path) -> None:
    nums = np.arange(1, 13)
    a = [counts.get(f"{n}A", 0) for n in nums]
    b = [counts.get(f"{n}B", 0) for n in nums]
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)
    _style(ax)
    w = 0.4
    ax.bar(nums - w / 2, a, w, color=SERIES_2, edgecolor=SURFACE, linewidth=2,
           label="A (minor)")
    ax.bar(nums + w / 2, b, w, color=SERIES_1, edgecolor=SURFACE, linewidth=2,
           label="B (major)")
    ax.set_xticks(nums)
    ax.set_xlabel("Camelot number", color=TEXT_2)
    ax.set_ylabel("Tracks", color=TEXT_2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_title("Camelot code distribution (cleaned dataset)", color=TEXT, loc="left")
    ax.legend(frameon=False, labelcolor=TEXT_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_hist(scores: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)
    _style(ax)
    ax.hist(scores, bins=50, range=(0, 1), color=SERIES_1, edgecolor=SURFACE, linewidth=1)
    med = float(np.median(scores))
    ax.axvline(med, color=TEXT_2, linewidth=1, linestyle="--")
    ax.text(med, ax.get_ylim()[1] * 0.95, f" median {med:.2f}", color=TEXT_2, va="top")
    ax.set_xlabel("transition_score_v0", color=TEXT_2)
    ax.set_ylabel("Pairs", color=TEXT_2)
    ax.set_title(f"transition_score_v0 over {len(scores):,} random pairs",
                 color=TEXT, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fmt_track(r: pd.Series) -> str:
    artists = ", ".join(r["artists"])
    return f"{r['name']} - {artists}"


def seed_table(seed: pd.Series, df: pd.DataFrame, cfg: dict, top_n: int) -> str:
    cands = df[df["id"] != seed["id"]]
    res = score_candidates(seed, cands, cfg)
    top = res.nlargest(top_n, "score")
    rows = cands.loc[top.index]
    lines = [
        f"SEED: {fmt_track(seed)} ({seed['year']})",
        f"      camelot={seed['camelot']}  tempo={seed['tempo']:.1f} "
        f"(folded {seed['tempo_folded']:.1f})  energy={seed['energy']:.2f}  "
        f"loudness={seed['loudness']:.1f} dB",
        f"  {'#':>2} {'track':<52} {'cam':>4} {'bpm':>6} {'nrg':>5} {'year':>5} "
        f"{'score':>6} | " + " ".join(f"{SHORT[c]:>6}" for c in COMPONENTS) + f" {'dE':>6}",
    ]
    for i, (idx, r) in enumerate(rows.iterrows(), 1):
        s = top.loc[idx]
        name = fmt_track(r)
        name = name if len(name) <= 52 else name[:49] + "..."
        lines.append(
            f"  {i:>2} {name:<52} {r['camelot']:>4} {r['tempo']:>6.1f} {r['energy']:>5.2f} "
            f"{r['year']:>5} {s['score']:>6.3f} | "
            + " ".join(f"{s[c]:>6.2f}" for c in COMPONENTS) + f" {s['energy_delta']:>+6.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=PROJECT_ROOT / "configs" / "paths.yaml", type=Path)
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "transitions.yaml",
                        type=Path)
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text(encoding="utf-8"))["paths"]
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sc = cfg["sanity"]
    fig_dir = PROJECT_ROOT / paths["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROJECT_ROOT / paths["features_dir"] / "track_features.parquet")

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("TRANSITION SCORER v0 - SANITY CHECK")
        print("=" * 100)

        # 1. Camelot distribution
        counts = df["camelot"].value_counts(dropna=False)
        print(f"\nCAMELOT DISTRIBUTION ({len(df):,} tracks, null = {df['camelot'].isna().sum()})")
        for n in range(1, 13):
            ca, cb = counts.get(f"{n}A", 0), counts.get(f"{n}B", 0)
            print(f"  {n:>2}A {ca:>8,} ({ca / len(df):5.2%})    {n:>2}B {cb:>8,} ({cb / len(df):5.2%})")
        plot_camelot(counts, fig_dir / "camelot_distribution.png")

        # 2. Random pairs
        rng = np.random.default_rng(sc["random_seed"])
        n = sc["random_pairs"]
        ia = rng.integers(0, len(df), n)
        ib = rng.integers(0, len(df), n)
        keep = ia != ib
        pa = df.iloc[ia[keep]].reset_index(drop=True)
        pb = df.iloc[ib[keep]].reset_index(drop=True)
        res = transition_score_v0(pa, pb, cfg)
        scores = res["score"]
        q = np.quantile(scores, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        print(f"\nRANDOM PAIRS (n={len(scores):,}, seed={sc['random_seed']})")
        print(f"  score mean={scores.mean():.3f} std={scores.std():.3f} "
              f"min={scores.min():.3f} max={scores.max():.3f}")
        print("  quantiles: " + "  ".join(
            f"p{p}={v:.3f}" for p, v in zip([1, 5, 25, 50, 75, 95, 99], q)))
        print("  component means: " + "  ".join(
            f"{c}={np.mean(res[c]):.3f}" for c in COMPONENTS))
        h = res["harmonic"]
        print("  harmonic levels: " + "  ".join(
            f"{v:.1f}: {np.mean(h == v):.1%}" for v in sorted(np.unique(h))))
        plot_hist(scores, fig_dir / "transition_score_hist.png")

        # 3. Seed songs
        print(f"\nTOP {sc['top_n']} NEXT TRACKS PER SEED (full dataset, seed itself excluded)")
        print("bpm = raw tempo; component columns are the per-component scores; "
              "dE = signed energy change B - A")
        for s in sc["seeds"]:
            seed = df.loc[df["id"] == s["id"]].iloc[0]
            print(f"\n[{s['label'].upper()}]")
            print(seed_table(seed, df, cfg, sc["top_n"]))

    report = buf.getvalue()
    out = PROJECT_ROOT / paths["reports_dir"] / "transition_sanity.txt"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {out}")
    print(f"Wrote {fig_dir / 'camelot_distribution.png'}")
    print(f"Wrote {fig_dir / 'transition_score_hist.png'}")


if __name__ == "__main__":
    main()
