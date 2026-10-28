"""Evaluate next-track methods against held-out genre labels.

Genre comes from a separate Kaggle dataset (paths.genre_csv) and is used only here.
Metrics per seed track (top-N results):
  genre_agree@N  share of genre-labeled results that share a genre with the seed
                 (a track can carry several genres in that dataset; any overlap counts)
  labeled@N      how many of the N results have a genre label at all
  trans@N        mean transition_score_v0_1(seed -> result)
  dbpm@N         mean |BPM difference| after half/double-time alignment

Usage:
    python src/djlite/evaluate.py
"""

from __future__ import annotations

import argparse
import io
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from djlite.retrieval import Recommender, load_configs, load_tracks
from djlite.transitions import tempo_octave_align

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_genres(paths: dict, ecfg: dict) -> pd.Series:
    """track_id -> frozenset of genres."""
    g = ecfg["genre"]
    path = PROJECT_ROOT / paths["genre_csv"]
    df = pd.read_csv(path, usecols=[g["id_column"], g["genre_column"]])
    return df.groupby(g["id_column"])[g["genre_column"]].agg(frozenset)


def evaluate_run(rec: Recommender, genres: list, seeds: np.ndarray, methods: list[str],
                 n: int) -> dict[str, dict[str, float]]:
    """Mean metrics over `seeds` for each method. genres[i] = frozenset or None."""
    cands = {
        "faiss": rec.retrieve(seeds),
        "audio": rec.retrieve(seeds, k=rec.rcfg["retrieval"]["k"], audio_only=True),
    }
    out = {}
    for m in methods:
        agree, labeled, trans, dbpm = [], [], [], []
        for row, i in enumerate(seeds):
            cand = None if m == "transition" else cands["audio" if m == "audio" else "faiss"][row]
            res = rec.rank(int(i), m, n, cand=cand)
            idx = res["idx"].to_numpy()
            g = [genres[j] for j in idx if genres[j] is not None]
            labeled.append(len(g))
            if g:
                agree.append(np.mean([bool(x & genres[i]) for x in g]))
            trans.append(res["transition"].mean())
            pct, _, b2 = tempo_octave_align(rec.arr["tempo"][i], rec.arr["tempo"][idx])
            dbpm.append(np.mean(np.abs(rec.arr["tempo"][i] - b2)))
        out[m] = {
            "genre_agree": float(np.mean(agree)) if agree else np.nan,
            "seeds_with_label": len(agree) / len(seeds),
            "labeled": float(np.mean(labeled)),
            "trans": float(np.mean(trans)),
            "dbpm": float(np.mean(dbpm)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=PROJECT_ROOT / "configs" / "paths.yaml", type=Path)
    parser.add_argument("--n-seed-tracks", type=int, help="override evaluation.yaml")
    parser.add_argument("--sampling-seeds", type=int, nargs="+", help="override evaluation.yaml")
    args = parser.parse_args()

    paths, rcfg, tcfg = load_configs(args.paths)
    ecfg = yaml.safe_load((PROJECT_ROOT / "configs" / "evaluation.yaml")
                          .read_text(encoding="utf-8"))
    n_seed_tracks = args.n_seed_tracks or ecfg["n_seed_tracks"]
    if args.sampling_seeds:
        ecfg["sampling_seeds"] = args.sampling_seeds
    n = ecfg["top_n"]

    tracks, style = load_tracks(paths)
    genre_map = load_genres(paths, ecfg)
    dedup = pd.read_parquet(PROJECT_ROOT / paths["processed_dir"] / "dedup_map.parquet")
    raw_ids = pd.read_csv(PROJECT_ROOT / paths["raw_csv"], usecols=["id"])["id"]

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("GENRE EVALUATION")
        print("=" * 100)
        gids = pd.Index(genre_map.index)
        print(f"genre dataset: {len(gids):,} unique track IDs, "
              f"{len(set().union(*genre_map.to_numpy())):,} genres, "
              f"{(genre_map.map(len) > 1).mean():.1%} of IDs carry >1 genre")
        print(f"  match raw tracks_features.csv by ID:   {gids.isin(raw_ids).sum():,}")
        print(f"  match cleaned data by ID:              {gids.isin(tracks['id']).sum():,}")
        print(f"  match a dropped duplicate (dedup map): {gids.isin(dedup['dropped_id']).sum():,} "
              "(not used)")

        labels = tracks["id"].map(genre_map)
        has = labels.notna().to_numpy()
        print(f"  cleaned tracks with a genre label: {has.sum():,} "
              f"({has.mean():.2%} of {len(tracks):,})")
        if has.sum() < n_seed_tracks:
            raise SystemExit("Not enough genre-labeled tracks to sample seeds.")

        methods = list(ecfg["methods"])
        rows = []
        for pool in ecfg["pools"]:
            if pool == "full":
                rec, lab = Recommender(tracks, style, rcfg, tcfg), labels.tolist()
            else:
                sub = np.flatnonzero(has)
                rec = Recommender(tracks.iloc[sub], style[sub], rcfg, tcfg)
                lab = labels.iloc[sub].tolist()
            labeled_idx = np.flatnonzero([x is not None and x == x for x in lab])
            lab = [x if isinstance(x, frozenset) else None for x in lab]
            for s in ecfg["sampling_seeds"]:
                t0 = time.time()
                rng = np.random.default_rng(s)
                seeds = rng.choice(labeled_idx, size=n_seed_tracks, replace=False)
                res = evaluate_run(rec, lab, seeds, methods, n)
                for m, r in res.items():
                    rows.append({"pool": pool, "seed": s, "method": m, **r})
                print(f"  [{pool} pool, sampling seed {s}] done in {time.time() - t0:.0f}s",
                      flush=True)

        df = pd.DataFrame(rows)
        agg = df.groupby(["pool", "method"], sort=False).agg(["mean", "std"])
        print(f"\nRESULTS: {n_seed_tracks} seed tracks x {len(ecfg['sampling_seeds'])} "
              f"sampling seeds, top-{n}, mean +- std over runs")
        print(f"exclude_same_artist={rcfg['retrieval']['exclude_same_artist']}, "
              f"K={rcfg['retrieval']['k']}, alpha={rcfg['ranking']['alpha']}, "
              f"beta={rcfg['ranking']['beta']}")
        hdr = (f"  {'pool':<8} {'method':<40} {'genre_agree@10':>16} {'labeled@10':>12} "
               f"{'seeds w/ label':>15} {'trans@10':>14} {'|dBPM|@10':>14}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for (pool, m), r in agg.iterrows():
            f = lambda c, p=3: f"{r[(c, 'mean')]:.{p}f} +- {r[(c, 'std')]:.{p}f}"  # noqa: E731
            print(f"  {pool:<8} {ecfg['methods'][m]:<40} {f('genre_agree'):>16} "
                  f"{f('labeled', 2):>12} {f('seeds_with_label', 2):>15} {f('trans'):>14} "
                  f"{f('dbpm', 2):>14}")
        print("\ngenre_agree is averaged over seeds with >=1 labeled result; "
              "'seeds w/ label' is that share.")

    report = buf.getvalue()
    out = PROJECT_ROOT / paths["reports_dir"] / "evaluation.txt"
    out.write_text(report, encoding="utf-8")
    df.to_parquet(PROJECT_ROOT / paths["reports_dir"] / "evaluation_runs.parquet", index=False)
    print(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
