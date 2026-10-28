"""Evaluate next-track methods against held-out genre labels.

Genre comes from a separate Kaggle dataset (paths.genre_csv) and is used only here.
Metrics per seed track (top-N results):
  genre_agree@N  share of genre-labeled results that share a genre with the seed
                 (a track can carry several genres in that dataset; any overlap counts)
  labeled@N      how many of the N results have a genre label at all
  trans@N        mean transition_score_v0_1(seed -> result)
  dbpm@N         mean |BPM difference| after half/double-time alignment
  ild@N          intra-list diversity: mean pairwise cosine distance (1 - cos) between
                 the results' combined audio + style vectors (the FAISS search space)
  artist_rep@N   artist repetition rate on the UNCAPPED top-N (max_per_artist off):
                 share of results that share an artist with a higher-ranked result
All other metrics use the capped list (max_per_artist from retrieval.yaml).

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
    cap = rec.rcfg["ranking"].get("max_per_artist")
    out = {}
    for m in methods:
        agree, labeled, trans, dbpm, ild, rep, length = [], [], [], [], [], [], []
        for row, i in enumerate(seeds):
            cand = None if m == "transition" else cands["audio" if m == "audio" else "faiss"][row]
            lists = rec.rank_lists(int(i), m, n, cand=cand, caps=(cap, None))
            res = lists[cap]
            idx = res["idx"].to_numpy()
            g = [genres[j] for j in idx if genres[j] is not None]
            labeled.append(len(g))
            if g:
                agree.append(np.mean([bool(x & genres[i]) for x in g]))
            trans.append(res["transition"].mean())
            pct, _, b2 = tempo_octave_align(rec.arr["tempo"][i], rec.arr["tempo"][idx])
            dbpm.append(np.mean(np.abs(rec.arr["tempo"][i] - b2)))
            ild.append(intra_list_diversity(rec.combined_vectors(idx)))
            rep.append(artist_repetition(rec, lists[None]["idx"].to_numpy()))
            length.append(len(idx))
        out[m] = {
            "genre_agree": float(np.mean(agree)) if agree else np.nan,
            "seeds_with_label": len(agree) / len(seeds),
            "labeled": float(np.mean(labeled)),
            "trans": float(np.mean(trans)),
            "dbpm": float(np.mean(dbpm)),
            "ild": float(np.nanmean(ild)),
            "artist_rep": float(np.mean(rep)),
            "list_len": float(np.mean(length)),
        }
    return out


def intra_list_diversity(vecs: np.ndarray) -> float:
    """Mean pairwise cosine distance between unit vectors (nan for < 2 items)."""
    k = len(vecs)
    if k < 2:
        return np.nan
    sims = vecs @ vecs.T
    return float(1.0 - sims[np.triu_indices(k, 1)].mean())


def artist_repetition(rec: Recommender, idx: np.ndarray) -> float:
    """Share of results sharing any artist with a higher-ranked result in the list."""
    seen: set = set()
    repeats = 0
    for j in idx:
        artists = rec.artist_sets[j]
        repeats += bool(artists & seen)
        seen |= artists
    return repeats / len(idx) if len(idx) else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=PROJECT_ROOT / "configs" / "paths.yaml", type=Path)
    parser.add_argument("--n-seed-tracks", type=int, help="override evaluation.yaml")
    parser.add_argument("--sampling-seeds", type=int, nargs="+", help="override evaluation.yaml")
    parser.add_argument("--out-name", default="evaluation",
                        help="basename for reports/<name>.{txt,csv,md}")
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
        table = summarize(df, ecfg)
        print(f"\nRESULTS: {n_seed_tracks} seed tracks x {len(ecfg['sampling_seeds'])} "
              f"sampling seed(s), top-{n}, mean +- std over runs")
        print(settings_line(rcfg))
        print(format_table(table, ecfg))
        print("\ngenre_agree@10 is averaged over seeds with >=1 labeled result "
              "('seeds w/ label' = that share). Primary protocol: labeled pool.")
        print("artist rep is measured on the uncapped top-10; all other columns on the "
              "capped list.\n\nKNOWN LIMITATIONS\n" + LIMITATIONS)

    report = buf.getvalue()
    rdir = PROJECT_ROOT / paths["reports_dir"]
    (rdir / f"{args.out_name}.txt").write_text(report, encoding="utf-8")
    table.to_csv(rdir / f"{args.out_name}.csv", index=False)
    (rdir / f"{args.out_name}.md").write_text(
        to_markdown(table, ecfg, n_seed_tracks, len(ecfg["sampling_seeds"]), rcfg),
        encoding="utf-8")
    df.to_parquet(rdir / f"{args.out_name}_runs.parquet", index=False)
    print(report)
    print(f"Wrote {rdir / args.out_name}.{{txt,csv,md}}")


METRICS = [  # (column, header, decimals)
    ("genre_agree", "genre agree@10", 3),
    ("trans", "transition@10", 3),
    ("dbpm", "|dBPM|@10", 2),
    ("ild", "ILD@10", 3),
    ("artist_rep", "artist rep (uncapped)", 3),
    ("labeled", "labeled/list", 2),
    ("seeds_with_label", "seeds w/ label", 2),
]


def summarize(df: pd.DataFrame, ecfg: dict) -> pd.DataFrame:
    """One row per (pool, method): <metric>_mean / <metric>_std over sampling seeds."""
    agg = df.groupby(["pool", "method"], sort=False).agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg = agg.drop(columns=["seed_mean", "seed_std"]).reset_index()
    agg.insert(2, "method_label", agg["method"].map(ecfg["methods"]))
    agg.insert(3, "n_runs", df.groupby(["pool", "method"], sort=False).size().to_numpy())
    return agg


def _pm(r: pd.Series, col: str, d: int) -> str:
    m, sd = r[f"{col}_mean"], r[f"{col}_std"]
    if pd.isna(m):
        return "n/a"
    return f"{m:.{d}f}" if pd.isna(sd) else f"{m:.{d}f} +- {sd:.{d}f}"


def settings_line(rcfg: dict) -> str:
    return (f"K={rcfg['retrieval']['k']}, alpha={rcfg['ranking']['alpha']}, "
            f"beta={rcfg['ranking']['beta']}, "
            f"exclude_same_artist={rcfg['retrieval']['exclude_same_artist']}, "
            f"max_per_artist={rcfg['ranking'].get('max_per_artist')}, "
            f"missing_style={rcfg['relevance']['missing_style']}")


def format_table(table: pd.DataFrame, ecfg: dict) -> str:
    hdr = f"  {'pool':<8} {'method':<40} " + " ".join(f"{h:>18}" for _, h, _ in METRICS)
    lines = [hdr, "  " + "-" * (len(hdr) - 2)]
    for _, r in table.iterrows():
        lines.append(f"  {r['pool']:<8} {r['method_label']:<40} "
                     + " ".join(f"{_pm(r, c, d):>18}" for c, _, d in METRICS))
    return "\n".join(lines)


def to_markdown(table: pd.DataFrame, ecfg: dict, n_seeds: int, n_runs: int,
                rcfg: dict) -> str:
    out = ["# Step 3 evaluation", "",
           f"{n_seeds} genre-labeled seed tracks x {n_runs} sampling seed(s); "
           "mean ± std over sampling seeds. Genre labels are evaluation-only.", "",
           f"Settings: {settings_line(rcfg)}", ""]
    titles = {"labeled": "Primary: labeled pool (candidates = genre-labeled tracks)",
              "full": "Secondary: full pool (candidates = all cleaned tracks)"}
    for pool in table["pool"].unique():
        out += [f"## {titles.get(pool, pool)}", "",
                "| method | " + " | ".join(h for _, h, _ in METRICS) + " |",
                "|---|" + "---:|" * len(METRICS)]
        for _, r in table[table["pool"] == pool].iterrows():
            out.append(f"| {r['method_label']} | "
                       + " | ".join(_pm(r, c, d).replace("+-", "±") for c, _, d in METRICS)
                       + " |")
        out.append("")
    out += ["genre agree@10 = share of genre-labeled results sharing a genre with the seed, "
            "averaged over seeds with ≥1 labeled result (seeds w/ label). "
            "|dBPM| is after half/double-time alignment. transition@10 = mean "
            "transition_score_v0_1(seed → result). ILD@10 = mean pairwise cosine distance "
            "between results' combined audio + style vectors. artist rep (uncapped) = share "
            "of results in the top-10 *without* the max_per_artist cap that share an artist "
            "with a higher-ranked result; every other column uses the capped list.", "",
            "## Known limitations", "",
            LIMITATIONS, ""]
    return "\n".join(out)


LIMITATIONS = (
    "- **FAISS missing-style retrieval bias.** Retrieval searches "
    "[sqrt(w_audio)·audio, sqrt(w_style)·style]. A candidate without a style vector "
    "(artist on no shared album; ~38% of tracks) has a zero style block, so for a seed "
    "*with* style its inner product is only w_audio·cos_audio and it ranks below "
    "styled candidates in the top-K. Exact re-ranking renormalizes the relevance for "
    "such candidates (0.70 audio / 0.30 era), but it can only re-rank what FAISS "
    "returned, so style-less tracks are under-represented for methods (c) and (d). "
    "For a seed without style, the query's style block is zero and retrieval is "
    "audio-only, which is unbiased.\n"
    "- Method (a) scans the whole pool; (b) uses an audio-only index; (c) and (d) "
    "re-rank the top K=500 from the combined index.\n"
    "- Genre labels come from a different Kaggle dataset matched by track ID; a track "
    "can carry several genres there, and any overlap counts as agreement."
)


if __name__ == "__main__":
    main()
