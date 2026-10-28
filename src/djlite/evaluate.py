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
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from djlite.clean import normalize_name, strip_featuring
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
    parser.add_argument("--version", choices=["v1", "v2"], help="override evaluation.yaml")
    parser.add_argument("--n-seed-tracks", type=int,
                        help="v1: seeds per run; v2: cap on seeds per pool (debugging)")
    parser.add_argument("--sampling-seeds", type=int, nargs="+", help="v1 only")
    parser.add_argument("--out-name", help="basename for reports/<name>.{txt,csv,md}")
    args = parser.parse_args()

    paths, rcfg, tcfg = load_configs(args.paths)
    ecfg = yaml.safe_load((PROJECT_ROOT / "configs" / "evaluation.yaml")
                          .read_text(encoding="utf-8"))
    version = args.version or ecfg.get("version", "v1")
    args.out_name = args.out_name or ("evaluation" if version == "v1" else "eval_v2")
    (main_v1 if version == "v1" else main_v2)(args, paths, rcfg, tcfg, ecfg)


def main_v1(args, paths: dict, rcfg: dict, tcfg: dict, ecfg: dict) -> None:
    """v1 protocol: ID-matched labels, n random seeds x several sampling seeds, mean +- std."""
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

        methods = [m for m in ecfg["methods"] if m != "random"]  # v1 had no baseline
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
    "- Genre labels come from a different Kaggle dataset matched by track ID (v1) or by "
    "track ID plus normalized name + primary artist (v2); a track can carry several "
    "genres there, and any overlap counts as agreement. Name-only matches cannot be "
    "verified automatically: the precision proxy (ID-matched rows) checks that the "
    "normalization routes keys correctly, not that same-title same-artist tracks are the "
    "same recording; inspect the listed examples.\n"
    "- Genres are imbalanced and some have very few labeled tracks, so macro agreement "
    "is driven partly by small genres and has wide CIs."
)


# ====================================================================================
# v2: ID + name matching, every labeled seed once, bootstrap CIs, random baseline,
#     macro genre agreement, alpha sweep.
# ====================================================================================

V2_METRICS = [  # (column, header, decimals)
    ("genre_agree", "genre agree (micro)", 3),
    ("genre_macro", "genre agree (macro)", 3),
    ("trans", "transition@10", 3),
    ("dbpm", "|dBPM|@10", 2),
    ("ild", "ILD@10", 3),
    ("artist_rep", "artist rep (uncapped)", 3),
    ("labeled", "labeled/list", 2),
]


def genre_name_keys(g: pd.DataFrame) -> pd.DataFrame:
    """Normalized (name, primary artist) keys for genre-dataset rows, using the same
    normalization as dedup (remaster tags, then featured artists)."""
    g = g.dropna(subset=["track_name", "artists"]).copy()
    g["name_norm"] = g["track_name"].map(lambda s: strip_featuring(normalize_name(s)))
    g["key_artist"] = g["artists"].str.split(";").str[0].str.strip().str.casefold()
    return g


def build_labels_v2(tracks: pd.DataFrame, clean_keys: pd.DataFrame, paths: dict,
                    ecfg: dict) -> tuple[pd.Series, pd.Series, list[str], str]:
    """Genre sets for cleaned tracks from ID matches plus (optionally) name matches.

    Returns (labels aligned to `tracks`, source in {'id','name','both',None},
    report lines, formatted name-match examples).
    """
    gc, nm = ecfg["genre"], ecfg["v2"]["name_matching"]
    g = pd.read_csv(PROJECT_ROOT / paths["genre_csv"],
                    usecols=[gc["id_column"], gc["genre_column"], "track_name", "artists",
                             "duration_ms"])
    g = g.rename(columns={gc["id_column"]: "track_id", gc["genre_column"]: "genre"})
    by_id = g.groupby("track_id")["genre"].agg(frozenset)
    dedup = pd.read_parquet(PROJECT_ROOT / paths["processed_dir"] / "dedup_map.parquet")

    lines = [f"genre dataset: {len(g):,} rows, {len(by_id):,} unique track IDs, "
             f"{g['genre'].nunique():,} genres; {(by_id.map(len) > 1).sum():,} IDs "
             f"({(by_id.map(len) > 1).mean():.1%}) carry >1 genre"]
    id_kept = by_id.index.isin(tracks["id"])
    id_dropped = by_id.index.isin(dedup["dropped_id"]) & ~id_kept
    lines.append(f"ID match: {id_kept.sum():,} genre IDs -> kept cleaned tracks "
                 f"({id_kept.mean():.1%}); {id_dropped.sum():,} match only rows dropped "
                 "as duplicates (not used)")
    id_labels = tracks["id"].map(by_id)

    name_labels = pd.Series([None] * len(tracks), index=tracks.index, dtype=object)
    examples = ""
    if nm["enabled"]:
        gk = genre_name_keys(g)
        key = ["name_norm", "key_artist"]
        ck = clean_keys.assign(key_artist=clean_keys["primary"].str.strip().str.casefold())
        counts = ck.groupby(key)["idx"].transform("size")
        n_amb = int(ck.loc[counts > 1].drop_duplicates(key).shape[0])
        if nm["skip_ambiguous"]:
            ck = ck[counts == 1]
        genre_by_key = gk.groupby(key).agg(
            genre=("genre", frozenset), g_track_id=("track_id", "first"),
            g_name=("track_name", "first"), g_artists=("artists", "first"),
            g_dur=("duration_ms", "first")).reset_index()
        m = genre_by_key.merge(ck, on=key, how="inner")
        m["gap_s"] = (m["g_dur"] - tracks["duration_ms"].to_numpy()[m["idx"]]).abs() / 1000
        if nm["max_duration_gap_s"] is not None:
            m = m[m["gap_s"] <= nm["max_duration_gap_s"]]
        name_labels.iloc[m["idx"].to_numpy()] = m["genre"].to_numpy()

        # Precision proxy: genre rows whose ID matches a cleaned track -- does the name
        # key land on that same track?
        id_rows = gk[gk["track_id"].isin(tracks["id"])].drop_duplicates("track_id")
        chk = id_rows.merge(ck, on=key, how="inner")
        same = (tracks["id"].to_numpy()[chk["idx"]] == chk["track_id"].to_numpy())
        new = m[id_labels.iloc[m["idx"]].isna().to_numpy()]
        both = m[id_labels.iloc[m["idx"]].notna().to_numpy()]
        agree_src = np.mean([bool(a & b) for a, b in
                             zip(both["genre"], id_labels.iloc[both["idx"]])]) if len(both) else np.nan
        lines += [
            f"name match (normalized name + primary artist, dedup normalization): "
            f"{len(m):,} cleaned tracks matched; {n_amb:,} ambiguous keys "
            f"{'skipped' if nm['skip_ambiguous'] else 'kept'}",
            f"  NEW tracks labeled only via name: {len(new):,}",
            f"  precision proxy: for ID-matched genre rows with a name match, the name match "
            f"hits the same track {same.mean():.1%} of the time (n={len(chk):,})",
            f"  where both sources label a track, genre sets overlap {agree_src:.1%} "
            f"(n={len(both):,})",
            "  duration gap of new name matches (s): " + "  ".join(
                f"p{int(q * 100)}={v:.1f}" for q, v in
                new["gap_s"].quantile([.5, .75, .9, .95]).items())
            + f"  (>10 s: {(new['gap_s'] > 10).mean():.1%})",
        ]
        ex = new.sample(min(nm["n_examples"], len(new)), random_state=nm["random_seed"])
        ex_lines = []
        for _, r in ex.iterrows():
            t = tracks.iloc[r["idx"]]
            ex_lines += [
                f"  GENRE  {r['g_name']}  [{r['g_artists']}]  {sorted(r['genre'])}",
                f"  OURS   {t['name']}  {list(t['artists'])}  ({t['year']})  "
                f"gap={r['gap_s']:.1f}s",
            ]
        examples = "\n".join(ex_lines)

    labels, source = [], []
    for a, b in zip(id_labels, name_labels):
        a = a if isinstance(a, frozenset) else None
        b = b if isinstance(b, frozenset) else None
        labels.append((a or frozenset()) | (b or frozenset()) or None)
        source.append("both" if a and b else "id" if a else "name" if b else None)
    labels = pd.Series(labels, index=tracks.index, dtype=object)
    source = pd.Series(source, index=tracks.index, dtype=object)
    lines.append(f"labeled cleaned tracks: {labels.notna().sum():,} "
                 f"(id only {(source == 'id').sum():,}, name only {(source == 'name').sum():,}, "
                 f"both {(source == 'both').sum():,})")
    return labels, source, lines, examples


def list_metrics(rec: Recommender, lab: list, i: int, res: pd.DataFrame,
                 uncapped: pd.DataFrame) -> dict:
    idx = res["idx"].to_numpy()
    g = [lab[j] for j in idx if lab[j] is not None]
    _, _, b2 = tempo_octave_align(rec.arr["tempo"][i], rec.arr["tempo"][idx])
    return {
        "genre_agree": float(np.mean([bool(x & lab[i]) for x in g])) if g else np.nan,
        "labeled": len(g),
        "trans": float(res["transition"].mean()),
        "dbpm": float(np.mean(np.abs(rec.arr["tempo"][i] - b2))),
        "ild": intra_list_diversity(rec.combined_vectors(idx)),
        "artist_rep": artist_repetition(rec, uncapped["idx"].to_numpy()),
        "list_len": len(idx),
    }


def per_seed_v2(rec: Recommender, lab: list, seeds: np.ndarray, ecfg: dict,
                pool: str) -> pd.DataFrame:
    """One row per (seed, method[, alpha]) with the list metrics."""
    v2, n = ecfg["v2"], ecfg["top_n"]
    cap = rec.rcfg["ranking"].get("max_per_artist")
    caps = (cap, None)
    d_alpha = rec.rcfg["ranking"]["alpha"]
    rows = []
    t0 = time.time()
    for start in range(0, len(seeds), 1000):
        chunk = seeds[start:start + 1000]
        faiss_c = rec.retrieve(chunk)
        audio_c = rec.retrieve(chunk, audio_only=True)
        for row, i in enumerate(chunk):
            i = int(i)

            def add(method, lists, alpha=np.nan):
                rows.append({"seed_idx": i, "method": method, "alpha": alpha,
                             **list_metrics(rec, lab, i, lists[cap], lists[None])})

            add("transition", rec.rank_lists(i, "transition", n, caps=caps))
            add("audio", rec.rank_lists(i, "audio", n, cand=audio_c[row], caps=caps))
            rng = np.random.default_rng(v2["random_baseline_seed"] + i)
            add("random", rec.rank_lists(i, "random", n, caps=caps, rng=rng))
            cand, parts = rec.score(i, "full", faiss_c[row])
            add("relevance", rec.select(cand, parts, parts["relevance"], n, caps))
            for a in v2["alpha_sweep"]:
                key = a * parts["relevance"] + (1 - a) * parts["transition"]
                add("full", rec.select(cand, parts, key, n, caps), alpha=a)
        done = start + len(chunk)
        el = time.time() - t0
        print(f"  [{pool}] {done:,}/{len(seeds):,} seeds, {el:.0f}s elapsed, "
              f"~{el / done * (len(seeds) - done):.0f}s left", file=sys.stderr, flush=True)
    df = pd.DataFrame(rows)
    df["is_d"] = (df["method"] == "full") & np.isclose(df["alpha"], d_alpha)
    return df


def bootstrap_v2(per_seed: pd.DataFrame, seed_genres: pd.Series, bcfg: dict) -> pd.DataFrame:
    """Point estimate + percentile CI per (method, alpha) for every metric.

    Resamples seeds with replacement (paired: the same resamples for every method).
    genre_macro = mean over genres of the mean per-seed agreement of seeds in that genre
    (multi-genre seeds count toward each of their genres).
    """
    seeds = np.sort(per_seed["seed_idx"].unique())
    n, B = len(seeds), bcfg["n_resamples"]
    rng = np.random.default_rng(bcfg["random_seed"])
    draws = rng.integers(0, n, size=(B, n))
    C = np.zeros((B + 1, n), dtype=np.float32)
    C[0] = 1.0  # row 0 = the observed sample (point estimate)
    for b in range(B):
        C[b + 1] = np.bincount(draws[b], minlength=n)
    genres = sorted(set().union(*seed_genres.loc[seeds]))
    gpos = {gname: k for k, gname in enumerate(genres)}
    inc = np.zeros((n, len(genres)), dtype=np.float32)
    for r, s in enumerate(seeds):
        for gname in seed_genres.loc[s]:
            inc[r, gpos[gname]] = 1.0
    lo_q, hi_q = (1 - bcfg["ci"]) / 2 * 100, (1 + bcfg["ci"]) / 2 * 100

    def stat(values):
        v = np.asarray(values, dtype=np.float64)
        valid = ~np.isnan(v)
        num, den = C @ np.where(valid, v, 0.0), C @ valid.astype(np.float64)
        return np.divide(num, den, out=np.full(len(C), np.nan), where=den > 0)

    def macro(values):
        v = np.asarray(values, dtype=np.float64)
        valid = ~np.isnan(v)
        num = C @ (np.where(valid, v, 0.0)[:, None] * inc)
        den = C @ (valid[:, None] * inc)
        per_g = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
        return np.nanmean(per_g, axis=1)

    out = []
    for (method, alpha, is_d), grp in per_seed.groupby(["method", "alpha", "is_d"],
                                                       dropna=False, sort=False):
        grp = grp.set_index("seed_idx").loc[seeds]
        rec = {"method": method, "alpha": alpha, "is_d": is_d, "n_seeds": n,
               "seeds_with_label": float(grp["genre_agree"].notna().mean())}
        for col, _, _ in V2_METRICS:
            boot = macro(grp["genre_agree"]) if col == "genre_macro" else stat(grp[col])
            rec[col] = boot[0]
            rec[f"{col}_lo"], rec[f"{col}_hi"] = np.nanpercentile(boot[1:], [lo_q, hi_q])
        out.append(rec)
    return pd.DataFrame(out)


def _ci(r: pd.Series, col: str, d: int) -> str:
    if pd.isna(r[col]):
        return "n/a"
    return f"{r[col]:.{d}f} [{r[f'{col}_lo']:.{d}f}, {r[f'{col}_hi']:.{d}f}]"


def _row_label(r: pd.Series, ecfg: dict) -> str:
    if r["method"] == "full":
        return (ecfg["methods"]["full"] if r["is_d"] else "(d) sweep") + f", alpha={r['alpha']:g}"
    return ecfg["methods"][r["method"]]


def v2_tables(summary: pd.DataFrame, ecfg: dict, md: bool) -> str:
    """Main-methods table and alpha-sweep table per pool, as text or markdown."""
    out = []
    order = {"transition": 0, "audio": 1, "relevance": 2, "full": 3, "random": 4}
    titles = {"labeled": "Primary: labeled pool (candidates = genre-labeled tracks)",
              "full": "Secondary: full pool (candidates = all cleaned tracks)"}
    for pool in summary["pool"].unique():
        s = summary[summary["pool"] == pool]
        main = s[(s["method"] != "full") | s["is_d"]].sort_values(
            "method", key=lambda c: c.map(order))
        sweep = s[s["method"] == "full"].sort_values("alpha")
        n_seeds = int(s["n_seeds"].iat[0])
        for title, part in [(f"{titles.get(pool, pool)}, {n_seeds:,} seeds", main),
                            (f"Alpha sweep, method (d), {pool} pool (beta = 1 - alpha)", sweep)]:
            hdrs = [h for _, h, _ in V2_METRICS] + ["seeds w/ label"]
            if md:
                out += [f"### {title}", "", "| method | " + " | ".join(hdrs) + " |",
                        "|---|" + "---:|" * len(hdrs)]
            else:
                out += ["", title, "  " + f"{'method':<44}" + "".join(f"{h:>24}" for h in hdrs)]
            for _, r in part.iterrows():
                cells = [_ci(r, c, d) for c, _, d in V2_METRICS] + [f"{r['seeds_with_label']:.2f}"]
                if md:
                    out.append(f"| {_row_label(r, ecfg)} | " + " | ".join(cells) + " |")
                else:
                    out.append(f"  {_row_label(r, ecfg):<44}" + "".join(f"{c:>24}" for c in cells))
            out.append("")
    return "\n".join(out)


def plot_pareto(summary: pd.DataFrame, ecfg: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, text, text2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    sweep_c = "#2a78d6"
    base = {"transition": ("#eb6834", "s", "(a) transition only"),
            "audio": ("#1baf7a", "^", "(b) audio only"),
            "relevance": ("#6250d6", "D", "(c) audio + style"),
            "random": ("#52514e", "X", "(r) random")}
    pools = list(summary["pool"].unique())
    fig, axes = plt.subplots(1, len(pools), figsize=(6.5 * len(pools), 5.2),
                             facecolor=surface, squeeze=False)
    for ax, pool in zip(axes[0], pools):
        s = summary[summary["pool"] == pool]
        ax.set_facecolor(surface)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=text2, length=0)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)

        def err(r):
            return ([[r["trans"] - r["trans_lo"]], [r["trans_hi"] - r["trans"]]],
                    [[r["genre_agree"] - r["genre_agree_lo"]],
                     [r["genre_agree_hi"] - r["genre_agree"]]])

        sw = s[s["method"] == "full"].sort_values("alpha")
        ax.plot(sw["trans"], sw["genre_agree"], color=sweep_c, linewidth=2, zorder=2,
                label="(d) alpha sweep")
        for _, r in sw.iterrows():
            xe, ye = err(r)
            ax.errorbar(r["trans"], r["genre_agree"], xerr=xe, yerr=ye, fmt="o", ms=8,
                        color=sweep_c, mec=surface, mew=2, elinewidth=1, capsize=0, zorder=3)
            ax.annotate(f"α={r['alpha']:g}", (r["trans"], r["genre_agree"]),
                        textcoords="offset points", xytext=(7, 5), color=text2, fontsize=9)
        for m, (c, mk, lbl) in base.items():
            r = s[s["method"] == m]
            if r.empty:
                continue
            r = r.iloc[0]
            xe, ye = err(r)
            ax.errorbar(r["trans"], r["genre_agree"], xerr=xe, yerr=ye, fmt=mk, ms=9,
                        color=c, mec=surface, mew=2, elinewidth=1, capsize=0, zorder=3,
                        label=lbl)
            ax.annotate(lbl.split(" ")[0], (r["trans"], r["genre_agree"]),
                        textcoords="offset points", xytext=(7, -12), color=text2, fontsize=9)
        n_seeds = int(s["n_seeds"].iat[0])
        ax.set_xlabel("mean transition score @10 (v0.1)", color=text2)
        ax.set_ylabel("genre agreement @10 (micro)", color=text2)
        ax.set_title(f"{pool} pool ({n_seeds:,} seeds), 95% bootstrap CIs",
                     color=text, loc="left", fontsize=11)
        ax.legend(frameon=False, labelcolor=text2, fontsize=9, loc="best")
    fig.suptitle("Relevance vs transition trade-off: alpha sweep of method (d)",
                 color=text, x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def main_v2(args, paths: dict, rcfg: dict, tcfg: dict, ecfg: dict) -> None:
    v2 = ecfg["v2"]
    tracks, style = load_tracks(paths)
    clean = pd.read_parquet(PROJECT_ROOT / paths["processed_dir"] / "tracks_clean.parquet",
                            columns=["id", "name_norm", "artists"])
    clean = clean.set_index("id").loc[tracks["id"]]
    clean_keys = pd.DataFrame({
        "idx": np.arange(len(tracks)),
        "name_norm": clean["name_norm"].to_numpy(),
        "primary": [a[0] if len(a) else "" for a in clean["artists"]],
    })
    labels, source, label_lines, examples = build_labels_v2(tracks, clean_keys, paths, ecfg)
    has = labels.notna().to_numpy()

    log = ["GENRE EVALUATION v2", "=" * 100, *label_lines, "",
           f"{v2['name_matching']['n_examples']} RANDOM NAME-ONLY MATCHES "
           "(check for false matches)", examples, ""]
    print("\n".join(log), flush=True)

    per_seed_all, summaries = [], []
    for pool in v2["pools"]:
        if pool == "full":
            rec, lab = Recommender(tracks, style, rcfg, tcfg), labels.tolist()
            seed_pos = np.flatnonzero(has)
            seed_genres = pd.Series(labels.to_numpy()[seed_pos], index=seed_pos)
        else:
            sub = np.flatnonzero(has)
            rec = Recommender(tracks.iloc[sub], style[sub], rcfg, tcfg)
            lab = labels.iloc[sub].tolist()
            seed_pos = np.arange(len(sub))
            seed_genres = pd.Series(lab, index=seed_pos)
        if args.n_seed_tracks:  # debugging cap only; the protocol uses every seed
            seed_pos = np.random.default_rng(0).choice(seed_pos, args.n_seed_tracks,
                                                       replace=False)
        t0 = time.time()
        ps = per_seed_v2(rec, lab, seed_pos, ecfg, pool)
        ps.insert(0, "pool", pool)
        per_seed_all.append(ps)
        sm = bootstrap_v2(ps, seed_genres, v2["bootstrap"])
        sm.insert(0, "pool", pool)
        summaries.append(sm)
        print(f"[{pool} pool] {len(seed_pos):,} seeds in {time.time() - t0:.0f}s", flush=True)
        del rec

    summary = pd.concat(summaries, ignore_index=True)
    per_seed = pd.concat(per_seed_all, ignore_index=True)
    rdir = PROJECT_ROOT / paths["reports_dir"]
    fig_path = PROJECT_ROOT / paths["figures_dir"] / "pareto_alpha.png"
    plot_pareto(summary, ecfg, fig_path)

    bc = v2["bootstrap"]
    settings = (f"{settings_line(rcfg)}; bootstrap {bc['n_resamples']} resamples, "
                f"{bc['ci']:.0%} percentile CIs; every labeled seed evaluated once")
    notes = ("genre agree (micro) = mean per-seed share of labeled results whose genre set "
             "overlaps the seed's; seeds with no labeled result are excluded (seeds w/ label). "
             "(macro) = mean over genres of that per-seed value, seeds counted in each of "
             "their genres. artist rep is on the uncapped top-10; all other columns on the "
             "capped list. alpha=1 ranks FAISS candidates by relevance only (= method c); "
             "alpha=0 ranks the same candidates by transition only (unlike method a, which "
             "scans the whole pool).")
    text = "\n".join(log + ["RESULTS (point estimate [95% CI])", settings,
                            v2_tables(summary, ecfg, md=False), notes, "",
                            "KNOWN LIMITATIONS", LIMITATIONS])
    md = "\n".join([
        "# Evaluation v2", "", settings, "", "## Labels", "",
        *[f"- {line}" for line in label_lines], "",
        v2_tables(summary, ecfg, md=True), notes, "",
        f"![alpha sweep](figures/{fig_path.name})", "",
        "## 20 random name-only matches", "", "```", examples, "```", "",
        "## Known limitations", "", LIMITATIONS, ""])
    (rdir / f"{args.out_name}.txt").write_text(text, encoding="utf-8")
    (rdir / f"{args.out_name}.md").write_text(md, encoding="utf-8")
    summary.to_csv(rdir / f"{args.out_name}.csv", index=False)
    per_seed.to_parquet(rdir / f"{args.out_name}_per_seed.parquet", index=False)
    print(text)
    print(f"Wrote {rdir / args.out_name}.{{txt,md,csv}} and {fig_path}")


if __name__ == "__main__":
    main()
