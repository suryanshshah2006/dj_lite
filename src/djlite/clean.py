"""Pipeline step 1: clean the raw Spotify tracks CSV.

Reads data/raw/tracks_features.csv, applies the cleaning rules in order, and writes
data/processed/tracks_clean.parquet, data/processed/dedup_map.parquet, and
reports/cleaning_report.txt. Thresholds live in configs/cleaning.yaml.

Usage:
    python src/djlite/clean.py
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# A tag that says nothing except "this is a remaster", e.g. "2011 remaster",
# "remastered 2009", "digital remaster", "remastered version", "remasterisé en 2016".
# Mono/stereo, re-recorded, live, remix, edit, acoustic, version tags are NOT matched.
_REMASTER_CORE = (
    r"(?:\d{4}\s+)?(?:digital(?:ly)?\s+)?"
    r"re-?master(?:ed|is[ée]e?)?"
    r"(?:\s+(?:in\s+|en\s+)?\d{4})?"
    r"(?:\s+(?:version|edition))?"
)
_REMASTER_PATTERNS = [
    # "(2009 Remaster)", "[Remastered]" anywhere in the name
    re.compile(r"\s*[(\[]\s*" + _REMASTER_CORE + r"\s*[)\]]"),
    # "Song - Remastered 2011" (the whole dash segment is the tag)
    re.compile(r"\s+-\s+" + _REMASTER_CORE + r"\s*$"),
    # "Song - Original Demo; 2019 Remaster" / "Song - Live / Remastered"
    re.compile(r"\s*[;/]\s*" + _REMASTER_CORE + r"\s*$"),
]


def normalize_name(name: str) -> str:
    """Lowercase, trim, collapse whitespace, and strip remaster tags only."""
    s = re.sub(r"\s+", " ", str(name).lower()).strip()
    for pat in _REMASTER_PATTERNS:
        s = pat.sub("", s)
    return s.strip()


def fold_tempo(tempo, low: float = 80.0, high: float = 160.0):
    """Halve or double tempo until it lies in [low, high). Requires high == 2 * low.

    Works on scalars and numpy/pandas arrays. Tempo must be > 0.
    """
    if not np.isclose(high, 2 * low):
        raise ValueError("fold_tempo requires high == 2 * low")
    t = np.asarray(tempo, dtype=float)
    if np.any(~(t > 0)):
        raise ValueError("tempo must be > 0 to fold")
    k = np.floor(np.log2(t / low))
    folded = t / np.exp2(k)
    # Guard against floating-point edge cases right at the boundaries.
    folded = np.where(folded >= high, folded / 2, folded)
    folded = np.where(folded < low, folded * 2, folded)
    return folded.item() if folded.ndim == 0 else folded


def recover_year(release_date: str, min_valid: int, max_valid: int) -> int | None:
    """Year from the first 4 chars of release_date, or None if invalid."""
    s = str(release_date)[:4]
    if not s.isdigit():
        return None
    y = int(s)
    return y if min_valid <= y <= max_valid else None


def parse_list(s: str) -> list[str]:
    """Parse a Python-literal list string like "['A', 'B']" into a list."""
    value = ast.literal_eval(s)
    if not isinstance(value, list):
        raise ValueError(f"not a list: {s!r}")
    return [str(v) for v in value]


def clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Apply all cleaning rules in order. Returns (clean_df, dedup_map, log lines)."""
    log: list[str] = []
    n_start = len(df)
    log.append(f"{'Rows loaded':<58} {n_start:>10,}")

    def drop(mask: pd.Series, label: str) -> None:
        nonlocal df
        n = int(mask.sum())
        df = df.loc[~mask]
        log.append(f"{'drop: ' + label:<58} {-n:>10,}   -> {len(df):,}")

    def note(label: str, n: int) -> None:
        log.append(f"{'  (no drop) ' + label:<58} {n:>10,} rows affected")

    t, d = cfg["tempo"], cfg["duration"]

    # 1. Tempo / time signature
    drop(df["tempo"] == 0, "tempo == 0")
    drop(df["time_signature"] == 0, "time_signature == 0")
    drop(df["tempo"] < t["min_bpm"], f"tempo < {t['min_bpm']}")

    # 2. Duration
    dur_s = df["duration_ms"] / 1000
    drop(dur_s < d["min_s"], f"duration < {d['min_s']} s")
    dur_s = df["duration_ms"] / 1000
    drop(dur_s > d["max_s"], f"duration > {d['max_s']} s")

    # 3. Missing name / album
    drop(df["name"].isna(), "missing name")
    note("missing album filled with ''", int(df["album"].isna().sum()))
    df = df.assign(album=df["album"].fillna(""))

    # 4. year == 0 -> recover from release_date
    y = cfg["year"]
    zero = df["year"] == 0
    recovered = df.loc[zero, "release_date"].map(
        lambda r: recover_year(r, y["min_valid"], y["max_valid"])
    )
    ok = recovered.dropna()
    df.loc[ok.index, "year"] = ok.astype(int)
    note("year == 0 recovered from release_date", len(ok))
    drop(df.index.isin(recovered.index[recovered.isna()]), "year == 0, recovery failed")

    # 5. Spoken word
    smax = cfg["speechiness"]["max"]
    drop(df["speechiness"] > smax, f"speechiness > {smax}")

    df = df.copy()

    # 6. Loudness clip
    lmax = cfg["loudness"]["max_db"]
    df["loudness_clipped"] = df["loudness"] > lmax
    df["loudness"] = df["loudness"].clip(upper=lmax)
    note(f"loudness > {lmax} dB clipped", int(df["loudness_clipped"].sum()))

    # 7. Live flag
    lt = cfg["liveness"]["live_threshold"]
    df["is_live"] = df["liveness"] > lt
    note(f"is_live (liveness > {lt})", int(df["is_live"].sum()))

    # 8. Types
    df["time_signature"] = df["time_signature"].astype(int)
    df["artists"] = df["artists"].map(parse_list)
    df["artist_ids"] = df["artist_ids"].map(parse_list)

    # 9. Folded tempo
    df["tempo_folded"] = fold_tempo(df["tempo"].to_numpy(), t["fold_low"], t["fold_high"])

    # 10. Deduplicate on (normalized name, sorted artists), keep earliest release
    df["name_norm"] = df["name"].map(normalize_name)
    artist_key = df["artists"].map(lambda a: "\x1f".join(sorted(a)))
    order = (
        df.assign(_artist_key=artist_key)
        .sort_values(["name_norm", "_artist_key", "release_date", "id"])
    )
    kept_id = order.groupby(["name_norm", "_artist_key"], sort=False)["id"].transform("first")
    is_dup = order["id"] != kept_id
    dedup_map = pd.DataFrame({
        "dropped_id": order.loc[is_dup, "id"].to_numpy(),
        "kept_id": kept_id[is_dup].to_numpy(),
    })
    drop(df["id"].isin(dedup_map["dropped_id"]), "duplicate song (name_norm + artists)")

    log.append(f"{'Final rows':<58} {len(df):>10,}   "
               f"({len(df) / n_start:.1%} of input, {n_start - len(df):,} removed)")
    return df.reset_index(drop=True), dedup_map, log


def dedup_examples(raw: pd.DataFrame, dedup_map: pd.DataFrame, n: int, seed: int,
                   names_differ: bool) -> str:
    """Format n random dropped->kept pairs for manual inspection."""
    cols = ["id", "name", "artists", "release_date"]
    pairs = (
        dedup_map
        .merge(raw[cols].add_prefix("dropped_"), left_on="dropped_id", right_on="dropped_id")
        .merge(raw[cols].add_prefix("kept_"), left_on="kept_id", right_on="kept_id")
    )
    if names_differ:
        pairs = pairs[pairs["dropped_name"] != pairs["kept_name"]]
    sample = pairs.sample(min(n, len(pairs)), random_state=seed)
    lines = []
    for _, r in sample.iterrows():
        lines.append(f"  artists: {r['kept_artists']}")
        lines.append(f"    KEPT    {r['kept_release_date']:<10}  {r['kept_name']}")
        lines.append(f"    DROPPED {r['dropped_release_date']:<10}  {r['dropped_name']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=PROJECT_ROOT / "configs" / "paths.yaml", type=Path)
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "cleaning.yaml", type=Path)
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text(encoding="utf-8"))["paths"]
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    csv_path = PROJECT_ROOT / paths["raw_csv"]
    print(f"Loading {csv_path} ...")
    raw = pd.read_csv(csv_path)

    df, dedup_map, log = clean(raw, cfg)

    out_dir = PROJECT_ROOT / paths["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "tracks_clean.parquet", index=False)
    dedup_map.to_parquet(out_dir / "dedup_map.parquet", index=False)

    rep = cfg["report"]
    n_groups = dedup_map["kept_id"].nunique()
    report = "\n".join([
        "CLEANING REPORT (rules applied in order)",
        "=" * 80,
        *log,
        "",
        f"Deduplication: {len(dedup_map):,} rows merged into {n_groups:,} kept songs.",
        f"Columns: {list(df.columns)}",
        "",
        f"{rep['n_dedup_examples']} RANDOM DEDUP MERGES",
        "=" * 80,
        dedup_examples(raw, dedup_map, rep["n_dedup_examples"], rep["random_seed"], False),
        "",
        f"{rep['n_dedup_examples']} RANDOM DEDUP MERGES WHERE THE RAW NAMES DIFFER",
        "=" * 80,
        dedup_examples(raw, dedup_map, rep["n_dedup_examples"], rep["random_seed"], True),
        "",
    ])
    report_path = PROJECT_ROOT / paths["reports_dir"] / "cleaning_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Wrote {out_dir / 'tracks_clean.parquet'}")
    print(f"Wrote {out_dir / 'dedup_map.parquet'}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
