"""Read-only inventory of the raw Spotify tracks CSV.

Reports shape, columns/dtypes, missing values, duplicates, key feature
distributions, and suspicious rows. Does NOT clean or modify anything.

Usage:
    python src/djlite/data_inventory.py [--config configs/paths.yaml]
"""

from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_COLUMNS = [
    "id", "name", "album", "album_id", "artists", "artist_ids", "track_number",
    "disc_number", "explicit", "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms", "time_signature", "year", "release_date",
]

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def numeric_summary(s: pd.Series) -> None:
    q = s.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
    print(f"  mean={s.mean():.3f}  std={s.std():.3f}")
    print("  quantiles: " + "  ".join(f"p{int(k * 100)}={v:.3f}" for k, v in q.items()))


def run_inventory(df: pd.DataFrame, min_dur_s: float, max_dur_s: float) -> None:
    section("SHAPE")
    print(f"rows={len(df):,}  columns={df.shape[1]}")
    print(f"memory in pandas: {df.memory_usage(deep=True).sum() / 1e6:,.0f} MB")

    section("COLUMNS AND DTYPES")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<18} {dtype}")
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    print(f"\nmissing vs CLAUDE.md: {missing or 'none'}")
    print(f"extra vs CLAUDE.md:   {extra or 'none'}")

    section("MISSING VALUES PER COLUMN")
    na = df.isna().sum()
    for col, n in na.items():
        if n:
            print(f"  {col:<18} {n:>9,}  ({n / len(df):.4%})")
    if not na.any():
        print("  none")
    # Empty-list artists like "[]" are not NaN but are effectively missing.
    empty_artists = (df["artists"].astype(str).str.strip() == "[]").sum()
    print(f"  artists == '[]':   {empty_artists:,}")

    section("DUPLICATES")
    print(f"duplicate id rows:                 {df['id'].duplicated().sum():,}")
    print(f"fully duplicated rows:             {df.duplicated().sum():,}")
    name_artist_dups = df.duplicated(subset=["name", "artists"]).sum()
    print(f"duplicate (name, artists) rows:    {name_artist_dups:,}  "
          "(same song on multiple releases - informational)")

    section("TEMPO (BPM)")
    numeric_summary(df["tempo"])
    print(f"  tempo == 0:   {(df['tempo'] == 0).sum():,}")
    print(f"  tempo < 50:   {(df['tempo'] < 50).sum():,}")
    print(f"  tempo > 200:  {(df['tempo'] > 200).sum():,}")
    bins = [0, 1, 60, 80, 100, 120, 140, 160, 180, 200, 300]
    counts = pd.cut(df["tempo"], bins=bins, right=False).value_counts().sort_index()
    for interval, n in counts.items():
        print(f"  {str(interval):<14} {n:>9,}")

    section("KEY (pitch class)")
    kc = df["key"].value_counts().sort_index()
    for k, n in kc.items():
        label = KEY_NAMES[int(k)] if 0 <= k < 12 else "?"
        print(f"  {int(k):>2} {label:<3} {n:>9,}  ({n / len(df):.2%})")

    section("MODE (1 = major, 0 = minor)")
    for m, n in df["mode"].value_counts().sort_index().items():
        print(f"  {m}  {n:>9,}  ({n / len(df):.2%})")

    section("ENERGY")
    numeric_summary(df["energy"])
    ec = pd.cut(df["energy"], bins=[i / 10 for i in range(11)], include_lowest=True)
    for interval, n in ec.value_counts().sort_index().items():
        print(f"  {str(interval):<16} {n:>9,}")

    section("YEAR")
    print(f"  min={df['year'].min()}  max={df['year'].max()}  "
          f"median={df['year'].median():.0f}")
    print(f"  year == 0:     {(df['year'] == 0).sum():,}")
    print(f"  year < 1900:   {(df['year'] < 1900).sum():,}")
    decade = (df.loc[df["year"] >= 1900, "year"] // 10) * 10
    for d, n in decade.value_counts().sort_index().items():
        print(f"  {d}s  {n:>9,}")

    section("DURATION")
    dur_s = df["duration_ms"] / 1000
    numeric_summary(dur_s)
    print(f"  < {min_dur_s:.0f} s:          {(dur_s < min_dur_s).sum():,}")
    print(f"  > {max_dur_s / 60:.0f} min:        {(dur_s > max_dur_s).sum():,}")
    print(f"  > 60 min:        {(dur_s > 3600).sum():,}")

    section("OTHER SANITY CHECKS")
    print(f"time_signature counts: {df['time_signature'].value_counts().sort_index().to_dict()}")
    print(f"loudness > 0 dB:       {(df['loudness'] > 0).sum():,}")
    unusual = (
        (df["tempo"] == 0)
        | (dur_s < min_dur_s)
        | (dur_s > max_dur_s)
    )
    print(f"rows with tempo == 0 OR unusual duration: {unusual.sum():,} "
          f"({unusual.mean():.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.yaml", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    csv_path = PROJECT_ROOT / cfg["paths"]["raw_csv"]
    inv = cfg["inventory"]

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    buf = io.StringIO()
    with redirect_stdout(buf):
        run_inventory(df, inv["min_duration_s"], inv["max_duration_s"])
    report = buf.getvalue()
    print(report)

    out = PROJECT_ROOT / cfg["paths"]["reports_dir"] / "data_inventory.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Saved report to {out}")


if __name__ == "__main__":
    main()
