"""Pipeline step 2a: track-level features.

Reads data/processed/tracks_clean.parquet, adds Camelot codes and standardized audio
features, and writes data/features/track_features.parquet plus models/scaler.json.

Usage:
    python src/djlite/features.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def camelot_from_key_mode(key, mode):
    """Map Spotify (key, mode) to Camelot (number 1-12, letter 'A'/'B').

    key: pitch class 0-11 (C=0), -1 = unknown. mode: 1 = major (B), 0 = minor (A).
    C major = 8B, A minor = 8A; each step clockwise is a fifth up.
    Scalars return (float, str | None); arrays return (float array, object array).
    Unknown keys give (nan, None).
    """
    k = np.asarray(key, dtype=float)
    m = np.asarray(mode, dtype=float)
    valid = (k >= 0) & (k <= 11)
    k_safe = np.where(valid, k, 0).astype(int)
    major = m == 1
    offset = np.where(major, 7, 4)
    num = np.where(valid, (7 * k_safe + offset) % 12 + 1, np.nan)
    letter = np.where(valid, np.where(major, "B", "A"), None).astype(object)
    if num.ndim == 0:
        return float(num), letter.item()
    return num, letter


def add_camelot(df: pd.DataFrame) -> pd.DataFrame:
    num, letter = camelot_from_key_mode(df["key"].to_numpy(), df["mode"].to_numpy())
    df = df.copy()
    df["camelot_num"] = pd.array(num, dtype="Int8")
    df["camelot_letter"] = pd.array(letter, dtype="string")
    df["camelot"] = df["camelot_num"].astype("string") + df["camelot_letter"]
    return df


def fit_scaler(df: pd.DataFrame, columns: list[str]) -> dict:
    """Population mean/std (ddof=0, same as sklearn StandardScaler)."""
    return {
        "columns": columns,
        "mean": {c: float(df[c].mean()) for c in columns},
        "std": {c: float(df[c].std(ddof=0)) for c in columns},
    }


def apply_scaler(df: pd.DataFrame, scaler: dict, suffix: str) -> pd.DataFrame:
    df = df.copy()
    for c in scaler["columns"]:
        df[c + suffix] = (df[c] - scaler["mean"][c]) / scaler["std"][c]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=PROJECT_ROOT / "configs" / "paths.yaml", type=Path)
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "features.yaml", type=Path)
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text(encoding="utf-8"))["paths"]
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    std_cols = cfg["standardize"]["columns"]
    suffix = cfg["standardize"]["suffix"]

    src = PROJECT_ROOT / paths["processed_dir"] / "tracks_clean.parquet"
    df = pd.read_parquet(src, columns=list(dict.fromkeys(cfg["keep_columns"] + std_cols)))
    print(f"Loaded {len(df):,} rows from {src}")

    n_unknown = int((df["key"] == -1).sum())
    print(f"Rows with key == -1 (no key detected): {n_unknown:,} -> Camelot set to null")

    df = add_camelot(df)
    scaler = fit_scaler(df, std_cols)
    df = apply_scaler(df, scaler, suffix)

    out_dir = PROJECT_ROOT / paths["features_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "track_features.parquet"
    df.to_parquet(out, index=False)

    scaler_path = PROJECT_ROOT / paths["models_dir"] / cfg["scaler_file"]
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.write_text(json.dumps(scaler, indent=2), encoding="utf-8")

    z = df[[c + suffix for c in std_cols]]
    print(f"Null Camelot codes: {int(df['camelot'].isna().sum()):,}")
    print("Standardized columns (mean / std after scaling):")
    for c in z.columns:
        print(f"  {c:<22} {z[c].mean():+.2e} / {z[c].std(ddof=0):.4f}")
    print(f"Columns: {list(df.columns)}")
    print(f"Wrote {out}")
    print(f"Wrote {scaler_path}")


if __name__ == "__main__":
    main()
