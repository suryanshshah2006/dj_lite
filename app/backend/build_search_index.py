"""Rebuild the track search index (app/backend/cache/search.sqlite by default).

Run from the project root:  python -m app.backend.build_search_index
Re-run whenever data/features/track_features.parquet changes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from djlite.retrieval import load_configs

from .main import PROJECT_ROOT, load_settings
from .search import SearchIndex, build_index


def main() -> None:
    settings = load_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / settings["search"]["db_path"])
    args = ap.parse_args()

    paths, _, _ = load_configs()
    t0 = time.perf_counter()
    tracks = pd.read_parquet(PROJECT_ROOT / paths["features_dir"] / "track_features.parquet",
                             columns=["id", "name", "artists"])
    dedup = pd.read_parquet(PROJECT_ROOT / paths["processed_dir"] / "dedup_map.parquet",
                            columns=["kept_id"])
    # Popularity proxy (same as the engine): how many duplicate releases a song had.
    pop = tracks["id"].map(dedup["kept_id"].value_counts()).fillna(0)
    rows = zip(tracks["id"], tracks["name"], tracks["artists"], pop)
    n = build_index(rows, args.out)
    size = args.out.stat().st_size / 2**20
    print(f"indexed {n:,} tracks -> {args.out} ({size:.0f} MB) in {time.perf_counter() - t0:.1f}s")
    SearchIndex(args.out).close()  # sanity: opens and schema matches


if __name__ == "__main__":
    main()
