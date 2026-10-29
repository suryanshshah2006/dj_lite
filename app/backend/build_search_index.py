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


def artist_prominence(tracks: pd.DataFrame) -> pd.DataFrame:
    """Per-track prominence of its primary artist (first entry of artist_ids):

    - tracks:     number of cleaned tracks by that artist (any position). Favours prolific
                  catalogues, including cover/tribute acts (Vitamin String Quartet: 1,474).
    - co_artists: number of distinct other artists the artist shares an album with
                  (compilations, soundtracks, features). Tribute acts mostly release
                  single-artist albums, so originals rank above covers.
    Needs columns artist_ids and album_id.
    """
    primary = tracks["artist_ids"].map(lambda a: a[0] if len(a) else None)
    links = (tracks[["album_id", "artist_ids"]].explode("artist_ids")
             .dropna().drop_duplicates())
    track_counts = tracks["artist_ids"].explode().value_counts()
    pairs = links.merge(links, on="album_id")
    pairs = pairs[pairs["artist_ids_x"] != pairs["artist_ids_y"]]
    co = pairs.drop_duplicates(["artist_ids_x", "artist_ids_y"]).groupby("artist_ids_x").size()
    return pd.DataFrame({"co_artists": primary.map(co).fillna(0).to_numpy(),
                         "tracks": primary.map(track_counts).fillna(0).to_numpy()},
                        index=tracks.index)


def main() -> None:
    settings = load_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / settings["search"]["db_path"])
    args = ap.parse_args()

    paths, _, _ = load_configs()
    t0 = time.perf_counter()
    tracks = pd.read_parquet(PROJECT_ROOT / paths["features_dir"] / "track_features.parquet",
                             columns=["id", "name", "artists", "artist_ids"])
    albums = pd.read_parquet(PROJECT_ROOT / paths["processed_dir"] / "tracks_clean.parquet",
                             columns=["id", "album_id"]).set_index("id")["album_id"]
    tracks["album_id"] = tracks["id"].map(albums)
    dedup = pd.read_parquet(PROJECT_ROOT / paths["processed_dir"] / "dedup_map.parquet",
                            columns=["kept_id"])
    # Popularity proxy (same as the engine): how many duplicate releases a song had.
    pop = tracks["id"].map(dedup["kept_id"].value_counts()).fillna(0)
    prom = artist_prominence(tracks)
    rows = zip(tracks["id"], tracks["name"], tracks["artists"], pop,
               prom["co_artists"], prom["tracks"])
    n = build_index(rows, args.out)
    size = args.out.stat().st_size / 2**20
    print(f"indexed {n:,} tracks -> {args.out} ({size:.0f} MB) in {time.perf_counter() - t0:.1f}s")
    SearchIndex(args.out).close()  # sanity: opens and schema matches


if __name__ == "__main__":
    main()
