"""Search regression set: 20 famous songs whose original recording is in the dataset.

    python -m app.backend.search_regression     # pass rate per ranking configuration

A case passes when the top result for the title is by the expected (original) artist.
The first six were given; the other fourteen were fixed in advance as the first 14 of a
priority list (spread over eras and genres) whose original exists in the cleaned data,
chosen by existence only, before looking at rankings.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .main import PROJECT_ROOT, load_settings
from .search import SearchIndex, normalize

CASES = [  # (query, expected primary artist)
    ("Bohemian Rhapsody", "Queen"),
    ("Billie Jean", "Michael Jackson"),
    ("Fast Car", "Tracy Chapman"),
    ("Shape of You", "Ed Sheeran"),
    ("Halo", "Beyoncé"),
    ("Everlong", "Foo Fighters"),
    ("Respect", "Aretha Franklin"),
    ("Take Five", "The Dave Brubeck Quartet"),
    ("Jolene", "Dolly Parton"),
    ("Africa", "TOTO"),
    ("Blue Monday", "New Order"),
    ("Purple Rain", "Prince"),
    ("Juicy", "The Notorious B.I.G."),
    ("Starlight", "Muse"),
    ("Rolling in the Deep", "Adele"),
    ("Do I Wanna Know?", "Arctic Monkeys"),
    ("Lean On", "Major Lazer"),
    ("SICKO MODE", "Travis Scott"),
    ("bad guy", "Billie Eilish"),
    ("Levitating", "Dua Lipa"),
]


def top_artist(index: SearchIndex, db: sqlite3.Connection, query: str) -> str | None:
    ids = index.search(query, 1)
    if not ids:
        return None
    return db.execute("SELECT artists_key FROM tracks WHERE id = ?", (ids[0],)).fetchone()[0] \
        .split(" | ")[0]


def evaluate(index: SearchIndex, db: sqlite3.Connection) -> list[tuple[str, str, str | None, bool]]:
    out = []
    for q, artist in CASES:
        got = top_artist(index, db, q)
        out.append((q, artist, got, got == normalize(artist)))
    return out


def main() -> None:
    s = load_settings()["search"]
    path = PROJECT_ROOT / s["db_path"]
    db = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
    configs = [("tracks (no boost)", "tracks", None), ("co_artists (no boost)", "co_artists", None)]
    configs += [(f"co_artists + boost >= {t}", "co_artists", t) for t in (10, 20, 30, 50, 100)]
    results = {}
    for name, prom, thr in configs:
        idx = SearchIndex(path, tier_cap=s["tier_cap"], prominence=prom,
                          tier_window=s["tier_window"], original_boost_min_prom=thr)
        results[name] = evaluate(idx, db)
        idx.close()
    base = {q for q, _, _, ok in results["co_artists (no boost)"] if ok}
    print(f"{'configuration':<28} pass   broken vs co_artists baseline")
    for name, res in results.items():
        passed = {q for q, _, _, ok in res if ok}
        print(f"{name:<28} {len(passed):>2}/20  {sorted(base - passed) or '-'}")
    print()
    names = list(results)
    print(f"{'query':<22}" + "".join(f"{n[:22]:>24}" for n in names))
    for k, (q, artist) in enumerate(CASES):
        cells = [("OK " if results[n][k][3] else "-- ") + (results[n][k][2] or "")[:19] for n in names]
        print(f"{q[:21]:<22}" + "".join(f"{c:>24}" for c in cells))


if __name__ == "__main__":
    main()
