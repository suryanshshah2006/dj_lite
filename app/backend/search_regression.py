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


# Held-out set: never used to choose any search setting. A fixed priority list (eras and
# genres spread; no song from the tuning candidates, no artist from CASES); the first 20
# whose original recording exists in the cleaned data under the exact title are used,
# selected by existence only (`heldout_cases`), then evaluated once with the configured
# ranking: python -m app.backend.search_regression --heldout
HELDOUT_CANDIDATES = [
    ("Like a Rolling Stone", "Bob Dylan"), ("What's Going On", "Marvin Gaye"),
    ("Imagine", "John Lennon"), ("Dancing Queen", "ABBA"), ("Stayin' Alive", "Bee Gees"),
    ("Sweet Dreams (Are Made of This)", "Eurythmics"), ("Like a Prayer", "Madonna"),
    ("Born to Run", "Bruce Springsteen"), ("Losing My Religion", "R.E.M."),
    ("Enter Sandman", "Metallica"), ("No Scrubs", "TLC"), ("Ms. Jackson", "Outkast"),
    ("Crazy", "Gnarls Barkley"), ("Toxic", "Britney Spears"), ("Umbrella", "Rihanna"),
    ("Clocks", "Coldplay"), ("Hips Don't Lie", "Shakira"),
    ("Somebody That I Used to Know", "Gotye"), ("Royals", "Lorde"), ("Get Lucky", "Daft Punk"),
    ("Blinding Lights", "The Weeknd"), ("Old Town Road", "Lil Nas X"),
    ("Uptown Funk", "Mark Ronson"), ("Hotline Bling", "Drake"), ("Rehab", "Amy Winehouse"),
    ("Hey Jude", "The Beatles"), ("One", "U2"), ("Wannabe", "Spice Girls"),
    ("September", "Earth, Wind & Fire"), ("Gangsta's Paradise", "Coolio"),
    ("Thunderstruck", "AC/DC"), ("Zombie", "The Cranberries"), ("Despacito", "Luis Fonsi"),
    ("Sweet Caroline", "Neil Diamond"), ("Feel Good Inc.", "Gorillaz"),
    ("Mr. Blue Sky", "Electric Light Orchestra"), ("Paranoid", "Black Sabbath"),
    ("Lose Yourself", "Eminem"), ("Hound Dog", "Elvis Presley"), ("Respect Yourself", "The Staple Singers"),
    # Extension (only 15 of the 40 above exist under the exact title): appended once, fixed
    # before any of these were ranked; same rules. The first 15 cases are unchanged.
    ("Purple Haze", "Jimi Hendrix"), ("Johnny B. Goode", "Chuck Berry"),
    ("Good Vibrations", "The Beach Boys"), ("My Girl", "The Temptations"),
    ("I Will Survive", "Gloria Gaynor"), ("Le Freak", "Chic"), ("Super Freak", "Rick James"),
    ("Take On Me", "a-ha"), ("Don't You (Forget About Me)", "Simple Minds"),
    ("Smooth", "Santana"), ("Bitter Sweet Symphony", "The Verve"), ("Song 2", "Blur"),
    ("Firework", "Katy Perry"), ("Poker Face", "Lady Gaga"), ("Shake It Off", "Taylor Swift"),
    ("Sorry", "Justin Bieber"), ("HUMBLE.", "Kendrick Lamar"),
    ("Get Ur Freak On", "Missy Elliott"), ("California Love", "2Pac"),
    ("Dreams", "Fleetwood Mac"), ("Heroes", "David Bowie"), ("La Bamba", "Ritchie Valens"),
    ("Sultans of Swing", "Dire Straits"), ("Hallelujah", "Jeff Buckley"),
    ("Juicy Fruit", "Mtume"),
]


def heldout_cases(db: sqlite3.Connection, n: int = 20) -> list[tuple[str, str]]:
    """First n candidates whose original artist has a track with exactly that title."""
    out = []
    for title, artist in HELDOUT_CANDIDATES:
        t, a = normalize(title), normalize(artist)
        hit = db.execute("SELECT 1 FROM tracks WHERE name_key = ? AND (artists_key = ? OR "
                         "artists_key LIKE ?) LIMIT 1", (t, a, a + " | %")).fetchone()
        if hit:
            out.append((title, artist))
        if len(out) == n:
            break
    return out


def top_artist(index: SearchIndex, db: sqlite3.Connection, query: str) -> str | None:
    ids = index.search(query, 1)
    if not ids:
        return None
    return db.execute("SELECT artists_key FROM tracks WHERE id = ?", (ids[0],)).fetchone()[0] \
        .split(" | ")[0]


def evaluate(index: SearchIndex, db: sqlite3.Connection,
             cases=None) -> list[tuple[str, str, str | None, bool]]:
    out = []
    for q, artist in cases or CASES:
        got = top_artist(index, db, q)
        out.append((q, artist, got, got == normalize(artist)))
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--heldout", action="store_true",
                    help="evaluate the held-out set once with the configured ranking")
    args = ap.parse_args()
    s = load_settings()["search"]
    path = PROJECT_ROOT / s["db_path"]
    db = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
    if args.heldout:
        cases = heldout_cases(db)
        idx = SearchIndex(path, tier_cap=s["tier_cap"], prominence=s["prominence"],
                          tier_window=s["tier_window"],
                          original_boost_min_prom=s.get("original_boost_min_prominence"))
        res = evaluate(idx, db, cases)
        print(f"held-out: {sum(ok for *_, ok in res)}/{len(res)} (configured ranking: "
              f"prominence={s['prominence']}, boost>={s.get('original_boost_min_prominence')})")
        for q, artist, got, ok in res:
            print(f"  {'OK' if ok else '--'}  {q:<34} expected {artist:<26} got {got}")
        return
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
