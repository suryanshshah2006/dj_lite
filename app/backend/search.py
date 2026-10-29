"""Track search over a SQLite FTS5 index of title + artists.

Build:  python -m app.backend.build_search_index   (writes app/backend/cache/search.sqlite)

Ranking tiers; within a tier: tier-specific refinement, then the primary artist's
prominence (precomputed at build time, see build_search_index.artist_prominence),
then the duplicate-release popularity proxy, then shorter title. Optional original-release
boost (`original_boost_min_prom`): among results of one tier with the same normalized title,
the earliest release whose artist prominence is at least the threshold is ranked first:
  0  exact title            ("one more time")
  1  title prefix           ("one more" -> "One More Time", "One More Night")
  2  every token in artists ("daft punk"; artists equal to the query come first)
  3  every token somewhere in title or artists ("daft one more")
Tiers 0-1 use a B-tree index on the normalized title; tiers 2-3 use FTS5. Each tier keeps
at most `tier_cap` rows. Prominence ordering looks at the first `tier_window` matches of a
tier, so it is exact for any query with fewer matches and bounded for very common prefixes
("a", "the"), which keeps every query well under 100 ms.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import unicodedata
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = 3
PROMINENCE = ("co_artists", "tracks")  # columns prom_<name> in the tracks table
_APOS = re.compile(r"['’`]")
_NONWORD = re.compile(r"[^\w]+")
_MAX_CHAR = "\U0010ffff"


def normalize(s: str) -> str:
    """Lowercase, strip accents, drop apostrophes, other punctuation -> space."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    s = _NONWORD.sub(" ", _APOS.sub("", s))
    return " ".join(s.replace("_", " ").split())


def build_index(rows: Iterable[tuple], db_path: Path, batch: int = 50_000) -> int:
    """rows: (track_id, name, artists, popularity, prom_co_artists, prom_tracks, year or None).
    Replaces db_path atomically."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.executescript("""
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE tracks (rowid INTEGER PRIMARY KEY, id TEXT NOT NULL, name_key TEXT NOT NULL,
                             artists_key TEXT NOT NULL, pop REAL NOT NULL,
                             prom_co_artists REAL NOT NULL, prom_tracks REAL NOT NULL,
                             year INTEGER);
        CREATE VIRTUAL TABLE fts USING fts5(name_key, artists_key, content='tracks',
            content_rowid='rowid', tokenize='unicode61', prefix='2 3');
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    """)
    n, buf = 0, []

    def flush():
        db.executemany("INSERT INTO tracks VALUES (?, ?, ?, ?, ?, ?, ?, ?)", buf)
        buf.clear()

    for tid, name, artists, pop, prom_co, prom_tr, year in rows:
        n += 1
        year = None if year is None or year != year else int(year)  # NaN -> NULL
        buf.append((n, tid, normalize(name), " | ".join(normalize(a) for a in artists),
                    float(pop), float(prom_co), float(prom_tr), year))
        if len(buf) >= batch:
            flush()
    flush()
    db.executescript("""
        INSERT INTO fts(fts) VALUES ('rebuild');
        INSERT INTO fts(fts) VALUES ('optimize');
        CREATE INDEX tracks_name_co ON tracks(name_key, prom_co_artists);
        CREATE INDEX tracks_name_tr ON tracks(name_key, prom_tracks);
    """)
    db.executemany("INSERT INTO meta VALUES (?, ?)",
                   [("schema_version", str(SCHEMA_VERSION)), ("n_tracks", str(n))])
    db.commit()
    db.execute("VACUUM")
    db.close()
    tmp.replace(db_path)
    return n


class SearchIndex:
    """Read-only, one SQLite connection per thread (FastAPI runs sync routes in a pool)."""

    def __init__(self, db_path: Path, tier_cap: int = 400, prominence: str = "co_artists",
                 tier_window: int = 5000, original_boost_min_prom: float | None = None):
        if not db_path.exists():
            raise FileNotFoundError(db_path)
        if prominence not in PROMINENCE:
            raise ValueError(f"prominence must be one of {PROMINENCE}")
        self.prom = f"prom_{prominence}"
        self.uri = f"{db_path.resolve().as_uri()}?mode=ro"
        self.tier_cap, self.tier_window = tier_cap, tier_window
        self.boost_min_prom = original_boost_min_prom
        self._local = threading.local()
        meta = dict(self._db().execute("SELECT key, value FROM meta"))
        if int(meta.get("schema_version", 0)) != SCHEMA_VERSION:
            raise RuntimeError(f"{db_path} has an old schema; rebuild it")
        self.n_tracks = int(meta["n_tracks"])

    def _db(self) -> sqlite3.Connection:
        db = getattr(self._local, "db", None)
        if db is None:
            db = self._local.db = sqlite3.connect(self.uri, uri=True, check_same_thread=False)
        return db

    @staticmethod
    def _fts_query(tokens: list[str], column: str | None) -> str:
        parts = []
        for k, tok in enumerate(tokens):
            term = f'"{tok}"' + ("*" if k == len(tokens) - 1 and len(tok) >= 2 else "")
            parts.append(f"{column} : {term}" if column else term)
        return " AND ".join(parts)

    def search(self, q: str, limit: int = 20) -> list[str]:
        """Track ids, best first."""
        key = normalize(q)
        if not key:
            return []
        db, cap = self._db(), self.tier_cap
        tokens = key.split()
        prom, win = self.prom, self.tier_window
        cols = f"t.rowid, t.id, t.name_key, t.artists_key, t.pop, t.{prom}, t.year"

        def by_prom(matches: str) -> str:
            # the most prominent `cap` rows among the first `win` matches
            return (f"SELECT {cols} FROM ({matches} LIMIT {win}) m JOIN tracks t "
                    f"ON t.rowid = m.rowid ORDER BY t.{prom} DESC LIMIT ?")

        fts = f"SELECT {cols} FROM fts JOIN tracks t ON t.rowid = fts.rowid WHERE fts MATCH ?"
        tiers = [
            (by_prom("SELECT rowid FROM tracks WHERE name_key = ?"), (key, cap)),
            (by_prom("SELECT rowid FROM tracks WHERE name_key > ? AND name_key < ?"),
             (key, key + _MAX_CHAR, cap)),
            (by_prom("SELECT rowid FROM fts WHERE fts MATCH ?"),
             (self._fts_query(tokens, "artists_key"), cap)),
            (fts + " ORDER BY rank LIMIT ?", (self._fts_query(tokens, None), cap)),
        ]
        seen: set[int] = set()
        ranked: list[tuple] = []
        for tier, (sql, params) in enumerate(tiers):
            for rowid, tid, name, artists, pop, pr, year in db.execute(sql, params):
                if rowid in seen:
                    continue
                seen.add(rowid)
                if tier == 1:  # whole-word prefix ("love" -> "love story") before "lovely"
                    sub = 0 if name[len(key)] == " " else 1
                elif tier == 2:  # an artist equal to the query first
                    sub = 0 if key in artists.split(" | ") else 1
                elif tier == 3:  # all tokens in the title before tokens spread across fields
                    sub = 0 if all(t in name for t in tokens) else 1
                else:
                    sub = 0
                ranked.append([tier, sub, 1, -pr, -pop, len(name), rowid, tid, name, year])
            if tier >= 1 and len(ranked) >= limit:
                break  # better tiers already fill the page
        if self.boost_min_prom is not None:
            # per (tier, title): the earliest release by a prominent-enough artist goes first
            best: dict = {}
            for r in ranked:
                if r[9] is not None and -r[3] >= self.boost_min_prom:
                    k = (r[0], r[8])
                    if k not in best or (r[9], r[3], r[6]) < (best[k][9], best[k][3], best[k][6]):
                        best[k] = r
            for r in best.values():
                r[2] = 0
        ranked.sort(key=lambda r: tuple(r[:7]))
        return [r[7] for r in ranked[:limit]]

    def close(self) -> None:
        db = getattr(self._local, "db", None)
        if db is not None:
            db.close()
            self._local.db = None
