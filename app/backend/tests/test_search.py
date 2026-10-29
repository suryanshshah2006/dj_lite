"""FTS5 search index: normalization, tier ranking, engine integration (small fixture)."""

from __future__ import annotations

import numpy as np
import pytest

from app.backend.engine import Engine
from app.backend.main import load_settings
from app.backend.search import SearchIndex, build_index, normalize
from app.backend.tests.test_api import fixture_tracks
from djlite.retrieval import load_configs


def _id(k: int) -> str:
    return f"{k:022d}"


ROWS = [  # (id, name, artists, popularity)
    (_id(1), "Love Story", ["Taylor Swift"], 3),
    (_id(2), "Lovely", ["Billie Eilish", "Khalid"], 9),
    (_id(3), "Love", ["Kendrick Lamar"], 0),
    (_id(4), "Love", ["Lana Del Rey"], 5),
    (_id(5), "Crazy in Love", ["Beyoncé"], 8),
    (_id(6), "Halo", ["Beyoncé"], 2),
    (_id(7), "Don't Stop Me Now", ["Queen"], 1),
    (_id(8), "Queen of Hearts", ["Juice Newton"], 4),
    (_id(9), "Killer Queen", ["Queen"], 0),
    (_id(10), "Somebody to Love", ["Queens of the Stone Age"], 7),
]


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    db = tmp_path_factory.mktemp("search") / "search.sqlite"
    assert build_index(iter(ROWS), db, batch=3) == len(ROWS)
    idx = SearchIndex(db)
    yield idx
    idx.close()


def test_normalize():
    assert normalize("  Don’t  Stop—Me NOW! ") == "dont stop me now"
    assert normalize("Beyoncé") == "beyonce"
    assert normalize("AC/DC") == "ac dc"
    assert normalize("!!!") == ""


def test_exact_then_prefix_then_artist_then_other(index):
    # exact "love" (by popularity), whole-word prefix, partial prefix, then other matches
    assert index.search("love") == [_id(4), _id(3), _id(1), _id(2), _id(5), _id(10)]


def test_title_prefix_while_typing(index):
    assert index.search("dont stop m")[0] == _id(7)
    assert index.search("Don't stop me now") == [_id(7)]


def test_artist_matches_rank_after_titles_exact_artist_first(index):
    # titles starting with "queen" first, then artist == "queen", then partial artist
    # ("queens of the stone age" via prefix), then the remaining title match
    assert index.search("queen") == [_id(8), _id(7), _id(9), _id(10)]


def test_accents_and_multi_field(index):
    assert set(index.search("beyonce")) == {_id(5), _id(6)}
    assert index.search("beyonce halo") == [_id(6)]
    assert index.search("zzz") == []
    assert index.search("   ") == []


def test_limit(index):
    assert len(index.search("love", limit=2)) == 2


def test_engine_uses_index_and_falls_back(tmp_path):
    paths, rcfg, tcfg = load_configs()
    settings = load_settings()
    df, style = fixture_tracks()
    db = tmp_path / "s.sqlite"
    build_index(zip(df["id"], df["name"], df["artists"], np.zeros(len(df))), db)
    idx = SearchIndex(db)
    fts = Engine(df, style, rcfg, tcfg, settings, search_index=idx)
    scan = Engine(df, style, rcfg, tcfg, settings)
    name = df["name"].iloc[7]
    for eng in (fts, scan):
        assert eng.search(name)[0]["name"] == name
    idx.close()
