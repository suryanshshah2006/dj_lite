"""FTS5 search index: normalization, tier ranking, engine integration (small fixture)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backend.build_search_index import artist_prominence
from app.backend.engine import Engine
from app.backend.main import load_settings
from app.backend.search import SearchIndex, build_index, normalize
from app.backend.tests.test_api import fixture_tracks
from djlite.retrieval import load_configs


def _id(k: int) -> str:
    return f"{k:022d}"


ROWS = [  # (id, name, artists, popularity, prom_co_artists, prom_tracks)
    (_id(1), "Love Story", ["Taylor Swift"], 3, 0, 0),
    (_id(2), "Lovely", ["Billie Eilish", "Khalid"], 9, 0, 0),
    (_id(3), "Love", ["Kendrick Lamar"], 0, 0, 0),
    (_id(4), "Love", ["Lana Del Rey"], 5, 0, 0),
    (_id(5), "Crazy in Love", ["Beyoncé"], 8, 0, 0),
    (_id(6), "Halo", ["Beyoncé"], 2, 0, 0),
    (_id(7), "Don't Stop Me Now", ["Queen"], 1, 0, 0),
    (_id(8), "Queen of Hearts", ["Juice Newton"], 4, 0, 0),
    (_id(9), "Killer Queen", ["Queen"], 0, 0, 0),
    (_id(10), "Somebody to Love", ["Queens of the Stone Age"], 7, 0, 0),
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
    z = np.zeros(len(df))
    build_index(zip(df["id"], df["name"], df["artists"], z, z, z), db)
    idx = SearchIndex(db)
    fts = Engine(df, style, rcfg, tcfg, settings, search_index=idx)
    scan = Engine(df, style, rcfg, tcfg, settings)
    name = df["name"].iloc[7]
    for eng in (fts, scan):
        assert eng.search(name)[0]["name"] == name
    idx.close()


COVERS = [  # one title by the original artist and by prolific cover acts
    (_id(21), "Bohemian Rhapsody", ["Queen"], 0, 57, 5),
    (_id(22), "Bohemian Rhapsody", ["Vitamin String Quartet"], 0, 1, 1474),
    (_id(23), "Bohemian Rhapsody", ["Richard Cheese"], 9, 16, 163),
    (_id(24), "Bohemian Rhapsody - Live", ["Queen"], 0, 57, 5),
]


@pytest.mark.parametrize("prominence, first", [("co_artists", _id(21)), ("tracks", _id(22))])
def test_prominence_breaks_ties_within_tier(tmp_path, prominence, first):
    db = tmp_path / "covers.sqlite"
    build_index(iter(COVERS), db)
    idx = SearchIndex(db, prominence=prominence)
    res = idx.search("bohemian rhapsody")
    assert res[0] == first
    assert res[-1] == _id(24)  # prefix tier stays below every exact-title match
    # prominence outranks the duplicate-release popularity proxy: Richard Cheese (pop 9,
    # middle prominence under both scores) is second, not first
    assert res.index(_id(23)) == 1
    idx.close()


def test_prominence_setting_validated(tmp_path):
    db = tmp_path / "s.sqlite"
    build_index(iter(COVERS), db)
    with pytest.raises(ValueError):
        SearchIndex(db, prominence="plays")


def test_artist_prominence():
    t = pd.DataFrame({
        "artist_ids": [["q"], ["q", "x"], ["v"], ["v"], ["v"], ["y"]],
        "album_id": ["comp", "comp", "tribute1", "tribute2", "tribute2", "comp"],
    })
    p = artist_prominence(t)
    # q shares album "comp" with x and y; v only with itself
    assert p["co_artists"].tolist() == [2, 2, 0, 0, 0, 2]
    assert p["tracks"].tolist() == [2, 2, 3, 3, 3, 1]
