"""Search regression on the real index (skipped when app/backend/cache/search.sqlite is
missing or outdated). Top result for each famous title must be the original artist."""

from __future__ import annotations

import sqlite3

import pytest

from app.backend.main import PROJECT_ROOT, load_settings
from app.backend.search import SearchIndex, normalize
from app.backend.search_regression import CASES, top_artist

# Known misses with the configured ranking (see `python -m app.backend.search_regression`);
# kept as xfail so a regression elsewhere still fails and a fix shows up as XPASS.
KNOWN_FAILURES: set[str] = {
    "Billie Jean", "Fast Car", "Take Five", "Africa", "Blue Monday", "Purple Rain",
    "Starlight", "Rolling in the Deep",
}

S = load_settings()["search"]
DB = PROJECT_ROOT / S["db_path"]


@pytest.fixture(scope="module")
def index():
    try:
        idx = SearchIndex(DB, tier_cap=S["tier_cap"], prominence=S["prominence"],
                          tier_window=S["tier_window"],
                          original_boost_min_prom=S.get("original_boost_min_prominence"))
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(f"search index unavailable: {exc}")
    db = sqlite3.connect(f"{DB.resolve().as_uri()}?mode=ro", uri=True)
    yield idx, db
    idx.close()
    db.close()


@pytest.mark.parametrize("query, artist", [
    pytest.param(q, a, marks=pytest.mark.xfail(reason="known miss", strict=False))
    if q in KNOWN_FAILURES else (q, a) for q, a in CASES])
def test_original_artist_first(index, query, artist):
    idx, db = index
    assert top_artist(idx, db, query) == normalize(artist)
