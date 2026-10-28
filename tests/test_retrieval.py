import numpy as np
import pandas as pd
import pytest

from djlite.features import add_camelot
from djlite.retrieval import Recommender, load_configs
from djlite.style import l2_normalize
from djlite.transitions import transition_score_v0_1

AUDIO = ["danceability", "energy", "loudness", "speechiness", "acousticness",
         "instrumentalness", "liveness", "valence", "tempo_folded"]


@pytest.fixture(scope="module")
def setup():
    _, rcfg, tcfg = load_configs()
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "id": [f"t{i}" for i in range(n)],
        "name": [f"song {i}" for i in range(n)],
        "artists": [[f"artist {i % 40}"] for i in range(n)],
        "artist_ids": [[f"a{i % 40}"] for i in range(n)],
        "key": rng.integers(0, 12, n), "mode": rng.integers(0, 2, n),
        "tempo": rng.uniform(60, 180, n), "energy": rng.uniform(0, 1, n),
        "loudness": rng.uniform(-20, -2, n), "year": rng.integers(1980, 2021, n),
    })
    df["tempo_folded"] = df["tempo"].where(df["tempo"] >= 80, df["tempo"] * 2)
    df["tempo_folded"] = df["tempo_folded"].where(df["tempo_folded"] < 160,
                                                  df["tempo_folded"] / 2)
    for c in AUDIO:
        df[c + "_z"] = rng.normal(size=n)
    df = add_camelot(df)
    style = l2_normalize(rng.normal(size=(n, 8)).astype(np.float32))
    style[:20] = 0  # some tracks without a style vector
    rcfg = {**rcfg, "retrieval": {**rcfg["retrieval"], "k": 100}}
    return Recommender(df, style, rcfg, tcfg), df, style, rcfg, tcfg


def test_relevance_matches_formula(setup):
    rec, df, style, rcfg, tcfg = setup
    w = rcfg["relevance"]
    i, cand = 5, np.arange(50, 60)
    r = rec.relevance(i, cand)
    a = l2_normalize(df[[c + "_z" for c in AUDIO]].to_numpy())
    audio = (a[cand] @ a[i] + 1) / 2
    sty = (style[cand] @ style[i] + 1) / 2
    gap = np.abs(df["year"].to_numpy()[cand] - df["year"].iat[i])
    era = 1 - 0.5 * np.clip((gap - 5) / 25, 0, 1)
    expected = w["w_audio"] * audio + w["w_style"] * sty + w["w_era"] * era
    assert np.allclose(r["relevance"], expected, atol=1e-6)


def test_zero_style_is_neutral(setup):
    rec, *_ = setup
    r = rec.relevance(0, np.arange(30, 40))  # track 0 has no style vector
    assert np.allclose(r["style_sim"], 0.5)


def test_full_ranking_combines_relevance_and_transition(setup):
    rec, df, _, rcfg, tcfg = setup
    out = rec.rank(7, "full", n=10)
    a, b = rcfg["ranking"]["alpha"], rcfg["ranking"]["beta"]
    assert np.allclose(out["final"], a * out["relevance"] + b * out["transition"])
    assert np.all(np.diff(out["final"].to_numpy()) <= 1e-12)
    j = int(out["idx"].iat[0])
    single = transition_score_v0_1(df.iloc[7], df.iloc[j], tcfg)["score"]
    assert out["transition"].iat[0] == pytest.approx(single)


@pytest.mark.parametrize("method", ["transition", "audio", "relevance", "full"])
def test_excludes_seed_and_same_artist(setup, method):
    rec, df, *_ = setup
    i = 11
    out = rec.rank(i, method, n=10)
    assert len(out) == 10
    assert i not in set(out["idx"])
    seed_artist = df["artist_ids"].iat[i][0]
    assert all(df["artist_ids"].iat[j][0] != seed_artist for j in out["idx"])


def test_faiss_retrieval_matches_exact_scores(setup):
    rec, *_ = setup
    rel = rec.rcfg["relevance"]
    i = 3
    cand = rec.retrieve(np.array([i]), k=20)[0]
    exact = rel["w_audio"] * (rec.audio @ rec.audio[i]) + rel["w_style"] * (rec.style @ rec.style[i])
    assert set(cand) == set(np.argsort(-exact)[:20])
