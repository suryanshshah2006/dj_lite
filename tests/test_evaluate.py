import numpy as np
import pandas as pd
import pytest

from djlite.evaluate import artist_repetition, bootstrap_v2, intra_list_diversity

from test_retrieval import setup  # noqa: F401  (shared synthetic Recommender fixture)


def test_ild_identical_and_orthogonal():
    same = np.tile([1.0, 0.0, 0.0], (4, 1))
    assert intra_list_diversity(same) == pytest.approx(0.0)
    assert intra_list_diversity(np.eye(3)) == pytest.approx(1.0)
    assert np.isnan(intra_list_diversity(np.eye(3)[:1]))


def test_ild_matches_pairwise_mean():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(10, 5))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pairs = [1 - v[i] @ v[j] for i in range(10) for j in range(i + 1, 10)]
    assert intra_list_diversity(v) == pytest.approx(np.mean(pairs))


class _FakeRec:
    def __init__(self, artist_sets):
        self.artist_sets = [frozenset(a) for a in artist_sets]


def test_artist_repetition():
    rec = _FakeRec([{"a"}, {"b"}, {"a"}, {"c", "b"}, {"d"}])
    # idx 2 repeats "a", idx 3 repeats "b" -> 2 of 5
    assert artist_repetition(rec, np.arange(5)) == pytest.approx(0.4)
    assert artist_repetition(rec, np.array([0, 1, 4])) == 0.0


def test_rank_lists_capped_vs_uncapped(setup):  # noqa: F811
    rec, df, *_ = setup
    lists = rec.rank_lists(11, "full", 40, caps=(1, None))
    capped, uncapped = lists[1], lists[None]
    assert capped["idx"].iat[0] == uncapped["idx"].iat[0]
    assert np.all(np.diff(uncapped["final"].to_numpy()) <= 1e-12)
    a_cap = [df["artist_ids"].iat[j][0] for j in capped["idx"]]
    assert len(set(a_cap)) == len(a_cap)
    assert artist_repetition(rec, uncapped["idx"].to_numpy()) > 0
    assert artist_repetition(rec, capped["idx"].to_numpy()) == 0
    # rank() returns the configured (capped) list
    assert rec.rank(11, "full", 40)["idx"].tolist() == capped["idx"].tolist()


def test_combined_vectors_are_unit(setup):  # noqa: F811
    rec, *_ = setup
    v = rec.combined_vectors(np.arange(30))
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


# ---------- v2 bootstrap / macro ----------

def _per_seed(agree, trans):
    n = len(agree)
    base = {"method": "relevance", "alpha": np.nan, "is_d": False, "labeled": 10,
            "dbpm": 1.0, "ild": 0.5, "artist_rep": 0.0}
    return pd.DataFrame([{**base, "seed_idx": s, "genre_agree": agree[s], "trans": trans[s]}
                         for s in range(n)])


def test_bootstrap_point_estimates_micro_and_macro():
    # genre x: seeds 0,1 (agree 1.0, 0.0); genre y: seeds 2,3,4 (1.0 each); seed 1 also in y
    agree = [1.0, 0.0, 1.0, 1.0, 1.0]
    genres = pd.Series([frozenset("x"), frozenset("xy"), frozenset("y"), frozenset("y"),
                        frozenset("y")])
    out = bootstrap_v2(_per_seed(agree, [0.5] * 5), genres,
                       {"n_resamples": 200, "ci": 0.95, "random_seed": 0}).iloc[0]
    assert out["genre_agree"] == pytest.approx(0.8)                 # micro: 4/5
    # macro: x -> mean(1, 0) = 0.5 ; y -> mean(0, 1, 1, 1) = 0.75 ; mean = 0.625
    assert out["genre_macro"] == pytest.approx(0.625)
    assert out["trans"] == pytest.approx(0.5)
    assert out["trans_lo"] == pytest.approx(0.5) and out["trans_hi"] == pytest.approx(0.5)
    assert out["genre_agree_lo"] <= 0.8 <= out["genre_agree_hi"]


def test_bootstrap_ignores_nan_and_ci_covers_mean():
    rng = np.random.default_rng(1)
    agree = rng.uniform(size=400)
    agree[::4] = np.nan  # seeds without a labeled result
    trans = rng.normal(0.7, 0.1, size=400)
    genres = pd.Series([frozenset({f"g{s % 5}"}) for s in range(400)])
    out = bootstrap_v2(_per_seed(agree, trans), genres,
                       {"n_resamples": 500, "ci": 0.95, "random_seed": 0}).iloc[0]
    assert out["genre_agree"] == pytest.approx(np.nanmean(agree))
    assert out["seeds_with_label"] == pytest.approx(0.75)
    se = np.std(trans) / np.sqrt(400)
    assert out["trans_lo"] == pytest.approx(trans.mean() - 1.96 * se, abs=se)
    assert out["trans_hi"] == pytest.approx(trans.mean() + 1.96 * se, abs=se)
