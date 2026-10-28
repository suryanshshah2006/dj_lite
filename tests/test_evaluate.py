import numpy as np
import pytest

from djlite.evaluate import artist_repetition, intra_list_diversity

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
