import itertools

import numpy as np
import pandas as pd
import pytest

from djlite import generator as G
from djlite.features import add_camelot
from djlite.retrieval import Recommender, load_configs
from djlite.style import l2_normalize
from djlite.transitions import tempo_octave_align

AUDIO = ["danceability", "energy", "loudness", "speechiness", "acousticness",
         "instrumentalness", "liveness", "valence", "tempo_folded"]


@pytest.fixture(scope="module")
def rec():
    _, rcfg, tcfg = load_configs()
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({
        "id": [f"t{i}" for i in range(n)],
        "name": [f"song {i}" for i in range(n)],
        "artists": [[f"artist {i % 90}"] for i in range(n)],
        # some tracks have a featured artist shared with other tracks
        "artist_ids": [[f"a{i % 90}"] + ([f"f{i % 5}"] if i % 11 == 0 else []) for i in range(n)],
        "key": rng.integers(0, 12, n), "mode": rng.integers(0, 2, n),
        "tempo": rng.uniform(60, 180, n), "energy": rng.uniform(0, 1, n),
        "loudness": rng.uniform(-20, -2, n), "year": rng.integers(1970, 2021, n),
    })
    df["tempo_folded"] = df["tempo"].where(df["tempo"] >= 80, df["tempo"] * 2)
    df["tempo_folded"] = df["tempo_folded"].where(df["tempo_folded"] < 160,
                                                  df["tempo_folded"] / 2)
    for c in AUDIO:
        df[c + "_z"] = rng.normal(size=n)
    df = add_camelot(df)
    df.loc[df.index[::37], ["camelot_num", "camelot_letter", "camelot"]] = None  # null keys
    style = l2_normalize(rng.normal(size=(n, 8)).astype(np.float32))
    style[:30] = 0
    return Recommender(df, style, rcfg, tcfg)


@pytest.fixture(scope="module")
def gcfg():
    return G.load_generator_config()


def _pool(rec, seeds=(5,), k=150, dbpm=False):
    return G.build_pool(rec, list(seeds), k=k, per_seed_k=200, need_dbpm=dbpm)


def test_resample_arc():
    t = G.resample_arc([0.2, 0.8, 0.4], 5)
    assert t.tolist() == pytest.approx([0.2, 0.5, 0.8, 0.6, 0.4])
    assert G.resample_arc([0.7], 4).tolist() == [0.7] * 4
    assert G.resample_arc([-1, 2], 3).tolist() == [0.0, 0.5, 1.0]  # clipped


def test_pool_structure_and_relevance(rec):
    pool = _pool(rec, seeds=(5, 77), k=120)
    assert pool.idx[0] == 5 and len(pool.idx) == 121 and len(set(pool.idx)) == 121
    expect = np.mean([rec.relevance(s, pool.idx[1:])["relevance"] for s in (5, 77)], axis=0)
    assert pool.rel[1:] == pytest.approx(expect)
    assert np.all(np.diff(pool.rel[1:]) <= 1e-12)  # top-k by relevance to the seed set
    assert pool.T.dtype == np.float32 and pool.T.shape == (121, 121)


def test_transition_matrix_matches_scorer(rec):
    pool = _pool(rec, k=80, dbpm=True)
    for p in [0, 3, 41, 80]:
        ref = np.asarray(rec.transition(int(pool.idx[p]), pool.idx)["score"])
        assert pool.T[p] == pytest.approx(ref, abs=1e-6)
        _, _, b2 = tempo_octave_align(rec.arr["tempo"][pool.idx[p]], rec.arr["tempo"][pool.idx])
        assert pool.dbpm[p] == pytest.approx(np.abs(rec.arr["tempo"][pool.idx[p]] - b2), abs=1e-4)


def _check_constraints(rec, pool, seq, max_dbpm=None, seed_pos=0):
    idx = pool.idx[seq]
    assert seq[seed_pos] == 0 and len(set(seq)) == len(seq)
    artists = [a for i in idx for a in rec.artist_sets[int(i)]]
    assert len(artists) == len(set(artists))
    if max_dbpm is not None:
        assert all(pool.dbpm[a, b] <= max_dbpm for a, b in zip(seq, seq[1:]))


@pytest.mark.parametrize("seed, alpha, gamma, cap, lam, placement", [
    (5, 0.5, 0.5, None, 0.0, "start"), (17, 0.75, 0.0, None, 0.0, "start"),
    (123, 0.0, 1.0, None, 1.0, "start"), (250, 0.5, 0.5, 4.0, 0.0, "start"),
    (33, 0.5, 0.25, 6.0, 2.0, "best_fit"), (300, 0.5, 0.5, None, 0.5, "best_fit")])
def test_beam_width_1_equals_greedy(rec, seed, alpha, gamma, cap, lam, placement):
    pool = _pool(rec, seeds=(seed,), dbpm=cap is not None)
    for arc in ([0.3, 0.9, 0.4], [0.7], [0.2, 0.4, 0.6, 0.8]):
        target = G.resample_arc(arc, 12)
        sp = G.seed_slot(pool, target, placement)
        g = G.greedy(pool, target, alpha, gamma, cap, lam=lam, seed_pos=sp)
        b = G.beam(pool, target, alpha, gamma, 1, cap, lam=lam, seed_pos=sp)
        assert b == g and len(b) == 12 and b[sp] == 0
        _check_constraints(rec, pool, b, cap, sp)


def test_seed_slot_and_fill_order(rec):
    pool = _pool(rec, seeds=(8,))
    target = G.resample_arc([0.0, 1.0], 11)  # targets 0.0, 0.1, ..., 1.0
    assert G.seed_slot(pool, target, "start") == 0
    assert G.seed_slot(pool, target, "best_fit") == int(np.argmin(np.abs(target - pool.energy[0])))
    with pytest.raises(ValueError):
        G.seed_slot(pool, target, "middle")
    order = G.fill_order(5, 2)
    assert order == [(3, 2, 1), (4, 3, 1), (1, 2, -1), (0, 1, -1)]
    assert sorted(p for p, _, _ in order) == [0, 1, 3, 4]


def test_weak_link_penalty_in_objective(rec):
    pool = _pool(rec)
    target = G.resample_arc([0.3, 0.8], 10)
    seq = G.beam(pool, target, 0.5, 0.5, 5)
    s = np.asarray(seq)
    tr = pool.T[s[:-1], s[1:]].astype(float)
    base = G.objective(pool, seq, target, 0.5, 0.5)
    assert G.objective(pool, seq, target, 0.5, 0.5, lam=2.0, tau=0.85) == \
        pytest.approx(base - 2.0 * np.maximum(0, 0.85 - tr).sum())


def test_penalty_raises_worst_transition_on_average(rec):
    target = G.resample_arc([0.1, 0.9, 0.1], 12)
    lows = {0.0: [], 2.0: []}
    for seed in range(0, 400, 25):
        pool = _pool(rec, seeds=(seed,))
        for lam in lows:
            s = np.asarray(G.beam(pool, target, 0.5, 0.5, 10, lam=lam, tau=0.85))
            lows[lam].append(pool.T[s[:-1], s[1:]].min())
    assert np.mean(lows[2.0]) > np.mean(lows[0.0])


def test_stratified_pool_covers_energy_bands(rec):
    plain = G.build_pool(rec, [5], k=100, per_seed_k=100)
    strat = G.build_pool(rec, [5], k=100, per_seed_k=400, energy_bands=5)
    assert len(strat.idx) == 101 and len(set(strat.idx)) == 101 and strat.idx[0] == 5
    cand = rec.retrieve(np.array([5]), k=400).ravel()
    cand = cand[(cand >= 0) & (cand != 5)]
    edges = np.quantile(rec.arr["energy"][cand], [0.2, 0.4, 0.6, 0.8])
    bands = np.searchsorted(edges, strat.energy[1:], side="right")
    assert np.bincount(bands, minlength=5).tolist() == [20] * 5
    assert np.all(np.diff(strat.rel[1:]) <= 1e-12)  # still ordered by relevance
    assert strat.energy[1:].min() <= plain.energy[1:].min()
    assert strat.energy[1:].max() >= plain.energy[1:].max()


def test_beam_constraints_and_not_worse_on_average(rec):
    target = G.resample_arc([0.35, 0.5, 0.68, 0.85, 0.9, 0.7, 0.45], 12)
    gains = []
    for seed in range(0, 400, 40):
        pool = _pool(rec, seeds=(seed,), dbpm=True)
        g = G.greedy(pool, target, 0.5, 0.5)
        b = G.beam(pool, target, 0.5, 0.5, 20)
        assert len(b) == 12
        _check_constraints(rec, pool, b)
        _check_constraints(rec, pool, G.beam(pool, target, 0.5, 0.5, 20, 3.0), 3.0)
        gains.append(G.objective(pool, b, target, .5, .5) - G.objective(pool, g, target, .5, .5))
    assert np.mean(gains) > 0 and min(gains) > -0.05


@pytest.mark.parametrize("lam, seed_pos", [(0.0, 0), (1.5, 0), (1.0, 2), (0.0, 3)])
def test_wide_beam_finds_brute_force_optimum(rec, lam, seed_pos):
    pool = _pool(rec, seeds=(9,), k=7)
    target = G.resample_arc([0.2, 0.9], 4)
    best, best_seq = -np.inf, None
    for perm in itertools.permutations(range(1, len(pool.idx)), 3):
        seq = [*perm[:seed_pos], 0, *perm[seed_pos:]]
        try:
            _check_constraints(rec, pool, seq, seed_pos=seed_pos)
        except AssertionError:
            continue
        v = G.objective(pool, seq, target, 0.5, 0.5, lam=lam, seed_pos=seed_pos)
        if v > best:
            best, best_seq = v, seq
    b = G.beam(pool, target, 0.5, 0.5, 10_000, lam=lam, seed_pos=seed_pos)
    assert G.objective(pool, b, target, 0.5, 0.5, lam=lam, seed_pos=seed_pos) == pytest.approx(best)
    assert b == best_seq


def test_reorder_baselines(rec):
    pool = _pool(rec)
    target = G.resample_arc([0.1, 0.9], 8)
    seq = G.beam(pool, target, 0.5, 0.5, 5)
    r = G.random_order(seq, np.random.default_rng(0))
    e = G.energy_sorted(seq, pool, target)
    for s in (r, e):
        assert s[0] == seq[0] and sorted(s) == sorted(seq)
    mae = lambda s: np.abs(pool.energy[s[1:]] - target[1:len(s)]).sum()  # noqa: E731
    best = min(mae([seq[0], *p]) for p in itertools.permutations(seq[1:]))
    assert mae(e) == pytest.approx(best)  # sorted assignment is L1-optimal


def test_reorder_baselines_keep_seed_position(rec):
    pool = _pool(rec)
    target = G.resample_arc([0.9, 0.1], 8)
    seq = G.beam(pool, target, 0.5, 0.5, 5, seed_pos=3)
    for s in (G.random_order(seq, np.random.default_rng(1), 3), G.energy_sorted(seq, pool, target, 3)):
        assert s[3] == 0 and sorted(s) == sorted(seq)


def test_generate_api(rec, gcfg):
    target = G.resample_arc([0.4, 0.8], 10)
    cfg = {**gcfg, "pool": {**gcfg["pool"], "k": 100, "per_seed_k": 200, "energy_bands": None},
           "constraints": {**gcfg["constraints"], "max_dbpm": None},
           "objective": {**gcfg["objective"], "weak_link": {"lambda": 0.0, "tau": 0.85}},
           "seed_position": "start"}
    out = G.generate(rec, [12], target, cfg)
    assert len(out["idx"]) == 10 and out["idx"][0] == 12
    assert out["total_ms"] >= out["search_ms"] >= 0
    g = G.generate(rec, [12], target, cfg, generator="greedy_v0", pool=out["pool"])
    b1 = G.generate(rec, [12], target, cfg, width=1, pool=out["pool"])
    assert g["seq"] == b1["seq"]
    with pytest.raises(ValueError):
        G.generate(rec, [12], target, cfg, generator="nope")
