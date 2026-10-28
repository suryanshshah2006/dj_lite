import numpy as np
import pandas as pd
import pytest

from djlite.features import add_camelot, camelot_from_key_mode
from djlite.transitions import (
    harmonic_score, load_config, tempo_score, transition_score_v0,
)

# Standard Camelot wheel. Spotify key: C=0 ... B=11; mode 1 = major (B), 0 = minor (A).
CAMELOT_MAJOR = {0: "8B", 7: "9B", 2: "10B", 9: "11B", 4: "12B", 11: "1B",
                 6: "2B", 1: "3B", 8: "4B", 3: "5B", 10: "6B", 5: "7B"}
CAMELOT_MINOR = {9: "8A", 4: "9A", 11: "10A", 6: "11A", 1: "12A", 8: "1A",
                 3: "2A", 10: "3A", 5: "4A", 0: "5A", 7: "6A", 2: "7A"}
ALL_24 = [(k, 1, c) for k, c in CAMELOT_MAJOR.items()] + \
         [(k, 0, c) for k, c in CAMELOT_MINOR.items()]

HARM = load_config()["harmonic"]


# ---------- Camelot mapping ----------

def test_mapping_covers_24_distinct_codes():
    assert len(ALL_24) == 24
    assert len({c for _, _, c in ALL_24}) == 24


@pytest.mark.parametrize("key, mode, expected", ALL_24)
def test_camelot_scalar(key, mode, expected):
    num, letter = camelot_from_key_mode(key, mode)
    assert f"{int(num)}{letter}" == expected


def test_camelot_vectorized_and_null():
    keys = [k for k, _, _ in ALL_24] + [-1]
    modes = [m for _, m, _ in ALL_24] + [1]
    df = add_camelot(pd.DataFrame({"key": keys, "mode": modes}))
    assert df["camelot"].iloc[:24].tolist() == [c for _, _, c in ALL_24]
    assert df[["camelot_num", "camelot_letter", "camelot"]].iloc[24].isna().all()


# ---------- harmonic score ----------

@pytest.mark.parametrize("a, b, expected", [
    ("8A", "8A", 1.0),     # same code
    ("8A", "9A", 1.0),     # +1 same letter
    ("8A", "7A", 1.0),     # -1 same letter
    ("8A", "8B", 1.0),     # relative major/minor
    ("8B", "8A", 1.0),
    ("12A", "1A", 1.0),    # wraparound
    ("1A", "12A", 1.0),
    ("12B", "1B", 1.0),
    ("8A", "9B", 0.6),     # diagonal
    ("12A", "1B", 0.6),    # diagonal across the wrap
    ("8A", "10A", 0.3),    # two steps
    ("11A", "1A", 0.3),    # two steps across the wrap
    ("8A", "10B", 0.0),
    ("8A", "2A", 0.0),     # opposite side of the wheel
])
def test_harmonic_score(a, b, expected):
    s = harmonic_score(int(a[:-1]), a[-1], int(b[:-1]), b[-1], **HARM)
    assert s == pytest.approx(expected)


def test_harmonic_null_key_is_neutral():
    assert harmonic_score(np.nan, None, 8, "A", **HARM) == 0.5
    assert harmonic_score(8, "A", None, None, **HARM) == 0.5


# ---------- tempo score ----------

@pytest.mark.parametrize("pct, expected", [(0.0, 1.0), (0.02, 1.0), (0.05, 0.5), (0.08, 0.0)])
def test_tempo_score_points(pct, expected):
    s = tempo_score(100.0, 100.0 * (1 + pct), full_pct=0.02, zero_pct=0.08, octave_wrap=False)
    assert s == pytest.approx(expected)


def test_tempo_score_symmetric_and_beyond_zero():
    kw = dict(full_pct=0.02, zero_pct=0.08, octave_wrap=False)
    assert tempo_score(100, 105, **kw) == pytest.approx(tempo_score(105, 100, **kw))
    assert tempo_score(100, 130, **kw) == 0.0


def test_tempo_octave_wrap_across_fold_seam():
    kw = dict(full_pct=0.02, zero_pct=0.08)
    assert tempo_score(159.0, 81.0, octave_wrap=False, **kw) == 0.0
    assert tempo_score(159.0, 81.0, octave_wrap=True, **kw) == 1.0


# ---------- single pair vs vectorized ----------

def _random_tracks(n, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "key": rng.integers(0, 12, n), "mode": rng.integers(0, 2, n),
        "tempo_folded": rng.uniform(80, 160, n), "energy": rng.uniform(0, 1, n),
        "loudness": rng.uniform(-30, 0, n), "year": rng.integers(1950, 2021, n),
    })
    for c in ["danceability", "acousticness", "instrumentalness", "speechiness", "valence"]:
        df[c + "_z"] = rng.normal(size=n)
    df.loc[3, "key"] = -1  # one unknown key
    return add_camelot(df)


def test_single_pair_matches_vectorized():
    cfg = load_config()
    df = _random_tracks(200)
    seed = df.iloc[0]
    vec = transition_score_v0(seed, df, cfg)
    for i in range(len(df)):
        single = transition_score_v0(seed, df.iloc[i], cfg)
        for k, v in single.items():
            assert isinstance(v, float)
            assert v == pytest.approx(vec[k][i]), (k, i)


def test_pairwise_dataframes_match_single():
    cfg = load_config()
    a, b = _random_tracks(100, seed=1), _random_tracks(100, seed=2)
    vec = transition_score_v0(a, b, cfg)
    for i in range(len(a)):
        single = transition_score_v0(a.iloc[i], b.iloc[i], cfg)
        assert single["score"] == pytest.approx(vec["score"][i])


def test_score_in_unit_interval():
    cfg = load_config()
    df = _random_tracks(500)
    s = transition_score_v0(df.iloc[5], df, cfg)["score"]
    assert np.all((s >= 0) & (s <= 1))
