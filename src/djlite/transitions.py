"""Rule-based transition scorer v0: how well does track A flow into track B?

Every component function broadcasts with numpy, so it works for a single pair
(scalars in, float out) and for one track against many candidates (scalar A,
array B, array out), or for many aligned pairs (array A, array B).
Thresholds and weights come from configs/transitions.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPONENTS = ["harmonic", "tempo", "energy_smoothness", "timbre", "loudness", "era"]


def load_config(path: Path | None = None) -> dict:
    path = path or PROJECT_ROOT / "configs" / "transitions.yaml"
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _out(x: np.ndarray):
    """Return a Python float for 0-d results, the array otherwise."""
    return float(x) if np.ndim(x) == 0 else x


def _ramp_down(x, full, zero):
    """1.0 at x <= full, linear down to 0.0 at x >= zero."""
    return np.clip((zero - x) / (zero - full), 0.0, 1.0)


def harmonic_score(num_a, letter_a, num_b, letter_b, *, same, adjacent, relative, diagonal,
                   two_step, other, null_key):
    """Camelot compatibility with circular distance on the 12-number wheel."""
    na = np.asarray(num_a, dtype=float)
    nb = np.asarray(num_b, dtype=float)
    la = np.asarray(letter_a, dtype=object)
    lb = np.asarray(letter_b, dtype=object)
    diff = np.abs(na - nb) % 12
    dist = np.minimum(diff, 12 - diff)
    same_letter = la == lb
    score = np.select(
        [same_letter & (dist == 0), same_letter & (dist == 1), ~same_letter & (dist == 0),
         ~same_letter & (dist == 1), same_letter & (dist == 2)],
        [same, adjacent, relative, diagonal, two_step],
        default=other,
    ).astype(float)
    score = np.where(np.isnan(na) | np.isnan(nb), null_key, score)
    return _out(score)


def tempo_pct_diff(tempo_a, tempo_b, *, octave_wrap):
    """|a - b| / min(a, b) on folded tempos; optionally also across the fold seam."""
    a = np.asarray(tempo_a, dtype=float)
    b = np.asarray(tempo_b, dtype=float)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    pct = (hi - lo) / lo
    if octave_wrap:
        # 159 vs 81 BPM: doubling the slower gives 162 vs 159, i.e. ~1.9%.
        lo2 = 2 * lo
        pct = np.minimum(pct, np.abs(lo2 - hi) / np.minimum(lo2, hi))
    return _out(pct)


def tempo_score(tempo_a, tempo_b, *, full_pct, zero_pct, octave_wrap):
    pct = tempo_pct_diff(tempo_a, tempo_b, octave_wrap=octave_wrap)
    return _out(_ramp_down(np.asarray(pct), full_pct, zero_pct))


def tempo_octave_align(tempo_a, tempo_b):
    """Halve/double B to the power-of-2 multiple nearest A (in log space).

    Returns (pct difference |a - b'| / min(a, b'), octave shift k, aligned b').
    k != 0 means a half- or double-time match.
    """
    a = np.asarray(tempo_a, dtype=float)
    b = np.asarray(tempo_b, dtype=float)
    k = np.round(np.log2(b / a))
    b2 = b / np.exp2(k)
    pct = np.abs(a - b2) / np.minimum(a, b2)
    return pct, k, b2


def tempo_score_v0_1(tempo_a, tempo_b, *, sigma, octave_penalty):
    """exp(-(pct/sigma)^2) on raw tempos after octave alignment, penalized if k != 0."""
    pct, k, _ = tempo_octave_align(tempo_a, tempo_b)
    score = np.exp(-((pct / sigma) ** 2)) * np.where(k != 0, octave_penalty, 1.0)
    return _out(score)


def energy_delta(energy_a, energy_b):
    """Signed energy change B - A (raw, used later by the energy-arc logic)."""
    return _out(np.asarray(energy_b, dtype=float) - np.asarray(energy_a, dtype=float))


def energy_smoothness(energy_a, energy_b):
    d = np.asarray(energy_delta(energy_a, energy_b))
    return _out(np.clip(1.0 - np.abs(d), 0.0, 1.0))


def loudness_score(loud_a, loud_b, *, full_db, zero_db):
    d = np.abs(np.asarray(loud_b, dtype=float) - np.asarray(loud_a, dtype=float))
    return _out(_ramp_down(d, full_db, zero_db))


def timbre_similarity(vec_a, vec_b):
    """Cosine similarity of standardized feature vectors (last axis), mapped to [0, 1].

    Zero vectors (a track exactly at the dataset mean) get a neutral 0.5.
    """
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    dot = np.sum(a * b, axis=-1)
    norm = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cos = np.divide(dot, norm, out=np.zeros_like(dot, dtype=float), where=norm > 0)
    return _out(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))


def era_score(year_a, year_b, *, full_years, floor_years, min_score):
    gap = np.abs(np.asarray(year_b, dtype=float) - np.asarray(year_a, dtype=float))
    frac = np.clip((gap - full_years) / (floor_years - full_years), 0.0, 1.0)
    return _out(1.0 - (1.0 - min_score) * frac)


# ---------- track-level wrapper ----------

TrackLike = pd.Series | pd.DataFrame | Mapping


def _col(x: TrackLike, name: str, dtype=float):
    """Column as a numpy array (DataFrame) or a scalar (Series / dict); NA -> nan/None."""
    na_value = np.nan if dtype is float else None
    if isinstance(x, pd.DataFrame):
        return x[name].to_numpy(dtype=dtype, na_value=na_value)
    v = x[name]
    if isinstance(v, np.ndarray):  # dict of precomputed arrays (see retrieval.py)
        return v
    return na_value if pd.isna(v) else (float(v) if dtype is float else v)


def _vec(x: TrackLike, names: list[str]) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        return x[names].to_numpy(dtype=float)
    if isinstance(x[names[0]], np.ndarray):
        return np.column_stack([x[n] for n in names])
    return np.array([float(x[n]) for n in names])


def transition_score_v0(a: TrackLike, b: TrackLike, cfg: dict, z_suffix: str = "_z") -> dict:
    """Score A -> B. Returns {component: value, 'energy_delta': ..., 'score': ...}.

    a, b: a single track (Series/dict) or many (DataFrame, or dict of arrays). One track
    vs many scores against every row; two equal-length inputs score row-aligned pairs.
    """
    tempo = tempo_score(_col(a, "tempo_folded"), _col(b, "tempo_folded"), **cfg["tempo"])
    return _combine(a, b, cfg, cfg["harmonic"], tempo, z_suffix)


def transition_score_v0_1(a: TrackLike, b: TrackLike, cfg: dict,
                          z_suffix: str = "_z") -> dict:
    """v0.1: v0 with the v0_1 harmonic table and the Gaussian, octave-aware tempo score."""
    v = cfg["v0_1"]
    tempo = tempo_score_v0_1(_col(a, "tempo"), _col(b, "tempo"), **v["tempo"])
    return _combine(a, b, cfg, v["harmonic"], tempo, z_suffix)


def _combine(a: TrackLike, b: TrackLike, cfg: dict, harm_cfg: dict, tempo,
             z_suffix: str) -> dict:
    ld, e = cfg["loudness"], cfg["era"]
    timbre_cols = [c + z_suffix for c in cfg["timbre"]["columns"]]

    parts = {
        "harmonic": harmonic_score(
            _col(a, "camelot_num"), _col(a, "camelot_letter", object),
            _col(b, "camelot_num"), _col(b, "camelot_letter", object), **harm_cfg),
        "tempo": tempo,
        "energy_smoothness": energy_smoothness(_col(a, "energy"), _col(b, "energy")),
        "timbre": timbre_similarity(_vec(a, timbre_cols), _vec(b, timbre_cols)),
        "loudness": loudness_score(_col(a, "loudness"), _col(b, "loudness"), **ld),
        "era": era_score(_col(a, "year"), _col(b, "year"), **e),
    }
    w = cfg["weights"]
    total_w = sum(w[k] for k in COMPONENTS)
    score = sum(w[k] * np.asarray(parts[k]) for k in COMPONENTS) / total_w
    parts["energy_delta"] = energy_delta(_col(a, "energy"), _col(b, "energy"))
    parts["score"] = _out(np.asarray(score))
    return parts


SCORERS = {"v0": transition_score_v0, "v0_1": transition_score_v0_1}


def score_candidates(seed: TrackLike, candidates: pd.DataFrame, cfg: dict,
                     version: str = "v0") -> pd.DataFrame:
    """Score one seed against every candidate row; returns a breakdown DataFrame."""
    res = SCORERS[version](seed, candidates, cfg)
    return pd.DataFrame(res, index=candidates.index)
