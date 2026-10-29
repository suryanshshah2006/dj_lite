"""Arc-following playlist generators (Step 5).

Given seed tracks, a target energy arc e_1..e_N and weights (alpha, gamma), build an
N-track playlist starting with the first seed that maximizes, summed over positions 2..N,

    alpha * relevance(track, seed set) + (1 - alpha) * transition(prev, track)
    + gamma * (1 - |energy(track) - e_i|)

Relevance is always to the seed set (mean over seeds), never to the previous track, so the
playlist cannot drift away from the seeds. Hard constraints: no repeated tracks, at most
one track per artist (the first seed's artists included), optional max |dBPM| per
transition (octave-aligned, off by default).

Everything runs on a fixed candidate pool per request (`build_pool`): the union of FAISS
top-k per seed, ranked by relevance to the seed set and capped at k, plus a float32
pool x pool transition matrix computed once.

Generators:
  greedy_v0  argmax of the step objective at each position (on the pool)
  beam_v1    beam search over sequences, width B; B = 1 is exactly greedy_v0
Baselines that reuse a playlist's tracks: random order, energy-sorted to the arc.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from djlite.retrieval import PROJECT_ROOT, Recommender
from djlite.transitions import tempo_octave_align

GENERATORS = ("beam_v1", "greedy_v0")


def load_generator_config(path: Path | None = None) -> dict:
    p = path or PROJECT_ROOT / "configs" / "generator.yaml"
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def resample_arc(points, n: int) -> np.ndarray:
    """Linear interpolation of control points onto n evenly spaced positions (same as the
    web app's resampleArc)."""
    pts = np.clip(np.asarray(points, dtype=float), 0.0, 1.0)
    if n <= 1 or len(pts) == 1:
        return np.full(max(n, 1), pts[0])
    return np.interp(np.linspace(0, len(pts) - 1, n), np.arange(len(pts)), pts)


@dataclass
class Pool:
    """Candidate pool for one request. Position 0 is the first seed (the start track)."""
    idx: np.ndarray        # global track indices, shape (P,)
    seeds: np.ndarray      # global indices of the seed tracks
    rel: np.ndarray        # relevance to the seed set (mean over seeds), float64 (P,)
    T: np.ndarray          # transition score matrix, float32 (P, P)
    energy: np.ndarray     # (P,)
    blocks: list           # blocks[p]: pool positions sharing an artist with p (incl. p)
    dbpm: np.ndarray | None  # octave-aligned |dBPM| matrix, float32 (P, P), if a cap is set
    build_ms: float


def build_pool(rec: Recommender, seeds, k: int = 2000, per_seed_k: int = 2000,
               need_dbpm: bool = False, chunk: int = 512) -> Pool:
    t0 = time.perf_counter()
    seeds = np.asarray(seeds, dtype=np.int64)
    cand = rec.retrieve(seeds, k=min(per_seed_k, len(rec.tracks))).ravel()
    cand = np.unique(cand[cand >= 0])
    cand = cand[cand != seeds[0]]
    rel = np.mean([rec.relevance(int(s), cand)["relevance"] for s in seeds], axis=0)
    keep = np.argsort(-rel, kind="stable")[:k]
    cand, rel = cand[keep], rel[keep]
    start_rel = float(np.mean([rec.relevance(int(s), seeds[:1])["relevance"][0] for s in seeds]))
    idx = np.concatenate([seeds[:1], cand])
    rel = np.concatenate([[start_rel], rel])

    P = len(idx)
    T = np.empty((P, P), dtype=np.float32)
    for s in range(0, P, chunk):  # row chunks keep float64 temporaries ~P*chunk*8 bytes
        T[s:s + chunk] = rec.transition_block(idx[s:s + chunk], idx)
    dbpm = None
    if need_dbpm:
        tempo = rec.arr["tempo"][idx]
        dbpm = np.empty((P, P), dtype=np.float32)
        for s in range(0, P, chunk):
            a = tempo[s:s + chunk, None]
            _, _, b2 = tempo_octave_align(a, tempo[None, :])
            dbpm[s:s + chunk] = np.abs(a - b2)

    by_artist: dict = {}
    for p, i in enumerate(idx):
        for a in rec.artist_sets[int(i)]:
            by_artist.setdefault(a, []).append(p)
    blocks = []
    for p, i in enumerate(idx):
        b = {p}
        for a in rec.artist_sets[int(i)]:
            b.update(by_artist[a])
        blocks.append(np.fromiter(b, dtype=np.int64))
    return Pool(idx=idx, seeds=seeds, rel=rel, T=T, energy=rec.arr["energy"][idx], blocks=blocks,
                dbpm=dbpm, build_ms=(time.perf_counter() - t0) * 1000)


def _initial_block(pool: Pool) -> np.ndarray:
    blocked = np.zeros(len(pool.idx), dtype=bool)
    blocked[pool.blocks[0]] = True  # the start track and its artists
    return blocked


def _step_scores(pool: Pool, last: np.ndarray, target: float, alpha: float,
                 gamma: float) -> np.ndarray:
    """Step objective for every candidate after each of `last` (pool positions): (len, P)."""
    fit = 1.0 - np.abs(pool.energy - target)
    return (alpha * pool.rel)[None, :] + (1 - alpha) * pool.T[last].astype(np.float64) \
        + (gamma * fit)[None, :]


def _mask(pool: Pool, S: np.ndarray, last: np.ndarray, blocked: np.ndarray,
          max_dbpm: float | None) -> None:
    S[blocked] = -np.inf
    if max_dbpm is not None:
        if pool.dbpm is None:
            raise ValueError("max_dbpm needs build_pool(..., need_dbpm=True)")
        S[pool.dbpm[last] > max_dbpm] = -np.inf


def greedy(pool: Pool, target: np.ndarray, alpha: float, gamma: float,
           max_dbpm: float | None = None) -> list[int]:
    """greedy_v0 on the pool: at each position the argmax of the step objective
    (ties -> lowest pool position). Returns pool positions, starting with 0."""
    seq = [0]
    blocked = _initial_block(pool)
    for t in range(1, len(target)):
        S = _step_scores(pool, np.array([seq[-1]]), target[t], alpha, gamma)
        _mask(pool, S, np.array([seq[-1]]), blocked[None, :], max_dbpm)
        c = int(np.argmax(S[0]))
        if not np.isfinite(S[0, c]):
            break  # pool exhausted under the constraints
        seq.append(c)
        blocked[pool.blocks[c]] = True
    return seq


def beam(pool: Pool, target: np.ndarray, alpha: float, gamma: float, width: int,
         max_dbpm: float | None = None) -> list[int]:
    """beam_v1: keep the `width` best partial playlists (by summed objective) at each
    position. Ties are broken by (beam rank, pool position), so width = 1 reproduces
    greedy() exactly. Returns pool positions of the best full playlist."""
    P = len(pool.idx)
    seqs = np.zeros((1, 1), dtype=np.int64)
    scores = np.zeros(1)
    blocked = _initial_block(pool)[None, :]
    for t in range(1, len(target)):
        last = seqs[:, -1]
        S = scores[:, None] + _step_scores(pool, last, target[t], alpha, gamma)
        _mask(pool, S, last, blocked, max_dbpm)
        flat = S.ravel()
        if width == 1:
            top = np.array([int(np.argmax(flat))])
        else:
            m = min(width, flat.size)
            part = np.argpartition(-flat, m - 1)[:m]
            top = part[np.lexsort((part, -flat[part]))]  # by score desc, then flat index
        top = top[np.isfinite(flat[top])]
        if len(top) == 0:
            break  # no beam can be extended under the constraints
        b, c = np.divmod(top, P)
        seqs = np.concatenate([seqs[b], c[:, None]], axis=1)
        scores = flat[top]
        blocked = blocked[b].copy()
        for r, cc in enumerate(c):
            blocked[r, pool.blocks[cc]] = True
    return [int(x) for x in seqs[int(np.argmax(scores))]]


def random_order(seq: list[int], rng: np.random.Generator) -> list[int]:
    """Same tracks, start track first, the rest shuffled (isolates the value of ordering)."""
    rest = list(seq[1:])
    rng.shuffle(rest)
    return [seq[0], *rest]


def energy_sorted(seq: list[int], pool: Pool, target: np.ndarray) -> list[int]:
    """Same tracks, start track first; the k-th lowest-energy track goes to the position
    with the k-th lowest target (the L1-optimal assignment of energies to the arc)."""
    rest = np.asarray(seq[1:])
    tgt = np.asarray(target[1:len(seq)])
    by_energy = rest[np.argsort(pool.energy[rest], kind="stable")]
    out = np.empty_like(rest)
    out[np.argsort(tgt, kind="stable")] = by_energy
    return [seq[0], *[int(x) for x in out]]


def objective(pool: Pool, seq: list[int], target: np.ndarray, alpha: float,
              gamma: float) -> float:
    s = np.asarray(seq)
    fit = 1.0 - np.abs(pool.energy[s[1:]] - target[1:len(s)])
    return float(np.sum(alpha * pool.rel[s[1:]] + (1 - alpha) * pool.T[s[:-1], s[1:]]
                        + gamma * fit))


def generate(rec: Recommender, seeds, target: np.ndarray, gcfg: dict, *,
             generator: str = "beam_v1", alpha: float | None = None,
             gamma: float | None = None, width: int | None = None,
             max_dbpm: float | None = None, pool: Pool | None = None) -> dict:
    """One playlist. Returns {"pool", "seq" (pool positions), "idx" (global indices),
    "search_ms", "total_ms"}."""
    if generator not in GENERATORS:
        raise ValueError(f"generator must be one of {GENERATORS}")
    o, c = gcfg["objective"], gcfg["constraints"]
    if c["max_per_artist"] != 1:
        raise ValueError("only max_per_artist = 1 is supported")
    alpha = o["alpha"] if alpha is None else alpha
    gamma = o["gamma"] if gamma is None else gamma
    max_dbpm = c["max_dbpm"] if max_dbpm is None else max_dbpm
    if pool is None:
        pool = build_pool(rec, seeds, k=gcfg["pool"]["k"], per_seed_k=gcfg["pool"]["per_seed_k"],
                          need_dbpm=max_dbpm is not None)
    t0 = time.perf_counter()
    if generator == "greedy_v0":
        seq = greedy(pool, target, alpha, gamma, max_dbpm)
    else:
        seq = beam(pool, target, alpha, gamma, width or gcfg["beam"]["width"], max_dbpm)
    search_ms = (time.perf_counter() - t0) * 1000
    return {"pool": pool, "seq": seq, "idx": pool.idx[seq], "alpha": alpha, "gamma": gamma,
            "search_ms": search_ms, "total_ms": pool.build_ms + search_ms}
