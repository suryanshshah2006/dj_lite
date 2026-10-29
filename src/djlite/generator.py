"""Arc-following playlist generators (Step 5, 5.1).

Given seed tracks, a target energy arc e_1..e_N and weights, build an N-track playlist
containing the first seed (at position 1, or at the arc position that best fits its
energy) that maximizes, summed over every non-seed position i,

    alpha * relevance(track_i, seed set)
    + (1 - alpha) * T_i  -  lambda * max(0, tau - T_i)      (weak-link penalty)
    + gamma * (1 - |energy(track_i) - e_i|)

where T_i is the transition score between track_i and its already-placed neighbour (the
previous track when filling forward, the next one when filling backward), so every
transition is counted exactly once. Relevance is always to the seed set (mean over seeds),
never to the neighbour, so the playlist cannot drift away from the seeds. Hard constraints:
no repeated tracks, at most one track per artist (the seed's artists included), optional
max |dBPM| per transition (octave-aligned).

Everything runs on a fixed candidate pool per request (`build_pool`): FAISS top-k per seed,
ranked by relevance to the seed set and capped at k -- optionally stratified into energy
bands so that the extremes of an arc have candidates -- plus a float32 pool x pool
transition matrix computed once.

Generators (same fill order: seed position, then forward to the end, then backward):
  greedy_v0  argmax of the step objective at each position
  beam_v1    beam search over partial playlists, width B; B = 1 is exactly greedy_v0
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
SEED_POSITIONS = ("start", "best_fit")


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
    """Candidate pool for one request. Position 0 is the first seed."""
    idx: np.ndarray        # global track indices, shape (P,)
    seeds: np.ndarray      # global indices of the seed tracks
    rel: np.ndarray        # relevance to the seed set (mean over seeds), float64 (P,)
    T: np.ndarray          # transition score matrix, float32 (P, P): T[a, b] = a -> b
    energy: np.ndarray     # (P,)
    blocks: list           # blocks[p]: pool positions sharing an artist with p (incl. p)
    dbpm: np.ndarray | None  # octave-aligned |dBPM| matrix, float32 (P, P), if requested
    build_ms: float


def _stratified(cand: np.ndarray, rel: np.ndarray, energy: np.ndarray, k: int,
                bands: int) -> np.ndarray:
    """Indices into cand: the k // bands most relevant candidates of each energy-quantile
    band (quantiles of the candidates' energy); unused budget (a band with too few
    candidates) goes to the most relevant remaining candidates. Sorted by relevance."""
    edges = np.quantile(energy, np.linspace(0, 1, bands + 1)[1:-1])
    band = np.searchsorted(edges, energy, side="right")
    order = np.argsort(-rel, kind="stable")
    per = k // bands
    take = np.zeros(len(cand), dtype=bool)
    for b in range(bands):
        take[order[band[order] == b][:per]] = True
    rest = order[~take[order]][:max(0, k - int(take.sum()))]
    take[rest] = True
    sel = np.flatnonzero(take)
    return sel[np.argsort(-rel[sel], kind="stable")]


def build_pool(rec: Recommender, seeds, k: int = 2000, per_seed_k: int = 2000,
               need_dbpm: bool = False, energy_bands: int | None = None,
               chunk: int = 512) -> Pool:
    """energy_bands: None = top-k by relevance; an int = stratify the k slots over that
    many energy-quantile bands (use a per_seed_k well above k so each band has depth)."""
    t0 = time.perf_counter()
    seeds = np.asarray(seeds, dtype=np.int64)
    cand = rec.retrieve(seeds, k=min(per_seed_k, len(rec.tracks))).ravel()
    cand = np.unique(cand[cand >= 0])
    cand = cand[cand != seeds[0]]
    rel = np.mean([rec.relevance(int(s), cand)["relevance"] for s in seeds], axis=0)
    if energy_bands:
        keep = _stratified(cand, rel, rec.arr["energy"][cand], k, energy_bands)
    else:
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


def seed_slot(pool: Pool, target: np.ndarray, seed_position: str = "start") -> int:
    """Playlist position of the first seed: 0, or where the target is closest to its energy
    (earliest on ties)."""
    if seed_position == "start":
        return 0
    if seed_position == "best_fit":
        return int(np.argmin(np.abs(np.asarray(target) - pool.energy[0])))
    raise ValueError(f"seed_position must be one of {SEED_POSITIONS}")


def fill_order(n: int, seed_pos: int) -> list[tuple[int, int, int]]:
    """(position, neighbour position, direction): forward from the seed to the end, then
    backward from the seed to the start. direction +1: neighbour -> track; -1: track ->
    neighbour."""
    return ([(t, t - 1, 1) for t in range(seed_pos + 1, n)]
            + [(t, t + 1, -1) for t in range(seed_pos - 1, -1, -1)])


def _initial_block(pool: Pool) -> np.ndarray:
    blocked = np.zeros(len(pool.idx), dtype=bool)
    blocked[pool.blocks[0]] = True  # the seed and its artists
    return blocked


def _step_scores(pool: Pool, nb: np.ndarray, direction: int, target: float, alpha: float,
                 gamma: float, lam: float, tau: float, max_dbpm: float | None,
                 blocked: np.ndarray) -> np.ndarray:
    """Step objective for every candidate next to each neighbour in `nb`: (len(nb), P),
    -inf where a constraint is violated."""
    tr = (pool.T[nb] if direction > 0 else pool.T[:, nb].T).astype(np.float64)
    fit = 1.0 - np.abs(pool.energy - target)
    S = (alpha * pool.rel)[None, :] + (1 - alpha) * tr + (gamma * fit)[None, :]
    if lam:
        S -= lam * np.maximum(0.0, tau - tr)
    S[blocked] = -np.inf
    if max_dbpm is not None:
        if pool.dbpm is None:
            raise ValueError("max_dbpm needs build_pool(..., need_dbpm=True)")
        d = pool.dbpm[nb] if direction > 0 else pool.dbpm[:, nb].T
        S[d > max_dbpm] = -np.inf
    return S


def _compact(seq: np.ndarray) -> list[int]:
    # filled positions always form one contiguous block (the fill extends its ends)
    return [int(x) for x in seq if x >= 0]


def greedy(pool: Pool, target: np.ndarray, alpha: float, gamma: float,
           max_dbpm: float | None = None, *, lam: float = 0.0, tau: float = 0.85,
           seed_pos: int = 0) -> list[int]:
    """greedy_v0: at each position (in fill order) the argmax of the step objective
    (ties -> lowest pool position). Returns pool positions in playlist order."""
    n = len(target)
    seq = np.full(n, -1, dtype=np.int64)
    seq[seed_pos] = 0
    blocked = _initial_block(pool)
    for t, nb, d in fill_order(n, seed_pos):
        S = _step_scores(pool, np.array([seq[nb]]), d, target[t], alpha, gamma, lam, tau,
                         max_dbpm, blocked[None, :])
        c = int(np.argmax(S[0]))
        if not np.isfinite(S[0, c]):
            break  # pool exhausted under the constraints
        seq[t] = c
        blocked[pool.blocks[c]] = True
    return _compact(seq)


def beam(pool: Pool, target: np.ndarray, alpha: float, gamma: float, width: int,
         max_dbpm: float | None = None, *, lam: float = 0.0, tau: float = 0.85,
         seed_pos: int = 0) -> list[int]:
    """beam_v1: keep the `width` best partial playlists (by summed objective) after each
    position. Ties are broken by (beam rank, pool position), so width = 1 reproduces
    greedy() exactly. Returns pool positions of the best playlist, in playlist order."""
    n, P = len(target), len(pool.idx)
    seqs = np.full((1, n), -1, dtype=np.int64)
    seqs[0, seed_pos] = 0
    scores = np.zeros(1)
    blocked = _initial_block(pool)[None, :]
    for t, nb, d in fill_order(n, seed_pos):
        S = scores[:, None] + _step_scores(pool, seqs[:, nb], d, target[t], alpha, gamma,
                                           lam, tau, max_dbpm, blocked)
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
        seqs = seqs[b].copy()
        seqs[:, t] = c
        scores = flat[top]
        blocked = blocked[b].copy()
        for r, cc in enumerate(c):
            blocked[r, pool.blocks[cc]] = True
    return _compact(seqs[int(np.argmax(scores))])


def random_order(seq: list[int], rng: np.random.Generator, seed_pos: int = 0) -> list[int]:
    """Same tracks, seed kept at its position, the rest shuffled (value of ordering)."""
    rest = [x for k, x in enumerate(seq) if k != seed_pos]
    rng.shuffle(rest)
    return rest[:seed_pos] + [seq[seed_pos]] + rest[seed_pos:]


def energy_sorted(seq: list[int], pool: Pool, target: np.ndarray, seed_pos: int = 0) -> list[int]:
    """Same tracks, seed kept at its position; the k-th lowest-energy track goes to the
    free position with the k-th lowest target (the L1-optimal assignment)."""
    free = np.array([k for k in range(len(seq)) if k != seed_pos])
    rest = np.array([x for k, x in enumerate(seq) if k != seed_pos])
    out = np.array(seq)
    out[free[np.argsort(np.asarray(target)[free], kind="stable")]] = \
        rest[np.argsort(pool.energy[rest], kind="stable")]
    return [int(x) for x in out]


def objective(pool: Pool, seq: list[int], target: np.ndarray, alpha: float, gamma: float,
              *, lam: float = 0.0, tau: float = 0.85, seed_pos: int = 0) -> float:
    """The search objective of a full playlist (pool positions in playlist order)."""
    s = np.asarray(seq)
    tr = pool.T[s[:-1], s[1:]].astype(np.float64)
    other = np.array([k for k in range(len(s)) if k != seed_pos])
    fit = 1.0 - np.abs(pool.energy[s[other]] - np.asarray(target)[other])
    return float(np.sum(alpha * pool.rel[s[other]] + gamma * fit)
                 + np.sum((1 - alpha) * tr - lam * np.maximum(0.0, tau - tr)))


def pool_kwargs(gcfg: dict, need_dbpm: bool) -> dict:
    p = gcfg["pool"]
    bands = p.get("energy_bands")
    return {"k": p["k"], "need_dbpm": need_dbpm, "energy_bands": bands,
            "per_seed_k": p["stratified_retrieve_k"] if bands else p["per_seed_k"]}


def generate(rec: Recommender, seeds, target: np.ndarray, gcfg: dict, *,
             generator: str = "beam_v1", alpha: float | None = None,
             gamma: float | None = None, width: int | None = None,
             max_dbpm: float | None = None, lam: float | None = None,
             seed_position: str | None = None, pool: Pool | None = None) -> dict:
    """One playlist with the config's defaults unless overridden. Returns {"pool", "seq"
    (pool positions, playlist order), "idx" (global indices), "seed_pos", weights used,
    "search_ms", "total_ms"}. max_dbpm=None means the config's cap (0 or less: no cap)."""
    if generator not in GENERATORS:
        raise ValueError(f"generator must be one of {GENERATORS}")
    o, c = gcfg["objective"], gcfg["constraints"]
    if c["max_per_artist"] != 1:
        raise ValueError("only max_per_artist = 1 is supported")
    alpha = o["alpha"] if alpha is None else alpha
    gamma = o["gamma"] if gamma is None else gamma
    wl = o.get("weak_link", {"lambda": 0.0, "tau": 0.85})
    lam = wl["lambda"] if lam is None else lam
    cap = c["max_dbpm"] if max_dbpm is None else max_dbpm
    cap = cap if cap is not None and cap > 0 else None
    if pool is None:
        pool = build_pool(rec, seeds, **pool_kwargs(gcfg, need_dbpm=cap is not None))
    sp = seed_slot(pool, target, seed_position or gcfg.get("seed_position", "start"))
    t0 = time.perf_counter()
    kw = dict(lam=lam, tau=wl["tau"], seed_pos=sp)
    if generator == "greedy_v0":
        seq = greedy(pool, target, alpha, gamma, cap, **kw)
    else:
        seq = beam(pool, target, alpha, gamma, width or gcfg["beam"]["width"], cap, **kw)
    search_ms = (time.perf_counter() - t0) * 1000
    return {"pool": pool, "seq": seq, "idx": pool.idx[seq], "seed_pos": sp, "alpha": alpha,
            "gamma": gamma, "lam": lam, "tau": wl["tau"], "max_dbpm": cap,
            "search_ms": search_ms, "total_ms": pool.build_ms + search_ms}
