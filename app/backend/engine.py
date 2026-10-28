"""Recommendation engine for the web app: search, track details, next tracks, mixes.

Wraps djlite's Recommender. Construct with `Engine.from_project()` (loads the full
dataset + FAISS index; ~1 minute) or directly with a small DataFrame for tests.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from djlite.retrieval import Recommender, load_configs, load_tracks
from djlite.transitions import COMPONENTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRACK_FIELDS = ["id", "name", "album", "year", "duration_ms", "key", "mode", "energy",
                "valence", "danceability", "acousticness", "instrumentalness", "speechiness",
                "liveness", "loudness"]


def _num(v):
    """JSON-safe number (NaN -> None)."""
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) else v


class Engine:
    def __init__(self, tracks: pd.DataFrame, style: np.ndarray, rcfg: dict, tcfg: dict,
                 settings: dict, popularity: np.ndarray | None = None):
        self.rec = Recommender(tracks, style, rcfg, tcfg)
        self.t = self.rec.tracks
        self.settings = settings
        n = len(self.t)
        # Popularity proxy for search ranking: how many duplicate releases a song had.
        self.popularity = popularity if popularity is not None else np.zeros(n)
        self.name_lc = self.t["name"].str.lower().to_numpy(dtype=object)
        self.artists_lc = np.array([" | ".join(a).lower() for a in self.t["artists"]], dtype=object)
        self._name_s = pd.Series(self.name_lc)
        self._art_s = pd.Series(self.artists_lc)

    @classmethod
    def from_project(cls, settings: dict) -> "Engine":
        paths, rcfg, tcfg = load_configs()
        tracks, style = load_tracks(paths)
        dedup = pd.read_parquet(PROJECT_ROOT / paths["processed_dir"] / "dedup_map.parquet",
                                columns=["kept_id"])
        counts = dedup["kept_id"].value_counts()
        pop = tracks["id"].map(counts).fillna(0).to_numpy(dtype=float)
        return cls(tracks, style, rcfg, tcfg, settings, popularity=pop)

    # ---------- serialization ----------

    def track_json(self, i: int) -> dict:
        r = self.t.iloc[int(i)]
        out = {f: (r[f].item() if hasattr(r[f], "item") else r[f]) for f in TRACK_FIELDS}
        out["year"] = None if pd.isna(r["year"]) else int(r["year"])
        out["artists"] = [str(a) for a in r["artists"]]
        out["bpm"] = round(float(r["tempo"]), 1)
        out["tempo_folded"] = round(float(r["tempo_folded"]), 1)
        has_key = not pd.isna(r["camelot_num"])
        out["camelot"] = str(r["camelot"]) if has_key else None
        out["camelot_num"] = int(r["camelot_num"]) if has_key else None
        out["camelot_letter"] = str(r["camelot_letter"]) if has_key else None
        out["has_style"] = bool(self.rec.has_style[int(i)])
        return out

    @staticmethod
    def transition_json(parts: dict, j: int) -> dict:
        out = {k: float(np.asarray(parts[k])[j]) for k in COMPONENTS}
        out["score"] = float(np.asarray(parts["transition"])[j])
        out["energy_delta"] = float(np.asarray(parts["energy_delta"])[j])
        return out

    @staticmethod
    def relevance_json(parts: dict, j: int) -> dict:
        return {"score": float(parts["relevance"][j]), "audio": float(parts["audio_sim"][j]),
                "style": _num(parts["style_sim"][j]), "era": float(parts["era_rel"][j])}

    # ---------- queries ----------

    def index_of(self, track_id: str) -> int:
        try:
            return int(self.rec.id_to_idx[track_id])
        except KeyError as exc:
            raise KeyError(track_id) from exc

    def search(self, q: str, limit: int | None = None) -> list[dict]:
        limit = limit or self.settings["search"]["limit"]
        tokens = [t for t in q.lower().split() if t]
        if not tokens:
            return []
        mask = np.ones(len(self.t), dtype=bool)
        for tok in tokens:  # every token must appear in the name or the artists
            mask &= (self._name_s.str.contains(tok, regex=False)
                     | self._art_s.str.contains(tok, regex=False)).to_numpy()
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return []
        ql = q.lower().strip()
        names = self.name_lc[idx]
        arts = self.artists_lc[idx]
        score = (
            4.0 * (names == ql)
            + 2.0 * np.array([n.startswith(ql) for n in names])
            + 1.5 * np.array([ql in a for a in arts])
            + 1.0 * np.array([ql in n for n in names])
            + 0.3 * np.log1p(self.popularity[idx])
            - 0.002 * np.array([len(n) for n in names])
        )
        order = idx[np.argsort(-score, kind="stable")][:limit]
        return [self.track_json(i) for i in order]

    def track(self, track_id: str) -> dict:
        return self.track_json(self.index_of(track_id))

    def next_tracks(self, seed_id: str, n: int, alpha: float) -> dict:
        i = self.index_of(seed_id)
        cand, parts = self.rec.score(i, "full")
        key = alpha * parts["relevance"] + (1 - alpha) * parts["transition"]
        cap = self.rec.rcfg["ranking"].get("max_per_artist")
        top = self.rec.select(cand, parts, key, n, caps=(cap,))[cap]
        pos = {c: p for p, c in enumerate(cand)}
        items = []
        for c in top["idx"]:
            j = pos[c]
            items.append({"track": self.track_json(c), "final": float(key[j]),
                          "relevance": self.relevance_json(parts, j),
                          "transition": self.transition_json(parts, j)})
        return {"seed": self.track_json(i), "alpha": alpha, "items": items}

    def mix(self, seed_ids: list[str], arc: list[float], alpha: float, length: int) -> dict:
        """greedy_v0: at each step pick argmax of
        alpha * relevance(seeds) + (1 - alpha) * transition(prev -> c) + w_arc * arc_fit."""
        ms = self.settings["mix"]
        seeds = [self.index_of(s) for s in seed_ids]
        target = np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(arc)),
                           np.clip(arc, 0, 1)) if len(arc) > 1 else np.full(length, arc[0])
        energy = self.rec.arr["energy"]
        cap = ms["max_per_artist"]
        artist_count: dict[str, int] = {}
        used: set[int] = set()

        def take(i: int) -> None:
            used.add(i)
            for a in self.rec.artist_sets[i]:
                artist_count[a] = artist_count.get(a, 0) + 1

        first = seeds[0]
        take(first)
        steps = [{"track": self.track_json(first), "target_energy": float(target[0]),
                  "relevance": 1.0, "transition": None,
                  "arc_fit": float(1 - abs(energy[first] - target[0])), "objective": None}]
        seed_pool = self.rec.retrieve(np.array(seeds), k=ms["k"]).ravel()
        current = first
        for t in range(1, length):
            pool = np.concatenate([self.rec.retrieve(np.array([current]), k=ms["k"])[0],
                                   seed_pool])
            pool = np.unique(pool[pool >= 0])
            ok = np.array([c not in used and all(artist_count.get(a, 0) < cap
                                                 for a in self.rec.artist_sets[c])
                           for c in pool], dtype=bool)
            cand = pool[ok]
            if len(cand) == 0:
                break
            rel = np.mean([self.rec.relevance(s, cand)["relevance"] for s in seeds], axis=0)
            tr = self.rec.transition(current, cand)
            fit = 1 - np.abs(energy[cand] - target[t])
            obj = alpha * rel + (1 - alpha) * np.asarray(tr["score"]) + ms["arc_weight"] * fit
            j = int(np.argmax(obj))
            c = int(cand[j])
            parts = {**tr, "transition": tr["score"]}
            steps.append({"track": self.track_json(c), "target_energy": float(target[t]),
                          "relevance": float(rel[j]), "transition": self.transition_json(parts, j),
                          "arc_fit": float(fit[j]), "objective": float(obj[j])})
            take(c)
            current = c
        return {"generator": ms["generator"], "alpha": alpha,
                "arc": [float(x) for x in target], "steps": steps}
