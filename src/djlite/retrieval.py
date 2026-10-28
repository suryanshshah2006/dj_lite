"""Candidate retrieval, relevance, and next-track ranking.

relevance(seed, c) = w_audio * audio_sim + w_style * style_sim + w_era * era_score
next_tracks(seed)  = FAISS top-K on [sqrt(w_audio) * audio, sqrt(w_style) * style],
                     re-ranked by alpha * relevance + beta * transition_score_v0_1.

Usage (prints the top-N for the sanity seeds in configs/transitions.yaml):
    python src/djlite/retrieval.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import yaml

from djlite.style import l2_normalize
from djlite.transitions import COMPONENTS, SCORERS, era_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METHODS = ("transition", "audio", "relevance", "full")


def load_configs(paths_file: Path | None = None) -> tuple[dict, dict, dict]:
    cdir = PROJECT_ROOT / "configs"
    load = lambda p: yaml.safe_load(Path(p).read_text(encoding="utf-8"))  # noqa: E731
    return (load(paths_file or cdir / "paths.yaml")["paths"],
            load(cdir / "retrieval.yaml"), load(cdir / "transitions.yaml"))


def load_tracks(paths: dict) -> tuple[pd.DataFrame, np.ndarray]:
    feats = pd.read_parquet(PROJECT_ROOT / paths["features_dir"] / "track_features.parquet")
    style = pd.read_parquet(PROJECT_ROOT / paths["features_dir"] / "track_style.parquet")
    if not feats["id"].equals(style["id"]):
        style = style.set_index("id").loc[feats["id"]].reset_index()
    return feats, style.drop(columns="id").to_numpy(dtype=np.float32)


class Recommender:
    """Holds the precomputed arrays and FAISS indexes for one candidate pool."""

    def __init__(self, tracks: pd.DataFrame, style: np.ndarray, rcfg: dict, tcfg: dict,
                 z_suffix: str = "_z"):
        self.tracks = tracks.reset_index(drop=True)
        self.rcfg, self.tcfg = rcfg, tcfg
        rel = rcfg["relevance"]
        self.audio = l2_normalize(
            self.tracks[[c + z_suffix for c in rel["audio_columns"]]].to_numpy(np.float32))
        self.style = np.ascontiguousarray(style, dtype=np.float32)
        self.has_style = np.linalg.norm(self.style, axis=1) > 0

        combined = np.hstack([np.sqrt(rel["w_audio"]) * self.audio,
                              np.sqrt(rel["w_style"]) * self.style]).astype(np.float32)
        self.index = faiss.IndexFlatIP(combined.shape[1])
        self.index.add(combined)
        self.audio_index = faiss.IndexFlatIP(self.audio.shape[1])
        self.audio_index.add(self.audio)

        # Columns the transition scorer needs, as plain numpy arrays.
        t = self.tracks
        self.arr = {
            "camelot_num": t["camelot_num"].to_numpy(dtype=float, na_value=np.nan),
            "camelot_letter": t["camelot_letter"].to_numpy(dtype=object, na_value=None),
            **{c: t[c].to_numpy(dtype=float) for c in
               ["tempo", "tempo_folded", "energy", "loudness", "year"]},
            **{c + z_suffix: t[c + z_suffix].to_numpy(dtype=float)
               for c in tcfg["timbre"]["columns"]},
        }
        self.artist_sets = [frozenset(a) for a in t["artist_ids"]]
        self.id_to_idx = pd.Series(np.arange(len(t)), index=t["id"])

    # ---------- pieces ----------

    def _take(self, idx: np.ndarray) -> dict:
        return {k: v[idx] for k, v in self.arr.items()}

    def _seed(self, i: int) -> dict:
        return {k: v[i] for k, v in self.arr.items()}

    def relevance(self, i: int, cand: np.ndarray) -> dict:
        rel = self.rcfg["relevance"]
        audio = (self.audio[cand] @ self.audio[i] + 1) / 2
        style = (self.style[cand] @ self.style[i] + 1) / 2
        era = np.asarray(era_score(self.arr["year"][i], self.arr["year"][cand],
                                   **self.tcfg["era"]))
        w_a, w_s, w_e = rel["w_audio"], rel["w_style"], rel["w_era"]
        score = w_a * audio + w_s * style + w_e * era
        if rel["missing_style"] == "renormalize":
            # No style vector on either side: drop the style term and rescale the
            # audio/era weights to sum to 1 (0.35/0.15 -> 0.70/0.30).
            missing = ~(self.has_style[cand] & self.has_style[i])
            alt = (w_a * audio + w_e * era) / (w_a + w_e) * (w_a + w_s + w_e)
            score = np.where(missing, alt, score)
            style = np.where(missing, np.nan, style)
        elif rel["missing_style"] != "neutral":
            raise ValueError(f"unknown missing_style: {rel['missing_style']!r}")
        return {"relevance": score, "audio_sim": audio, "style_sim": style, "era_rel": era}

    def transition(self, i: int, cand: np.ndarray) -> dict:
        scorer = SCORERS[self.rcfg["ranking"]["transition_version"]]
        return scorer(self._seed(i), self._take(cand), self.tcfg)

    def _allowed(self, i: int, cand: np.ndarray) -> np.ndarray:
        ok = cand != i
        if self.rcfg["retrieval"]["exclude_same_artist"]:
            seed_artists = self.artist_sets[i]
            ok &= np.array([not (self.artist_sets[j] & seed_artists) for j in cand], bool)
        return ok

    def combined_vectors(self, idx: np.ndarray) -> np.ndarray:
        """The audio + style vectors the FAISS index searches over (unit-normalized)."""
        v = np.asarray(self.index.reconstruct_batch(np.asarray(idx, dtype=np.int64)))
        return l2_normalize(v)

    def retrieve(self, seeds: np.ndarray, k: int | None = None,
                 audio_only: bool = False) -> np.ndarray:
        """Batched FAISS top-k candidate indices, shape (len(seeds), k)."""
        k = k or self.rcfg["retrieval"]["k"]
        index = self.audio_index if audio_only else self.index
        q = (self.audio[seeds] if audio_only else
             np.asarray(index.reconstruct_batch(seeds.astype(np.int64))))
        _, idx = index.search(np.ascontiguousarray(q, dtype=np.float32), k)
        return idx

    # ---------- ranking ----------

    def rank(self, i: int, method: str = "full", n: int = 10,
             cand: np.ndarray | None = None) -> pd.DataFrame:
        """Top-n next tracks for seed index i. `cand` = precomputed FAISS candidates."""
        cap = self.rcfg["ranking"].get("max_per_artist")
        return self.rank_lists(i, method, n, cand, caps=(cap,))[cap]

    def rank_lists(self, i: int, method: str, n: int, cand: np.ndarray | None = None,
                   caps: tuple = (None,)) -> dict:
        """Score candidates once; return {cap: top-n DataFrame} for each max_per_artist cap
        (None = uncapped). Lets evaluation get capped and uncapped lists in one pass."""
        if method == "transition":
            # Full scan over the pool; keep a margin for exclusions.
            s = self.transition(i, np.arange(len(self.tracks)))["score"]
            margin = min(300, len(s) - 1)
            cand = np.argpartition(-s, margin)[:margin]
        elif cand is None:
            cand = self.retrieve(np.array([i]), audio_only=(method == "audio"))[0]
        cand = cand[cand >= 0]
        cand = cand[self._allowed(i, cand)]

        parts = {**self.relevance(i, cand), **self.transition(i, cand)}
        parts["transition"] = parts.pop("score")
        a, b = self.rcfg["ranking"]["alpha"], self.rcfg["ranking"]["beta"]
        parts["final"] = a * parts["relevance"] + b * parts["transition"]
        key = {"transition": "transition", "audio": "audio_sim",
               "relevance": "relevance", "full": "final"}[method]
        order = np.argsort(-parts[key], kind="stable")
        lists = {}
        for cap in caps:
            top = self._diversify(cand, order, n, cap)
            out = pd.DataFrame({k: np.asarray(v)[top] for k, v in parts.items()})
            out.insert(0, "idx", cand[top])
            lists[cap] = out
        return lists

    def _diversify(self, cand: np.ndarray, order: np.ndarray, n: int,
                   cap: int | None) -> np.ndarray:
        """Greedy top-n: skip a candidate once any of its artists has `cap` picks."""
        if not cap:
            return order[:n]
        counts: dict[str, int] = {}
        picked = []
        for p in order:
            artists = self.artist_sets[cand[p]]
            if any(counts.get(a, 0) >= cap for a in artists):
                continue
            picked.append(p)
            for a in artists:
                counts[a] = counts.get(a, 0) + 1
            if len(picked) == n:
                break
        return np.array(picked, dtype=int)

    def next_tracks(self, seed_id: str, n: int = 10, method: str = "full") -> pd.DataFrame:
        i = int(self.id_to_idx[seed_id])
        res = self.rank(i, method, n)
        meta = self.tracks.loc[res["idx"], ["id", "name", "artists", "camelot", "tempo",
                                            "energy", "year"]].reset_index(drop=True)
        meta = meta.rename(columns={"tempo": "bpm"})  # "tempo" is the score component
        return pd.concat([meta, res.drop(columns="idx")], axis=1)


def format_list(rec: Recommender, seed_id: str, n: int, method: str = "full") -> str:
    s = rec.tracks.loc[rec.id_to_idx[seed_id]]
    res = rec.next_tracks(seed_id, n, method)
    short = {"harmonic": "harm", "tempo": "tempo", "energy_smoothness": "nrg",
             "timbre": "timb", "loudness": "loud", "era": "era"}
    lines = [
        f"SEED: {s['name']} - {', '.join(s['artists'])} ({s['year']})  "
        f"camelot={s['camelot']} tempo={s['tempo']:.1f} energy={s['energy']:.2f}",
        f"  {'#':>2} {'track':<44} {'cam':>4} {'bpm':>6} {'nrg':>5} {'year':>5} "
        f"{'FINAL':>6} | {'REL':>5} {'audio':>5} {'style':>5} {'era':>5} | {'TRANS':>5} "
        + " ".join(f"{short[c]:>5}" for c in COMPONENTS) + f" {'dE':>5}",
    ]
    for j, r in res.iterrows():
        name = f"{r['name']} - {', '.join(r['artists'])}"
        name = name if len(name) <= 44 else name[:41] + "..."
        lines.append(
            f"  {j + 1:>2} {name:<44} {r['camelot']:>4} {r['bpm']:>6.1f} {r['energy']:>5.2f} "
            f"{r['year']:>5} {r['final']:>6.3f} | {r['relevance']:>5.2f} {r['audio_sim']:>5.2f} "
            f"{r['style_sim']:>5.2f} {r['era_rel']:>5.2f} | {r['transition']:>5.2f} "
            + " ".join(f"{r[c]:>5.2f}" for c in COMPONENTS) + f" {r['energy_delta']:>+5.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="full", choices=METHODS)
    args = parser.parse_args()

    paths, rcfg, tcfg = load_configs()
    tracks, style = load_tracks(paths)
    rec = Recommender(tracks, style, rcfg, tcfg)
    n = rcfg["sanity"]["top_n"]
    blocks = [f"NEXT TRACKS, method={args.method}  (alpha={rcfg['ranking']['alpha']}, "
              f"beta={rcfg['ranking']['beta']}, K={rcfg['retrieval']['k']}, "
              f"exclude_same_artist={rcfg['retrieval']['exclude_same_artist']})",
              "REL = relevance (audio/style cosines mapped to [0,1], era); "
              "TRANS = transition v0.1 with its components; dE = energy B - A", ""]
    for s in tcfg["sanity"]["seeds"]:
        blocks += [f"[{s['label'].upper()}]", format_list(rec, s["id"], n, args.method), ""]
    report = "\n".join(blocks)
    out = PROJECT_ROOT / paths["reports_dir"] / f"next_tracks_{args.method}.txt"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
