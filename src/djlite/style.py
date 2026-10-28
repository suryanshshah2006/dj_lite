"""Artist style embeddings from album co-occurrence.

Builds a TF-IDF-weighted sparse artist x album matrix from
data/processed/artist_album_links.parquet, reduces it with truncated SVD, and writes:
  models/artist_embeddings.parquet       artist_id, artist, n_tracks, linked, emb_0..63
  data/features/track_style.parquet      id, style_0..63 (mean of artists, L2-normalized)
  reports/style_report.txt               coverage + nearest-artist sanity check

Usage:
    python src/djlite/style.py
"""

from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from sklearn.decomposition import TruncatedSVD

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-normalize; all-zero rows stay zero."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > 0)


def build_matrix(links: pd.DataFrame, sublinear_tf: bool, drop_single_artist_albums: bool):
    """TF-IDF artist x album matrix with L2-normalized rows.

    Returns (X, artist ids, linked mask, B) where B is the binary artist x album
    incidence over the same columns (used for random-walk smoothing).
    """
    a_codes, artists = pd.factorize(links["artist_id"])
    b_codes, _ = pd.factorize(links["album_id"])
    n_art = len(artists)
    album_size = np.bincount(b_codes)
    # An artist is "linked" if at least one of its albums has another artist on it.
    on_shared = album_size[b_codes] >= 2
    linked = np.zeros(n_art, dtype=bool)
    linked[a_codes[on_shared]] = True

    # Single-artist albums carry no co-occurrence signal; as unique columns they make
    # every artist's row nearly orthogonal to all others and swamp the SVD.
    keep = on_shared if drop_single_artist_albums else np.ones(len(links), dtype=bool)
    tf = links["n_tracks"].to_numpy(dtype=float)
    tf = 1.0 + np.log(tf) if sublinear_tf else tf
    idf = np.log(n_art / album_size)
    vals = (tf * idf[b_codes])[keep]
    shape = (n_art, len(album_size))
    X = sp.csr_matrix((vals, (a_codes[keep], b_codes[keep])), shape=shape)
    row_norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    X = sp.diags(np.divide(1.0, row_norm, out=np.zeros_like(row_norm), where=row_norm > 0)) @ X
    B = sp.csr_matrix((np.ones(keep.sum()), (a_codes[keep], b_codes[keep])), shape=shape)
    return X.tocsr(), artists, linked, B


def smooth(Z: np.ndarray, B: sp.csr_matrix, hops: int) -> np.ndarray:
    """Z + W Z + W^2 Z + ..., W = artist -> album -> artist random-walk step."""
    def inv(d):
        return sp.diags(np.divide(1.0, d, out=np.zeros_like(d), where=d > 0))
    to_album = inv(np.asarray(B.sum(axis=1)).ravel()) @ B
    to_artist = inv(np.asarray(B.sum(axis=0)).ravel()) @ B.T.tocsr()
    acc, E = Z.copy(), Z
    for _ in range(hops):
        E = to_album @ (to_artist @ E)
        acc += E
    return acc


def artist_embeddings(links: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    w = cfg["weighting"]
    X, artists, linked, B = build_matrix(links, w["sublinear_tf"],
                                         w["drop_single_artist_albums"])
    s = cfg["svd"]
    svd = TruncatedSVD(n_components=s["n_components"], n_iter=s["n_iter"],
                       random_state=s["random_seed"])
    Z = svd.fit_transform(X)
    captured = np.linalg.norm(Z, axis=1)
    Z = smooth(Z, B, cfg["propagation_hops"])
    usable = linked & (captured >= cfg["min_captured_norm"])
    emb = l2_normalize(Z) * usable[:, None]

    names = links.dropna(subset=["artist"]).groupby("artist_id")["artist"].first()
    n_tracks = links.groupby("artist_id")["n_tracks"].sum()
    meta = pd.DataFrame({
        "artist_id": artists,
        "artist": names.reindex(artists).to_numpy(),
        "n_tracks": n_tracks.reindex(artists).to_numpy(),
        "n_albums": links.groupby("artist_id").size().reindex(artists).to_numpy(),
        "linked": linked,
        "captured_norm": captured,
        "usable": usable,
    })
    meta.attrs["explained_variance"] = float(svd.explained_variance_ratio_.sum())
    return meta, emb.astype(np.float32)


def track_style(artist_ids: pd.Series, meta: pd.DataFrame, emb: np.ndarray) -> np.ndarray:
    """Mean of each track's artist embeddings, L2-normalized (zero if none usable)."""
    index = pd.Index(meta["artist_id"])
    lens = artist_ids.map(len).to_numpy()
    flat = np.concatenate([np.asarray(a, dtype=object) for a in artist_ids])
    cols = index.get_indexer(flat)
    rows = np.repeat(np.arange(len(artist_ids)), lens)
    ok = cols >= 0
    M = sp.csr_matrix((np.ones(ok.sum(), dtype=np.float32), (rows[ok], cols[ok])),
                      shape=(len(artist_ids), len(index)))
    return l2_normalize(np.asarray(M @ emb)).astype(np.float32)


def nearest_artists(meta: pd.DataFrame, emb: np.ndarray, name: str, top_n: int) -> str:
    rows = meta.index[meta["artist"] == name]
    if len(rows) == 0:
        return f"  {name}: not found"
    i = meta.loc[rows, "n_tracks"].idxmax()  # most prolific id if the name repeats
    r = meta.loc[i]
    head = (f"  {name}  (tracks={r['n_tracks']}, albums={r['n_albums']}, "
            f"linked={r['linked']}, captured_norm={r['captured_norm']:.3f})")
    if not r["usable"]:
        return head + "\n    -> no usable embedding"
    sims = emb @ emb[i]
    sims[i] = -np.inf
    top = np.argsort(-sims)[:top_n]
    lines = [head]
    for j in top:
        lines.append(f"    {sims[j]:.3f}  {meta.at[j, 'artist']}  (tracks={meta.at[j, 'n_tracks']})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=PROJECT_ROOT / "configs" / "paths.yaml", type=Path)
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "style.yaml", type=Path)
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text(encoding="utf-8"))["paths"]
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    links = pd.read_parquet(PROJECT_ROOT / paths["processed_dir"] / "artist_album_links.parquet")
    meta, emb = artist_embeddings(links, cfg)

    tracks = pd.read_parquet(PROJECT_ROOT / paths["features_dir"] / "track_features.parquet",
                             columns=["id", "artist_ids"])
    tstyle = track_style(tracks["artist_ids"], meta, emb)
    has_style = np.linalg.norm(tstyle, axis=1) > 0

    k = emb.shape[1]
    art_out = pd.concat([meta, pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(k)])],
                        axis=1)
    art_path = PROJECT_ROOT / paths["models_dir"] / cfg["files"]["artist_embeddings"]
    art_out.to_parquet(art_path, index=False)
    trk_out = pd.concat([tracks[["id"]],
                         pd.DataFrame(tstyle, columns=[f"style_{i}" for i in range(k)])], axis=1)
    trk_path = PROJECT_ROOT / paths["features_dir"] / cfg["files"]["track_style"]
    trk_out.to_parquet(trk_path, index=False)

    # Artists that appear in the cleaned track set
    in_clean = meta["artist_id"].isin(set(np.concatenate(tracks["artist_ids"].to_numpy())))

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("ARTIST STYLE EMBEDDINGS")
        print("=" * 80)
        print(f"links={len(links):,}  artists={len(meta):,}  albums={links['album_id'].nunique():,}")
        print(f"SVD: {k} components, explained variance ratio (sum) = "
              f"{meta.attrs['explained_variance']:.3f}")
        print("\nCOVERAGE")
        for label, m in [("all artists", np.ones(len(meta), bool)),
                         ("artists in cleaned tracks", in_clean.to_numpy())]:
            sub = meta[m]
            print(f"  {label}: {len(sub):,}")
            print(f"    share >=1 album with another artist (linked): "
                  f"{sub['linked'].mean():.1%}  ({sub['linked'].sum():,})")
            print(f"    no links -> zero embedding:                  "
                  f"{(~sub['linked']).mean():.1%}  ({(~sub['linked']).sum():,})")
            weak = sub["linked"] & ~sub["usable"]
            print(f"    linked but captured_norm < {cfg['min_captured_norm']} -> zero:   "
                  f"{weak.mean():.1%}  ({weak.sum():,})")
            print(f"    usable embedding:                             "
                  f"{sub['usable'].mean():.1%}  ({sub['usable'].sum():,})")
        lk = meta.loc[meta["linked"], "captured_norm"]
        print("  captured_norm quantiles (linked artists, before smoothing): " + "  ".join(
            f"p{int(q * 100)}={v:.4f}" for q, v in lk.quantile([.1, .25, .5, .75, .9]).items()))
        # Anisotropy: if random pairs are already very similar, cosines carry little signal.
        rng = np.random.default_rng(cfg["svd"]["random_seed"])
        u = np.flatnonzero(meta["usable"].to_numpy())
        pa, pb = rng.choice(u, 20000), rng.choice(u, 20000)
        rc = np.sum(emb[pa] * emb[pb], axis=1)
        print(f"  cosine of random usable-artist pairs: mean={rc.mean():.3f} "
              f"p50={np.median(rc):.3f} p90={np.quantile(rc, .9):.3f} "
              f"p99={np.quantile(rc, .99):.3f}")
        print(f"\n  tracks with a non-zero style vector: {has_style.mean():.1%} "
              f"({has_style.sum():,} of {len(tracks):,})")

        print(f"\nNEAREST {cfg['sanity']['top_n']} ARTISTS BY STYLE (cosine)")
        for a in cfg["sanity"]["artists"]:
            print(f"\n[{a['label'].upper()}]")
            print(nearest_artists(meta, emb, a["name"], cfg["sanity"]["top_n"]))

    report = buf.getvalue()
    out = PROJECT_ROOT / paths["reports_dir"] / "style_report.txt"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {art_path}\nWrote {trk_path}\nWrote {out}")


if __name__ == "__main__":
    main()
