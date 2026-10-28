"""Read finished evaluation reports for the How It Works page (read-only)."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE = "pareto_alpha.png"


def _f(v):
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) else v


def _labels() -> dict:
    cfg = PROJECT_ROOT / "configs" / "evaluation.yaml"
    try:
        return yaml.safe_load(cfg.read_text(encoding="utf-8")).get("methods", {})
    except OSError:
        return {}


def eval_summary(settings: dict, root: Path = PROJECT_ROOT) -> dict:
    rep = settings["reports"]
    fig = root / rep["figures_dir"] / FIGURE
    figure_url = f"/api/figures/{FIGURE}" if fig.exists() else None
    labels = _labels()
    v2, v1 = root / rep["eval_csv_v2"], root / rep["eval_csv_v1"]

    if v2.exists():
        df = pd.read_csv(v2)
        main = df[(df["method"] != "full") | (df["is_d"].astype(str) == "True")]
        rows = [{
            "pool": r["pool"],
            "label": labels.get(r["method"], r["method"]),
            "genre_agree": _f(r["genre_agree"]), "genre_agree_lo": _f(r["genre_agree_lo"]),
            "genre_agree_hi": _f(r["genre_agree_hi"]), "genre_macro": _f(r["genre_macro"]),
            "trans": float(r["trans"]), "trans_lo": float(r["trans_lo"]),
            "trans_hi": float(r["trans_hi"]), "dbpm": float(r["dbpm"]), "ild": float(r["ild"]),
            "artist_rep": float(r["artist_rep"]),
        } for _, r in main.iterrows()]
        return {"available": True, "source": str(v2.relative_to(root)), "rows": rows,
                "figure_url": figure_url}

    if v1.exists():
        df = pd.read_csv(v1)
        rows = [{
            "pool": r["pool"], "label": r.get("method_label", r["method"]),
            "genre_agree": _f(r["genre_agree_mean"]), "genre_agree_lo": None,
            "genre_agree_hi": None, "genre_macro": None,
            "trans": float(r["trans_mean"]), "trans_lo": float(r["trans_mean"]),
            "trans_hi": float(r["trans_mean"]), "dbpm": float(r["dbpm_mean"]),
            "ild": float(r.get("ild_mean", float("nan"))),
            "artist_rep": float(r.get("artist_rep_mean", float("nan"))),
        } for _, r in df.iterrows()]
        return {"available": True, "source": str(v1.relative_to(root)), "rows": rows,
                "figure_url": figure_url}

    return {"available": False, "rows": [], "figure_url": figure_url}


def figure_path(settings: dict, name: str, root: Path = PROJECT_ROOT) -> Path | None:
    if name != FIGURE:  # only serve the one known figure
        return None
    p = root / settings["reports"]["figures_dir"] / name
    return p if p.exists() else None
