"""Step 5 evaluation: beam-search arc generator vs baselines and ablations.

    python -m djlite.evaluate_generator [--n-seeds N]

Seeds: random genre-labeled tracks (eval v2 labels; genre is used only for evaluation).
Candidates: the full catalogue (as in the app). Per seed the pool and its transition matrix
are built once and shared by every arc and configuration. Outputs:
reports/eval_step5.{txt,md,csv}, reports/eval_step5_per_playlist.parquet,
reports/figures/step5_examples.png.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import yaml

from djlite import generator as G
from djlite.evaluate import _clean_keys, build_labels_v2
from djlite.retrieval import PROJECT_ROOT, Recommender, load_configs, load_tracks
from djlite.transitions import tempo_octave_align

METRICS = [  # (column, header, decimals)
    ("trans_mean", "transition mean", 3),
    ("trans_min", "transition min", 3),
    ("harm_compat", "% harmonic compat", 1),
    ("dbpm_mean", "|dBPM| mean", 2),
    ("dbpm_max", "|dBPM| max", 2),
    ("arc_mae", "arc MAE", 3),
    ("genre_agree", "genre agree", 3),
    ("unique_artists", "unique artists", 2),
    ("objective", "objective", 3),
    ("runtime_ms", "runtime ms", 0),
]


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (reverts automatically on exit)."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def configs(gcfg: dict) -> list[dict]:
    """Every (generator, alpha, gamma, width) to run, with the table rows each feeds."""
    o, ab = gcfg["evaluation"]["objective"], gcfg["evaluation"]["ablations"]
    a0, g0, w0 = o["alpha"], o["gamma"], gcfg["beam"]["width"]
    out = {}

    def add(table, label, gen="beam_v1", alpha=a0, gamma=g0, width=w0, max_dbpm=None):
        key = (gen, alpha, gamma, width if gen == "beam_v1" else None, max_dbpm)
        out.setdefault(key, {"generator": gen, "alpha": alpha, "gamma": gamma,
                             "width": key[3], "max_dbpm": max_dbpm,
                             "rows": []})["rows"].append((table, label))

    add("main", f"beam_v1 (B={w0})")
    add("main", "greedy_v0", gen="greedy_v0")
    for b in ab["beam_width"]:
        add("beam_width", f"B={b}", width=b)
    for g in ab["gamma"]:
        add("gamma", f"gamma={g:g}", gamma=g)
    for a in ab["alpha"]:
        add("alpha", f"alpha={a:g}", alpha=a)
    for cap in ab.get("max_dbpm", []):
        add("max_dbpm", "no cap" if cap is None else f"max |dBPM| {cap:g}", max_dbpm=cap)
    return list(out.values())


def playlist_metrics(rec: Recommender, pool: G.Pool, seq: list[int], target: np.ndarray,
                     lab: list, seed: int, primary: np.ndarray, hc: float, alpha0: float,
                     gamma0: float, seed_pos: int = 0) -> dict:
    s = np.asarray(seq)
    g = pool.idx[s]
    tr = pool.T[s[:-1], s[1:]].astype(float)
    code = rec._camelot_code
    harm = rec._harm_table[code[g[:-1]], code[g[1:]]]
    _, _, b2 = tempo_octave_align(rec.arr["tempo"][g[:-1]], rec.arr["tempo"][g[1:]])
    dbpm = np.abs(rec.arr["tempo"][g[:-1]] - b2)
    other = np.array([k for k in range(len(s)) if k != seed_pos])
    labs = [lab[j] for j in g[other] if lab[j] is not None]
    err = np.abs(pool.energy[s] - np.asarray(target)[:len(s)])
    return {
        "length": len(s),
        "trans_mean": tr.mean(), "trans_min": tr.min(),
        "harm_compat": 100.0 * np.mean((harm >= hc) & (code[g[:-1]] < 24) & (code[g[1:]] < 24)),
        "dbpm_mean": dbpm.mean(), "dbpm_max": dbpm.max(),
        "arc_mae": float(err[other].mean()),      # positions the generator chose
        "arc_mae_all": float(err.mean()),         # including the seed's position
        "genre_agree": float(np.mean([bool(x & lab[seed]) for x in labs])) if labs else np.nan,
        "n_labeled": len(labs),
        "unique_artists": len(set(primary[g])),
        # always the default Step 5 objective weights, so every row is comparable
        "objective": G.objective(pool, seq, target, alpha0, gamma0, seed_pos=seed_pos),
    }


def run(rec, lab, seeds, gcfg, primary) -> tuple[pd.DataFrame, int]:
    ev = gcfg["evaluation"]; o = ev["objective"]
    arcs = {name: G.resample_arc(pts, ev["length"]) for name, pts in gcfg["arcs"].items()}
    cfgs = configs(gcfg)
    rows, mismatches, t0 = [], 0, time.time()
    for n_done, i in enumerate(seeds, 1):
        i = int(i)
        pool = G.build_pool(rec, [i], k=gcfg["pool"]["k"], per_seed_k=gcfg["pool"]["per_seed_k"],
                            need_dbpm=any(c["max_dbpm"] is not None for c in cfgs))
        for arc_name, target in arcs.items():
            seqs = {}
            for c in cfgs:
                t = time.perf_counter()
                if c["generator"] == "greedy_v0":
                    seq = G.greedy(pool, target, c["alpha"], c["gamma"], c["max_dbpm"])
                else:
                    seq = G.beam(pool, target, c["alpha"], c["gamma"], c["width"], c["max_dbpm"])
                ms = pool.build_ms + (time.perf_counter() - t) * 1000
                seqs[(c["generator"], c["alpha"], c["gamma"], c["width"], c["max_dbpm"])] = seq
                m = playlist_metrics(rec, pool, seq, target, lab, i, primary,
                                     ev["harmonic_compatible"], o["alpha"], o["gamma"])
                for table, label in c["rows"]:
                    rows.append({"seed_idx": i, "arc": arc_name, "table": table, "method": label,
                                 **m, "runtime_ms": ms, "seq": [int(x) for x in pool.idx[seq]]})
            main = seqs[("beam_v1", o["alpha"], o["gamma"], gcfg["beam"]["width"], None)]
            if seqs[("greedy_v0", o["alpha"], o["gamma"], None, None)] != \
                    seqs[("beam_v1", o["alpha"], o["gamma"], 1, None)]:
                mismatches += 1
            beam_ms = next(r["runtime_ms"] for r in reversed(rows)
                           if r["table"] == "main" and r["method"].startswith("beam_v1"))
            rng = np.random.default_rng(ev["random_seed"] * 1_000_003 + i)
            for label, seq in [("beam tracks, random order", G.random_order(main, rng)),
                               ("beam tracks, energy-sorted", G.energy_sorted(main, pool, target))]:
                m = playlist_metrics(rec, pool, seq, target, lab, i, primary,
                                     ev["harmonic_compatible"], o["alpha"], o["gamma"])
                rows.append({"seed_idx": i, "arc": arc_name, "table": "main", "method": label,
                             **m, "runtime_ms": beam_ms, "seq": [int(x) for x in pool.idx[seq]]})
        if n_done % 25 == 0 or n_done == len(seeds):
            el = time.time() - t0
            print(f"  {n_done}/{len(seeds)} seeds, {el:.0f}s elapsed, "
                  f"~{el / n_done * (len(seeds) - n_done):.0f}s left", file=sys.stderr, flush=True)
    return pd.DataFrame(rows), mismatches


def bootstrap(df: pd.DataFrame, group: list[str], bcfg: dict, metrics=METRICS) -> pd.DataFrame:
    """Mean and percentile CI over seeds (paired resamples) for each group. Each seed
    contributes the mean over its rows in the group (e.g. over the 4 arcs)."""
    per_seed = df.groupby(group + ["seed_idx"], sort=False)[[c for c, _, _ in metrics]].mean()
    seeds = np.sort(df["seed_idx"].unique())
    rng = np.random.default_rng(bcfg["random_seed"])
    draws = rng.integers(0, len(seeds), size=(bcfg["n_resamples"], len(seeds)))
    C = np.vstack([np.ones(len(seeds))] + [np.bincount(d, minlength=len(seeds)) for d in draws])
    lo_q, hi_q = (1 - bcfg["ci"]) / 2 * 100, (1 + bcfg["ci"]) / 2 * 100
    out = []
    for key, g in per_seed.groupby(level=list(range(len(group))), sort=False):
        g = g.droplevel(list(range(len(group)))).reindex(seeds)
        rec = dict(zip(group, key if isinstance(key, tuple) else (key,)))
        for col, _, _ in metrics:
            v = g[col].to_numpy(dtype=float)
            ok = ~np.isnan(v)
            num, den = C @ np.where(ok, v, 0.0), C @ ok
            boot = np.divide(num, den, out=np.full(len(C), np.nan), where=den > 0)
            rec[col] = boot[0]
            rec[f"{col}_lo"], rec[f"{col}_hi"] = np.nanpercentile(boot[1:], [lo_q, hi_q])
        if "genre_agree" in g:
            rec["seeds_with_label"] = float(np.mean(~np.isnan(g["genre_agree"].to_numpy(float))))
        out.append(rec)
    return pd.DataFrame(out)


def _cell(r, col, d):
    return f"{r[col]:.{d}f} [{r[f'{col}_lo']:.{d}f}, {r[f'{col}_hi']:.{d}f}]"


def table(summary: pd.DataFrame, title: str, label_col: str, md: bool,
          cols=METRICS) -> list[str]:
    hdr = [h for _, h, _ in cols]
    if md:
        out = [f"### {title}", "", f"| {label_col} | " + " | ".join(hdr) + " |",
               "|---|" + "---:|" * len(hdr)]
    else:
        out = ["", title, f"  {label_col:<30}" + "".join(f"{h:>24}" for h in hdr)]
    for _, r in summary.iterrows():
        cells = [_cell(r, c, d) for c, _, d in cols]
        out.append(f"| {r[label_col]} | " + " | ".join(cells) + " |" if md
                   else f"  {r[label_col]:<30}" + "".join(f"{c:>24}" for c in cells))
    return out + [""]


def plot_examples(rec: Recommender, gcfg: dict, seed: int, path, title: str) -> list[str]:
    """One beam_v1 playlist per arc preset with the config's defaults: target vs achieved
    energy, the seed marked, transition scores between tracks. Returns markdown lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, text, text2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    tgt_c, ach_c = "#eb6834", "#6250d6"
    ev = gcfg["evaluation"]
    cap = gcfg["constraints"]["max_dbpm"]
    pool = G.build_pool(rec, [seed], **G.pool_kwargs(gcfg, need_dbpm=cap is not None))
    t = rec.tracks
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), facecolor=surface)
    md = []
    for ax, (name, pts) in zip(axes.ravel(), gcfg["arcs"].items()):
        target = G.resample_arc(pts, ev["length"])
        out = G.generate(rec, [seed], target, gcfg, pool=pool)
        s, sp = np.asarray(out["seq"]), out["seed_pos"]
        x = np.arange(1, len(s) + 1)
        tr = pool.T[s[:-1], s[1:]]
        ax.set_facecolor(surface)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=text2, length=0)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.plot(x, target[:len(s)], "--", color=tgt_c, linewidth=2, label="target energy")
        ax.plot(x, pool.energy[s], "-o", color=ach_c, linewidth=2, ms=6, mec=surface, mew=1.5,
                label="achieved energy")
        ax.plot([x[sp]], [pool.energy[s[sp]]], "*", color=ach_c, ms=16, mec=surface, mew=1.5,
                label="seed", zorder=4)
        for k, v in enumerate(tr):
            ax.annotate(f"{v:.2f}", ((x[k] + x[k + 1]) / 2, 0.04), ha="center", fontsize=8,
                        color=text2)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xlabel("position (numbers along the bottom: transition score between neighbours)",
                      color=text2, fontsize=8.5)
        ax.set_ylabel("energy", color=text2)
        other = [k for k in range(len(s)) if k != sp]
        mae = np.mean(np.abs(pool.energy[s[other]] - target[other]))
        ax.set_title(f"{name.replace('_', ' ')}: transition mean {tr.mean():.2f}, "
                     f"min {tr.min():.2f}, arc MAE {mae:.3f}", color=text, loc="left", fontsize=10)
        ax.legend(frameon=False, labelcolor=text2, fontsize=8.5, loc="upper right")
        md += [f"**{name.replace('_', ' ')}** ({out['total_ms']:.0f} ms, seed at position {sp + 1})",
               "", "| # | track | artist | Camelot | BPM | energy | target | transition in |",
               "|---:|---|---|---|---:|---:|---:|---:|"]
        for k, p in enumerate(s):
            r = t.iloc[int(pool.idx[p])]
            label = f"**{r['name']}** (seed)" if k == sp else r["name"]
            md.append(f"| {k + 1} | {label} | {r['artists'][0]} | {r['camelot'] or '-'} | "
                      f"{r['tempo']:.0f} | {r['energy']:.2f} | {target[k]:.2f} | "
                      f"{'' if k == 0 else f'{tr[k - 1]:.2f}'} |")
        md.append("")
    seed_row = t.iloc[seed]
    fig.suptitle(f"{title}, seed: {seed_row['name']} - {seed_row['artists'][0]}", color=text,
                 x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    return md


def defaults_line(gcfg: dict) -> str:
    o, c, p = gcfg["objective"], gcfg["constraints"], gcfg["pool"]
    return (f"alpha {o['alpha']}, gamma {o['gamma']}, lambda {o['weak_link']['lambda']} "
            f"(tau {o['weak_link']['tau']}), |dBPM| cap {c['max_dbpm']}, energy bands "
            f"{p['energy_bands']}, seed position {gcfg['seed_position']}, B {gcfg['beam']['width']}")


# ---------------------------------------------------------------- Step 5.1

METRICS_51 = [
    ("trans_mean", "transition mean", 3),
    ("trans_min", "transition min", 3),
    ("harm_compat", "% harmonic compat", 1),
    ("dbpm_mean", "|dBPM| mean", 2),
    ("dbpm_max", "|dBPM| max", 2),
    ("arc_mae", "arc MAE", 3),
    ("arc_mae_all", "arc MAE incl. seed", 3),
    ("genre_agree", "genre agree", 3),
    ("runtime_ms", "runtime ms", 0),
]


def configs_5_1(gcfg: dict) -> list[dict]:
    e, n = gcfg["evaluation_5_1"], gcfg["evaluation"]["length"]
    old, new = e["old_defaults"], e["new_defaults"]
    out = []

    def add(table, label, pool="plain", gen="beam_v1", d=new, lam=0.0, sp="start", length=n):
        out.append({"table": table, "method": label, "pool": pool, "generator": gen,
                    "gamma": d["gamma"], "max_dbpm": d["max_dbpm"], "lam": lam,
                    "seed_position": sp, "n": length})

    for lam in e["lambdas"]:
        add("lambda_old", f"lambda={lam:g}", d=old, lam=lam)
        add("new_defaults", f"lambda={lam:g}", lam=lam)
        add("stratified", f"lambda={lam:g}", pool="strat", lam=lam)
        add("best_fit", f"lambda={lam:g}", sp="best_fit", lam=lam)
        add("strat_best_fit", f"lambda={lam:g}", pool="strat", sp="best_fit", lam=lam)
        for gen in G.GENERATORS:
            add("long", f"{gen}, lambda={lam:g}", gen=gen, lam=lam, length=e["long_length"])
    return out


def run_5_1(rec, lab, seeds, gcfg, primary) -> pd.DataFrame:
    ev, e = gcfg["evaluation"], gcfg["evaluation_5_1"]
    alpha, width, tau = (ev["objective"]["alpha"], gcfg["beam"]["width"],
                         gcfg["objective"]["weak_link"]["tau"])
    cfgs = configs_5_1(gcfg)
    lengths = sorted({c["n"] for c in cfgs})
    arcs = {(name, n): G.resample_arc(pts, n) for name, pts in gcfg["arcs"].items() for n in lengths}
    rows, t0 = [], time.time()
    p = gcfg["pool"]
    for n_done, i in enumerate(seeds, 1):
        i = int(i)
        pools = {"plain": G.build_pool(rec, [i], k=p["k"], per_seed_k=p["per_seed_k"],
                                       need_dbpm=True),
                 "strat": G.build_pool(rec, [i], k=p["k"], per_seed_k=p["stratified_retrieve_k"],
                                       need_dbpm=True, energy_bands=e["energy_bands"])}
        for arc_name in gcfg["arcs"]:
            for c in cfgs:
                pool, target = pools[c["pool"]], arcs[(arc_name, c["n"])]
                sp = G.seed_slot(pool, target, c["seed_position"])
                kw = dict(lam=c["lam"], tau=tau, seed_pos=sp)
                t = time.perf_counter()
                if c["generator"] == "greedy_v0":
                    seq = G.greedy(pool, target, alpha, c["gamma"], c["max_dbpm"], **kw)
                else:
                    seq = G.beam(pool, target, alpha, c["gamma"], width, c["max_dbpm"], **kw)
                ms = pool.build_ms + (time.perf_counter() - t) * 1000
                m = playlist_metrics(rec, pool, seq, target, lab, i, primary,
                                     ev["harmonic_compatible"], alpha, c["gamma"], seed_pos=sp)
                rows.append({"seed_idx": i, "arc": arc_name, "table": c["table"],
                             "method": c["method"], "lam": c["lam"], "seed_pos": sp, **m,
                             "runtime_ms": ms, "seq": [int(x) for x in pool.idx[seq]]})
        if n_done % 25 == 0 or n_done == len(seeds):
            el = time.time() - t0
            print(f"  {n_done}/{len(seeds)} seeds, {el:.0f}s elapsed, "
                  f"~{el / n_done * (len(seeds) - n_done):.0f}s left", file=sys.stderr, flush=True)
    return pd.DataFrame(rows)


def best_lambda(pooled: pd.DataFrame, gcfg: dict) -> float:
    """Pre-registered rule: under the new defaults, the lambda with the highest mean
    min-transition whose arc MAE is within `lambda_rule_max_mae_increase` of lambda = 0."""
    s = pooled[pooled["table"] == "new_defaults"].copy()
    s["lam"] = s["method"].str.replace("lambda=", "").astype(float)
    base = float(s.loc[s["lam"] == 0, "arc_mae"].iloc[0])
    ok = s[s["arc_mae"] <= base + gcfg["evaluation_5_1"]["lambda_rule_max_mae_increase"]]
    return float(ok.sort_values(["trans_min", "lam"], ascending=[False, True])["lam"].iloc[0])


def report_5_1(df: pd.DataFrame, gcfg: dict, n_seeds: int, out_base) -> str:
    bc, e = gcfg["evaluation"]["bootstrap"], gcfg["evaluation_5_1"]
    pooled = bootstrap(df, ["table", "method"], bc, METRICS_51)
    lam = best_lambda(pooled, gcfg)
    lam_label = f"lambda={lam:g}"
    focus = df[df["method"].isin(["lambda=0", lam_label])
               & df["table"].isin(["new_defaults", "stratified", "best_fit", "strat_best_fit"])]
    per_arc = bootstrap(focus, ["arc", "table", "method"], bc, METRICS_51)
    od, nd = e["old_defaults"], e["new_defaults"]
    head = (f"{n_seeds} genre-labeled seeds (same as Step 5) x {len(gcfg['arcs'])} arcs; full "
            f"catalogue; alpha {gcfg['evaluation']['objective']['alpha']}, B "
            f"{gcfg['beam']['width']}, tau {gcfg['objective']['weak_link']['tau']}; plain pool = top "
            f"{gcfg['pool']['k']} by relevance (FAISS top {gcfg['pool']['per_seed_k']}); stratified "
            f"= {e['energy_bands']} energy-quantile bands x {gcfg['pool']['k'] // e['energy_bands']} "
            f"(FAISS top {gcfg['pool']['stratified_retrieve_k']}); {bc['n_resamples']} bootstrap "
            f"resamples over seeds, 95% CIs (each seed = mean over the 4 arcs).")
    rule = (f"Best lambda (rule fixed before the run: highest mean min-transition under the new "
            f"defaults with arc MAE within +{e['lambda_rule_max_mae_increase']} of lambda = 0): "
            f"**{lam:g}**")
    sections = [
        ("lambda_old", f"Weak-link penalty, Step 5 defaults (gamma {od['gamma']}, no cap), N 12"),
        ("new_defaults", f"New defaults (gamma {nd['gamma']}, |dBPM| cap {nd['max_dbpm']}) x lambda, "
                         "plain pool, seed at start, N 12"),
        ("stratified", "Energy-stratified pool, new defaults x lambda, seed at start, N 12"),
        ("best_fit", "Seed placement best_fit, plain pool, new defaults x lambda, N 12"),
        ("strat_best_fit", "Stratified pool + best_fit, new defaults x lambda, N 12"),
        ("long", f"N = {e['long_length']}: beam vs greedy, new defaults (cap on), plain pool"),
    ]
    md = ["# Step 5.1 evaluation: weak-link penalty, stratified pool, seed placement", "",
          head, "", rule, ""]
    for tbl, title in sections:
        md += table(pooled[pooled["table"] == tbl], title, "method", True, METRICS_51)
    md += ["## Per arc (lambda = 0 and the best lambda)", ""]
    for arc in gcfg["arcs"]:
        part = per_arc[per_arc["arc"] == arc].copy()
        part["config"] = part["table"] + ", " + part["method"]
        md += table(part, f"Per arc: {arc}", "config", True,
                    [m for m in METRICS_51 if m[0] in
                     ("trans_mean", "trans_min", "dbpm_max", "arc_mae", "arc_mae_all")])
    md += ["Transitions: N - 1 per playlist. arc MAE = mean |energy - target| over the positions "
           "the generator filled; 'incl. seed' also counts the seed's position (where best_fit "
           "helps). Other columns as in eval_step5.md. runtime = pool (incl. |dBPM| matrix) + search.",
           ""]
    pd.concat([pooled.assign(scope="pooled"), per_arc.assign(scope="per_arc")]) \
        .to_csv(f"{out_base}.csv", index=False)
    return "\n".join(md)


def run_lengths(rec, lab, seeds, gcfg, primary, lengths=(12, 24)) -> pd.DataFrame:
    """greedy_v0 and beam_v1 at several N with the configured (final) defaults; both
    generators and all lengths share each seed's pool."""
    ev = gcfg["evaluation"]
    cap = gcfg["constraints"]["max_dbpm"]
    rows, t0 = [], time.time()
    for n_done, i in enumerate(seeds, 1):
        i = int(i)
        pool = G.build_pool(rec, [i], **G.pool_kwargs(gcfg, need_dbpm=cap is not None))
        for arc_name, pts in gcfg["arcs"].items():
            for n in lengths:
                target = G.resample_arc(pts, n)
                for gen in G.GENERATORS:
                    out = G.generate(rec, [i], target, gcfg, generator=gen, pool=pool)
                    m = playlist_metrics(rec, pool, out["seq"], target, lab, i, primary,
                                         ev["harmonic_compatible"], out["alpha"], out["gamma"],
                                         seed_pos=out["seed_pos"])
                    m["objective"] = G.objective(pool, out["seq"], target, out["alpha"],
                                                 out["gamma"], lam=out["lam"], tau=out["tau"],
                                                 seed_pos=out["seed_pos"])
                    rows.append({"seed_idx": i, "arc": arc_name, "table": f"N={n}",
                                 "method": gen, **m, "runtime_ms": out["total_ms"]})
        if n_done % 25 == 0 or n_done == len(seeds):
            el = time.time() - t0
            print(f"  {n_done}/{len(seeds)} seeds, {el:.0f}s elapsed, "
                  f"~{el / n_done * (len(seeds) - n_done):.0f}s left", file=sys.stderr, flush=True)
    return pd.DataFrame(rows)


METRICS_LEN = [m for m in METRICS_51 if m[0] != "genre_agree"] + [("objective", "objective", 3)]


def report_lengths(df: pd.DataFrame, gcfg: dict, n_seeds: int, out_base) -> str:
    bc = gcfg["evaluation"]["bootstrap"]
    pooled = bootstrap(df, ["table", "method"], bc, METRICS_LEN)
    # paired per-seed differences beam - greedy (mean over arcs), bootstrapped
    cols = [c for c, _, _ in METRICS_LEN]
    per_seed = df.groupby(["table", "method", "seed_idx"])[cols].mean()
    diffs = []
    for n in df["table"].unique():
        d = (per_seed.loc[(n, "beam_v1")] - per_seed.loc[(n, "greedy_v0")]).reset_index()
        diffs.append(d.assign(table=n, method="beam - greedy"))
    diff = bootstrap(pd.concat(diffs), ["table", "method"], bc, METRICS_LEN)
    md = ["# Generator length check: greedy_v0 vs beam_v1 at N = 12 and N = 24", "",
          f"{n_seeds} genre-labeled seeds (same as Steps 5 and 5.1) x {len(gcfg['arcs'])} arcs; "
          f"final defaults for both generators and both lengths: {defaults_line(gcfg)}. "
          f"objective = the configured objective (incl. the weak-link penalty). "
          f"{bc['n_resamples']} bootstrap resamples over seeds, 95% CIs (each seed = mean over "
          "the 4 arcs); differences are paired per seed.", ""]
    md += table(pooled.assign(method=pooled["table"] + ", " + pooled["method"]),
                "Per generator and length", "method", True, METRICS_LEN)
    md += table(diff.assign(method=diff["table"] + ": beam - greedy"),
                "Paired difference (beam - greedy)", "method", True, METRICS_LEN)
    pd.concat([pooled.assign(scope="pooled"), diff.assign(scope="diff")]) \
        .to_csv(f"{out_base}.csv", index=False)
    return "\n".join(md)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--suite", choices=["step5", "step5_1", "lengths", "examples"],
                    default="step5_1")
    ap.add_argument("--n-seeds", type=int, help="override evaluation.n_seeds (debugging)")
    ap.add_argument("--out-name", help="default: eval_<suite>")
    args = ap.parse_args()
    keep_awake()

    paths, rcfg, tcfg = load_configs()
    gcfg = G.load_generator_config()
    ecfg = yaml.safe_load((PROJECT_ROOT / "configs" / "evaluation.yaml").read_text(encoding="utf-8"))
    ev = gcfg["evaluation"]
    out_name = args.out_name or f"eval_{args.suite}"
    rdir = PROJECT_ROOT / paths["reports_dir"]
    fig_dir = PROJECT_ROOT / paths["figures_dir"]
    t0 = time.time()
    tracks, style = load_tracks(paths)
    rec = Recommender(tracks, style, rcfg, tcfg)
    ex_seed = int(rec.id_to_idx[ev["example_seed"]])

    if args.suite == "examples":
        ex_md = plot_examples(rec, gcfg, ex_seed, fig_dir / "step5_examples.png",
                              "beam_v1 example playlists (final defaults)")
        md = ["# Example playlists (final generator defaults)", "", defaults_line(gcfg), "",
              "![examples](figures/step5_examples.png)", "", *ex_md]
        (rdir / "generator_examples.md").write_text("\n".join(md), encoding="utf-8")
        print("\n".join(md))
        return

    labels, _, _, _ = build_labels_v2(tracks, _clean_keys(tracks, paths), paths, ecfg)
    lab = labels.tolist()
    primary = np.array([a[0] if len(a) else "" for a in rec.tracks["artist_ids"]], dtype=object)
    labeled = np.flatnonzero(labels.notna().to_numpy())
    n = args.n_seeds or ev["n_seeds"]
    seeds = np.random.default_rng(ev["random_seed"]).choice(labeled, n, replace=False)
    print(f"loaded in {time.time() - t0:.0f}s; {len(labeled):,} labeled tracks, {n} seeds",
          flush=True)

    if args.suite == "lengths":
        df = run_lengths(rec, lab, seeds, gcfg, primary)
        md = report_lengths(df, gcfg, n, rdir / out_name)
        (rdir / f"{out_name}.md").write_text(md, encoding="utf-8")
        df.to_parquet(rdir / f"{out_name}_per_playlist.parquet", index=False)
        print(md)
        print(f"total {time.time() - t0:.0f}s")
        return

    if args.suite == "step5_1":
        df = run_5_1(rec, lab, seeds, gcfg, primary)
        md = report_5_1(df, gcfg, n, rdir / out_name)
        (rdir / f"{out_name}.md").write_text(md, encoding="utf-8")
        df.to_parquet(rdir / f"{out_name}_per_playlist.parquet", index=False)
        print(md)
        print(f"total {time.time() - t0:.0f}s")
        return

    main_step5(rec, lab, seeds, gcfg, primary, n, ex_seed, rdir, fig_dir, out_name, t0)


def main_step5(rec, lab, seeds, gcfg, primary, n, ex_seed, rdir, fig_dir, out_name, t0) -> None:
    ev = gcfg["evaluation"]
    df, mismatches = run(rec, lab, seeds, gcfg, primary)
    bc = ev["bootstrap"]
    pooled = bootstrap(df, ["table", "method"], bc)
    per_arc = bootstrap(df[df["table"] == "main"], ["arc", "method"], bc)
    # the Step 5 figure uses the Step 5 defaults
    g5 = {**gcfg, "objective": {**gcfg["objective"], **ev["objective"],
                                "weak_link": {"lambda": 0.0, "tau": 0.85}},
          "constraints": {**gcfg["constraints"], "max_dbpm": None},
          "pool": {**gcfg["pool"], "energy_bands": None}, "seed_position": "start"}
    ex_md = plot_examples(rec, g5, ex_seed, fig_dir / "step5_examples.png",
                          "beam_v1 example playlists")
    o = ev["objective"]
    settings = (f"{n} genre-labeled seeds (random_seed {ev['random_seed']}) x {len(gcfg['arcs'])} "
                f"arc presets, N = {ev['length']}; candidates = full catalogue; pool = top "
                f"{gcfg['pool']['k']} by relevance to the seed (FAISS top {gcfg['pool']['per_seed_k']}); "
                f"alpha = {o['alpha']}, gamma = {o['gamma']}, B = {gcfg['beam']['width']}, max 1 "
                f"track per artist, no |dBPM| cap; {bc['n_resamples']} bootstrap resamples over "
                f"seeds, {bc['ci']:.0%} CIs (each seed = mean over the 4 arcs)")
    check = (f"beam(B=1) == greedy_v0 on {len(seeds) * len(gcfg['arcs']) - mismatches:,} of "
             f"{len(seeds) * len(gcfg['arcs']):,} playlists")
    notes = ("Transitions: 11 per playlist (seed -> 2 -> ... -> 12). % harmonic compat = share "
             f"with Camelot score >= {ev['harmonic_compatible']} (same key, +-1, relative; unknown "
             "keys count as incompatible). |dBPM| after octave alignment. arc MAE over positions "
             "2..12 (position 1 is the seed). genre agree = share of labeled tracks in positions "
             "2..12 whose genre set overlaps the seed's; only playlists with >=1 labeled track "
             "count (seeds w/ label). objective = the default objective (alpha 0.5, gamma 0.5) for "
             "every row. runtime = pool + transition matrix + search, single-threaded BLAS/OpenMP; "
             "for the reordered baselines it is the beam's runtime.")
    sections = []
    for md in (False, True):
        lines = []
        main = pooled[pooled["table"] == "main"]
        lines += table(main, "Main: generators and baselines (pooled over arcs)", "method", md)
        for arc in gcfg["arcs"]:
            lines += table(per_arc[per_arc["arc"] == arc], f"Per arc: {arc}", "method", md,
                           METRICS[:6])
        for tbl, title in [("beam_width", "Ablation: beam width (alpha 0.5, gamma 0.5)"),
                           ("gamma", "Ablation: gamma (B = 20, alpha 0.5)"),
                           ("alpha", "Ablation: alpha (B = 20, gamma 0.5)"),
                           ("max_dbpm", "Extra ablation: |dBPM| cap per transition "
                                        "(B = 20, alpha 0.5, gamma 0.5)")]:
            lines += table(pooled[pooled["table"] == tbl], title, "method", md)
        sections.append(lines)
    text = "\n".join(["STEP 5 EVALUATION: beam-search arc generator", "=" * 100, settings, check,
                      *sections[0], notes])
    md = "\n".join(["# Step 5 evaluation: beam-search arc generator", "", settings, "",
                    f"Sanity check: {check}.", "", *sections[1], notes, "",
                    "## Example playlists (seed: Duke Dumont - Worship)", "",
                    "![examples](figures/step5_examples.png)", "", *ex_md])
    (rdir / f"{out_name}.txt").write_text(text, encoding="utf-8")
    (rdir / f"{out_name}.md").write_text(md, encoding="utf-8")
    pd.concat([pooled.assign(scope="pooled"), per_arc.assign(scope="per_arc")]) \
        .to_csv(rdir / f"{out_name}.csv", index=False)
    df.to_parquet(rdir / f"{out_name}_per_playlist.parquet", index=False)
    print(text)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
