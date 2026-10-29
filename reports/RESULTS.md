# DJ-Lite: final results (frozen)

Transition-aware playlist recommender built only from precomputed Spotify audio features
(no raw audio). This file collects the final numbers from Steps 3, 5 and 5.1 and their
held-out validation in one place; each section names the report it comes from. All
intervals are 95% bootstrap CIs over seeds.

**Headline (held-out, section 5).** On 500 seeds never used for tuning, the final generator
(beam search, energy-stratified pool, weak-link penalty, 3 BPM cap, seed at its best-fit arc
position) builds 12-track mixes with a mean transition score of 0.951 and a worst
transition of 0.909, 99.8% harmonically compatible transitions, a largest BPM jump of
1.8 BPM and an energy-arc error of 0.106, in ~1.2 s on a laptop CPU. The transition numbers
match the tuning set exactly; the arc error is 0.014 worse than on the tuning set (0.092).
Ordering is what matters most: the same tracks in random order have a worst transition of
0.61. Beam beats greedy consistently but modestly, by the same margin on both seed sets and
at 12 and 24 tracks. Search returns the original artist first for 10 of 20 held-out
famous-song queries (12/20 on the tuning queries).

---

## 1. Data

| | |
|---|---|
| Source | Kaggle "Spotify 1.2M+ Songs" (`data/raw/tracks_features.csv`), 1,204,025 rows |
| After cleaning + 2-pass dedup | **1,089,816 tracks** (90.5% kept; 67,844 duplicate releases merged) |
| Style vectors (artist-album SVD, 64 dims, 2-hop smoothing) | 61.5% of tracks |
| Genre labels (evaluation only, never a model input) | separate Kaggle dataset (114 genres); 7,250 cleaned tracks labeled by track ID or normalized name + primary artist |

Sources: `reports/cleaning_report.txt`, `reports/style_report.txt`, `reports/eval_v2.md`.

---

## 2. Step 3: next-track retrieval (top-10 lists)

Methods: (a) transition score only (v0.1), (b) audio similarity, (c) audio + style relevance,
(d) alpha * relevance + (1 - alpha) * transition, (r) random (artist-capped). Max 1 track
per artist in every list.

### 2.1 Primary: labeled pool, every labeled seed once (7,250 seeds) — `reports/eval_v2.md`

| method | genre agree (micro) | genre agree (macro) | transition@10 | \|dBPM\|@10 | ILD@10 | artist rep (uncapped) |
|---|---:|---:|---:|---:|---:|---:|
| (a) transition only | 0.062 [0.060, 0.064] | 0.051 [0.049, 0.054] | **0.916** [0.915, 0.917] | **0.80** | 0.649 | 0.037 |
| (b) audio only | 0.097 [0.094, 0.100] | 0.078 [0.075, 0.082] | 0.566 [0.565, 0.568] | 7.03 | 0.464 | 0.069 |
| (c) audio + style | **0.165** [0.160, 0.169] | **0.136** [0.132, 0.143] | 0.523 [0.521, 0.524] | 11.13 | 0.291 | 0.214 |
| (d) alpha = 0.5 | 0.118 [0.115, 0.122] | 0.100 [0.097, 0.105] | 0.847 [0.846, 0.848] | 2.52 | 0.472 | 0.083 |
| (r) random | 0.021 [0.019, 0.022] | 0.018 [0.017, 0.020] | 0.375 [0.374, 0.377] | 21.40 | 0.902 | 0.006 |

### 2.2 Alpha sweep, method (d), labeled pool — `reports/eval_v2.md`, `reports/figures/pareto_alpha.png`

| alpha | 0 | 0.1 | 0.25 | **0.5** | 0.75 | 0.9 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| genre agree (micro) | 0.092 | 0.094 | 0.100 | 0.118 | 0.147 | 0.161 | 0.165 |
| transition@10 | 0.870 | 0.870 | 0.868 | 0.847 | 0.756 | 0.649 | 0.523 |
| \|dBPM\|@10 | 1.58 | 1.59 | 1.68 | 2.52 | 5.60 | 8.35 | 11.13 |

Knee: alpha ~0.5 by marginal cost (beyond it each unit of genre agreement costs 3-8x more
transition score; |dBPM| doubles from 0.5 to 0.75); 0.75 by elbow detection. The app's
"DJ mode" is 0.5 and "Discovery mode" 0.75.

### 2.3 Label-source check (does name matching add noise?) — `reports/eval_v2_by_source.md`

| labeled pool, micro / macro | ID-matched seeds (4,142) | name-only seeds (3,108) |
|---|---:|---:|
| (a) | 0.067 / 0.051 | 0.055 / 0.054 |
| (b) | 0.106 / 0.077 | 0.085 / 0.081 |
| (c) | 0.170 / 0.130 | 0.157 / 0.148 |
| (d) alpha 0.5 | 0.125 / 0.096 | 0.110 / 0.110 |
| (r) random | 0.021 / 0.019 | 0.020 / 0.018 |

Same ranking in both groups and an identical random baseline: no sign that name matching
adds label noise.

### 2.4 Earlier protocol (v1: 1,000 seeds x 5 sampling seeds, ID-matched labels only) — `reports/eval_step3.md`

| labeled pool (4,142 tracks) | genre agree | transition@10 | \|dBPM\|@10 | ILD@10 | artist rep |
|---|---:|---:|---:|---:|---:|
| (a) | 0.061 +- 0.001 | 0.899 +- 0.001 | 0.93 | 0.678 | 0.044 |
| (b) | 0.106 +- 0.002 | 0.554 +- 0.001 | 7.88 | 0.465 | 0.090 |
| (c) | 0.146 +- 0.003 | 0.518 +- 0.001 | 11.39 | 0.313 | 0.206 |
| (d) | 0.112 +- 0.004 | 0.831 +- 0.001 | 2.98 | 0.508 | 0.087 |

Superseded by v2 (the +- understates uncertainty: runs share seeds); kept for reproducibility.

---

## 3. Step 5: beam-search arc generator — `reports/eval_step5.md`

500 genre-labeled seeds x 4 arc presets (Warm-up -> Peak -> Cool-down, Steady, Slow Build,
Sunset), N = 12, full catalogue, Step 5 defaults (alpha 0.5, gamma 0.5, B 20, no cap, seed
first). Each seed's value is the mean over the 4 arcs.

### 3.1 Generators and baselines

| method | transition mean | transition min | % harmonic | \|dBPM\| max | arc MAE |
|---|---:|---:|---:|---:|---:|
| beam_v1 (B=20) | **0.938** [0.936, 0.940] | 0.853 [0.847, 0.859] | **98.5** | 3.35 | 0.067 [0.061, 0.072] |
| greedy_v0 | 0.930 [0.928, 0.932] | 0.855 [0.850, 0.861] | 98.0 | 3.02 | 0.076 [0.070, 0.082] |
| beam tracks, random order | 0.798 | 0.578 | 65.7 | 6.22 | 0.159 |
| beam tracks, energy-sorted | 0.851 | 0.643 | 75.5 | 5.48 | **0.065** |

beam(B=1) reproduced greedy_v0 on 2,000 / 2,000 playlists.

### 3.2 Ablations

| beam width | transition min | \|dBPM\| max | arc MAE | objective |
|---|---:|---:|---:|---:|
| B=1 (= greedy) | 0.855 | 3.02 | 0.076 | 15.18 |
| B=5 | 0.860 | 3.02 | 0.070 | 15.27 |
| B=20 | 0.853 | 3.35 | 0.067 | 15.30 |
| B=50 | 0.848 | 3.54 | 0.065 | 15.31 |

| gamma (arc weight) | transition mean | transition min | \|dBPM\| max | arc MAE |
|---|---:|---:|---:|---:|
| 0 | 0.962 | 0.924 | 1.67 | 0.248 |
| 0.25 | 0.949 | 0.897 | 2.07 | 0.094 |
| 0.5 | 0.938 | 0.853 | 3.35 | 0.067 |
| 1.0 | 0.922 | 0.790 | 5.39 | 0.050 |

| alpha | transition min | \|dBPM\| max | genre agree |
|---|---:|---:|---:|
| 0.5 | 0.853 | 3.35 | 0.179 [0.132, 0.224] |
| 0.75 | 0.745 | 7.59 | 0.215 [0.160, 0.264] |

| \|dBPM\| cap | transition min | \|dBPM\| max | arc MAE |
|---|---:|---:|---:|
| none | 0.853 | 3.35 | 0.067 |
| 6 | 0.861 | 2.43 | 0.068 |
| 3 | 0.866 | 2.06 | 0.071 |

---

## 4. Step 5.1: weak-link penalty, stratified pool, seed placement — `reports/eval_step5_1.md`

Same 500 seeds x 4 arcs, N = 12. Weak-link penalty: lambda * max(0, tau - T) per transition,
tau = 0.85. New defaults: gamma 0.25, 3 BPM cap.

### 4.1 Weak-link penalty

| configuration | transition mean | transition min | \|dBPM\| max | arc MAE |
|---|---:|---:|---:|---:|
| Step 5 defaults, lambda 0 | 0.938 | 0.853 [0.847, 0.859] | 3.35 | 0.067 |
| Step 5 defaults, lambda 0.5 | 0.941 | 0.880 | 2.35 | 0.071 |
| Step 5 defaults, lambda 1 | 0.942 | 0.884 | 2.26 | 0.073 |
| Step 5 defaults, lambda 2 | 0.942 | 0.888 [0.886, 0.890] | 2.22 | 0.074 |
| gamma 0.25 + cap 3, lambda 0 | 0.950 | 0.899 [0.896, 0.902] | 1.87 | 0.095 |
| gamma 0.25 + cap 3, **lambda 2** | 0.950 | 0.905 [0.903, 0.908] | 1.85 | 0.097 |

lambda = 2 chosen by a rule fixed before the run (highest mean min-transition under the new
defaults with arc MAE within +0.01 of lambda = 0).

### 4.2 Energy-stratified pool (5 energy-quantile bands x 400 from FAISS top 10,000), arc MAE

| arc | lambda | plain pool | stratified pool |
|---|---|---:|---:|
| Sunset | 0 | 0.115 [0.108, 0.123] | **0.102** [0.095, 0.108] |
| Sunset | 2 | 0.117 | **0.103** |
| Warm-up -> Peak -> Cool-down | 0 | 0.085 [0.078, 0.092] | **0.079** [0.072, 0.087] |
| Warm-up -> Peak -> Cool-down | 2 | 0.086 | **0.081** |

Over all arcs (lambda 0): transition min 0.899 -> 0.903, \|dBPM\| max 1.87 -> 1.78, ~+90 ms.

### 4.3 Seed placement (new defaults, lambda 2)

| pool, placement | transition min | arc MAE (generated positions) | arc MAE incl. seed |
|---|---:|---:|---:|
| plain, start | 0.905 | 0.097 | 0.113 |
| plain, best_fit | 0.905 | 0.104 | 0.102 |
| stratified, start | 0.908 | 0.089 | 0.106 |
| **stratified, best_fit (final)** | **0.909** [0.907, 0.911] | 0.093 | **0.092** [0.086, 0.099] |

### 4.4 Beam vs greedy at N = 12 and N = 24, identical final settings — `reports/eval_lengths.md`

Final defaults for both generators and both lengths (stratified pool, best_fit, lambda 2,
gamma 0.25, cap 3); each seed's pool is shared by all four runs.

| configuration | transition mean | transition min | % harmonic | \|dBPM\| max | arc MAE incl. seed | objective | runtime ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| **N=12, beam_v1 (final)** | **0.952** [0.950, 0.953] | **0.909** [0.907, 0.911] | 99.8 | 1.81 | **0.092** [0.086, 0.099] | 12.786 | 1264 |
| N=12, greedy_v0 | 0.946 [0.944, 0.947] | 0.901 [0.899, 0.904] | 99.7 | 1.85 | 0.106 [0.099, 0.112] | 12.704 | 1254 |
| N=24, beam_v1 | 0.953 [0.952, 0.955] | 0.900 [0.898, 0.903] | 99.8 | 2.10 | 0.094 [0.088, 0.100] | 26.715 | 1278 |
| N=24, greedy_v0 | 0.947 [0.945, 0.948] | 0.891 [0.889, 0.893] | 99.7 | 2.14 | 0.109 [0.102, 0.115] | 26.527 | 1255 |

| paired difference, beam - greedy | transition mean | transition min | arc MAE incl. seed | objective | objective per placed track |
|---|---:|---:|---:|---:|---:|
| N = 12 | +0.006 [0.005, 0.007] | +0.008 [0.006, 0.009] | -0.013 [-0.015, -0.012] | +0.082 [0.078, 0.087] | +0.0075 |
| N = 24 | +0.007 [0.006, 0.007] | +0.009 [0.008, 0.010] | -0.015 [-0.016, -0.014] | +0.187 [0.179, 0.196] | +0.0081 |

With identical settings, beam's advantage per transition and per position is the same at
N = 24 as at N = 12: the total objective gap doubles only because the playlist is twice as
long. (Step 5.1's N = 24 comparison suggested a growing advantage, but it was confounded with
the change of defaults; this check supersedes it.) Beam is consistently but modestly better
than greedy (CIs exclude zero) at ~+10-25 ms.

---

## 5. Held-out validation — `reports/eval_heldout.md`, `reports/heldout_seeds.csv`

All defaults above were chosen on the tuning set (the 500 seeds of Steps 5 / 5.1 and the 20
search queries). To check them, the final configuration was run **once, with no tuning**, on
data never used to choose anything.

### 5.1 Generator: 500 held-out seeds

500 genre-labeled seeds drawn with a new fixed random seed (20261029) from the 6,750 labeled
tracks that are not tuning seeds (disjoint by construction; IDs, names and genres in
`reports/heldout_seeds.csv`). Final configuration, N = 12, 4 arcs; both seed sets were run in
the same process with the same code, so the tuning column matches section 4.4.

| method | set | transition mean | transition min | % harmonic | \|dBPM\| max | arc MAE incl. seed | genre agree |
|---|---|---:|---:|---:|---:|---:|---:|
| **beam_v1 (final)** | tuning | 0.952 [0.950, 0.953] | 0.909 [0.907, 0.911] | 99.8 | 1.81 [1.77, 1.85] | 0.092 [0.086, 0.099] | 0.199 [0.149, 0.249] |
| **beam_v1 (final)** | **held-out** | **0.951** [0.950, 0.953] | **0.909** [0.907, 0.911] | **99.8** | **1.80** [1.76, 1.83] | **0.106** [0.099, 0.113] | 0.217 [0.154, 0.286] |
| greedy_v0 | tuning | 0.946 [0.944, 0.947] | 0.901 [0.899, 0.904] | 99.7 | 1.85 | 0.106 [0.099, 0.112] | 0.173 |
| greedy_v0 | held-out | 0.946 [0.944, 0.947] | 0.902 [0.899, 0.904] | 99.7 | 1.84 | 0.120 [0.112, 0.128] | 0.196 |
| beam tracks, random order | tuning | 0.817 | 0.613 | 66.0 | 4.29 | 0.169 | — |
| beam tracks, random order | held-out | 0.815 | 0.612 | 65.4 | 4.28 | 0.176 | — |
| beam tracks, energy-sorted | tuning | 0.870 | 0.692 | 76.4 | 3.69 | 0.090 | — |
| beam tracks, energy-sorted | held-out | 0.864 | 0.684 | 75.0 | 3.73 | 0.104 | — |

(Genre agreement of the reordered baselines equals the beam's: same tracks.)

- **Confirmed:** every transition metric (mean 0.951, worst 0.909, 99.8% harmonic, largest
  jump 1.8 BPM) is the same on held-out seeds; beam beats greedy by the same margin
  (worst transition +0.008, arc MAE -0.014 on both sets); ordering still matters as much
  (random order: worst transition 0.61).
- **Revised:** arc MAE is 0.106 on held-out seeds, not 0.092. Greedy and the energy-sorted
  baseline worsen by the same ~0.014, so this looks like the held-out seeds being harder to
  fit to the four arcs rather than the tuned settings overfitting, but either way **0.106 is
  the estimate to quote**.
- Runtime 1.24 s per 12-track mix.

### 5.2 Search: 20 held-out famous-song queries

20 new queries, never used for tuning: a fixed priority list spread over eras and genres,
with no song from the tuning candidates and no artist from the tuning set, filtered by
existence only (first 20 whose original recording exists under the exact title). The first
40 candidates yielded only 15, so the list was extended once with 25 more candidates, fixed
before any was ranked. Evaluated once with the configured ranking
(`python -m app.backend.search_regression --heldout`).

| ranking | tuning set (20) | held-out (20) |
|---|---:|---:|
| **configured: co-artist prominence + original-release boost (>= 100)** | **12/20** | **10/20** |
| co-artist prominence, no boost | 11/20 | 9/20 |
| track-count prominence (the first specification) | 4/20 | 7/20 |

The configured ranking stays best on held-out queries and in the same order of choices, but
its pass rate drops from 60% to **50%**. Held-out misses: Like a Prayer, Crazy, Toxic,
September, Thunderstruck, Mr. Blue Sky, Hound Dog, Purple Haze, Johnny B. Goode, Good
Vibrations (mostly covers, re-recordings or same-title songs by artists with more co-artists).

---

## 6. Final configuration

`configs/generator.yaml` (used by the app's `/api/mix`):

```yaml
pool:        {per_seed_k: 2000, k: 2000, energy_bands: 5, stratified_retrieve_k: 10000}
objective:   {alpha: 0.5, gamma: 0.25, weak_link: {lambda: 2.0, tau: 0.85}}
constraints: {max_per_artist: 1, max_dbpm: 3}
seed_position: best_fit
beam:        {width: 20}
```

Other components: transition scorer v0.1 (`configs/transitions.yaml`: harmonic 0.30, tempo
0.30 with octave alignment, energy smoothness 0.15, timbre 0.15, loudness 0.05, era 0.05);
relevance 0.35 audio + 0.50 style + 0.15 era, renormalized to 0.70 / 0.30 when a style
vector is missing (`configs/retrieval.yaml`); FAISS IndexFlatIP, K = 500 for next-track
lists. Web app search: SQLite FTS5, title tiers then artist, tie-break by artist co-artist
prominence plus an original-release boost (top-1 original artist: 12/20 on the tuning
queries, 10/20 on held-out queries).

Runtime (laptop CPU, single-threaded BLAS/OpenMP, backend ~2.3 GB RAM, ~25 s startup):
12-track mix 1.26 s mean in the evaluation and 1.2-1.4 s via the API alone (target < 2 s);
2.2-2.5 s observed while a browser, the Next.js server and the backend shared the machine's
7.3 GB.

---

## 7. Figures

| figure | path |
|---|---|
| Relevance vs transition trade-off (alpha sweep, both pools) | `reports/figures/pareto_alpha.png` |
| Example playlists, final defaults (one per arc preset, seed marked) | `reports/figures/step5_examples.png` (track lists: `reports/generator_examples.md`) |
| Transition-score distribution over 100k random pairs (v0) | `reports/figures/transition_score_hist.png` |
| Camelot key distribution | `reports/figures/camelot_distribution.png` |
| App screenshots (local only, contain cover art; gitignored) | `reports/screenshots/` |

---

## 8. Known limitations

**Data and labels**
- No genre, user or playlist data in the main dataset; genre agreement uses a separate
  114-genre dataset matched by ID or by normalized name + primary artist (7,250 tracks).
  Name-only matches can be a different recording of the same song.
- Full-catalogue genre agreement rests on the ~6-25% of seeds whose results contain any
  labeled track, so its CIs are wide (e.g. 0.18 [0.13, 0.22] for the generator).
- Genres are imbalanced; macro agreement is driven partly by small genres.
- Catalogue coverage is uneven: some famous recordings are missing (e.g. Avicii "Levels")
  and some artists have only a handful of tracks (Queen: 6).

**Models**
- Transition scoring is rule-based (v0.1) and unvalidated against real DJ transitions; no
  ground-truth transition data was used.
- Style vectors exist for 61.5% of tracks; FAISS retrieval over [audio, style] favours
  styled candidates for styled seeds (missing-style bias), only partly corrected by exact
  re-ranking.
- Energy is Spotify's track-level feature: no intros/outros, no within-track energy.
- Beam search optimizes a sum; even with the weak-link penalty and the cap the worst
  transition is ~0.91, B > 20 adds nothing measurable, and the gain over greedy is small
  (+0.008 worst transition, -0.013 arc MAE) and does not grow with playlist length.
- Arc extremes depend on the seed: e.g. Sunset tails stay around 0.4 energy for house seeds
  because relevant low-energy tracks are scarce, even with the stratified pool.
- best_fit seed placement improves whole-playlist arc fit but means the seed is often not
  the first track (set `seed_position: start` to change).

**Evaluation**
- Tuned on the tuning set (500 seeds of Steps 5 / 5.1, random_seed 7): gamma 0.25 (from the
  Step 5 gamma ablation), the 3 BPM cap (Step 5 extra ablation), lambda 2 (rule fixed before
  the Step 5.1 run), the 5-band energy-stratified pool and best_fit seed placement (by
  inspection of the Step 5.1 tables); beam width 20 and alpha 0.5 were set earlier (alpha
  from the Step 3 sweep on the labeled pool, a different protocol). Tuned on the 20 search
  queries: co-artist prominence and the original-release boost threshold (100).
- Held-out check (section 5, run once, no tuning): **confirms** the transition results
  (mean 0.951, worst 0.909, 99.8% harmonic, largest jump 1.8 BPM; identical to the tuning
  set) and beam's margin over greedy; **revises** arc MAE from 0.092 to 0.106 (greedy and
  the energy-sorted baseline shift equally, so likely harder seeds rather than overfitting).
  Search: the chosen ranking stays best on held-out queries but drops from 12/20 to 10/20.
  Quote the held-out figures.
- The Step 3 alpha sweep, the Step 5 ablations and the Step 5.1 comparisons themselves have
  no held-out replication; only the final configuration was re-run.
- Single-seed playlists only; multi-seed sets are supported but not evaluated.
- Search misses (8/20 tuning, 10/20 held-out) are mostly covers, re-recordings or
  same-title songs by artists with larger catalogues or more co-artists; many famous
  originals are not in the dataset at all (only 20 of 65 held-out candidates were).

---

## 9. Reproduce

```powershell
# from the project root, venv active; single-threaded BLAS/OpenMP keeps RAM ~2.5 GB
$env:OPENBLAS_NUM_THREADS = "1"; $env:OMP_NUM_THREADS = "1"
python -m djlite.evaluate --version v2                       # Step 3 (eval_v2.*)
python -m djlite.evaluate --version v2 --by-source --replot  # label-source split, Pareto figure
python -m djlite.evaluate_generator --suite step5            # Step 5 (eval_step5.*)
python -m djlite.evaluate_generator --suite step5_1          # Step 5.1 (eval_step5_1.*)
python -m djlite.evaluate_generator --suite lengths          # N = 12 vs 24 (eval_lengths.*)
python -m djlite.evaluate_generator --suite heldout          # held-out validation (eval_heldout.*, heldout_seeds.csv)
python -m djlite.evaluate_generator --suite examples         # example figure, final defaults
python -m app.backend.search_regression                      # search, tuning queries (all rankings)
python -m app.backend.search_regression --heldout            # search, held-out queries (configured ranking)
python -m pytest -q tests app/backend/tests
```
