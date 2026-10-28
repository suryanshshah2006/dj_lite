# DJ-Lite: Transition-Aware Playlist Recommender

## Project summary
A recommender that builds playlists as DJ-style *mixes*: given one or more seed songs and a target
energy arc (e.g. warm-up → peak → cool-down), it generates an ordered playlist that maximizes
(1) similarity/relevance to the seeds, (2) smoothness of each song-to-song transition, and
(3) adherence to the target energy curve. No raw audio is used — only precomputed Spotify audio features.

## Data
- Main dataset: Kaggle "Spotify 1.2M+ Songs" (rodolfofigueroa/spotify-12m-songs)
- Location: `data/raw/tracks_features.csv` (~1.2M rows, ~345 MB). NEVER modify files in `data/raw/`.
- Expected columns: id, name, album, album_id, artists, artist_ids, track_number, disc_number, explicit,
  danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness,
  valence, tempo, duration_ms, time_signature, year, release_date
  (verify against the actual file before relying on this list).
- The dataset has NO genre column and NO user/playlist data. Do not invent these.
- Optional later: DJ Mix Dataset metadata (tracklists only, no audio) as ground-truth transitions.

## Pipeline (build in this order)
1. Data inventory + cleaning → `data/processed/tracks_clean.parquet`
   (drop duplicates, tempo == 0, missing values, extreme durations; parse artists lists).
2. Feature engineering → `data/features/`
   - Camelot wheel code from (key, mode)
   - Normalized tempo, with half/double-time awareness
   - Standardized audio features
3. Transition scoring f(A, B) → score in [0, 1]
   - v0: rule-based (Camelot distance, |ΔBPM| with half/double time, Δenergy, Δloudness, Δvalence)
   - v1: learned model (LightGBM) on pairwise features, once ground-truth transitions exist
4. Candidate retrieval: nearest neighbors in standardized feature space (FAISS or sklearn), top-K per seed.
5. Sequence generation with target energy arc:
   greedy baseline → beam search → (later) RL.
   Objective per step: α·relevance + β·transition_score + γ·(1 − |energy − target|) − δ·repetition penalty.
6. Evaluation: mean Camelot distance, mean |ΔBPM|, arc MAE, artist repetition rate, intra-list diversity.
   Baselines: random order, similarity-only greedy, rule-based DJ ordering.
7. Demo UI: Streamlit app — pick seed songs, draw/choose an arc, view the playlist + transition-score plot.

## Environment
- Python 3.11, virtual environment in `.venv/` (activate before running anything).
- Core libs: pandas, numpy, pyarrow, scikit-learn, lightgbm, faiss-cpu, rapidfuzz, matplotlib, plotly,
  streamlit, pyyaml, tqdm, pytest.
- Runs on a laptop CPU. No GPU assumptions. Keep RAM usage reasonable: read the CSV once,
  then work from parquet with only the needed columns.

## Conventions
- Code lives in `src/djlite/` as importable modules; notebooks in `notebooks/` are for exploration only.
- Configs (weights α/β/γ/δ, K, arc shapes, paths) live in `configs/*.yaml` — no hard-coded paths.
- Save intermediate outputs as parquet; save figures to `reports/figures/`.
- Fix random seeds. Report experiment results as mean ± std over 5 seeds.
- Write small pytest tests for the Camelot conversion and transition scoring functions.
- Do not commit anything in `data/` or `models/` to git.
- Before large changes, explain the plan briefly; after changes, say how to run them.
