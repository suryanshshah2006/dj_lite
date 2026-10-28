# Step 3 evaluation

1000 genre-labeled seed tracks x 5 sampling seed(s); mean ± std over sampling seeds. Genre labels are evaluation-only.

Settings: K=500, alpha=0.5, beta=0.5, exclude_same_artist=True, max_per_artist=1, missing_style=renormalize

## Primary: labeled pool (candidates = genre-labeled tracks)

| method | genre agree@10 | transition@10 | |dBPM|@10 | ILD@10 | artist rep (uncapped) | labeled/list | seeds w/ label |
|---|---:|---:|---:|---:|---:|---:|---:|
| (a) transition score only (v0.1) | 0.061 ± 0.001 | 0.899 ± 0.001 | 0.93 ± 0.01 | 0.678 ± 0.003 | 0.044 ± 0.002 | 10.00 ± 0.00 | 1.00 ± 0.00 |
| (b) audio similarity only | 0.106 ± 0.002 | 0.554 ± 0.001 | 7.88 ± 0.09 | 0.465 ± 0.002 | 0.090 ± 0.001 | 10.00 ± 0.00 | 1.00 ± 0.00 |
| (c) audio + style relevance | 0.146 ± 0.003 | 0.518 ± 0.001 | 11.39 ± 0.14 | 0.313 ± 0.003 | 0.206 ± 0.004 | 10.00 ± 0.00 | 1.00 ± 0.00 |
| (d) alpha*relevance + beta*transition | 0.112 ± 0.004 | 0.831 ± 0.001 | 2.98 ± 0.06 | 0.508 ± 0.004 | 0.087 ± 0.003 | 10.00 ± 0.00 | 1.00 ± 0.00 |

## Secondary: full pool (candidates = all cleaned tracks)

| method | genre agree@10 | transition@10 | |dBPM|@10 | ILD@10 | artist rep (uncapped) | labeled/list | seeds w/ label |
|---|---:|---:|---:|---:|---:|---:|---:|
| (a) transition score only (v0.1) | 0.081 ± 0.046 | 0.979 ± 0.000 | 0.35 ± 0.01 | 0.467 ± 0.002 | 0.015 ± 0.000 | 0.06 ± 0.01 | 0.06 ± 0.01 |
| (b) audio similarity only | 0.186 ± 0.046 | 0.646 ± 0.004 | 3.75 ± 0.08 | 0.352 ± 0.003 | 0.015 ± 0.001 | 0.06 ± 0.01 | 0.06 ± 0.01 |
| (c) audio + style relevance | 0.367 ± 0.043 | 0.565 ± 0.002 | 8.40 ± 0.13 | 0.195 ± 0.002 | 0.314 ± 0.007 | 0.15 ± 0.01 | 0.12 ± 0.01 |
| (d) alpha*relevance + beta*transition | 0.265 ± 0.028 | 0.896 ± 0.001 | 1.59 ± 0.07 | 0.274 ± 0.004 | 0.122 ± 0.004 | 0.14 ± 0.01 | 0.13 ± 0.01 |

genre agree@10 = share of genre-labeled results sharing a genre with the seed, averaged over seeds with ≥1 labeled result (seeds w/ label). |dBPM| is after half/double-time alignment. transition@10 = mean transition_score_v0_1(seed → result). ILD@10 = mean pairwise cosine distance between results' combined audio + style vectors. artist rep (uncapped) = share of results in the top-10 *without* the max_per_artist cap that share an artist with a higher-ranked result; every other column uses the capped list.

## Known limitations

- **FAISS missing-style retrieval bias.** Retrieval searches [sqrt(w_audio)·audio, sqrt(w_style)·style]. A candidate without a style vector (artist on no shared album; ~38% of tracks) has a zero style block, so for a seed *with* style its inner product is only w_audio·cos_audio and it ranks below styled candidates in the top-K. Exact re-ranking renormalizes the relevance for such candidates (0.70 audio / 0.30 era), but it can only re-rank what FAISS returned, so style-less tracks are under-represented for methods (c) and (d). For a seed without style, the query's style block is zero and retrieval is audio-only, which is unbiased.
- Method (a) scans the whole pool; (b) uses an audio-only index; (c) and (d) re-rank the top K=500 from the combined index.
- Genre labels come from a different Kaggle dataset matched by track ID; a track can carry several genres there, and any overlap counts as agreement.
