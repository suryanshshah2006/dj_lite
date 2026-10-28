# Step 3 evaluation

200 genre-labeled seed tracks x 1 sampling seed(s); mean ± std over sampling seeds. Genre labels are evaluation-only.

Settings: K=500, alpha=0.5, beta=0.5, exclude_same_artist=True, max_per_artist=1, missing_style=renormalize

## Primary: labeled pool (candidates = genre-labeled tracks)

| method | genre agree@10 | transition@10 | |dBPM|@10 | ILD@10 | artist rep (uncapped) | labeled/list | seeds w/ label |
|---|---:|---:|---:|---:|---:|---:|---:|
| (a) transition score only (v0.1) | 0.064 | 0.903 | 0.90 | 0.668 | 0.035 | 10.00 | 1.00 |
| (b) audio similarity only | 0.113 | 0.556 | 7.84 | 0.454 | 0.087 | 10.00 | 1.00 |
| (c) audio + style relevance | 0.163 | 0.517 | 11.54 | 0.300 | 0.221 | 10.00 | 1.00 |
| (d) alpha*relevance + beta*transition | 0.117 | 0.836 | 2.80 | 0.500 | 0.082 | 10.00 | 1.00 |

## Secondary: full pool (candidates = all cleaned tracks)

| method | genre agree@10 | transition@10 | |dBPM|@10 | ILD@10 | artist rep (uncapped) | labeled/list | seeds w/ label |
|---|---:|---:|---:|---:|---:|---:|---:|
| (a) transition score only (v0.1) | 0.000 | 0.980 | 0.31 | 0.460 | 0.015 | 0.04 | 0.04 |
| (b) audio similarity only | 0.167 | 0.648 | 3.77 | 0.345 | 0.014 | 0.10 | 0.09 |
| (c) audio + style relevance | 0.371 | 0.561 | 8.47 | 0.202 | 0.339 | 0.18 | 0.15 |
| (d) alpha*relevance + beta*transition | 0.265 | 0.899 | 1.50 | 0.279 | 0.113 | 0.19 | 0.17 |

genre agree@10 = share of genre-labeled results sharing a genre with the seed, averaged over seeds with ≥1 labeled result (seeds w/ label). |dBPM| is after half/double-time alignment. transition@10 = mean transition_score_v0_1(seed → result). ILD@10 = mean pairwise cosine distance between results' combined audio + style vectors. artist rep (uncapped) = share of results in the top-10 *without* the max_per_artist cap that share an artist with a higher-ranked result; every other column uses the capped list.

## Known limitations

- **FAISS missing-style retrieval bias.** Retrieval searches [sqrt(w_audio)·audio, sqrt(w_style)·style]. A candidate without a style vector (artist on no shared album; ~38% of tracks) has a zero style block, so for a seed *with* style its inner product is only w_audio·cos_audio and it ranks below styled candidates in the top-K. Exact re-ranking renormalizes the relevance for such candidates (0.70 audio / 0.30 era), but it can only re-rank what FAISS returned, so style-less tracks are under-represented for methods (c) and (d). For a seed without style, the query's style block is zero and retrieval is audio-only, which is unbiased.
- Method (a) scans the whole pool; (b) uses an audio-only index; (c) and (d) re-rank the top K=500 from the combined index.
- Genre labels come from a different Kaggle dataset matched by track ID; a track can carry several genres there, and any overlap counts as agreement.
