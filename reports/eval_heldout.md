# Held-out validation of the final generator

Final configuration, no tuning: alpha 0.5, gamma 0.25, lambda 2.0 (tau 0.85), |dBPM| cap 3, energy bands 5, seed position best_fit, B 20; N = 12, 4 arc presets. Tuning set = the 500 seeds used in Steps 5 / 5.1 (random_seed 7); held-out = 500 other genre-labeled seeds (random_seed 20261029, disjoint by construction; ids in reports/heldout_seeds.csv). Both sets run in the same process with the same code. 1000 bootstrap resamples over seeds, 95% CIs (each seed = mean over the 4 arcs).

### Main table: tuning set vs held-out set

| row | transition mean | transition min | % harmonic compat | |dBPM| mean | |dBPM| max | arc MAE | arc MAE incl. seed | genre agree | runtime ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| beam_v1 (final) — tuning | 0.952 [0.950, 0.953] | 0.909 [0.907, 0.911] | 99.8 [99.7, 99.8] | 0.70 [0.68, 0.71] | 1.81 [1.77, 1.85] | 0.093 [0.087, 0.100] | 0.092 [0.086, 0.099] | 0.199 [0.149, 0.249] | 1268 [1260, 1276] |
| beam_v1 (final) — held-out | 0.951 [0.950, 0.953] | 0.909 [0.907, 0.911] | 99.8 [99.7, 99.9] | 0.70 [0.68, 0.72] | 1.80 [1.76, 1.83] | 0.108 [0.100, 0.115] | 0.106 [0.099, 0.113] | 0.217 [0.154, 0.286] | 1244 [1238, 1249] |
| greedy_v0 — tuning | 0.946 [0.944, 0.947] | 0.901 [0.899, 0.904] | 99.7 [99.6, 99.8] | 0.74 [0.72, 0.76] | 1.85 [1.81, 1.89] | 0.108 [0.101, 0.115] | 0.106 [0.099, 0.112] | 0.173 [0.125, 0.224] | 1257 [1249, 1265] |
| greedy_v0 — held-out | 0.946 [0.944, 0.947] | 0.902 [0.899, 0.904] | 99.7 [99.6, 99.8] | 0.73 [0.71, 0.75] | 1.84 [1.81, 1.88] | 0.123 [0.115, 0.131] | 0.120 [0.112, 0.128] | 0.196 [0.138, 0.259] | 1233 [1227, 1238] |
| beam tracks, random order — tuning | 0.817 [0.811, 0.822] | 0.613 [0.603, 0.622] | 66.0 [64.5, 67.6] | 1.82 [1.75, 1.90] | 4.29 [4.14, 4.46] | 0.177 [0.173, 0.182] | 0.169 [0.165, 0.174] | 0.199 [0.149, 0.249] | 1268 [1260, 1276] |
| beam tracks, random order — held-out | 0.815 [0.809, 0.821] | 0.612 [0.603, 0.621] | 65.4 [63.7, 67.0] | 1.82 [1.75, 1.88] | 4.28 [4.13, 4.43] | 0.184 [0.180, 0.189] | 0.176 [0.172, 0.181] | 0.217 [0.154, 0.286] | 1244 [1238, 1249] |
| beam tracks, energy-sorted — tuning | 0.870 [0.866, 0.874] | 0.692 [0.683, 0.700] | 76.4 [75.3, 77.6] | 1.48 [1.42, 1.54] | 3.69 [3.57, 3.82] | 0.091 [0.084, 0.097] | 0.090 [0.083, 0.096] | 0.199 [0.149, 0.249] | 1268 [1260, 1276] |
| beam tracks, energy-sorted — held-out | 0.864 [0.859, 0.868] | 0.684 [0.676, 0.692] | 75.0 [73.6, 76.2] | 1.51 [1.45, 1.56] | 3.73 [3.60, 3.86] | 0.105 [0.097, 0.112] | 0.104 [0.096, 0.111] | 0.217 [0.154, 0.286] | 1244 [1238, 1249] |

### Side by side (tuning / held-out)

| method | trans_mean | trans_min | harm_compat | dbpm_max | arc_mae_all | genre_agree |
|---|---:|---:|---:|---:|---:|---:|
| beam_v1 (final) | 0.952 / 0.951 | 0.909 / 0.909 | 99.8 / 99.8 | 1.81 / 1.80 | 0.092 / 0.106 | 0.199 / 0.217 |
| greedy_v0 | 0.946 / 0.946 | 0.901 / 0.902 | 99.7 / 99.7 | 1.85 / 1.84 | 0.106 / 0.120 | 0.173 / 0.196 |
| beam tracks, random order | 0.817 / 0.815 | 0.613 / 0.612 | 66.0 / 65.4 | 4.29 / 4.28 | 0.169 / 0.176 | 0.199 / 0.217 |
| beam tracks, energy-sorted | 0.870 / 0.864 | 0.692 / 0.684 | 76.4 / 75.0 | 3.69 / 3.73 | 0.090 / 0.104 | 0.199 / 0.217 |
