# Evaluation v2: genre agreement by seed label source

Same per-seed results as eval_v2; seeds split by how their own genre label was obtained. Candidates' labels are unchanged (both sources). 1000 bootstrap resamples per group, 95% CIs.

### labeled pool, ID-matched seeds (id or id+name): 4,142 seeds

| method | genre agree (micro) | genre agree (macro) | seeds w/ label |
|---|---:|---:|---:|
| (a) transition score only (v0.1) | 0.067 [0.064, 0.070] | 0.051 [0.048, 0.055] | 1.00 |
| (b) audio similarity only | 0.106 [0.102, 0.111] | 0.077 [0.074, 0.082] | 1.00 |
| (c) audio + style relevance | 0.170 [0.163, 0.176] | 0.130 [0.126, 0.138] | 1.00 |
| (d) alpha*relevance + beta*transition, alpha=0.5 | 0.125 [0.120, 0.130] | 0.096 [0.093, 0.103] | 1.00 |
| (r) random (artist-capped) | 0.021 [0.019, 0.022] | 0.019 [0.017, 0.021] | 1.00 |

### labeled pool, name-only seeds: 3,108 seeds

| method | genre agree (micro) | genre agree (macro) | seeds w/ label |
|---|---:|---:|---:|
| (a) transition score only (v0.1) | 0.055 [0.052, 0.058] | 0.054 [0.050, 0.058] | 1.00 |
| (b) audio similarity only | 0.085 [0.081, 0.089] | 0.081 [0.075, 0.088] | 1.00 |
| (c) audio + style relevance | 0.157 [0.151, 0.165] | 0.148 [0.140, 0.159] | 1.00 |
| (d) alpha*relevance + beta*transition, alpha=0.5 | 0.110 [0.105, 0.115] | 0.110 [0.103, 0.117] | 1.00 |
| (r) random (artist-capped) | 0.020 [0.019, 0.022] | 0.018 [0.016, 0.020] | 1.00 |

### full pool, ID-matched seeds (id or id+name): 4,142 seeds

| method | genre agree (micro) | genre agree (macro) | seeds w/ label |
|---|---:|---:|---:|
| (a) transition score only (v0.1) | 0.131 [0.098, 0.168] | 0.130 [0.086, 0.180] | 0.09 |
| (b) audio similarity only | 0.166 [0.131, 0.200] | 0.107 [0.084, 0.140] | 0.11 |
| (c) audio + style relevance | 0.320 [0.291, 0.350] | 0.241 [0.218, 0.278] | 0.22 |
| (d) alpha*relevance + beta*transition, alpha=0.5 | 0.256 [0.230, 0.286] | 0.224 [0.195, 0.262] | 0.21 |
| (r) random (artist-capped) | 0.028 [0.008, 0.051] | 0.031 [0.008, 0.059] | 0.06 |

### full pool, name-only seeds: 3,108 seeds

| method | genre agree (micro) | genre agree (macro) | seeds w/ label |
|---|---:|---:|---:|
| (a) transition score only (v0.1) | 0.085 [0.052, 0.118] | 0.094 [0.054, 0.131] | 0.08 |
| (b) audio similarity only | 0.098 [0.066, 0.133] | 0.125 [0.075, 0.165] | 0.09 |
| (c) audio + style relevance | 0.324 [0.292, 0.357] | 0.298 [0.259, 0.342] | 0.25 |
| (d) alpha*relevance + beta*transition, alpha=0.5 | 0.273 [0.238, 0.308] | 0.305 [0.257, 0.347] | 0.21 |
| (r) random (artist-capped) | 0.011 [0.000, 0.029] | 0.019 [0.000, 0.049] | 0.06 |
