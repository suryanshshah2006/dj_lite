# Generator length check: greedy_v0 vs beam_v1 at N = 12 and N = 24

500 genre-labeled seeds (same as Steps 5 and 5.1) x 4 arcs; final defaults for both generators and both lengths: alpha 0.5, gamma 0.25, lambda 2.0 (tau 0.85), |dBPM| cap 3, energy bands 5, seed position best_fit, B 20. objective = the configured objective (incl. the weak-link penalty). 1000 bootstrap resamples over seeds, 95% CIs (each seed = mean over the 4 arcs); differences are paired per seed.

### Per generator and length

| method | transition mean | transition min | % harmonic compat | |dBPM| mean | |dBPM| max | arc MAE | arc MAE incl. seed | runtime ms | objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N=12, beam_v1 | 0.952 [0.950, 0.953] | 0.909 [0.907, 0.911] | 99.8 [99.7, 99.8] | 0.70 [0.68, 0.71] | 1.81 [1.77, 1.85] | 0.093 [0.087, 0.100] | 0.092 [0.086, 0.099] | 1264 [1257, 1272] | 12.786 [12.755, 12.815] |
| N=12, greedy_v0 | 0.946 [0.944, 0.947] | 0.901 [0.899, 0.904] | 99.7 [99.6, 99.8] | 0.74 [0.72, 0.76] | 1.85 [1.81, 1.89] | 0.108 [0.101, 0.115] | 0.106 [0.099, 0.112] | 1254 [1246, 1262] | 12.704 [12.672, 12.735] |
| N=24, beam_v1 | 0.953 [0.952, 0.955] | 0.900 [0.898, 0.903] | 99.8 [99.7, 99.9] | 0.70 [0.68, 0.72] | 2.10 [2.07, 2.14] | 0.095 [0.088, 0.101] | 0.094 [0.088, 0.100] | 1278 [1271, 1286] | 26.715 [26.646, 26.778] |
| N=24, greedy_v0 | 0.947 [0.945, 0.948] | 0.891 [0.889, 0.893] | 99.7 [99.6, 99.8] | 0.73 [0.72, 0.75] | 2.14 [2.11, 2.17] | 0.110 [0.103, 0.117] | 0.109 [0.102, 0.115] | 1255 [1248, 1263] | 26.527 [26.456, 26.594] |

### Paired difference (beam - greedy)

| method | transition mean | transition min | % harmonic compat | |dBPM| mean | |dBPM| max | arc MAE | arc MAE incl. seed | runtime ms | objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N=12: beam - greedy | 0.006 [0.005, 0.007] | 0.008 [0.006, 0.009] | 0.1 [-0.0, 0.2] | -0.04 [-0.05, -0.03] | -0.04 [-0.07, -0.01] | -0.015 [-0.016, -0.013] | -0.013 [-0.015, -0.012] | 11 [10, 11] | 0.082 [0.078, 0.087] |
| N=24: beam - greedy | 0.007 [0.006, 0.007] | 0.009 [0.008, 0.010] | 0.1 [0.1, 0.2] | -0.03 [-0.04, -0.02] | -0.04 [-0.07, -0.02] | -0.015 [-0.016, -0.014] | -0.015 [-0.016, -0.014] | 23 [23, 23] | 0.187 [0.179, 0.196] |
