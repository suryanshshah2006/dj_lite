# Step 5 evaluation: beam-search arc generator

500 genre-labeled seeds (random_seed 7) x 4 arc presets, N = 12; candidates = full catalogue; pool = top 2000 by relevance to the seed (FAISS top 2000); alpha = 0.5, gamma = 0.5, B = 20, max 1 track per artist, no |dBPM| cap; 1000 bootstrap resamples over seeds, 95% CIs (each seed = mean over the 4 arcs)

Sanity check: beam(B=1) == greedy_v0 on 2,000 of 2,000 playlists.

### Main: generators and baselines (pooled over arcs)

| method | transition mean | transition min | % harmonic compat | |dBPM| mean | |dBPM| max | arc MAE | genre agree | unique artists | objective | runtime ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| beam_v1 (B=20) | 0.938 [0.936, 0.940] | 0.853 [0.847, 0.859] | 98.5 [98.3, 98.7] | 0.96 [0.92, 1.01] | 3.35 [3.03, 3.66] | 0.067 [0.061, 0.072] | 0.179 [0.132, 0.224] | 12.00 [12.00, 12.00] | 15.299 [15.262, 15.335] | 1400 [1375, 1430] |
| greedy_v0 | 0.930 [0.928, 0.932] | 0.855 [0.850, 0.861] | 98.0 [97.6, 98.3] | 0.99 [0.95, 1.03] | 3.02 [2.79, 3.26] | 0.076 [0.070, 0.082] | 0.183 [0.136, 0.229] | 12.00 [12.00, 12.00] | 15.184 [15.145, 15.223] | 1392 [1367, 1422] |
| beam tracks, random order | 0.798 [0.793, 0.804] | 0.578 [0.569, 0.588] | 65.7 [64.2, 67.2] | 2.46 [2.30, 2.61] | 6.22 [5.82, 6.62] | 0.159 [0.155, 0.162] | 0.179 [0.132, 0.224] | 12.00 [12.00, 12.00] | 14.026 [13.976, 14.079] | 1400 [1375, 1430] |
| beam tracks, energy-sorted | 0.851 [0.846, 0.855] | 0.643 [0.633, 0.652] | 75.5 [74.4, 76.6] | 2.01 [1.90, 2.15] | 5.48 [5.11, 5.85] | 0.065 [0.059, 0.070] | 0.179 [0.132, 0.224] | 12.00 [12.00, 12.00] | 14.831 [14.781, 14.878] | 1400 [1375, 1430] |

### Per arc: warmup_peak_cooldown

| method | transition mean | transition min | % harmonic compat | |dBPM| mean | |dBPM| max | arc MAE |
|---|---:|---:|---:|---:|---:|---:|
| beam_v1 (B=20) | 0.938 [0.935, 0.940] | 0.854 [0.846, 0.862] | 98.6 [98.2, 98.9] | 0.91 [0.86, 0.96] | 3.19 [2.81, 3.61] | 0.059 [0.053, 0.065] |
| greedy_v0 | 0.930 [0.927, 0.932] | 0.857 [0.851, 0.863] | 98.0 [97.6, 98.4] | 0.96 [0.92, 1.02] | 2.97 [2.68, 3.34] | 0.067 [0.060, 0.073] |
| beam tracks, random order | 0.812 [0.805, 0.820] | 0.606 [0.595, 0.618] | 69.3 [67.5, 71.4] | 2.26 [2.07, 2.48] | 5.56 [5.11, 6.05] | 0.183 [0.178, 0.187] |
| beam tracks, energy-sorted | 0.853 [0.846, 0.860] | 0.638 [0.624, 0.652] | 77.5 [75.8, 79.3] | 2.05 [1.85, 2.27] | 5.35 [4.88, 5.88] | 0.057 [0.050, 0.062] |

### Per arc: steady

| method | transition mean | transition min | % harmonic compat | |dBPM| mean | |dBPM| max | arc MAE |
|---|---:|---:|---:|---:|---:|---:|
| beam_v1 (B=20) | 0.943 [0.941, 0.945] | 0.878 [0.872, 0.884] | 98.9 [98.6, 99.2] | 0.99 [0.92, 1.07] | 3.43 [2.94, 3.98] | 0.052 [0.046, 0.059] |
| greedy_v0 | 0.936 [0.934, 0.939] | 0.874 [0.868, 0.880] | 98.3 [97.8, 98.7] | 0.96 [0.91, 1.02] | 2.88 [2.57, 3.24] | 0.061 [0.053, 0.068] |
| beam tracks, random order | 0.793 [0.786, 0.800] | 0.572 [0.561, 0.584] | 61.6 [59.7, 63.4] | 2.81 [2.59, 3.06] | 6.90 [6.30, 7.52] | 0.054 [0.047, 0.061] |
| beam tracks, energy-sorted | 0.792 [0.785, 0.799] | 0.567 [0.556, 0.578] | 60.9 [58.9, 62.8] | 2.80 [2.59, 3.02] | 6.89 [6.29, 7.51] | 0.050 [0.043, 0.057] |

### Per arc: slow_build

| method | transition mean | transition min | % harmonic compat | |dBPM| mean | |dBPM| max | arc MAE |
|---|---:|---:|---:|---:|---:|---:|
| beam_v1 (B=20) | 0.936 [0.934, 0.938] | 0.832 [0.823, 0.840] | 98.3 [97.9, 98.6] | 0.97 [0.92, 1.03] | 3.61 [3.20, 4.07] | 0.071 [0.065, 0.077] |
| greedy_v0 | 0.928 [0.926, 0.931] | 0.848 [0.841, 0.855] | 98.1 [97.7, 98.4] | 1.00 [0.95, 1.05] | 2.99 [2.73, 3.29] | 0.081 [0.076, 0.087] |
| beam tracks, random order | 0.804 [0.797, 0.810] | 0.574 [0.562, 0.586] | 68.7 [66.8, 70.5] | 2.30 [2.12, 2.49] | 6.26 [5.71, 6.85] | 0.220 [0.216, 0.224] |
| beam tracks, energy-sorted | 0.888 [0.883, 0.893] | 0.684 [0.671, 0.697] | 83.9 [82.4, 85.3] | 1.51 [1.39, 1.63] | 4.86 [4.39, 5.35] | 0.070 [0.064, 0.076] |

### Per arc: sunset

| method | transition mean | transition min | % harmonic compat | |dBPM| mean | |dBPM| max | arc MAE |
|---|---:|---:|---:|---:|---:|---:|
| beam_v1 (B=20) | 0.935 [0.933, 0.937] | 0.849 [0.842, 0.856] | 98.2 [97.9, 98.6] | 0.99 [0.94, 1.03] | 3.18 [2.88, 3.49] | 0.085 [0.078, 0.092] |
| greedy_v0 | 0.925 [0.922, 0.928] | 0.841 [0.835, 0.848] | 97.5 [97.0, 97.9] | 1.05 [1.01, 1.11] | 3.25 [2.92, 3.61] | 0.097 [0.090, 0.104] |
| beam tracks, random order | 0.785 [0.778, 0.792] | 0.561 [0.549, 0.573] | 63.1 [61.0, 65.1] | 2.46 [2.29, 2.62] | 6.18 [5.73, 6.64] | 0.179 [0.175, 0.183] |
| beam tracks, energy-sorted | 0.870 [0.864, 0.876] | 0.682 [0.669, 0.694] | 79.8 [78.0, 81.4] | 1.70 [1.60, 1.82] | 4.82 [4.45, 5.21] | 0.083 [0.076, 0.090] |

### Ablation: beam width (alpha 0.5, gamma 0.5)

| method | transition mean | transition min | % harmonic compat | |dBPM| mean | |dBPM| max | arc MAE | genre agree | unique artists | objective | runtime ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B=20 | 0.938 [0.936, 0.940] | 0.853 [0.847, 0.859] | 98.5 [98.3, 98.7] | 0.96 [0.92, 1.01] | 3.35 [3.03, 3.66] | 0.067 [0.061, 0.072] | 0.179 [0.132, 0.224] | 12.00 [12.00, 12.00] | 15.299 [15.262, 15.335] | 1400 [1375, 1430] |
| B=1 | 0.930 [0.928, 0.932] | 0.855 [0.850, 0.861] | 98.0 [97.6, 98.3] | 0.99 [0.95, 1.03] | 3.02 [2.79, 3.26] | 0.076 [0.070, 0.082] | 0.183 [0.136, 0.229] | 12.00 [12.00, 12.00] | 15.184 [15.145, 15.223] | 1393 [1367, 1423] |
| B=5 | 0.937 [0.934, 0.939] | 0.860 [0.855, 0.866] | 98.6 [98.3, 98.8] | 0.94 [0.90, 0.98] | 3.02 [2.79, 3.28] | 0.070 [0.064, 0.075] | 0.172 [0.127, 0.215] | 12.00 [12.00, 12.00] | 15.269 [15.232, 15.306] | 1395 [1369, 1425] |
| B=50 | 0.938 [0.936, 0.940] | 0.848 [0.841, 0.854] | 98.5 [98.3, 98.7] | 0.98 [0.93, 1.03] | 3.54 [3.21, 3.87] | 0.065 [0.059, 0.071] | 0.202 [0.155, 0.249] | 12.00 [12.00, 12.00] | 15.311 [15.275, 15.347] | 1412 [1387, 1443] |

### Ablation: gamma (B = 20, alpha 0.5)

| method | transition mean | transition min | % harmonic compat | |dBPM| mean | |dBPM| max | arc MAE | genre agree | unique artists | objective | runtime ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gamma=0.5 | 0.938 [0.936, 0.940] | 0.853 [0.847, 0.859] | 98.5 [98.3, 98.7] | 0.96 [0.92, 1.01] | 3.35 [3.03, 3.66] | 0.067 [0.061, 0.072] | 0.179 [0.132, 0.224] | 12.00 [12.00, 12.00] | 15.299 [15.262, 15.335] | 1400 [1375, 1430] |
| gamma=0 | 0.962 [0.961, 0.963] | 0.924 [0.921, 0.927] | 99.9 [99.9, 100.0] | 0.60 [0.58, 0.63] | 1.67 [1.61, 1.74] | 0.248 [0.240, 0.256] | 0.211 [0.140, 0.283] | 12.00 [12.00, 12.00] | 14.534 [14.484, 14.581] | 1400 [1375, 1430] |
| gamma=0.25 | 0.949 [0.948, 0.951] | 0.897 [0.893, 0.900] | 99.6 [99.5, 99.7] | 0.74 [0.71, 0.76] | 2.07 [1.99, 2.20] | 0.094 [0.088, 0.101] | 0.178 [0.132, 0.225] | 12.00 [12.00, 12.00] | 15.245 [15.206, 15.284] | 1400 [1375, 1430] |
| gamma=1 | 0.922 [0.919, 0.925] | 0.790 [0.782, 0.799] | 96.4 [96.0, 96.8] | 1.33 [1.24, 1.41] | 5.39 [4.89, 5.86] | 0.050 [0.045, 0.055] | 0.176 [0.131, 0.218] | 12.00 [12.00, 12.00] | 15.274 [15.237, 15.311] | 1400 [1375, 1430] |

### Ablation: alpha (B = 20, gamma 0.5)

| method | transition mean | transition min | % harmonic compat | |dBPM| mean | |dBPM| max | arc MAE | genre agree | unique artists | objective | runtime ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| alpha=0.5 | 0.938 [0.936, 0.940] | 0.853 [0.847, 0.859] | 98.5 [98.3, 98.7] | 0.96 [0.92, 1.01] | 3.35 [3.03, 3.66] | 0.067 [0.061, 0.072] | 0.179 [0.132, 0.224] | 12.00 [12.00, 12.00] | 15.299 [15.262, 15.335] | 1400 [1375, 1430] |
| alpha=0.75 | 0.905 [0.902, 0.909] | 0.745 [0.736, 0.755] | 93.8 [93.3, 94.2] | 1.77 [1.66, 1.88] | 7.59 [6.93, 8.23] | 0.059 [0.053, 0.064] | 0.215 [0.160, 0.264] | 12.00 [12.00, 12.00] | 15.234 [15.196, 15.273] | 1400 [1375, 1430] |

### Extra ablation: |dBPM| cap per transition (B = 20, alpha 0.5, gamma 0.5)

| method | transition mean | transition min | % harmonic compat | |dBPM| mean | |dBPM| max | arc MAE | genre agree | unique artists | objective | runtime ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no cap | 0.938 [0.936, 0.940] | 0.853 [0.847, 0.859] | 98.5 [98.3, 98.7] | 0.96 [0.92, 1.01] | 3.35 [3.03, 3.66] | 0.067 [0.061, 0.072] | 0.179 [0.132, 0.224] | 12.00 [12.00, 12.00] | 15.299 [15.262, 15.335] | 1400 [1375, 1430] |
| max |dBPM| 6 | 0.939 [0.937, 0.941] | 0.861 [0.856, 0.866] | 98.4 [98.2, 98.7] | 0.87 [0.84, 0.89] | 2.43 [2.37, 2.50] | 0.068 [0.063, 0.074] | 0.171 [0.126, 0.216] | 12.00 [12.00, 12.00] | 15.294 [15.257, 15.331] | 1403 [1378, 1433] |
| max |dBPM| 3 | 0.940 [0.938, 0.942] | 0.866 [0.861, 0.871] | 98.2 [97.9, 98.5] | 0.81 [0.79, 0.83] | 2.06 [2.02, 2.09] | 0.071 [0.065, 0.076] | 0.168 [0.121, 0.214] | 12.00 [12.00, 12.00] | 15.286 [15.249, 15.324] | 1402 [1377, 1432] |

Transitions: 11 per playlist (seed -> 2 -> ... -> 12). % harmonic compat = share with Camelot score >= 0.9 (same key, +-1, relative; unknown keys count as incompatible). |dBPM| after octave alignment. arc MAE over positions 2..12 (position 1 is the seed). genre agree = share of labeled tracks in positions 2..12 whose genre set overlaps the seed's; only playlists with >=1 labeled track count (seeds w/ label). objective = the default objective (alpha 0.5, gamma 0.5) for every row. runtime = pool + transition matrix + search, single-threaded BLAS/OpenMP; for the reordered baselines it is the beam's runtime.

## Example playlists (seed: Duke Dumont - Worship)

![examples](figures/step5_examples.png)

**warmup peak cooldown** (1084 ms)

| # | track | artist | Camelot | BPM | energy | target | transition in |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | Worship | Duke Dumont | 10A | 126 | 0.78 | 0.35 |  |
| 2 | Boy | Juan Maclean's Peach Melba | 10A | 125 | 0.47 | 0.43 | 0.91 |
| 3 | It's Yours (feat. E-Man) - Frankie Feliciano's Reconstruction Mix | Jon Cutler | 10A | 125 | 0.51 | 0.52 | 0.98 |
| 4 | Read My Mind (Instrumental) | Metro Area | 10A | 125 | 0.61 | 0.61 | 0.96 |
| 5 | Larry's Exotic Blend | Zappe | 10A | 125 | 0.69 | 0.71 | 0.97 |
| 6 | Inside | Jovonn | 10A | 125 | 0.81 | 0.80 | 0.98 |
| 7 | Shinjuku Ramen | Fred Everything | 10A | 125 | 0.86 | 0.86 | 0.98 |
| 8 | Living In Brooklyn - Roach Motel Dub | Product Of Da Neighbourhood | 10A | 124 | 0.88 | 0.89 | 0.96 |
| 9 | My Offence (feat. Krystle Warren) - Brawther's Mean Dub | Hercules & Love Affair | 10A | 124 | 0.83 | 0.83 | 0.99 |
| 10 | Heartless Intentions (feat. Dr Moruti) | Black Motion | 11A | 123 | 0.74 | 0.72 | 0.94 |
| 11 | Cicadas - Maya Jane Coles Remix | Tom Middleton | 11A | 123 | 0.62 | 0.59 | 0.98 |
| 12 | Never Lies - Extended Mix | Sonny Fodera | 12A | 122 | 0.48 | 0.45 | 0.94 |

**steady** (1084 ms)

| # | track | artist | Camelot | BPM | energy | target | transition in |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | Worship | Duke Dumont | 10A | 126 | 0.78 | 0.70 |  |
| 2 | BOSS | Disclosure | 10A | 126 | 0.76 | 0.71 | 0.99 |
| 3 | Search Is Over | Spirit Catcher | 10A | 125 | 0.73 | 0.72 | 0.98 |
| 4 | Larry's Exotic Blend | Zappe | 10A | 125 | 0.69 | 0.71 | 0.98 |
| 5 | Higher - Benji Candelario Thump Mix | Benji Candelario | 10A | 124 | 0.69 | 0.70 | 0.98 |
| 6 | Right Thing (feat. Ben Westbeech) | Dennis Ferrer | 9A | 123 | 0.72 | 0.71 | 0.94 |
| 7 | Momma's Groove - Jimpster's Hip Replacement Mix, Mixed | Osunlade | 9B | 123 | 0.73 | 0.71 | 0.97 |
| 8 | Searching (feat. Melisa Whiskey) - Mixed | S-Man | 9B | 123 | 0.68 | 0.70 | 0.99 |
| 9 | Last Words - Dimitri from Paris Remix | Clubfeet | 9B | 122 | 0.70 | 0.71 | 0.98 |
| 10 | Flutter Vision | Shiny Objects | 9B | 121 | 0.72 | 0.72 | 0.98 |
| 11 | Sam - Jimmy V and S.pacini Mix | Jimmy V | 8B | 120 | 0.71 | 0.71 | 0.95 |
| 12 | Grant Land | M.ono | 8B | 120 | 0.71 | 0.70 | 0.98 |

**slow build** (1084 ms)

| # | track | artist | Camelot | BPM | energy | target | transition in |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | Worship | Duke Dumont | 10A | 126 | 0.78 | 0.25 |  |
| 2 | Black In My Soul - Tiger Stripes Main Remix | DJ Roland Clark | 10A | 127 | 0.46 | 0.30 | 0.92 |
| 3 | Yallah Habibe | CIRRUS | 10A | 126 | 0.41 | 0.36 | 0.98 |
| 4 | Boy | Juan Maclean's Peach Melba | 10A | 125 | 0.47 | 0.41 | 0.96 |
| 5 | Know What's Up | Ian Pooley | 10A | 125 | 0.48 | 0.47 | 0.97 |
| 6 | It's Yours (feat. E-Man) - Frankie Feliciano's Reconstruction Mix | Jon Cutler | 10A | 125 | 0.51 | 0.52 | 0.99 |
| 7 | Stomp The Ground - Mixed | Jovonn | 10A | 123 | 0.59 | 0.59 | 0.92 |
| 8 | Cicadas - Maya Jane Coles Remix | Tom Middleton | 11A | 123 | 0.62 | 0.66 | 0.96 |
| 9 | Heartless Intentions (feat. Dr Moruti) | Black Motion | 11A | 123 | 0.74 | 0.72 | 0.98 |
| 10 | A Thing Called J (feat. S L F) | Luca Cazal | 11B | 124 | 0.79 | 0.79 | 0.95 |
| 11 | Just No Good For Me - Mixed | The Deepshakerz | 10B | 123 | 0.82 | 0.85 | 0.95 |
| 12 | Up & Down | Charles Ramirez | 9B | 123 | 0.91 | 0.92 | 0.95 |

**sunset** (1084 ms)

| # | track | artist | Camelot | BPM | energy | target | transition in |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | Worship | Duke Dumont | 10A | 126 | 0.78 | 0.75 |  |
| 2 | Larry's Exotic Blend | Zappe | 10A | 125 | 0.69 | 0.71 | 0.96 |
| 3 | Deep Inside - Jesse Rose Play Prime Mix | Hardrive | 10A | 124 | 0.67 | 0.67 | 0.98 |
| 4 | Butterfly - Phil Asher's restless soul Vocal Mix | Osunlade | 10A | 123 | 0.63 | 0.63 | 0.98 |
| 5 | Cicadas - Maya Jane Coles Remix | Tom Middleton | 11A | 123 | 0.62 | 0.58 | 0.97 |
| 6 | Imali (feat. Nokwazi) | Black Motion | 11A | 123 | 0.56 | 0.53 | 0.99 |
| 7 | Never Lies - Extended Mix | Sonny Fodera | 12A | 122 | 0.48 | 0.48 | 0.94 |
| 8 | The Honeymoon - M.ono Remix | Kumquat | 1A | 122 | 0.43 | 0.43 | 0.95 |
| 9 | Can't Believe - Maxxi Soundsystem Rework | The Mekanism | 12A | 121 | 0.40 | 0.39 | 0.94 |
| 10 | Rendez-Vous (feat. Ingrid Hakanson) - Loui & Scibi Remix | Lumoon & Rob!n | 12B | 121 | 0.42 | 0.34 | 0.96 |
| 11 | The Velvet Slide | Shiny Objects | 11B | 122 | 0.43 | 0.29 | 0.95 |
| 12 | Healer | Chromatics | 11B | 120 | 0.39 | 0.25 | 0.94 |
