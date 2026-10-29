"""DJ-Lite web backend.

Thread limits are set here, before numpy/scipy/faiss load. With the defaults (one thread
per core) the three bundled OpenBLAS copies reserve ~2.2 GB of per-thread buffers at import
and faiss's OpenMP workers commit ~128 MB each on the first search; single-query flat search
is memory-bound, so one thread is only ~7 ms slower (31 vs 24 ms on 1.09M tracks).
Override with the environment variables if needed.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
