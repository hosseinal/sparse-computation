# Compressed Tensor Algebra

**Build:** copy this folder anywhere (zip/USB is fine). From inside it: `./build_local.sh 4 32`

The first run downloads two libraries into `benchmark/` and `aggregation/` (needs `git` on the machine). Your project does not need to be on GitHub.

**Run (large b_col):** `./run_local_spmm_large_b_col.sh /path/to/matrices 32`

**Run (small b_col):** `./run_local_spmm_small_b_col.sh /path/to/matrices 4`

Needs: CUDA, CMake, Python 3, `git` (once, for dependencies).
