# Compressed Tensor Algebra

**Build:** `./build_local.sh` or `./build_local.sh 4 32`

**Run (large b_col):** `./run_local_spmm_large_b_col.sh /path/to/matrices 32`

**Run (small b_col):** `./run_local_spmm_small_b_col.sh /path/to/matrices 4`

Needs: CUDA, CMake, Python 3.

`benchmark/` and `aggregation/` are normal folders in this repo (not submodules). Copy the whole project to build elsewhere.
