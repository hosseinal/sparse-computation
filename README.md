# Compressed Tensor Algebra

**Build:** `git submodule update --init --recursive` then `./build_local.sh 4 32`

**Run (large b_col):** Put `mat_list.txt` (one `.mtx` name per line) and your `.mtx` files in a folder. Generate `.cfmtx` with `format_generator.py` (see `run_local_spmm_large_b_col.sh`). Then:

`./run_local_spmm_large_b_col.sh /path/to/matrices 32`

**Run (small b_col):** `./run_local_spmm_small_b_col.sh /path/to/matrices 4`

Needs: CUDA, CMake, Python 3 (`numpy`, `scipy`).
