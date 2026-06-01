# Compressed Tensor Algebra

**Build:** `./build_local.sh`

**Run** (your matrix file + b_col width):

```bash
./run_local_spmm_large_b_col.sh data/ash85.mtx 32
./run_local_spmm_small_b_col.sh data/ash85.mtx 4
```

Creates `ash85-4-8.cfmtx` (or `ash85-4-1.cfmtx`) next to `ash85.mtx`, then runs the benchmark on that file.
