# Compressed Tensor Algebra

**Run** (builds codegen + binary, then benchmarks one matrix):

```bash
./run_local_spmm_large_b_col.sh data/ash85.mtx 32
./run_local_spmm_small_b_col.sh data/ash85.mtx 4
```

Input: path to `.mtx` file. Creates `ash85-4-1.cfmtx` beside it, then runs.

Optional: `./build_local.sh` to build all demos at once.
