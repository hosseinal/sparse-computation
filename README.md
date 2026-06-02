# Enumerate and Sparse Coarsen

**Run** (builds codegen + binary, then benchmarks one matrix):

The code run base on size of B in **AxB=C**

Two type of size we consider

1 - small which number of column of B are 1 2 4

2- large which number of column of B are 32 64 128

Note that other sizes are not supported

**Build**

Optional: `./build_local.sh` to build all demos at once.



**To run**

```bash
./run_local_spmm_large_b_col.sh ./matrices/body_decoder_layer_1_encdec_attention_multihead_attention_k_fully_connected.mtx 32

./run_local_spmm_small_b_col.sh ./matrices/body_decoder_layer_1_encdec_attention_multihead_attention_k_fully_connected.mtx 4
```

Input: path to `.mtx` file. Creates `ash85-4-1.cfmtx` beside it, then runs.

you can see some matrices in the **matrices** folder


