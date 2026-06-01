#!/bin/bash
# Example: ./run_local_spmm_large_b_col.sh data/ash85.mtx 32

source venv/bin/activate

matrix_name=$1
b_col=$2
kernel_m=4  
kernel_n=8
compressed_matrix_path="${matrix_name}-${kernel_m}-${kernel_n}.cfmtx"

python3 codeGen/ss_format/format_generator.py "$matrix_name" "$kernel_m" "$kernel_n" --pattern_dictionary false

build/example_cuda/spmm_demo_gpu \
  -m "$compressed_matrix_path" \
  --header --baseline=true \
  -n SPMM -s CSR -t "$b_col" \
  --b_matrix_columns="$b_col" \
  --number_of_warps=2 \
  --b_unroll_factor=8
