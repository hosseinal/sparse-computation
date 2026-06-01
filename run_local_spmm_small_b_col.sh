#!/bin/bash
# Example: ./run_local_spmm_small_b_col.sh data/ash85.mtx 4

cd "$(dirname "$0")"
source venv/bin/activate

matrix_path=$1
b_col=$2

matrix_dir=$(dirname "$matrix_path")
matrix_name=$(basename "$matrix_path" .mtx)
compressed_matrix_path="${matrix_dir}/${matrix_name}-4-1.cfmtx"

python3 codeGen/ss_format/format_generator.py "$matrix_path" 4 1 --pattern_dictionary false

build/example_cuda/spmm_small_b_col_demo_gpu \
  -m "$compressed_matrix_path" \
  --header --baseline=true \
  -n SPMM -s CSR -t "$b_col" \
  --b_matrix_columns="$b_col" \
  --number_of_warps=1
