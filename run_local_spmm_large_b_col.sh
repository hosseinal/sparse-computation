#!/bin/bash
# Example: ./run_local_spmm_large_b_col.sh data/ash85.mtx 32

python3 -m venv venv
source venv/bin/activate
pip install -q -r requirements.txt

matrix_path=$1
b_col=$2

kernel_m=4
kernel_n=1
b_unroll_factor=1
if [ "$b_col" -gt 32 ]; then
  warp_num=2
else
  warp_num=1
fi

matrix_dir=$(dirname "$matrix_path")
matrix_name=$(basename "$matrix_path" .mtx)
compressed_matrix_path="${matrix_dir}/${matrix_name}-${kernel_m}-${kernel_n}.cfmtx"

python3 code_gen/ss_format/format_generator.py "$matrix_path" "$kernel_m" "$kernel_n" --pattern_dictionary false

python3 code_gen/main_spmm.py example_cuda/spmm_demo_gpu_utils.h format "$kernel_m" "$kernel_n" "$b_unroll_factor" "$warp_num" i-j --pattern_dictionary false

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --target spmm_demo_gpu

build/example_cuda/spmm_demo_gpu \
  -m "$compressed_matrix_path" \
  --header --baseline=true \
  -n SPMM -s CSR -t "$b_col" \
  --b_matrix_columns="$b_col" \
  --number_of_warps="$warp_num" \
  --b_unroll_factor="$b_unroll_factor"
