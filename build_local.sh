#!/bin/bash
# Example: ./build_local.sh 4 32

cd "$(dirname "$0")"

kernel_m=${1:-4}
b_col=${2:-32}
kernel_n=2
b_unroll_factor=1
if [ "$b_col" -gt 32 ]; then
  warp_num=2
else
  warp_num=1
fi

python3 -m venv venv
source venv/bin/activate
pip install -q -r requirements.txt

python3 code_gen/main_spmm_small_b_col.py example_cuda/spmm_small_b_col_demo_gpu_utils.h "$kernel_m" 1 "$b_col" --pattern_dictionary false
python3 code_gen/main_spmm.py example_cuda/spmm_demo_gpu_utils.h format "$kernel_m" "$kernel_n" "$b_unroll_factor" "$warp_num" i-j --pattern_dictionary false
python3 code_gen/main_spmv.py example_cuda/spmv_demo_gpu_utils.h "$kernel_m" 1 --pattern_dictionary false

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
