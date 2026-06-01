#!/bin/bash
# Example: ./build_local.sh 4 32

cd "$(dirname "$0")"

kernel_m=${1:-4}
b_col=${2:-32}

python3 -m venv venv
source venv/bin/activate
pip install -q -r requirements.txt

python3 codeGen/main_spmm_small_b_col.py example_cuda/spmm_small_b_col_demo_gpu_utils.h "$kernel_m" 1 "$b_col" --pattern_dictionary false
python3 codeGen/main_spmm.py example_cuda/spmm_demo_gpu_utils.h format "$kernel_m" 8 8 2 i-j --pattern_dictionary false
python3 codeGen/main_spmv.py example_cuda/spmv_demo_gpu_utils.h "$kernel_m" 1 --pattern_dictionary false

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
