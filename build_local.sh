#!/bin/bash
# Build CUDA demos. Usage: ./build_local.sh [m] [b_col]   (defaults: 4, 32)
set -e
cd "$(dirname "$0")"

M=${1:-4}
B_COL=${2:-32}

echo "Python venv ..."
python3 -m venv venv
source venv/bin/activate
pip install -q -r requirements.txt

echo "Codegen: spmm_small_b_col ..."
python3 codeGen/main_spmm_small_b_col.py example_cuda/spmm_small_b_col_demo_gpu_utils.h "$M" 1 "$B_COL" --pattern_dictionary false

echo "Codegen: spmm (large b_col) ..."
python3 codeGen/main_spmm.py example_cuda/spmm_demo_gpu_utils.h format "$M" 8 8 2 i-j --pattern_dictionary false

echo "Codegen: spmv ..."
python3 codeGen/main_spmv.py example_cuda/spmv_demo_gpu_utils.h "$M" 1 --pattern_dictionary false

echo "CMake configure ..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

echo "Compile ..."
cmake --build build -j"$(nproc)"

echo "Done. Binaries in build/example_cuda/"
