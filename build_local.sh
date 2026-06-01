#!/bin/bash
# Build CUDA demos. Usage: ./build_local.sh [m] [b_col]   (defaults: 4, 32)
set -e
cd "$(dirname "$0")"

M=${1:-4}
B_COL=${2:-32}

if [ ! -d venv ]; then
  python3 -m venv venv
  ./venv/bin/pip install -q -r requirements.txt
fi
PY=./venv/bin/python3

$PY codeGen/main_spmm_small_b_col.py example_cuda/spmm_small_b_col_demo_gpu_utils.h "$M" 1 "$B_COL" --pattern_dictionary false
$PY codeGen/main_spmm.py example_cuda/spmm_demo_gpu_utils.h format "$M" 8 8 2 i-j --pattern_dictionary false

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

echo "Done. Binaries in build/example_cuda/"
