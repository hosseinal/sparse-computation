#!/bin/bash
# Build CUDA demos. Usage: ./build_local.sh [m] [b_col]   (defaults: 4, 32)

M=${1:-4}
B_COL=${2:-32}

source venv/bin/activate
pip install -q -r requirements.txt

python3 codeGen/main_spmm_small_b_col.py example_cuda/spmm_small_b_col_demo_gpu_utils.h "$M" 1 "$B_COL" --pattern_dictionary false
python3 codeGen/main_spmm.py example_cuda/spmm_demo_gpu_utils.h format "$M" 8 8 2 i-j --pattern_dictionary false

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

echo "Done. Binaries in build/example_cuda/"
