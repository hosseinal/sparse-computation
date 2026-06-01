#!/bin/bash
set -e

# Usage: ./build_local.sh [kernel_m] [b_col]
# Example: ./build_local.sh 4 32
m=${1:-4}
b_col=${2:-32}

n=8
bcol_unroll_factor=8
warp_num=2
spmv_warps=1
small_b_col_warps=1

BENCHMARK_URL="https://github.com/SwiftWare-Lab/benchmark.git"
AGGREGATION_URL="https://github.com/sympiler/aggregation.git"

fetch_deps() {
  if [ ! -f benchmark/CMakeLists.txt ]; then
    echo "Fetching benchmark/ ..."
    rm -rf benchmark
    git clone --depth 1 "${BENCHMARK_URL}" benchmark
  fi
  if [ ! -f aggregation/CMakeLists.txt ]; then
    echo "Fetching aggregation/ ..."
    rm -rf aggregation
    git clone --depth 1 "${AGGREGATION_URL}" aggregation
  fi
}

if [ -d .git ]; then
  git submodule update --init --recursive 2>/dev/null || fetch_deps
else
  fetch_deps
fi

if [ ! -f benchmark/CMakeLists.txt ] || [ ! -f aggregation/CMakeLists.txt ]; then
  echo "Error: need benchmark/ and aggregation/ (CMakeLists.txt in each)."
  echo "Install git and re-run this script, or copy those folders in by hand."
  exit 1
fi

python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

mkdir -p build
cd build

python3 ../codeGen/main_spmm_small_b_col.py \
  ../example_cuda/spmm_small_b_col_demo_gpu_utils.h \
  "${m}" "${small_b_col_warps}" "${b_col}" \
  --pattern_dictionary false

python3 ../codeGen/main_spmm.py \
  ../example_cuda/spmm_demo_gpu_utils.h format \
  "${m}" "${n}" "${bcol_unroll_factor}" "${warp_num}" i-j \
  --pattern_dictionary false

python3 ../codeGen/main_spmv.py \
  ../example_cuda/spmv_demo_gpu_utils.h \
  "${m}" "${spmv_warps}" \
  --pattern_dictionary false

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j "$(nproc)"
