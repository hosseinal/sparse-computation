#!/bin/bash

basepath=$1
b_col=$2

m=4
n=8
bcol_unroll_factor=8
warp_number=2

MATLIST=${basepath}/mat_list.txt
output_file=spmm_large_bcol_${b_col}.csv

mkdir build
cd build

while read line; do
  python3 ../codeGen/ss_format/format_generator.py ${basepath}/${line} ${m} ${n} --pattern_dictionary false --save_collection true
done < ${MATLIST}

python3 ../codeGen/main_spmm.py ../example_cuda/spmm_demo_gpu_utils.h format ${m} ${n} ${bcol_unroll_factor} ${warp_number} i-j --pattern_dictionary false
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 40

header=1
BASELINE=true
while read line; do
  new_filename="${line%.mtx}-${m}-${n}.cfmtx"
  if [ $header -eq 1 ]; then
    ./example_cuda/spmm_demo_gpu -m ${basepath}/${new_filename} --header --baseline=${BASELINE} -n SPMM -s CSR -t ${b_col} --b_matrix_columns=${b_col} --number_of_warps=${warp_number} --b_unroll_factor=${bcol_unroll_factor} > ${output_file}
    header=0
  else
    ./example_cuda/spmm_demo_gpu -m ${basepath}/${new_filename} --baseline=${BASELINE} -n SPMM -s CSR -t ${b_col} --b_matrix_columns=${b_col} --number_of_warps=${warp_number} --b_unroll_factor=${bcol_unroll_factor} >> ${output_file}
  fi
done < ${MATLIST}
