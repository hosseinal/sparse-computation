#!/bin/bash

basepath=$1
b_col=$2

m=4
n=1
warp_number=1

MATLIST=${basepath}/mat_list.txt
output_file=spmm_small_bcol_${b_col}.csv

mkdir build
cd build

while read line; do
  python3 ../codeGen/ss_format/format_generator.py ${basepath}/${line} ${m} ${n} --pattern_dictionary false
done < ${MATLIST}

python3 ../codeGen/main_spmm_small_b_col.py ../example_cuda/spmm_small_b_col_demo_gpu_utils.h ${m} ${warp_number} ${b_col} --pattern_dictionary false
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 40

header=1
BASELINE=true
while read line; do
  new_filename="${line%.mtx}-${m}-${n}.cfmtx"
  if [ $header -eq 1 ]; then
    ./example_cuda/spmm_small_b_col_demo_gpu -m ${basepath}/${new_filename} --header --baseline=${BASELINE} -n SPMM -s CSR -t ${b_col} --b_matrix_columns=${b_col} --number_of_warps=${warp_number} > ${output_file}
    header=0
  else
    ./example_cuda/spmm_small_b_col_demo_gpu -m ${basepath}/${new_filename} --baseline=${BASELINE} -n SPMM -s CSR -t ${b_col} --b_matrix_columns=${b_col} --number_of_warps=${warp_number} >> ${output_file}
  fi
done < ${MATLIST}
