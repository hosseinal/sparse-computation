from pattern_utiles.pattern_dictionary import PatternTypeDictionary


def kernel_code_generator(pattern_type_dictionary: PatternTypeDictionary):
    number_of_patterns = len(pattern_type_dictionary.pattern_bin_to_id)

    file = ""
    file += "__global__ void SpmvComplexPatternDecomposeByRowSeperated( \n"
    file += "   swiftware::compression::MixedFormatGPU<float> *mf, float *dX, float *dY) {\n"
    file += f"  int j = blockIdx.x / {number_of_patterns};\n"
    file += f"  int i = blockIdx.x % {number_of_patterns};\n\n"

    file += "  switch (i) {\n"
    for i, pattern in pattern_type_dictionary.id_to_patten_bin.items():
        nnz_number = pattern_type_dictionary.pattern_id_to_number_of_nnz(i)
        file += f"    case {i}:\n"
        file += (f"      PatternWith{nnz_number}NNZByRowPanelFull("
                 f"mf->cf[{i}]->RPP, mf->cf[{i}]->NPP, mf->cf[{i}]->NNZ, mf->cf[{i}]->CB, dX, dY, j ")
        for x in range(len(pattern)):
            if '1' == pattern[x]:
                file += f", {x}"
        file += ");\n"
        file += "      break;\n"

    file += "  }\n"
    file += "}\n"

    file += "\n"
    file += "__global__ void SpmvComplexPatternDecomposeByRowSeperatedHalf( \n"
    file += "   swiftware::compression::MixedFormatGPU<__half> *mf, __half *dX, __half *dY) {\n"
    file += f"  int j = blockIdx.x / {number_of_patterns};\n"
    file += f"  int i = blockIdx.x % {number_of_patterns};\n\n"

    file += "  switch (i) {\n"
    for i, pattern in pattern_type_dictionary.id_to_patten_bin.items():
        nnz_number = pattern_type_dictionary.pattern_id_to_number_of_nnz(i)
        file += f"    case {i}:\n"
        file += (f"      PatternWith{nnz_number}NNZByRowPanelFullHALF("
                 f"mf->cf[{i}]->RPP, mf->cf[{i}]->NPP, mf->cf[{i}]->NNZ, mf->cf[{i}]->CB, dX, dY, j ")
        for x in range(len(pattern)):
            if '1' == pattern[x]:
                file += f", {x}"
        file += ");\n"
        file += "      break;\n"

    file += "  }\n"
    file += "}\n"

    return file

def class_code_generator(m, number_of_warp, pattern_type_dictionary: PatternTypeDictionary):
    if number_of_warp <= 0:
        number_of_warp = 1

    number_of_patterns = len(pattern_type_dictionary.pattern_bin_to_id)

    file = ""
    file += "class SpMVMixePatternDecomposeByRowSeperated : public SpMMSerialFloat {\n"
    file += "\n"
    file += "protected:\n"
    file += "  float *x;\n"
    file += "\n"
    file += "  float *dX, *dY, *dY1, *dY2;\n"
    file += "  swiftware::compression::MixedFormatGPU<float> *mf;\n"
    file += "\n"
    file += "  float alpha = 1.0f;\n"
    file += "\n"
    file += "  TensorInputsMixed *InT;\n"
    file += "\n"
    file += "  bool verify(double &Error) override {\n"
    file += "    bool retValue = true;\n"
    file += "\n"
    file += "    double infNorm = 0;\n"
    file += "    for (int i = 0; i < InTensor->M * InTensor->Z - 4; ++i) {\n"
    file += "      if (std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]) > infNorm) {\n"
    file += "        infNorm = std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]);\n"
    file += "      }\n"
    file += "    }\n"
    file += "    Error = (double)infNorm;\n"
    file += "    if (infNorm > InTensor->Threshold) {\n"
    file += "      retValue = false;\n"
    file += "    }\n"
    file += "    return retValue;\n"
    file += "  }\n"
    file += "\n"
    file += "  swiftware::benchmark::Timer execute() override {\n"
    file += "\n"
    file += "    OutTensor->reset();\n"
    file += "    swiftware::benchmark::Timer t1;\n"
    file += "\n"
    file += "    t1.startGPU();\n"
    file += "    // lunch kernel\n"
    file += "\n"
    file += "    SpmvComplexPatternDecomposeByRowSeperated<<<\n"
    file += f"        (InT->MixedFormat->cf[0]->nrpp - 1) * {number_of_patterns}, InT->number_of_warps * 32>>>(mf, dX, dY);\n"
    file += "\n"
    file += "    t1.stopGPU(\"\");\n"
    file += "\n"
    file += "    cudaMemcpy(this->OutTensor->Dx, dY, this->InTensor->ACsr->m  * InTensor->Z * sizeof(float),\n"
    file += "               cudaMemcpyDeviceToHost);\n"
    file += "    cudaMemset(dY, 0, this->InTensor->ACsr->m  * InTensor->Z * sizeof(float));\n"
    file += "\n"
    file += "    return t1;\n"
    file += "  }\n"
    file += "\n"
    file += "public:\n"
    file += "  SpMVMixePatternDecomposeByRowSeperated(TensorInputsMixed *In1,\n"
    file += "                                         swiftware::benchmark::Stats *Stat1)\n"
    file += "      : SpMMSerialFloat(In1, Stat1), InT(In1) {\n"
    file += "\n"
    file += "    x = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));\n"
    file += "    for (int i = 0; i < In1->ACsr->nnz; i++) {\n"
    file += "      x[i] = static_cast<float>(In1->ACsr->x[i]);\n"
    file += "    }\n"
    file += "\n"
    file += "    cudaMalloc((void **)&dX, InTensor->N * InTensor->Z * sizeof(float));\n"
    file += "    cudaMalloc((void **)&dY, InTensor->M * InTensor->Z * sizeof(float));\n"
    file += "    cudaMalloc((void **)&dY1, InTensor->M * InTensor->Z * sizeof(float));\n"
    file += "    cudaMalloc((void **)&dY2, InTensor->M * InTensor->Z * sizeof(float));\n"
    file += "    //\n"
    file += "    cudaMemset(dY, 0, InTensor->M * InTensor->Z * sizeof(float));\n"
    file += "    cudaMemset(dY1, 0, InTensor->M * InTensor->Z * sizeof(float));\n"
    file += "    cudaMemset(dY2, 0, InTensor->M * InTensor->Z * sizeof(float));\n"
    file += "    cudaMemcpy(dX, this->InTensor->Bx, InTensor->N * InTensor->Z * sizeof(float),\n"
    file += "               cudaMemcpyHostToDevice);\n"
    file += "\n"
    file += "    mf = allocateAndCopyMixedFormat(In1->MixedFormat);\n"
    file += "  };\n"
    file += "\n"
    file += "  ~SpMVMixePatternDecomposeByRowSeperated() {\n"
    file += "    //    cudaFree(dX);\n"
    file += "    //    cudaFree(dY);\n"
    file += "  }\n"
    file += "};\n"

    file += "\n\n"
    file += "class SpMVMixePatternDecomposeByRowSeperatedHalf : public SpMMSerialFloat {\n"
    file += "\n"
    file += "protected:\n"
    file += "\n"
    file += "  __half *dX, *dY;\n"
    file += "  float *dY1;\n"
    file += "  swiftware::compression::MixedFormatGPU<__half> *mf;\n"
    file += "\n"
    file += "  float alpha = 1.0f;\n"
    file += "\n"
    file += "  TensorInputsMixed *InT;\n"
    file += "\n"
    file += "  bool verify(double &Error) override {\n"
    file += "    bool retValue = true;\n"
    file += "\n"
    file += "    double infNorm = 0;\n"
    file += "    for (int i = 0; i < InTensor->M * InTensor->Z - 4; ++i) {\n"
    file += "      if (std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]) > infNorm) {\n"
    file += "        infNorm = std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]);\n"
    file += "      }\n"
    file += "    }\n"
    file += "    Error = (double)infNorm;\n"
    file += "    if (infNorm > InTensor->Threshold) {\n"
    file += "      retValue = false;\n"
    file += "    }\n"
    file += "    return retValue;\n"
    file += "  }\n"
    file += "\n"
    file += "  swiftware::benchmark::Timer execute() override {\n"
    file += "\n"
    file += "    OutTensor->reset();\n"
    file += "    swiftware::benchmark::Timer t1;\n"
    file += "\n"
    file += "    t1.startGPU();\n"
    file += "    // lunch kernel\n"
    file += "\n"
    file += "    SpmvComplexPatternDecomposeByRowSeperatedHalf<<<\n"
    file += f"        (InT->MixedFormat->cf[0]->nrpp - 1) * {number_of_patterns}, InT->number_of_warps * 32>>>(mf, dX, dY);\n"
    file += "\n"
    file += "    t1.stopGPU(\"\");\n"
    file += "\n"
    file += "   cudaMemcpy(dY1, dY, this->InTensor->ACsr->m  * InTensor->Z * sizeof(__half), cudaMemcpyDeviceToHost);\n"
    file += "   for(int i = 0; i < InTensor->M * InTensor->Z; i++) {\n"
    file += "   OutTensor->Dx[i] = __half2float(dY1[i]);\n"
    file += "}\n"
    file += "   cudaMemset(dY, 0, this->InTensor->ACsr->m  * InTensor->Z * sizeof(float));\n"
    file += "\n"
    file += "    return t1;\n"
    file += "  }\n"
    file += "\n"
    file += "public:\n"
    file += "  SpMVMixePatternDecomposeByRowSeperatedHalf(TensorInputsMixed *In1,\n"
    file += "                                         swiftware::benchmark::Stats *Stat1)\n"
    file += "      : SpMMSerialFloat(In1, Stat1), InT(In1) {\n"
    file += "    dY1 = static_cast<float *>(aligned_alloc(32, sizeof(float) * InTensor->M * InTensor->Z));\n"
    file += "    std::fill(dY1,dY1+InTensor->M*InTensor->Z,0);\n"
    file += "    cudaMalloc((void **)&dX, InTensor->N * InTensor->Z * sizeof(__half));\n"
    file += "    cudaMalloc((void **)&dY, InTensor->M * InTensor->Z * sizeof(__half));\n"
    file += "    //\n"
    file += "    cudaMemset(dY, 0, InTensor->M * InTensor->Z * sizeof(__half));\n"
    file += "    cudaMemset(dY1, 0, InTensor->M * InTensor->Z * sizeof(float));\n"
    file += "    cudaMemcpy(dX, this->InTensor->Bx, InTensor->N * InTensor->Z * sizeof(__half),\n"
    file += "               cudaMemcpyHostToDevice);\n"
    file += "\n"
    file += "    mf = allocateAndCopyMixedFormatToHalf(In1->MixedFormat);\n"
    file += "  };\n"
    file += "\n"
    file += "  ~SpMVMixePatternDecomposeByRowSeperatedHalf() {\n"
    file += "    //    cudaFree(dX);\n"
    file += "    //    cudaFree(dY);\n"
    file += "  }\n"
    file += "};\n"

    return file

def codelet_code_generator(m, b_col):
    file = ""
    for i in range(1, m + 1):
        pack_size = 32 // i
        file += f"__device__ void PatternWith{i}NNZByRowPanelFull(int *RPP, int *NPP, float *NNZ,\n"
        file += "                                              int *CB, float *dx, float *dy, int i "
        for x in range(i):
            file += f", int offset{x}"
        file += ") {\n\n"
        file += "  int blockSize = blockDim.x;\n"
        file += "  int thread_id_in_warp = threadIdx.x % 32;\n"
        file += "  int warp_id = threadIdx.x / 32;\n"
        file += "\n\n"
        file += ""
        file += "  int rowStart_all = RPP[i];\n"
        file += "  int rowEnd_all = RPP[i + 1];\n"
        file += f"  int number_of_packs = (rowEnd_all - rowStart_all) / {pack_size};\n"
        file += "  if (rowEnd_all - rowStart_all == 0) {\n"
        file += "    return;\n"
        file += "  }\n\n"
        file += "  number_of_packs = number_of_packs == 0 ? 1 : number_of_packs;\n"
        file += "  int number_of_warps = blockSize / 32;\n"
        file += "  int pack_share = (number_of_packs + number_of_warps - 1) / number_of_warps;\n"
        file += f"  int warp_share = pack_share * {pack_size};\n"
        file += "  int rowStart = rowStart_all + warp_id * warp_share;\n"
        file += "  int rowEnd = rowStart + warp_share;\n"
        file += "  rowEnd = rowEnd > rowEnd_all ? rowEnd_all : rowEnd;\n"
        file += "  rowEnd = warp_id == number_of_warps - 1 && rowEnd < rowEnd_all ? rowEnd_all : rowEnd;\n"
        file += f"  int t_nnz = NPP[i] + warp_id * warp_share * {i};\n"

        for x in range(b_col):
            file += f'  float sum_all{x} = 0.0f;\n'
        file += f"  int j = rowStart;\n"
        file += f"  for ( j ; j < rowEnd - {pack_size - 1}; j += {pack_size}) {{\n"
        file += f"    float nnz = (thread_id_in_warp < {pack_size * i}) ? NNZ[t_nnz + thread_id_in_warp] : 0;\n"

        for x in range(b_col):
            file += f"    float B{x} = 0.0f;\n"

        for x in range(b_col):
            file += f"    B{x} = dx[CB[j + thread_id_in_warp / {i}] * {b_col} + {x}];\n"

        file += "\n"
        for x in range(b_col):
            file += f"    sum_all{x} += nnz * B{x};\n"

        file += f"    t_nnz += {pack_size * i};\n"
        file += "  }\n"

        file += "  int tail = rowEnd - j;\n"
        file += f"  int rest_nnz = {i} * tail; \n"

        file += "  if (thread_id_in_warp < rest_nnz && tail > 0) {\n"
        file += "    float nnz = NNZ[t_nnz + thread_id_in_warp];\n"
        for x in range(b_col):
            file += f"    float B{x} = 0.0f;\n"
        for x in range(b_col):
            file += f"    B{x} = dx[CB[j + thread_id_in_warp / {i}] * {b_col} + {x}];\n"

        for x in range(b_col):
            file += f"    sum_all{x} += nnz * B{x};\n"
        file += "  }\n"

        temp_text = ""
        start_reduction_point = i
        encounterd_threads = pack_size * i
        end_thread = (pack_size - 1) * i
        pack_size_temp = pack_size
        temp_counter = 0
        off_set_end = 1

        if pack_size_temp % 2 == 1:
            for x in range(b_col):
                file += f"  float temp_sum{temp_counter}_{x} = sum_all{x};\n"
                file += f"  sum_all{x} += __shfl_down_sync(0xFFFFFFFF, sum_all{x}, {start_reduction_point});\n"
                temp_text += f"  sum_all{x} += __shfl_down_sync(0xFFFFFFFF, temp_sum{temp_counter}_{x}, {end_thread});\n"
            temp_counter += 1
            start_reduction_point *= 2
            pack_size_temp = pack_size_temp // 2
            end_thread = end_thread - 2 * i

        while start_reduction_point <= encounterd_threads / 2:
            if pack_size_temp % 2 == 1 and start_reduction_point != 1:
                for x in range(b_col):
                    file += f"  float temp_sum{temp_counter}_{x} = sum_all{x};\n"
                    file += f"  sum_all{x} += __shfl_down_sync(0xFFFFFFFF, sum_all{x}, {start_reduction_point});\n"
                    temp_text += f"  sum_all{x} += __shfl_down_sync(0xFFFFFFFF, temp_sum{temp_counter}_{x}, {end_thread});\n"
                pack_size_temp = pack_size_temp // 2
                start_reduction_point *= 2
                end_thread = end_thread - 2 * i
                temp_counter += 1
            else:
                for x in range(b_col):
                    file += f"  sum_all{x} += __shfl_down_sync(0xFFFFFFFF, sum_all{x}, {start_reduction_point});\n"
                start_reduction_point *= 2
                pack_size_temp = pack_size_temp // 2
                off_set_end += 1
                end_thread = end_thread - i

        file += temp_text
        file += f"  if (thread_id_in_warp == 0) {{\n"
        for x in range(b_col):
            file += f"    atomicAdd(&dy[(i * {m} + offset0) * {b_col} + {x}], sum_all{x});\n"
        file += "  }\n"
        p = 1
        while p < i:
            file += f"  else if (thread_id_in_warp == {p}){{\n"
            for x in range(b_col):
                file += f"    atomicAdd(&dy[(i * {m} + offset{p}) * {b_col} + {x}], sum_all{x});\n"
            file += "  }\n"
            p += 1
        file += "}\n"

    file += "\n\n"
    for i in range(1, m + 1):
        pack_size = 32 // i
        file += f"__device__ void PatternWith{i}NNZByRowPanelFullHALF(int *RPP, int *NPP, __half *NNZ,\n"
        file += "                                              int *CB, __half *dx, __half *dy, int i "
        for x in range(i):
            file += f", int offset{x}"
        file += ") {\n\n"
        file += "  int blockSize = blockDim.x;\n"
        file += "  int thread_id_in_warp = threadIdx.x % 32;\n"
        file += "  int warp_id = threadIdx.x / 32;\n"
        file += "\n\n"
        file += ""
        file += "  int rowStart_all = RPP[i];\n"
        file += "  int rowEnd_all = RPP[i + 1];\n"
        file += f"  int number_of_packs = (rowEnd_all - rowStart_all) / {pack_size};\n"
        file += "  if (rowEnd_all - rowStart_all == 0) {\n"
        file += "    return;\n"
        file += "  }\n\n"
        file += "  number_of_packs = number_of_packs == 0 ? 1 : number_of_packs;\n"
        file += "  int number_of_warps = blockSize / 32;\n"
        file += "  int pack_share = (number_of_packs + number_of_warps - 1) / number_of_warps;\n"
        file += f"  int warp_share = pack_share * {pack_size};\n"
        file += "  int rowStart = rowStart_all + warp_id * warp_share;\n"
        file += "  int rowEnd = rowStart + warp_share;\n"
        file += "  rowEnd = rowEnd > rowEnd_all ? rowEnd_all : rowEnd;\n"
        file += "  rowEnd = warp_id == number_of_warps - 1 && rowEnd < rowEnd_all ? rowEnd_all : rowEnd;\n"
        file += f"  int t_nnz = NPP[i] + warp_id * warp_share * {i};\n"

        for x in range(b_col):
            file += f'  __half sum_all{x} = 0.0;\n'
        file += f"  int j = rowStart;\n"
        file += f"  for ( j ; j < rowEnd - {pack_size - 1}; j += {pack_size}) {{\n"
        file += f"    __half nnz = (thread_id_in_warp < {pack_size * i}) ? NNZ[t_nnz + thread_id_in_warp] : __float2half_rn(0);\n"

        for x in range(b_col):
            file += f"    __half B{x} = 0.0;\n"

        for x in range(b_col):
            file += f"    B{x} = dx[CB[j + thread_id_in_warp / {i}] * {b_col} + {x}];\n"

        file += "\n"
        for x in range(b_col):
            file += f"    sum_all{x} += __hmul(nnz, B{x});\n"

        file += f"    t_nnz += {pack_size * i};\n"
        file += "  }\n"

        file += "  int tail = rowEnd - j;\n"
        file += f"  int rest_nnz = {i} * tail; \n"

        file += "  if (thread_id_in_warp < rest_nnz && tail > 0) {\n"
        file += "    __half nnz = NNZ[t_nnz + thread_id_in_warp];\n"
        for x in range(b_col):
            file += f"    __half B{x} = 0.0;\n"
        for x in range(b_col):
            file += f"    B{x} = dx[CB[j + thread_id_in_warp / {i}] * {b_col} + {x}];\n"

        for x in range(b_col):
            file += f"    sum_all{x} += __hmul(nnz , B{x});\n"
        file += "  }\n"

        temp_text = ""
        start_reduction_point = i
        encounterd_threads = pack_size * i
        end_thread = (pack_size - 1) * i
        pack_size_temp = pack_size
        temp_counter = 0
        off_set_end = 1

        if pack_size_temp % 2 == 1:
            for x in range(b_col):
                file += f"  __half temp_sum{temp_counter}_{x} = sum_all{x};\n"
                file += f"  sum_all{x} += warp_shfl_down_half(sum_all{x}, {start_reduction_point});\n"
                temp_text += f"  sum_all{x} += warp_shfl_down_half(temp_sum{temp_counter}_{x}, {end_thread});\n"
            temp_counter += 1
            start_reduction_point *= 2
            pack_size_temp = pack_size_temp // 2
            end_thread = end_thread - 2 * i

        while start_reduction_point <= encounterd_threads / 2:
            if pack_size_temp % 2 == 1 and start_reduction_point != 1:
                for x in range(b_col):
                    file += f"  __half temp_sum{temp_counter}_{x} = sum_all{x};\n"
                    file += f"  sum_all{x} += warp_shfl_down_half(sum_all{x}, {start_reduction_point});\n"
                    temp_text += f"  sum_all{x} += warp_shfl_down_half(temp_sum{temp_counter}_{x}, {end_thread});\n"
                pack_size_temp = pack_size_temp // 2
                start_reduction_point *= 2
                end_thread = end_thread - 2 * i
                temp_counter += 1
            else:
                for x in range(b_col):
                    file += f"  sum_all{x} += warp_shfl_down_half(sum_all{x}, {start_reduction_point});\n"
                start_reduction_point *= 2
                pack_size_temp = pack_size_temp // 2
                off_set_end += 1
                end_thread = end_thread - i

        file += temp_text
        file += f"  if (thread_id_in_warp == 0) {{\n"
        for x in range(b_col):
            file += f"    atomicAdd(&dy[(i * {m} + offset0) * {b_col} + {x}], sum_all{x});\n"
        file += "  }\n"
        p = 1
        while p < i:
            file += f"  else if (thread_id_in_warp == {p}){{\n"
            for x in range(b_col):
                file += f"    atomicAdd(&dy[(i * {m} + offset{p}) * {b_col} + {x}], sum_all{x});\n"
            file += "  }\n"
            p += 1
        file += "}\n"
    return file

def base_file(codelet_text, kernel_text, class_text, m, pattern_type_dictionary):
    file_content = ""
    file_content += "#ifndef COMPRESSED_TENSOR_ALGEBRA_SPMM_SMALL_B_COL_DEMO_GPU_UTILS_H\n"
    file_content += "#define COMPRESSED_TENSOR_ALGEBRA_SPMM_SMALL_B_COL_DEMO_GPU_UTILS_H\n\n"
    file_content += "#include \"SWTensorBench.h\"\n"
    file_content += "#include \"aggregation/def.h\"\n"
    file_content += "#include <cublas_v2.h>\n"
    file_content += "#include \"compressed_format/CompressedFormat.h\"\n"
    file_content += "#include <cuda/pipeline>\n"
    file_content += "#include <cuda_runtime.h>\n"
    file_content += "#include <cuda_fp16.h>\n"
    file_content += "#include <cuda_runtime_api.h>\n"
    file_content += "#include <cusparse.h>\n"
    file_content += "#include <stdio.h>\n"

    file_content += "#define CUDA_CHECK(err) \\\n"
    file_content += "  do { \\\n"
    file_content += "    cudaError_t err_ = (err); \\\n"
    file_content += "    if (err_ != cudaSuccess) { \\\n"
    file_content += "      std::printf(\"CUDA error %d at %s:%d\\n\", err_, __FILE__, __LINE__); \\\n"
    file_content += "      throw std::runtime_error(\"CUDA error\"); \\\n"
    file_content += "    } \\\n"
    file_content += "  } while (0)\n\n"

    file_content += "__device__ __half warp_shfl_down_half(__half val, int delta) {\n"
    file_content += "uint16_t val_bits = *reinterpret_cast<uint16_t*>(&val);\n"
    file_content += "uint16_t shuffled = __shfl_down_sync(0xffffffff, val_bits, delta);\n"
    file_content += "return *reinterpret_cast<__half*>(&shuffled);\n"
    file_content += "}\n\n"

    file_content += "void spmm_csr_float(int m, int n, const int *Ap, const int *Ai, const float *Ax,\n"
    file_content += "                    const float *B, float *C) {\n"
    file_content += "  for (int i = 0; i < m; i++) {\n"
    file_content += "    for (int j = Ap[i]; j < Ap[i + 1]; j++) {\n"
    file_content += "      int col_index = Ai[j];\n"
    file_content += "      float value = Ax[j];\n"
    file_content += "      for (int k = 0; k < n; k++) {\n"
    file_content += "        C[i * n + k] += value * B[col_index * n + k];\n"
    file_content += "      }\n"
    file_content += "    }\n"
    file_content += "  }\n"
    file_content += "}\n\n"

    file_content += "template <typename T>\n"
    file_content += "swiftware::compression::CompressedFormatGPU<T> *allocateAndCopyCompressedFormat(\n"
    file_content += "    swiftware::compression::CompressedFormat<T> *h_cf) {\n"
    file_content += "  swiftware::compression::CompressedFormatGPU<T> *d_cf;\n"
    file_content += "  cudaMalloc((void **)&d_cf,\n"
    file_content += "             sizeof(swiftware::compression::CompressedFormatGPU<T>));\n"
    file_content += "  cudaMemcpy(d_cf, h_cf, sizeof(swiftware::compression::CompressedFormat<T>),\n"
    file_content += "             cudaMemcpyHostToDevice);\n"
    file_content += "  T *d_NNZ;\n"
    file_content += "  int *d_cols, *d_RPP, *d_NPP;\n"
    file_content += "  int *d_PT, *d_CB, *d_CSRRows;\n"
    file_content += "  cudaMalloc((void **)&d_NNZ, h_cf->nnz * sizeof(T));\n"
    file_content += "  cudaMemcpy(d_NNZ, h_cf->NNZ, h_cf->nnz * sizeof(T), cudaMemcpyHostToDevice);\n"
    file_content += "  cudaMalloc((void **)&d_RPP, h_cf->nrpp * sizeof(int));\n"
    file_content += "  cudaMemcpy(d_RPP, h_cf->RPP, h_cf->nrpp * sizeof(int), cudaMemcpyHostToDevice);\n"
    file_content += "  cudaMalloc((void **)&d_NPP, h_cf->nrpp * sizeof(int));\n"
    file_content += "  cudaMemcpy(d_NPP, h_cf->NPP, h_cf->nrpp * sizeof(int), cudaMemcpyHostToDevice);\n"
    file_content += "  cudaMalloc((void **)&d_PT, h_cf->PT.size() * sizeof(int));\n"
    file_content += "  cudaMemcpy(d_PT, h_cf->PT.data(), h_cf->PT.size() * sizeof(int), cudaMemcpyHostToDevice);\n"
    file_content += "  cudaMalloc((void **)&d_CB, h_cf->CB.size() * sizeof(int));\n"
    file_content += "  cudaMemcpy(d_CB, h_cf->CB.data(), h_cf->CB.size() * sizeof(int), cudaMemcpyHostToDevice);\n"
    file_content += "  CUDA_CHECK(\n"
    file_content += "      cudaMemcpy(&(d_cf->NNZ), &d_NNZ, sizeof(T *), cudaMemcpyHostToDevice));\n"
    file_content += "  CUDA_CHECK(\n"
    file_content += "      cudaMemcpy(&(d_cf->RPP), &d_RPP, sizeof(int *), cudaMemcpyHostToDevice));\n"
    file_content += "  cudaMemcpy(&(d_cf->NPP), &d_NPP, sizeof(int *), cudaMemcpyHostToDevice);\n"
    file_content += "  cudaMemcpy(&(d_cf->PT), &d_PT, sizeof(int *), cudaMemcpyHostToDevice);\n"
    file_content += "  cudaMemcpy(&(d_cf->CB), &d_CB, sizeof(int *), cudaMemcpyHostToDevice);\n"
    file_content += "  return d_cf;\n"
    file_content += "}\n"

    file_content += "void copyCSR(sym_lib::CSR *A, sym_lib::CSR *ACsr) {\n"
    file_content += "  ACsr->m = A->m;\n"
    file_content += "  ACsr->n = A->n;\n"
    file_content += "  ACsr->nnz = A->nnz;\n"

    file_content += "  for (int i = 0; i < A->m + 1; ++i) {\n"
    file_content += "    ACsr->p[i] = A->p[i];\n"
    file_content += "  }\n"

    file_content += "  for (int i = 0; i < A->m; ++i) {\n"
    file_content += "    for (int j = A->p[i]; j < A->p[i + 1]; ++j) {\n"
    file_content += "      ACsr->x[j] = A->x[j];\n"
    file_content += "      ACsr->i[j] = A->i[j];\n"
    file_content += "    }\n"
    file_content += "  }\n"
    file_content += "}\n"

    file_content += "swiftware::compression::CompressedFormatGPU<__half> *\n"
    file_content += "allocateAndCopyCompressedFormatToHalf(\n"
    file_content += "    swiftware::compression::CompressedFormat<float> *h_cf) {\n\n"
    file_content += "  swiftware::compression::CompressedFormatGPU<__half> *d_cf;\n"
    file_content += "  cudaMalloc((void **)&d_cf,\n"
    file_content += "             sizeof(swiftware::compression::CompressedFormatGPU<__half>));\n\n"
    file_content += "  // Allocate and convert arrays from float to half before copying\n"
    file_content += "  __half *d_NNZ;\n"
    file_content += "  int *d_RPP, *d_NPP;\n"
    file_content += "  int *d_PT, *d_CB;\n\n"
    file_content += "  // Allocate temporary array to hold half-precision values\n"
    file_content += "  __half *temp_NNZ = new __half[h_cf->nnz];\n"
    file_content += "  for (size_t i = 0; i < h_cf->nnz; ++i) {\n"
    file_content += "    temp_NNZ[i] = __float2half(h_cf->NNZ[i]);\n"
    file_content += "  }\n\n"
    file_content += "  // Copy the converted half-precision array to the device\n"
    file_content += "  cudaMalloc((void **)&d_NNZ, h_cf->nnz * sizeof(__half));\n"
    file_content += "  cudaMemcpy(d_NNZ, temp_NNZ, h_cf->nnz * sizeof(__half), cudaMemcpyHostToDevice);\n"
    file_content += "  delete[] temp_NNZ;\n\n"
    file_content += "  // Copy integer arrays directly\n"
    file_content += "  cudaMalloc((void **)&d_RPP, h_cf->nrpp * sizeof(int));\n"
    file_content += "  cudaMemcpy(d_RPP, h_cf->RPP, h_cf->nrpp * sizeof(int),\n"
    file_content += "             cudaMemcpyHostToDevice);\n\n"
    file_content += "  cudaMalloc((void **)&d_NPP, h_cf->nrpp * sizeof(int));\n"
    file_content += "  cudaMemcpy(d_NPP, h_cf->NPP, h_cf->nrpp * sizeof(int),\n"
    file_content += "             cudaMemcpyHostToDevice);\n\n"
    file_content += "  cudaMalloc((void **)&d_PT, h_cf->PT.size() * sizeof(int));\n"
    file_content += "  cudaMemcpy(d_PT, h_cf->PT.data(), h_cf->PT.size() * sizeof(int),\n"
    file_content += "             cudaMemcpyHostToDevice);\n\n"
    file_content += "  cudaMalloc((void **)&d_CB, h_cf->CB.size() * sizeof(int));\n"
    file_content += "  cudaMemcpy(d_CB, h_cf->CB.data(), h_cf->CB.size() * sizeof(int),\n"
    file_content += "             cudaMemcpyHostToDevice);\n\n"
    file_content += "  // Copy pointers to the device\n"
    file_content += "  CUDA_CHECK(\n"
    file_content += "      cudaMemcpy(&(d_cf->NNZ), &d_NNZ, sizeof(__half *), cudaMemcpyHostToDevice));\n"
    file_content += "  CUDA_CHECK(\n"
    file_content += "      cudaMemcpy(&(d_cf->RPP), &d_RPP, sizeof(int *), cudaMemcpyHostToDevice));\n"
    file_content += "  cudaMemcpy(&(d_cf->NPP), &d_NPP, sizeof(int *), cudaMemcpyHostToDevice);\n"
    file_content += "  cudaMemcpy(&(d_cf->PT), &d_PT, sizeof(int *), cudaMemcpyHostToDevice);\n"
    file_content += "  cudaMemcpy(&(d_cf->CB), &d_CB, sizeof(int *), cudaMemcpyHostToDevice);\n\n"
    file_content += "  return d_cf;\n"
    file_content += "}\n\n"
    file_content += "swiftware::compression::MixedFormatGPU<__half> *allocateAndCopyMixedFormatToHalf(\n"
    file_content += "    swiftware::compression::MixedFormat<float> *h_mixed) {\n\n"
    file_content += "  swiftware::compression::MixedFormatGPU<__half> *d_mixed;\n"
    file_content += "  cudaMalloc((void **)&d_mixed,\n"
    file_content += "             sizeof(swiftware::compression::MixedFormatGPU<__half> *));\n\n"
    file_content += "  // Allocate and copy the array of CompressedFormat pointers\n"
    file_content += "  swiftware::compression::CompressedFormatGPU<__half> **d_cf;\n"
    file_content += "  cudaMalloc((void **)&d_cf,\n"
    file_content += "             h_mixed->cf.size() *\n"
    file_content += "                 sizeof(swiftware::compression::CompressedFormatGPU<__half> *));\n\n"
    file_content += "  for (size_t i = 0; i < h_mixed->cf.size(); ++i) {\n"
    file_content += "    swiftware::compression::CompressedFormatGPU<__half> *d_cf_element =\n"
    file_content += "        allocateAndCopyCompressedFormatToHalf(h_mixed->cf[i]);\n"
    file_content += "    cudaMemcpy(&(d_cf[i]), &d_cf_element,\n"
    file_content += "               sizeof(swiftware::compression::CompressedFormatGPU<__half> *),\n"
    file_content += "               cudaMemcpyHostToDevice);\n"
    file_content += "  }\n\n"
    file_content += "  cudaMemcpy(&(d_mixed->cf), &d_cf,\n"
    file_content += "             sizeof(swiftware::compression::CompressedFormatGPU<__half> *),\n"
    file_content += "             cudaMemcpyHostToDevice);\n\n"
    file_content += "  return d_mixed;\n"
    file_content += "}\n"

    file_content += "template <typename T>\n"
    file_content += "struct TensorOutputs : public swiftware::benchmark::Outputs<T> {\n"
    file_content += "  int M;\n"
    file_content += "  T *Dx;\n\n"

    file_content += "  TensorOutputs(int M) : M(M) { Dx = new T[M](); }\n\n"

    file_content += "  ~TensorOutputs() { delete[] Dx; }\n\n"

    file_content += "  void printDx() {\n";
    file_content += "    std::cout << \"\\n Dx:\\n\";\n"
    file_content += "    std::cout << \"\\n\";\n"
    file_content += "  }\n\n"

    file_content += "  void reset() { std::fill_n(Dx, M, 0.0); }\n"
    file_content += "};\n"

    file_content += "struct TensorInputsFloat : public swiftware::benchmark::Inputs<float> {\n"
    file_content += "  int M, N, Z;\n"
    file_content += "  sym_lib::CSR *ACsr;\n\n"

    file_content += "  float *Bx;\n"
    file_content += "  float *CorrectMul;\n"
    file_content += "  bool IsSolProvided;\n\n"

    file_content += "  TensorInputsFloat(int M1, int N1, int Z1, sym_lib::CSR *A1, int NumThreads1,\n"
    file_content += "                    int NumTrial1, std::string ExpN)\n"
    file_content += "      : swiftware::benchmark::Inputs<float>(NumTrial1, NumThreads1, ExpN) {\n"
    file_content += "    M = M1;\n"
    file_content += "    N = N1;\n"
    file_content += "    Z = Z1;\n"
    file_content += "    if (A1 != nullptr)\n"
    file_content += "      ACsr = new sym_lib::CSR(M, N, A1->nnz);\n\n"

    file_content += "    copyCSR(A1, ACsr);\n"
    file_content += "    IsSolProvided = false;\n"
    file_content += "    swiftware::benchmark::Inputs<float>::Threshold = 1e-6;\n"
    file_content += "  }\n\n"

    file_content += "  ~TensorInputsFloat() {\n"
    file_content += "    free(CorrectSol);\n"
    file_content += "    free(CorrectMul);\n"
    file_content += "    free(Bx);\n"
    file_content += "    delete ACsr;\n"
    file_content += "  }\n"
    file_content += "};\n"

    file_content += "template <typename T>\n"
    file_content += "swiftware::compression::MixedFormatGPU<T> *\n"
    file_content += "allocateAndCopyMixedFormat(swiftware::compression::MixedFormat<T> *h_mixed) {\n\n"

    file_content += "  swiftware::compression::MixedFormatGPU<T> *d_mixed;\n"
    file_content += "  cudaMalloc((void **)&d_mixed,\n"
    file_content += "             sizeof(swiftware::compression::MixedFormatGPU<T> *));\n\n"

    file_content += "  // Allocate and copy the array of CompressedFormat pointers\n"
    file_content += "  swiftware::compression::CompressedFormatGPU<T> **d_cf;\n"
    file_content += "  cudaMalloc((void **)&d_cf,\n"
    file_content += "             h_mixed->cf.size() *\n"
    file_content += "                 sizeof(swiftware::compression::CompressedFormatGPU<T> *));\n\n"

    file_content += "  for (size_t i = 0; i < h_mixed->cf.size(); ++i) {\n"
    file_content += "    swiftware::compression::CompressedFormatGPU<T> *d_cf_element =\n"
    file_content += "        allocateAndCopyCompressedFormat(h_mixed->cf[i]);\n"
    file_content += "    cudaMemcpy(&(d_cf[i]), &d_cf_element,\n"
    file_content += "               sizeof(swiftware::compression::CompressedFormatGPU<T> *),\n"
    file_content += "               cudaMemcpyHostToDevice);\n"
    file_content += "  }\n\n"

    file_content += "  cudaMemcpy(&(d_mixed->cf), &d_cf,\n"
    file_content += "             sizeof(swiftware::compression::CompressedFormatGPU<T> *),\n"
    file_content += "             cudaMemcpyHostToDevice);\n\n"

    file_content += "  //  int cf_len = h_mixed->cf.size();\n"
    file_content += "  //  cudaMemcpy(&(d_mixed->cflen), &cf_len, sizeof(int *),\n"
    file_content += "  //  cudaMemcpyHostToDevice);\n\n"

    file_content += "  // Allocate and copy the CSRFormat\n"
    file_content += "  //  swiftware::compression::CSRFormatGPU<T> *d_csr =\n"
    file_content += "  //      allocateAndCopyCSRFormat(h_mixed->csr);\n"
    file_content += "  //  cudaMemcpy(&(d_mixed->csr), &d_csr,\n"
    file_content += "  //             sizeof(swiftware::compression::CSRFormatGPU<T> *),\n"
    file_content += "  //             cudaMemcpyHostToDevice);\n\n"

    file_content += "  return d_mixed;\n"
    file_content += "}\n"

    file_content += "// Mixed Format\n"
    file_content += "struct TensorInputsMixed : public TensorInputsFloat {\n\n"
    file_content += "    int number_of_warps;\n"
    file_content += "public:\n"
    file_content += "  swiftware::compression::MixedFormat<float> *MixedFormat;\n"
    file_content += "  TensorInputsMixed(int M1, int N1, int Z1, sym_lib::CSR *A1,\n"
    file_content += "                    swiftware::compression::MixedFormat<float> *Mixed,\n"
    file_content += "                    int NumThreads1, int NumTrial1, std::string ExpN)\n"
    file_content += "      : TensorInputsFloat(M1, N1, Z1, A1, NumThreads1, NumTrial1, ExpN),\n"
    file_content += "        MixedFormat(Mixed){};\n"
    file_content += "};\n\n"

    file_content += "class SpMMSerialFloat : public swiftware::benchmark::SWTensorBench<float> {\n"
    file_content += "protected:\n"
    file_content += "  TensorInputsFloat *InTensor;\n"
    file_content += "  float *csr_x;\n"
    file_content += "  void setup() override { this->St->OtherStats[\"NTile\"] = {4}; }\n\n"

    file_content += "  void preExecute() override {}\n\n"

    file_content += "  swiftware::benchmark::Timer execute() override {\n"
    file_content += "    OutTensor->reset();\n"
    file_content += "    swiftware::benchmark::Timer t1;\n"
    file_content += "    t1.start();\n\n"

    file_content += "    spmm_csr_float(InTensor->M, InTensor->Z, InTensor->ACsr->p,\n"
    file_content += "                   InTensor->ACsr->i, csr_x, InTensor->Bx, OutTensor->Dx);\n\n"

    file_content += "    t1.stop();\n"
    file_content += "    return t1;\n"
    file_content += "  }\n\n"

    file_content += "  bool verify(double &Error) override {\n"
    file_content += "    bool retValue = true;\n"
    file_content += "    if (!InTensor->IsSolProvided) {\n"
    file_content += "      Error = 0;\n"
    file_content += "      return true;\n"
    file_content += "    }\n"
    file_content += "    double infNorm = 0;\n"
    file_content += "    for (int i = 0; i < InTensor->M * InTensor->Z - 1; ++i) {\n\n"

    file_content += "      if (std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]) > infNorm) {\n"
    file_content += "        infNorm = std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]);\n"
    file_content += "      }\n"
    file_content += "    }\n"
    file_content += "    Error = (double)infNorm;\n"
    file_content += "    if (infNorm > InTensor->Threshold) {\n"
    file_content += "      retValue = false;\n"
    file_content += "    }\n"
    file_content += "    return retValue;\n"
    file_content += "  }\n\n"

    file_content += "public:\n"
    file_content += "  TensorOutputs<float> *OutTensor;\n"
    file_content += "  SpMMSerialFloat(TensorInputsFloat *In1, swiftware::benchmark::Stats *Stat1)\n"
    file_content += "      : swiftware::benchmark::SWTensorBench<float>(In1, Stat1) {\n"
    file_content += "    OutTensor = new TensorOutputs<float>(In1->M * In1->Z);\n"
    file_content += "    InTensor = In1;\n"
    file_content += "    csr_x =\n"
    file_content += "        static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));\n"
    file_content += "    for (int i = 0; i < In1->ACsr->nnz; i++) {\n"
    file_content += "      csr_x[i] = static_cast<float>(In1->ACsr->x[i]);\n"
    file_content += "    }\n"
    file_content += "  };\n\n"

    file_content += "  virtual void envelopCalculation() {}\n\n"

    file_content += "  ~SpMMSerialFloat() {\n"
    file_content += "    delete OutTensor;\n"
    file_content += "    free(csr_x);\n"
    file_content += "  }\n"
    file_content += "};\n"

    file_content += "template <typename T>\n"
    file_content += "__global__ void spmm_complex_kernel(int *d_ptr, int *d_cols, T *d_val,\n"
    file_content += "                                    T *d_matrix, T *d_out, int N, int b_cols) {\n"
    file_content += "  int thread_id_in_warp = threadIdx.x % 32; // Lane index in the vector\n"
    file_content += "  int warp_id = threadIdx.x / 32;           // Vector index in the thread block\n"
    file_content += "  int number_of_warps =\n"
    file_content += "      blockDim.x / 32; // Number of vectors in the thread block\n\n"

    file_content += "  // Calculate the row range for this block\n"
    file_content += "  int startRow = blockIdx.x;\n"
    file_content += "  int endRow = min(startRow + 1, N);\n\n"

    file_content += "  int col_share = b_cols / number_of_warps;\n"
    file_content += "  int col_start = col_share * warp_id;\n"
    file_content += "  int col_end = col_start + col_share;\n\n"

    file_content += "  for (int col = col_start; col < col_end; ++col) {\n\n"
    file_content += "    // Process rows assigned to this block\n\n"
    file_content += "    // Loop over all columns of the dense matrix (b_cols)\n"
    file_content += "    T sum = 0;\n\n"

    file_content += "    int rowStart = d_ptr[startRow];\n"
    file_content += "    int rowEnd = d_ptr[endRow];\n\n"

    file_content += "    // Compute the dot product for this row and current column\n"
    file_content += "    for (int i = rowStart + thread_id_in_warp; i < rowEnd; i += 32) {\n"
    file_content += "      sum += d_val[i] * d_matrix[d_cols[i] * b_cols + col];\n"
    file_content += "      //      printf(\"sum: %f\\n\", sum);\n"
    file_content += "    }\n\n"

    file_content += "    // Intra-vector reduction\n"
    file_content += "    for (int i = 32 >> 1; i > 0; i >>= 1) {\n"
    file_content += "      sum += __shfl_down_sync(0xffffffff, sum, i);\n"
    file_content += "    }\n\n"

    file_content += "    // Save the result for the current column\n"
    file_content += "    if (thread_id_in_warp == 0) {\n"
    file_content += "      //      printf(\"sum: %f\\n\", sum);\n"
    file_content += "      d_out[startRow * b_cols + col] += sum;\n"
    file_content += "    }\n"
    file_content += "  }\n"
    file_content += "}\n"

    file_content += "class SpMMComplexKernel : public SpMMSerialFloat {\n\n"
    file_content += "  float *x;\n"
    file_content += "  int *dA_csrOffsets, *dA_columns;\n"
    file_content += "  float *dA_values, *dX, *dY;\n\n"

    file_content += "  int *m_gpu, *n_gpu, *nnz_gpu;\n\n"

    file_content += "  int BlockDim = 256;\n"
    file_content += "  float alpha = 1.0f;\n\n"

    file_content += "  swiftware::benchmark::Timer execute() override {\n\n"
    file_content += "    OutTensor->reset();\n"
    file_content += "    swiftware::benchmark::Timer t1;\n\n"

    file_content += "    t1.startGPU();\n\n"

    file_content += "    spmm_complex_kernel<float><<<InTensor->ACsr->m + 1, 128>>>(\n"
    file_content += "        dA_csrOffsets, dA_columns, dA_values, dX, dY, InTensor->ACsr->m,\n"
    file_content += "        InTensor->Z);\n\n"

    file_content += "    t1.stopGPU(\"\");\n\n"

    file_content += "    cudaMemcpy(this->OutTensor->Dx, dY,\n"
    file_content += "               this->InTensor->ACsr->m * InTensor->Z * sizeof(float),\n"
    file_content += "               cudaMemcpyDeviceToHost);\n\n"

    file_content += "    cudaMemset(dY, 0, this->InTensor->ACsr->m * InTensor->Z * sizeof(float));\n\n"

    file_content += "    return t1;\n"
    file_content += "  }\n\n"

    file_content += "public:\n"
    file_content += "  SpMMComplexKernel(TensorInputsFloat *In1, swiftware::benchmark::Stats *Stat1)\n"
    file_content += "      : SpMMSerialFloat(In1, Stat1) {\n"
    file_content += "    x = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));\n"
    file_content += "    for (int i = 0; i < In1->ACsr->nnz; i++) {\n"
    file_content += "      x[i] = static_cast<float>(In1->ACsr->x[i]);\n"
    file_content += "    }\n\n"

    file_content += "    cudaMalloc((void **)&dA_csrOffsets,\n"
    file_content += "               (this->InTensor->ACsr->m + 1) * sizeof(int));\n"
    file_content += "    cudaMalloc((void **)&dA_columns, this->InTensor->ACsr->nnz * sizeof(int));\n"
    file_content += "    cudaMalloc((void **)&dA_values, this->InTensor->ACsr->nnz * sizeof(float));\n"
    file_content += "    cudaMalloc((void **)&dX,\n"
    file_content += "               this->InTensor->ACsr->n * this->InTensor->Z * sizeof(float));\n"
    file_content += "    cudaMalloc((void **)&dY,\n"
    file_content += "               this->InTensor->ACsr->m * this->InTensor->Z * sizeof(float));\n\n"

    file_content += "    cudaMemcpy(dA_values, this->x, this->InTensor->ACsr->nnz * sizeof(float),\n"
    file_content += "               cudaMemcpyHostToDevice);\n"
    file_content += "    cudaMemcpy(dA_csrOffsets, this->InTensor->ACsr->p,\n"
    file_content += "               (this->InTensor->ACsr->m + 1) * sizeof(int),\n"
    file_content += "               cudaMemcpyHostToDevice);\n"
    file_content += "    cudaMemcpy(dA_columns, this->InTensor->ACsr->i,\n"
    file_content += "               this->InTensor->ACsr->nnz * sizeof(int), cudaMemcpyHostToDevice);\n\n"

    file_content += "    cudaMemcpy(dX, this->InTensor->Bx,\n"
    file_content += "               this->InTensor->ACsr->n * this->InTensor->Z * sizeof(float),\n"
    file_content += "               cudaMemcpyHostToDevice);\n"
    file_content += "    cudaMemcpy(dY, OutTensor->Dx,\n"
    file_content += "               this->InTensor->ACsr->m * this->InTensor->Z * sizeof(float),\n"
    file_content += "               cudaMemcpyHostToDevice);\n\n"

    file_content += "  };\n"
    file_content += "  ~SpMMComplexKernel() {\n"
    file_content += "    free(x);\n"
    file_content += "    cudaFree(dA_csrOffsets);\n"
    file_content += "    cudaFree(dA_columns);\n"
    file_content += "    cudaFree(dA_values);\n"
    file_content += "    cudaFree(dX);\n"
    file_content += "    cudaFree(dY);\n"
    file_content += "  }\n"
    file_content += "};\n"

    file_content += "class SpMMcuSparse : public SpMMSerialFloat {\n\n"
    file_content += "  float *x;\n"
    file_content += "  int *dA_csrOffsets, *dA_columns;\n"
    file_content += "  float *dA_values, *dX, *dY;\n\n"

    file_content += "  // CUSPARSE APIs\n"
    file_content += "  cusparseHandle_t handle = NULL;\n"
    file_content += "  cusparseSpMatDescr_t matA;\n"
    file_content += "  cusparseDnMatDescr_t matX, matY;\n"
    file_content += "  void *dBuffer = NULL;\n"
    file_content += "  size_t bufferSize = 0;\n\n"

    file_content += "  float alpha = 1.0f;\n"
    file_content += "  float beta = 0.0f;\n\n"

    file_content += "  bool verify(double &Error) override {\n"
    file_content += "    bool retValue = true;\n"
    file_content += "    if (!InTensor->IsSolProvided) {\n"
    file_content += "      Error = 0;\n"
    file_content += "      return true;\n"
    file_content += "    }\n"
    file_content += "    double infNorm = 0;\n"
    file_content += "    for (int i = 0; i < InTensor->M * InTensor->Z - 1; ++i) {\n\n"
    file_content += "      if (std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]) > infNorm) {\n"
    file_content += "        infNorm = std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]);\n"
    file_content += "      }\n"
    file_content += "    }\n"
    file_content += "    Error = (double)infNorm;\n"
    file_content += "    if (infNorm > InTensor->Threshold) {\n"
    file_content += "      retValue = false;\n"
    file_content += "    }\n"
    file_content += "    return retValue;\n"
    file_content += "  }\n\n"

    file_content += "  swiftware::benchmark::Timer execute() override {\n\n"
    file_content += "    OutTensor->reset();\n"
    file_content += "    swiftware::benchmark::Timer t1;\n"
    file_content += "    t1.startGPU();\n"
    file_content += "    cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,\n"
    file_content += "                 CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matX, &beta,\n"
    file_content += "                 matY, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);\n"
    file_content += "    t1.stopGPU(\"\");\n\n"

    file_content += "    cudaMemcpy(this->OutTensor->Dx, dY,\n"
    file_content += "               this->InTensor->ACsr->m * InTensor->Z * sizeof(float),\n"
    file_content += "               cudaMemcpyDeviceToHost);\n\n"

    file_content += "    return t1;\n"
    file_content += "  }\n\n"

    file_content += "public:\n"
    file_content += "  SpMMcuSparse(TensorInputsFloat *In1, swiftware::benchmark::Stats *Stat1)\n"
    file_content += "      : SpMMSerialFloat(In1, Stat1) {\n"
    file_content += "    x = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));\n"
    file_content += "    for (int i = 0; i < In1->ACsr->nnz; i++) {\n"
    file_content += "      x[i] = static_cast<float>(In1->ACsr->x[i]);\n"
    file_content += "    }\n\n"

    file_content += "    cudaMalloc((void **)&dA_csrOffsets,\n"
    file_content += "               (this->InTensor->ACsr->m + 1) * sizeof(int));\n"
    file_content += "    cudaMalloc((void **)&dA_columns, this->InTensor->ACsr->nnz * sizeof(int));\n"
    file_content += "    cudaMalloc((void **)&dA_values, this->InTensor->ACsr->nnz * sizeof(float));\n"
    file_content += "    cudaMalloc((void **)&dX,\n"
    file_content += "               this->InTensor->ACsr->n * InTensor->Z * sizeof(float));\n"
    file_content += "    cudaMalloc((void **)&dY,\n"
    file_content += "               this->InTensor->ACsr->m * this->InTensor->Z * sizeof(float));\n\n"

    file_content += "    cudaMemcpy(dA_values, this->x, this->InTensor->ACsr->nnz * sizeof(float),\n"
    file_content += "               cudaMemcpyHostToDevice);\n"
    file_content += "    cudaMemcpy(dA_csrOffsets, this->InTensor->ACsr->p,\n"
    file_content += "               (this->InTensor->ACsr->m + 1) * sizeof(int),\n"
    file_content += "               cudaMemcpyHostToDevice);\n"
    file_content += "    cudaMemcpy(dA_columns, this->InTensor->ACsr->i,\n"
    file_content += "               this->InTensor->ACsr->nnz * sizeof(int), cudaMemcpyHostToDevice);\n"
    file_content += "    cudaMemcpy(dX, this->InTensor->Bx,\n"
    file_content += "               this->InTensor->ACsr->n * InTensor->Z * sizeof(float),\n"
    file_content += "               cudaMemcpyHostToDevice);\n"
    file_content += "    cudaMemcpy(dY, OutTensor->Dx,\n"
    file_content += "               this->InTensor->ACsr->m * this->InTensor->Z * sizeof(float),\n"
    file_content += "               cudaMemcpyHostToDevice);\n\n"

    file_content += "    cusparseCreate(&handle);\n"
    file_content += "    cusparseCreateCsr(&matA, this->InTensor->ACsr->m, this->InTensor->ACsr->n,\n"
    file_content += "                      this->InTensor->ACsr->nnz, dA_csrOffsets, dA_columns,\n"
    file_content += "                      dA_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,\n"
    file_content += "                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);\n"
    file_content += "    cusparseCreateDnMat(&matX, this->InTensor->ACsr->n, this->InTensor->Z,\n"
    file_content += "                        this->InTensor->Z, dX, CUDA_R_32F, CUSPARSE_ORDER_ROW);\n"
    file_content += "    cusparseCreateDnMat(&matY, this->InTensor->ACsr->m, this->InTensor->Z,\n"
    file_content += "                        this->InTensor->Z, dY, CUDA_R_32F, CUSPARSE_ORDER_ROW);\n"
    file_content += "    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,\n"
    file_content += "                            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,\n"
    file_content += "                            matX, &beta, matY, CUDA_R_32F,\n"
    file_content += "                            CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);\n"
    file_content += "    cudaMalloc(&dBuffer, bufferSize);\n"
    file_content += "  };\n"
    file_content += "  ~SpMMcuSparse() {\n\n"

    file_content += "    free(x);\n"
    file_content += "    cusparseDestroySpMat(matA);\n"
    file_content += "    cusparseDestroyDnMat(matX);\n"
    file_content += "    cusparseDestroyDnMat(matY);\n"
    file_content += "    cusparseDestroy(handle);\n"
    file_content += "    cudaFree(dBuffer);\n"
    file_content += "    cudaFree(dA_csrOffsets);\n"
    file_content += "    cudaFree(dA_columns);\n"
    file_content += "    cudaFree(dA_values);\n"
    file_content += "    cudaFree(dX);\n"
    file_content += "    cudaFree(dY);\n"
    file_content += "  }\n"
    file_content += "};\n"

    file_content += "class SpMMcuBlas : public SpMMSerialFloat {\n\n"
    file_content += "  float *x;\n"
    file_content += "  float *dX, *dY, *dA_dense;\n\n"
    file_content += "  float *temp_c;\n"
    file_content += "  float alpha = 1.0f;\n"
    file_content += "  float beta = 0.0f;\n\n"
    file_content += "  // cuBLAS handle\n"
    file_content += "  cublasHandle_t cublasHandle = NULL;\n\n"

    file_content += "  bool verify(double &Error) override {\n"
    file_content += "    bool retValue = true;\n"
    file_content += "    if (!InTensor->IsSolProvided) {\n"
    file_content += "      Error = 0;\n"
    file_content += "      return true;\n"
    file_content += "    }\n"
    file_content += "    double infNorm = 0;\n"
    file_content += "    for (int i = 0; i < InTensor->M * InTensor->Z - 1; ++i) {\n"
    file_content += "      if (std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]) > infNorm) {\n"
    file_content += "        infNorm = std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]);\n"
    file_content += "      }\n"
    file_content += "    }\n"
    file_content += "    Error = (double)infNorm;\n"
    file_content += "    if (infNorm > InTensor->Threshold) {\n"
    file_content += "      retValue = false;\n"
    file_content += "    }\n"
    file_content += "    return retValue;\n"
    file_content += "  }\n\n"

    file_content += "  swiftware::benchmark::Timer execute() override {\n\n"
    file_content += "    OutTensor->reset();\n"
    file_content += "    swiftware::benchmark::Timer t1;\n"
    file_content += "    t1.startGPU();\n\n"

    file_content += "    // cuBLAS dense matrix multiplication\n"
    file_content += "    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,\n"
    file_content += "                this->InTensor->M,  // Rows of A\n"
    file_content += "                this->InTensor->Z,        // Columns of B (and Y)\n"
    file_content += "                this->InTensor->N,  // Columns of A (and rows of B)\n"
    file_content += "                &alpha, dA_dense, this->InTensor->M,  // A is now dense\n"
    file_content += "                dX, this->InTensor->N,                // B\n"
    file_content += "                &beta, dY, this->InTensor->M);        // Y (output)\n\n"

    file_content += "    t1.stopGPU(\"\");\n\n"

    file_content += "    cudaMemcpy(temp_c, dY,\n"
    file_content += "               this->InTensor->ACsr->m * InTensor->Z * sizeof(float),\n"
    file_content += "               cudaMemcpyDeviceToHost);\n\n"

    file_content += "    // Store transpose of temp_c to OutTensor->Dx\n"
    file_content += "    for (int i = 0; i < this->InTensor->ACsr->m; i++) {\n"
    file_content += "      for (int j = 0; j < this->InTensor->Z; j++) {\n"
    file_content += "        OutTensor->Dx[i * this->InTensor->Z + j] = temp_c[j * this->InTensor->M + i];\n"
    file_content += "      }\n"
    file_content += "    }\n\n"

    file_content += "    return t1;\n"
    file_content += "  }\n\n"

    file_content += "public:\n"
    file_content += "  SpMMcuBlas(TensorInputsFloat *In1, swiftware::benchmark::Stats *Stat1)\n"
    file_content += "      : SpMMSerialFloat(In1, Stat1) {\n"
    file_content += "    x = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));\n"
    file_content += "    for (int i = 0; i < In1->ACsr->nnz; i++) {\n"
    file_content += "      x[i] = static_cast<float>(In1->ACsr->x[i]);\n"
    file_content += "    }\n\n"

    file_content += "    temp_c = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->M * In1->Z));\n"
    file_content += "    // Allocate memory on GPU\n"
    file_content += "    cudaMalloc((void **)&dX, this->InTensor->ACsr->n * InTensor->Z * sizeof(float));\n\n"

    file_content += "    cudaMalloc((void **)&dY, this->InTensor->ACsr->m * this->InTensor->Z * sizeof(float));\n\n"

    file_content += "    // Allocate and copy B (X) matrix\n"
    file_content += "    cudaMemcpy(dX, this->InTensor->Bx,\n"
    file_content += "               this->InTensor->ACsr->n * InTensor->Z * sizeof(float),\n"
    file_content += "               cudaMemcpyHostToDevice);\n\n"

    file_content += "    cudaMemcpy(dY, OutTensor->Dx,\n"
    file_content += "               this->InTensor->ACsr->m * this->InTensor->Z * sizeof(float),\n"
    file_content += "               cudaMemcpyHostToDevice);\n\n"

    file_content += "    // Convert CSR to dense matrix on CPU\n"
    file_content += "    int rows = this->InTensor->M;\n"
    file_content += "    int cols = this->InTensor->N;\n"
    file_content += "    float *A_dense_cpu = static_cast<float *>(aligned_alloc(32, rows * cols * sizeof(float)));\n\n"

    file_content += "    for (int row = 0; row < rows; ++row) {\n"
    file_content += "      for (int idx = In1->ACsr->p[row]; idx < In1->ACsr->p[row + 1]; ++idx) {\n"
    file_content += "        int col = In1->ACsr->i[idx];\n"
    file_content += "        A_dense_cpu [col * rows + row] = x[idx];\n"
    file_content += "      }\n"
    file_content += "    }\n\n"

    file_content += "    float *B_trap = static_cast<float *>(aligned_alloc(32, this->InTensor->ACsr->n * InTensor->Z * sizeof(float)));\n\n"
    file_content += "    // Store transpose B\n"
    file_content += "    for (int i = 0; i < this->InTensor->ACsr->n; i++) {\n"
    file_content += "      for (int j = 0; j < this->InTensor->Z; j++) {\n"
    file_content += "        B_trap[j * this->InTensor->ACsr->n + i] = this->InTensor->Bx[i * this->InTensor->Z + j];\n"
    file_content += "      }\n"
    file_content += "    }\n\n"

    file_content += "    cudaMemcpy(dX, B_trap,\n"
    file_content += "               this->InTensor->ACsr->n * InTensor->Z * sizeof(float),\n"
    file_content += "               cudaMemcpyHostToDevice);\n\n"

    file_content += "    // Allocate and copy A (dense matrix) to GPU\n"
    file_content += "    cudaMalloc((void **)&dA_dense, rows * cols * sizeof(float));\n"
    file_content += "    cudaMemcpy(dA_dense, A_dense_cpu, rows * cols * sizeof(float), cudaMemcpyHostToDevice);\n\n"

    file_content += "    // Free CPU dense matrix memory\n"
    file_content += "    free(A_dense_cpu);\n"
    file_content += "    free(B_trap);\n"
    file_content += "    // cuBLAS handle creation\n"
    file_content += "    cublasCreate(&cublasHandle);\n"
    file_content += "  }\n\n"

    file_content += "  ~SpMMcuBlas() {\n"
    file_content += "    free(x);\n"
    file_content += "    cublasDestroy(cublasHandle);\n"
    file_content += "    cudaFree(dA_dense);\n"
    file_content += "    cudaFree(dX);\n"
    file_content += "    cudaFree(dY);\n"
    file_content += " }\n "
    file_content += "};\n"

    file_content += '''
    class SpMMcuBlasHalf : public SpMMSerialFloat {
    __half *x;
    __half *dX, *dY, *dA_dense;
    const float alpha = 1.0;
    const float beta = 0.0;

    cublasGemmAlgo_t CuBlasALG = static_cast<cublasGemmAlgo_t>(0);


    float *temp_c; // Keep as float for CPU operations
    const float alpha_f = 1.0f;
    const float beta_f = 0.0f;
    int m,n,k;

    // cuBLAS handle
    cublasHandle_t cublasHandle = NULL;

     bool verify(double &Error) override {
        bool retValue = true;
        if (!InTensor->IsSolProvided) {
          Error = 0;
          return true;
        }
    
        double infNorm = 0;
        for (int i = 0; i < InTensor->M * InTensor->Z - 1; ++i) {
          if (std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]) > infNorm) {
            infNorm = std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]);
          }
        }
        Error = (double)infNorm;
        if (infNorm > InTensor->Threshold) {
          retValue = false;
        }
        return retValue;
     }
    
     swiftware::benchmark::Timer execute() override {
        OutTensor->reset();
        swiftware::benchmark::Timer t1;
        
        cudaDeviceSynchronize();
        //    cudaEventRecord(start);
        t1.startGPU();
    
        cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                       dA_dense, CUDA_R_16F, k, dX, CUDA_R_16F, k, &beta, dY,
                       CUDA_R_16F, m, CUDA_R_32F, CuBlasALG);
    
    
        t1.stopGPU("");
    
        return t1;
     }
    
     public:
     SpMMcuBlasHalf(TensorInputsFloat *In1, swiftware::benchmark::Stats *Stat1)
         : SpMMSerialFloat(In1, Stat1) {
        m = this->InTensor->M;
        n = this->InTensor->Z;
        k = this->InTensor->N;
        __half *A_dense_cpu =
            static_cast<__half *>(aligned_alloc(32, m * k * sizeof(__half)));
    
        for (int row = 0; row < m; ++row) {
          for (int idx = InTensor->ACsr->p[row]; idx < InTensor->ACsr->p[row + 1]; ++idx) {
            int col = InTensor->ACsr->i[idx];
            A_dense_cpu[col * m + row] = __float2half_rn(InTensor->ACsr->x[idx]);
          }
        }
        // Convert B to half precision
        __half *B_trap = static_cast<__half *>(aligned_alloc(32, k * n * sizeof(__half)));
        for (int i = 0; i < k; i++) {
          for (int j = 0; j < n; j++) {
            B_trap[j * k + i] = __float2half_rn(InTensor->Bx[i * n + j]);
          }
        }
    
        // Allocate and copy A (dense matrix) to GPU
        cudaMalloc((void **)&dA_dense, m * k * sizeof(__half));
        cudaMemcpy(dA_dense, A_dense_cpu, m * k * sizeof(__half),
                   cudaMemcpyHostToDevice);
        // Allocate and copy B (X) matrix
        cudaMalloc((void **)&dX, k * n * sizeof(__half));
        cudaMemcpy(dX, B_trap, k * n * sizeof(__half), cudaMemcpyHostToDevice);
    
        // Allocate and copy Y (output) matrix
        cudaMalloc((void **)&dY, m * n * sizeof(__half));
        cudaMemset(dY, 0, m * n * sizeof(__half));
    
    
    
        cublasCreate(&cublasHandle);
        cublasSetStream(cublasHandle, 0);
        cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
        // Set cuBLAS math mode to use Tensor Cores
    
     }
    
    
     ~SpMMcuBlasHalf() {
        //    free(x);
        cublasDestroy(cublasHandle);
        cudaFree(dA_dense);
        cudaFree(dX);
        cudaFree(dY);
     }
     };
 '''

    file_content += '''
    class SpMMcuBlasGemmX : public SpMMSerialFloat {

      float *x;
      float *dX, *dY, *dA_dense;
    
      float *temp_c;
      float alpha = 1.0f;
      float beta = 0.0f;
    
      // cuBLAS handle
      cublasHandle_t cublasHandle = NULL;
    
      bool verify(double &Error) override {
        bool retValue = true;
        if (!InTensor->IsSolProvided) {
          Error = 0;
          return true;
        }
        double infNorm = 0;
        for (int i = 0; i < InTensor->M * InTensor->Z - 1; ++i) {
          if (std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]) > infNorm) {
            infNorm = std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]);
          }
        }
        Error = (double)infNorm;
        if (infNorm > InTensor->Threshold) {
          retValue = false;
        }
        return retValue;
      }
    
      swiftware::benchmark::Timer execute() override {
    
        OutTensor->reset();
        swiftware::benchmark::Timer t1;
        t1.startGPU();
    
    
        // First set the math mode to disable Tensor Cores
        cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH);
    
        // Then perform the GEMM operation
        cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                     this->InTensor->M,    // Rows of A
                     this->InTensor->Z,    // Columns of B (and Y)
                     this->InTensor->N,    // Columns of A (and rows of B)
                     &alpha,
                     dA_dense,             // A matrix (now dense)
                     CUDA_R_32F,           // Data type of A
                     this->InTensor->M,    // Leading dimension of A
                     dX,                   // B matrix
                     CUDA_R_32F,           // Data type of B
                     this->InTensor->N,    // Leading dimension of B
                     &beta,
                     dY,                   // Output matrix Y
                     CUDA_R_32F,           // Data type of C
                     this->InTensor->M,    // Leading dimension of C
                     CUDA_R_32F,           // Computation type
                     CUBLAS_GEMM_DEFAULT); // Algorithm can now be default
    
        t1.stopGPU("");
    
        cudaMemcpy(temp_c, dY,
                   this->InTensor->ACsr->m * InTensor->Z * sizeof(float),
                   cudaMemcpyDeviceToHost);
    
        // Store transpose of temp_c to OutTensor->Dx
        for (int i = 0; i < this->InTensor->ACsr->m; i++) {
          for (int j = 0; j < this->InTensor->Z; j++) {
            OutTensor->Dx[i * this->InTensor->Z + j] =
                temp_c[j * this->InTensor->M + i];
          }
        }
    
        return t1;
      }
    
    public:
      SpMMcuBlasGemmX(TensorInputsFloat *In1, swiftware::benchmark::Stats *Stat1)
          : SpMMSerialFloat(In1, Stat1) {
        x = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));
        for (int i = 0; i < In1->ACsr->nnz; i++) {
          x[i] = static_cast<float>(In1->ACsr->x[i]);
        }
    
        temp_c = static_cast<float *>(
            aligned_alloc(32, sizeof(float) * In1->M * In1->Z));
        // Allocate memory on GPU
        cudaMalloc((void **)&dX,
                   this->InTensor->ACsr->n * InTensor->Z * sizeof(float));
    
        cudaMalloc((void **)&dY,
                   this->InTensor->ACsr->m * this->InTensor->Z * sizeof(float));
    
        // Allocate and copy B (X) matrix
        cudaMemcpy(dX, this->InTensor->Bx,
                   this->InTensor->ACsr->n * InTensor->Z * sizeof(float),
                   cudaMemcpyHostToDevice);
    
        cudaMemcpy(dY, OutTensor->Dx,
                   this->InTensor->ACsr->m * this->InTensor->Z * sizeof(float),
                   cudaMemcpyHostToDevice);
    
        // Convert CSR to dense matrix on CPU
        int rows = this->InTensor->M;
        int cols = this->InTensor->N;
        float *A_dense_cpu =
            static_cast<float *>(aligned_alloc(32, rows * cols * sizeof(float)));
    
        for (int row = 0; row < rows; ++row) {
          for (int idx = In1->ACsr->p[row]; idx < In1->ACsr->p[row + 1]; ++idx) {
            int col = In1->ACsr->i[idx];
            A_dense_cpu[col * rows + row] = x[idx];
          }
        }
    
        float *B_trap = static_cast<float *>(aligned_alloc(
            32, this->InTensor->ACsr->n * InTensor->Z * sizeof(float)));
    
        // Store transpose B
        for (int i = 0; i < this->InTensor->ACsr->n; i++) {
          for (int j = 0; j < this->InTensor->Z; j++) {
            B_trap[j * this->InTensor->ACsr->n + i] =
                this->InTensor->Bx[i * this->InTensor->Z + j];
          }
        }
    
        cudaMemcpy(dX, B_trap,
                   this->InTensor->ACsr->n * InTensor->Z * sizeof(float),
                   cudaMemcpyHostToDevice);
    
        // Allocate and copy A (dense matrix) to GPU
        cudaMalloc((void **)&dA_dense, rows * cols * sizeof(float));
        cudaMemcpy(dA_dense, A_dense_cpu, rows * cols * sizeof(float),
                   cudaMemcpyHostToDevice);
    
        // Free CPU dense matrix memory
        free(A_dense_cpu);
        free(B_trap);
        // cuBLAS handle creation
        cublasCreate(&cublasHandle);
      }
    
      ~SpMMcuBlasGemmX() {
        free(x);
        cublasDestroy(cublasHandle);
        cudaFree(dA_dense);
        cudaFree(dX);
        cudaFree(dY);
      }
    };

    '''

    file_content += '''
    class SpMMcuBlasTF32 : public SpMMSerialFloat {
        float *x;
        float *dX, *dY, *dA_dense;  // Changed from __half to float
        const float alpha = 1.0f;
        const float beta = 0.0f;
    
        cublasGemmAlgo_t CuBlasALG = CUBLAS_GEMM_DEFAULT_TENSOR_OP;  // Changed for TF32 tensor ops
    
        float *temp_c;
        int m, n, k;
    
        // cuBLAS handle
        cublasHandle_t cublasHandle = NULL;
    
        bool verify(double &Error) override {
            bool retValue = true;
            if (!InTensor->IsSolProvided) {
                Error = 0;
                return true;
            }
    
            double infNorm = 0;
            for (int i = 0; i < InTensor->M * InTensor->Z - 1; ++i) {
                if (std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]) > infNorm) {
                    infNorm = std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]);
                }
            }
            Error = (double)infNorm;
            if (infNorm > InTensor->Threshold) {
                retValue = false;
            }
            return retValue;
        }
    
        swiftware::benchmark::Timer execute() override {
            OutTensor->reset();
            swiftware::benchmark::Timer t1;
            
            cudaDeviceSynchronize();
            t1.startGPU();
    
            // Using GemmEx with TF32
            cublasGemmEx(cublasHandle, 
                         CUBLAS_OP_T, CUBLAS_OP_N, 
                         m, n, k, 
                         &alpha,
                         dA_dense, CUDA_R_32F, k,    // Changed from CUDA_R_16F to CUDA_R_32F
                         dX, CUDA_R_32F, k,          // Changed from CUDA_R_16F to CUDA_R_32F
                         &beta, 
                         dY, CUDA_R_32F, m,          // Changed from CUDA_R_16F to CUDA_R_32F
                         CUDA_R_32F,                 // Computation type
                         CuBlasALG);                // Using tensor core algorithm
    
            t1.stopGPU("");
            return t1;
        }
    
    public:
        SpMMcuBlasTF32(TensorInputsFloat *In1, swiftware::benchmark::Stats *Stat1)
            : SpMMSerialFloat(In1, Stat1) {
            m = this->InTensor->M;
            n = this->InTensor->Z;
            k = this->InTensor->N;
    
            // Allocate dense matrix in CPU memory
            float *A_dense_cpu = static_cast<float *>(aligned_alloc(32, m * k * sizeof(float)));
            memset(A_dense_cpu, 0, m * k * sizeof(float));  // Initialize to zero
    
            // Convert sparse to dense format
            for (int row = 0; row < m; ++row) {
                for (int idx = InTensor->ACsr->p[row]; idx < InTensor->ACsr->p[row + 1]; ++idx) {
                    int col = InTensor->ACsr->i[idx];
                    A_dense_cpu[col * m + row] = InTensor->ACsr->x[idx];  // Direct float assignment
                }
            }
    
            // Prepare B matrix
            float *B_trap = static_cast<float *>(aligned_alloc(32, k * n * sizeof(float)));
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < n; j++) {
                    B_trap[j * k + i] = InTensor->Bx[i * n + j];  // Direct float assignment
                }
            }
    
            // Allocate and copy matrices to GPU
            cudaMalloc((void **)&dA_dense, m * k * sizeof(float));
            cudaMemcpy(dA_dense, A_dense_cpu, m * k * sizeof(float),
                       cudaMemcpyHostToDevice);
    
            cudaMalloc((void **)&dX, k * n * sizeof(float));
            cudaMemcpy(dX, B_trap, k * n * sizeof(float), 
                       cudaMemcpyHostToDevice);
    
            cudaMalloc((void **)&dY, m * n * sizeof(float));
            cudaMemset(dY, 0, m * n * sizeof(float));
    
            // Initialize cuBLAS and enable TF32 Tensor Cores
            cublasCreate(&cublasHandle);
            cublasSetStream(cublasHandle, 0);
            cublasSetMathMode(cublasHandle, CUBLAS_TF32_TENSOR_OP_MATH);  // Changed to TF32
    
            // Free CPU memory
            free(A_dense_cpu);
            free(B_trap);
        }
    
        ~SpMMcuBlasTF32() {
            cublasDestroy(cublasHandle);
            cudaFree(dA_dense);
            cudaFree(dX);
            cudaFree(dY);
        }
    };
    '''

    file_content += codelet_text

    file_content += kernel_text

    file_content += class_text

    file_content += "#endif // COMPRESSED_TENSOR_ALGEBRA_SPMM_SMALL_B_COL_DEMO_GPU_UTILS_H\n"

    return file_content
