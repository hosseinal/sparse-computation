import os
import sys

from pattern_utiles.pattern_dictionary import PatternTypeDictionary


def class_file_generator(m, n, tbs, second_grid_dim, pattern_type_dictionary: PatternTypeDictionary):
    thread_bloc_size = tbs * 32

    file_content = ""

    file_content += f"class SpMMMixePatternGeneralCacheLess: public SpMMSerialFloat {{\n"
    file_content += "    protected:\n"
    file_content += "        float *dX, *dY;\n\n"
    file_content += "float *x;\n"
    file_content += "swiftware::compression::MixedFormatGPU<float> *mf;\n\n"
    file_content += "TensorInputsMixed *InT;\n\n"
    file_content += "bool verify(double &Error) override {\n"
    file_content += "    bool retValue = true;\n\n"
    file_content += "double infNorm = 0;\n"
    file_content += "for (int i = 0; i < InTensor->M * InTensor->Z - 4; ++i) {\n"
    file_content += "if (std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]) > infNorm) {\n"
    file_content += "    infNorm = std::abs(OutTensor->Dx[i] - InTensor->CorrectMul[i]);\n"
    file_content += "}\n"
    file_content += "}\n"
    file_content += "Error = (double)infNorm;\n"
    file_content += "if (infNorm > InTensor->Threshold) {\n"
    file_content += "    retValue = false;\n"
    file_content += "}\n"
    file_content += "return retValue;\n"
    file_content += "}\n\n"

    file_content += "swiftware::benchmark::Timer execute() override {\n\n"
    file_content += "OutTensor->reset();\n"
    file_content += "swiftware::benchmark::Timer t1;\n\n"
    file_content += "t1.startGPU();\n"
    file_content += "// lunch kernel\n\n"
    file_content += f"dim3 gridDim((InT->MixedFormat->cf[0]->nrpp - 1) * {len(pattern_type_dictionary.pattern_bin_to_id)} , {second_grid_dim});\n"
    file_content += f"SpMMFullRowPanelGeneralKernel{m}With{n}BlockCacheLess<<<gridDim,\n{thread_bloc_size}>>>(mf, dX, dY, InTensor->Z);\n\n"
    file_content += "t1.stopGPU(\"\");\n\n"
    file_content += "cudaMemcpy(this->OutTensor->Dx, dY,\n"
    file_content += "this->InTensor->ACsr->m * InTensor->Z * sizeof(float),\n"
    file_content += "cudaMemcpyDeviceToHost);\n"
    file_content += "cudaMemset(dY, 0, InTensor->M * InTensor->Z * sizeof(float));\n\n"
    file_content += "return t1;\n"
    file_content += "}\n\n"

    file_content += "public:\n"
    file_content += f"SpMMMixePatternGeneralCacheLess(TensorInputsMixed *In1,\n"
    file_content += "swiftware::benchmark::Stats *Stat1)\n"
    file_content += ": SpMMSerialFloat(In1, Stat1), InT(In1) {\n\n"
    file_content += "x = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));\n"
    file_content += "for (int i = 0; i < In1->ACsr->nnz; i++) {\n"
    file_content += "    x[i] = static_cast<float>(In1->ACsr->x[i]);\n"
    file_content += "}\n\n"
    file_content += "cudaMalloc((void **)&dX, InTensor->N * InTensor->Z * sizeof(float));\n"
    file_content += "cudaMalloc((void **)&dY, InTensor->M * InTensor->Z * sizeof(float));\n"
    file_content += "//\n"
    file_content += "cudaMemset(dY, 0, InTensor->M * InTensor->Z * sizeof(float));\n"
    file_content += "cudaMemcpy(dX, this->InTensor->Bx,\n"
    file_content += "InTensor->N * InTensor->Z * sizeof(float),\n"
    file_content += "cudaMemcpyHostToDevice);\n\n"
    file_content += "mf = allocateAndCopyMixedFormat(In1->MixedFormat);\n"
    file_content += "};\n\n"
    file_content += f"~SpMMMixePatternGeneralCacheLess() {{\n"
    file_content += "//    cudaFree(dX);\n"
    file_content += "//    cudaFree(dY);\n"
    file_content += "}\n"
    file_content += "};\n"

    return file_content


def kernel_generator_nnz_base(m, n, pattern_type_dictionary: PatternTypeDictionary):
    file_content = ""

    file_content += "__global__ void\n"
    file_content += f"SpMMFullRowPanelGeneralKernel{m}With{n}BlockCacheLess(swiftware::compression::MixedFormatGPU<float> *mf,"
    file_content += " float *dX, float *dY, int c_cols) {\n"
    file_content += "  int i = blockIdx.x;\n"
    number_of_patterns = len(pattern_type_dictionary.pattern_bin_to_id)
    file_content += f"  int pattern = i % {number_of_patterns};\n"

    file_content += "  switch (pattern) {\n"
    for i, pattern in pattern_type_dictionary.id_to_patten_bin.items():
        nnz_number = pattern_type_dictionary.pattern_id_to_number_of_nnz(i)
        file_content += f"    case {i}:\n"
        file_content += f"      PatternWith{m}NNZByRowPanel{nnz_number}RowPerThread{n}BlockCacheLess(\n"
        file_content += f"          mf->cf[{i}]->RPP, mf->cf[{i}]->NPP, mf->cf[{i}]->NNZ, mf->cf[{i}]->CB, dX, dY,\n"
        file_content += f"          i / {number_of_patterns}, c_cols"
        for x in range(len(pattern)):
            if '1' == pattern[x]:
                file_content += f" , {x}"
        file_content += ");\n"
        file_content += "      break;\n"
    file_content += "  }\n"
    file_content += "}\n"
    file_content += "\n\n"

    file_content += "__global__ void\n"
    file_content += f"SingleSpMMFullRowPanelGeneralKernel{m}With{n}BlockCacheLess(swiftware::compression::MixedFormatGPU<float> *mf,"
    file_content += " float *dX, float *dY, int c_cols) {\n"
    file_content += "  int i = blockIdx.x;\n"
    number_of_patterns = len(pattern_type_dictionary.pattern_bin_to_id)
    file_content += f"  int pattern = i % {number_of_patterns};\n"

    pattern = pattern_type_dictionary.id_to_patten_bin[0]
    nnz_number = pattern_type_dictionary.pattern_id_to_number_of_nnz(0)
    file_content += f"  PatternWith{m}NNZByRowPanel{nnz_number}RowPerThread{n}BlockCacheLess(\n"
    file_content += f"          mf->cf[0]->RPP, mf->cf[0]->NPP, mf->cf[0]->NNZ, mf->cf[0]->CB, dX, dY,\n"
    file_content += f"          i / {number_of_patterns}, c_cols"
    for x in range(len(pattern)):
        if '1' == pattern[x]:
            file_content += f" , {x}"
    file_content += ");\n"
    file_content += "}\n"

    return file_content


def codelet_generator_nnz_base(m, n, b_col_iteration, pattern_type_dictionary: PatternTypeDictionary):
    iterator = b_col_iteration

    file_content = ""

    already_generated = []
    # part of cache less
    for key, pattern in pattern_type_dictionary.id_to_patten_bin.items():
        # i is equal to number of 1 in the pattern
        i = pattern_type_dictionary.pattern_id_to_number_of_nnz(key)
        if i in already_generated:
            continue
        already_generated.append(i)
        file_content += f"__device__ void PatternWith{m}NNZByRowPanel{i}RowPerThread{n}BlockCacheLess(\n"
        file_content += ("    int *RPP, int *NPP, float *NNZ, int *CB,"
                         " float *dx, float *dy,")
        file_content += "int i, int c_cols"
        for j in range(i):
            file_content += f", int offset{j}"

        file_content += ") {\n"

        file_content += "int blockSize = blockDim.x;\n"
        file_content += "int thread_id_in_warp = threadIdx.x % 32;\n"
        file_content += "int warp_id = threadIdx.x / 32;\n"
        file_content += "int number_of_warps = blockSize / 32;\n"

        file_content += "int rowStart_all = RPP[i];\n"
        file_content += "int rowEnd_all = RPP[i + 1];\n"
        file_content += "if (rowStart_all == rowEnd_all) {\n"
        file_content += "   return;\n"
        file_content += "}\n"

        file_content += f"int number_of_row_panel_packs = (rowEnd_all - rowStart_all + {n} - 1) / {n};\n"
        file_content += f"int row_panel_pack_share = ((number_of_row_panel_packs + gridDim.y - 1) / gridDim.y) * {n};\n"
        file_content += f"int row_panel_start = rowStart_all + blockIdx.y * row_panel_pack_share;\n"
        file_content += f"int row_panel_end = row_panel_start + row_panel_pack_share;\n"
        file_content += f"row_panel_end = row_panel_end > rowEnd_all ? rowEnd_all : row_panel_end;\n"

        file_content += "if (row_panel_start >= row_panel_end) {\n"
        file_content += "    return;\n"
        file_content += "}\n"

        file_content += f"int c_col_partition = (c_cols + {iterator * 32 - 1}) / {iterator * 32};\n"
        file_content += f"int c_col_share = ((c_col_partition + number_of_warps - 1) / number_of_warps) * {iterator * 32};\n"
        file_content += f"c_col_share = c_col_share == 0 ? {iterator * 32} : c_col_share;\n"
        file_content += "int c_col_start = c_col_share * warp_id;\n"
        file_content += "int c_col_end = c_col_start + c_col_share;\n"
        file_content += "c_col_end = c_col_end > c_cols ? c_cols : c_col_end;\n"

        file_content += "if (c_col_start >= c_col_end) {\n"
        file_content += "    return;\n"
        file_content += "}\n"

        file_content += f"for (int col = c_col_start; col < c_col_end; col += {iterator * 32}) {{\n"
        file_content += f"int t_nnz = NPP[i] + (row_panel_start - rowStart_all) * {i};\n"
        for x in range(iterator * i):
            file_content += f"float sum_{x} = 0.0f;\n"

        file_content += f"for (int j = row_panel_start; j < row_panel_end; j += {n}) {{\n"

        for x in range(n):
            file_content += f"int sub_b_row_start_{x} = CB[j + {x}];\n"

        for x in range(iterator):
            file_content += f"int sub_b_col_{x} = col + {32 * x} + thread_id_in_warp;\n"

        counter = 0
        for y in range(n):
            for x in range(i):
                file_content += f"float nnz{x}_{y} = NNZ[t_nnz + {counter}];\n"
                counter += 1

        b_counter = 0

        for xx in range(iterator):
            for x in range(n):
                file_content += f"float b{b_counter} = dx[sub_b_row_start_{x} * c_cols + sub_b_col_{xx}];\n"
                b_counter += 1

        for xx in range(iterator):
            for x in range(i):
                text = ""
                text += f"sum_{x + xx * i} += "
                for y in range(n):
                    text += f"nnz{x}_{y} * b{y + xx * n} + "
                file_content += text[:-2] + ";\n"

        file_content += f"t_nnz += {i * n};\n"
        file_content += "}\n"

        file_content += f"int c_row = i * {m};\n"
        file_content += f"int c_col = col + thread_id_in_warp;\n"

        for xx in range(iterator):
            for x in range(i):
                file_content += f"atomicAdd(&dy[(c_row + offset{x}) * c_cols + c_col + {xx * 32}], sum_{x + xx * i});\n"

        file_content += "}\n"
        file_content += "}\n"
        file_content += "\n\n"

    return file_content


def kernel_generator_pattern_base(m, n, pattern_type_dictionary: PatternTypeDictionary):
    file_content = ""
    file_content += "__global__ void\n"
    file_content += f"SpMMFullRowPanelGeneralKernel{m}With{n}Block(swiftware::compression::MixedFormatGPU<float> *mf,"
    file_content += " float *dX, float *dY, int c_cols) {\n"
    file_content += "  int i = blockIdx.x;\n"
    number_of_patterns = len(pattern_type_dictionary.pattern_bin_to_id)
    file_content += f"  int pattern = i % {number_of_patterns};\n"
    file_content += f"  __shared__ float NNZs[{m * n} * 32];\n"
    file_content += f"__shared__ float Bs[{n} * 64];\n\n"
    file_content += "  switch (pattern) {\n"
    for i, pattern in pattern_type_dictionary.id_to_patten_bin.items():
        nnz_number = pattern_type_dictionary.pattern_id_to_number_of_nnz(i)
        file_content += f"    case {i}:\n"
        file_content += f"      PatternWith{m}NNZByRowPanel{nnz_number}RowPerThread{n}BlockPattern{pattern}(\n"
        file_content += f"          mf->cf[{i}]->RPP, mf->cf[{i}]->NPP, mf->cf[{i}]->NNZ, mf->cf[{i}]->CB, dX, dY,\n"
        file_content += f"          NNZs, Bs, i / {number_of_patterns}, c_cols"
        for x in range(len(pattern)):
            if '1' == pattern[x]:
                file_content += f" , {x}"
        file_content += ");\n"
        file_content += "      break;\n"
    file_content += "  }\n"
    file_content += "}\n"
    return file_content


def codelet_generator_pattern_base(m, n, pattern_type_dictionary: PatternTypeDictionary):
    file_content = ""
    for key, value in pattern_type_dictionary.id_to_patten_bin.items():
        number_of_nnz = pattern_type_dictionary.pattern_id_to_number_of_nnz(key)
        file_content += f"__device__ void PatternWith{m}NNZByRowPanel{number_of_nnz}RowPerThread{n}BlockPattern{value}(\n"
        file_content += ("    int *RPP, int *NPP, float *NNZ, int *CB,"
                         " float *dx, float *dy, float *NNZs,")
        file_content += "float *Bs, int i, int c_cols"

        for j in range(number_of_nnz):
            file_content += f", int offset{j}"

        file_content += ") {\n"

        file_content += "int blockSize = blockDim.x;\n"
        file_content += "int thread_id_in_warp = threadIdx.x % 32;\n"
        file_content += "int warp_id = threadIdx.x / 32;\n"
        file_content += "int number_of_warps = blockSize / 32;\n"

        file_content += "int rowStart_all = RPP[i];\n"
        file_content += "int rowEnd_all = RPP[i + 1];\n"

        file_content += "int c_col_share = c_cols / number_of_warps;\n"
        file_content += "int c_col_start = c_col_share * warp_id;\n"
        file_content += "int c_col_end = c_col_start + c_col_share;\n"
        file_content += "c_col_end = c_col_end > c_cols ? c_cols : c_col_end;\n"

        file_content += "if (rowStart_all == rowEnd_all) {\n"
        file_content += "   return;\n"
        file_content += "}\n"

        file_content += "for (int col = c_col_start; col < c_col_end; col += 64) {\n"
        file_content += "int t_nnz = NPP[i];\n"
        for x in range(2 * number_of_nnz):
            file_content += f"float sum_{x} = 0.0f;\n"

        file_content += f"for (int j = rowStart_all; j < rowEnd_all; j += {n}) {{\n"

        file_content += f"for (int x = 0; x < {number_of_nnz * n} ; x++) {{\n"
        file_content += f"  NNZs[thread_id_in_warp + x * 32] = NNZ[t_nnz + x];\n"
        file_content += "}\n"

        for x in range(n):
            file_content += f"int sub_b_row_start_{x} = CB[j + {x}];\n"

        file_content += f"int sub_b_col = col + thread_id_in_warp;\n"
        file_content += f'int sub_b_col_1 = col + 32 + thread_id_in_warp;\n'

        for x in range(n):
            file_content += f"Bs[thread_id_in_warp + {x * 32}] = dx[sub_b_row_start_{x} * c_cols + sub_b_col];\n"

        for x in range(n):
            file_content += f"Bs[thread_id_in_warp + {(n + x) * 32}] = dx[sub_b_row_start_{x} * c_cols + sub_b_col_1]; \n"

        counter = 0
        for y in range(n):
            for x in range(number_of_nnz):
                file_content += f"float nnz{x}_{y} = NNZs[{counter} * 32 + thread_id_in_warp];\n"
                counter += 1

        b_counter = 0

        for x in range(n):
            file_content += f"float b{b_counter} = Bs[thread_id_in_warp + {b_counter * 32}];\n"
            b_counter += 1

        for x in range(n):
            file_content += f"float b{b_counter} = Bs[thread_id_in_warp + {b_counter * 32}];\n"
            b_counter += 1

        for x in range(number_of_nnz):
            text = ""
            text += f"sum_{x} += "
            for y in range(n):
                text += f"nnz{x}_{y} * b{y} + "
            file_content += text[:-2] + ";\n"

        for x in range(number_of_nnz):
            text = ""
            text += f"sum_{x + number_of_nnz} += "
            for y in range(n):
                text += f"nnz{x}_{y} * b{y + n} + "
            file_content += text[:-2] + ";\n"

        file_content += f"t_nnz += {number_of_nnz * n};\n"
        file_content += "}\n"

        file_content += f"int c_row = i * {m};\n"
        file_content += f"int c_col = col + thread_id_in_warp;\n"
        for x in range(number_of_nnz):
            file_content += f"atomicAdd(&dy[(c_row + offset{x}) * c_cols + c_col], sum_{x});\n"

        for x in range(number_of_nnz):
            file_content += f"atomicAdd(&dy[(c_row + offset{x}) * c_cols + c_col + 32], sum_{x + number_of_nnz});\n"

        file_content += "}\n"
        file_content += "}\n"
        file_content += "\n\n"

    return file_content


def base_file(codelet_text, kernel_text, class_text, m, n, pattern_type_dictionary):
    file_content = ""
    file_content += "#ifndef COMPRESSED_TENSOR_ALGEBRA_SPMM_DEMO_GPU_UTILS_H\n"
    file_content += "#define COMPRESSED_TENSOR_ALGEBRA_SPMM_DEMO_GPU_UTILS_H\n\n"
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
    file_content += "#include \"cooperative_groups.h\"\n"
    file_content += "#include <cooperative_groups/memcpy_async.h>\n"

    file_content += "#define CUDA_CHECK(err) \\\n"
    file_content += "  do { \\\n";
    file_content += "    cudaError_t err_ = (err); \\\n"
    file_content += "    if (err_ != cudaSuccess) { \\\n"
    file_content += "      std::printf(\"CUDA error %d at %s:%d\\n\", err_, __FILE__, __LINE__); \\\n"
    file_content += "      throw std::runtime_error(\"CUDA error\"); \\\n"
    file_content += "    } \\\n"
    file_content += "  } while (0)\n\n"

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

    file_content += "swiftware::compression::CompressedFormatGPU<half> *\n"
    file_content += "allocateAndCopyCompressedFormatToHalf(\n"
    file_content += "    swiftware::compression::CompressedFormat<float> *h_cf) {\n\n"
    file_content += "  swiftware::compression::CompressedFormatGPU<half> *d_cf;\n"
    file_content += "  cudaMalloc((void **)&d_cf,\n"
    file_content += "             sizeof(swiftware::compression::CompressedFormatGPU<half>));\n\n"
    file_content += "  // Allocate and convert arrays from float to half before copying\n"
    file_content += "  half *d_NNZ;\n"
    file_content += "  int *d_RPP, *d_NPP;\n"
    file_content += "  int *d_PT, *d_CB;\n\n"
    file_content += "  // Allocate temporary array to hold half-precision values\n"
    file_content += "  half *temp_NNZ = new half[h_cf->nnz];\n"
    file_content += "  for (size_t i = 0; i < h_cf->nnz; ++i) {\n"
    file_content += "    temp_NNZ[i] = __float2half(h_cf->NNZ[i]);\n"
    file_content += "  }\n\n"
    file_content += "  // Copy the converted half-precision array to the device\n"
    file_content += "  cudaMalloc((void **)&d_NNZ, h_cf->nnz * sizeof(half));\n"
    file_content += "  cudaMemcpy(d_NNZ, temp_NNZ, h_cf->nnz * sizeof(half), cudaMemcpyHostToDevice);\n"
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
    file_content += "      cudaMemcpy(&(d_cf->NNZ), &d_NNZ, sizeof(half *), cudaMemcpyHostToDevice));\n"
    file_content += "  CUDA_CHECK(\n"
    file_content += "      cudaMemcpy(&(d_cf->RPP), &d_RPP, sizeof(int *), cudaMemcpyHostToDevice));\n"
    file_content += "  cudaMemcpy(&(d_cf->NPP), &d_NPP, sizeof(int *), cudaMemcpyHostToDevice);\n"
    file_content += "  cudaMemcpy(&(d_cf->PT), &d_PT, sizeof(int *), cudaMemcpyHostToDevice);\n"
    file_content += "  cudaMemcpy(&(d_cf->CB), &d_CB, sizeof(int *), cudaMemcpyHostToDevice);\n\n"
    file_content += "  return d_cf;\n"
    file_content += "}\n\n"
    file_content += "swiftware::compression::MixedFormatGPU<half> *allocateAndCopyMixedFormatToHalf(\n"
    file_content += "    swiftware::compression::MixedFormat<float> *h_mixed) {\n\n"
    file_content += "  swiftware::compression::MixedFormatGPU<half> *d_mixed;\n"
    file_content += "  cudaMalloc((void **)&d_mixed,\n"
    file_content += "             sizeof(swiftware::compression::MixedFormatGPU<half> *));\n\n"
    file_content += "  // Allocate and copy the array of CompressedFormat pointers\n"
    file_content += "  swiftware::compression::CompressedFormatGPU<half> **d_cf;\n"
    file_content += "  cudaMalloc((void **)&d_cf,\n"
    file_content += "             h_mixed->cf.size() *\n"
    file_content += "                 sizeof(swiftware::compression::CompressedFormatGPU<half> *));\n\n"
    file_content += "  for (size_t i = 0; i < h_mixed->cf.size(); ++i) {\n"
    file_content += "    swiftware::compression::CompressedFormatGPU<half> *d_cf_element =\n"
    file_content += "        allocateAndCopyCompressedFormatToHalf(h_mixed->cf[i]);\n"
    file_content += "    cudaMemcpy(&(d_cf[i]), &d_cf_element,\n"
    file_content += "               sizeof(swiftware::compression::CompressedFormatGPU<half> *),\n"
    file_content += "               cudaMemcpyHostToDevice);\n"
    file_content += "  }\n\n"
    file_content += "  cudaMemcpy(&(d_mixed->cf), &d_cf,\n"
    file_content += "             sizeof(swiftware::compression::CompressedFormatGPU<half> *),\n"
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
        half *A_dense_cpu =
            static_cast<half *>(aligned_alloc(32, m * k * sizeof(half)));
    
        for (int row = 0; row < m; ++row) {
          for (int idx = InTensor->ACsr->p[row]; idx < InTensor->ACsr->p[row + 1]; ++idx) {
            int col = InTensor->ACsr->i[idx];
            A_dense_cpu[col * m + row] = __float2half_rn(InTensor->ACsr->x[idx]);
          }
        }
        // Convert B to half precision
        half *B_trap = static_cast<half *>(aligned_alloc(32, k * n * sizeof(half)));
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

    file_content += '''
    // Adding the BCSR PART
    #define MMA_M 16
    #define MMA_N 8
    #define MMA_K 16
    
    #define WARP_SIZE 32
    
    #define LDMATRIX_X2(R0, R1, addr) \\
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];" : "=r"(R0), "=r"(R1) : "r"(addr))
    
    #define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \\
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" \\
                     : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \\
                     : "r"(addr))
    
    #define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)                                                    \\
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};" \\
                     : "=r"(RD0), "=r"(RD1)                                                                                \\
                     : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))
    
    inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
      return (a % b != 0) ? (a / b + 1) : (a / b);
    }
    
    __global__ void
    mmaCBTKernelSparse(half *bcsrValuesA, int *bcsrRowPtrA, int *bcsrColIdxA, half *B, half *C, size_t M, size_t N,
                       size_t K) {
      //mmaCBTKernel
      const size_t K_tiles = div_ceil(K, MMA_K);
    
      const size_t warp_row = blockIdx.y * MMA_M;
      const size_t warp_col = blockIdx.x * MMA_N;
    
      size_t blockRow = blockIdx.y;
      size_t blockCol = blockIdx.x;
    
    
      if (warp_row >= M || warp_col >= N) {
        return;
      }
    
      __shared__ half A_smem[MMA_M][MMA_K];
      __shared__ half B_smem[MMA_N][MMA_K];
      __shared__ half C_smem[MMA_M][MMA_N];
    
      const size_t lane_id = threadIdx.x % WARP_SIZE;
      auto group = cooperative_groups::this_thread_block();
    
      size_t nnz_index = bcsrRowPtrA[blockRow] * MMA_M * MMA_K;
      uint32_t RC[2] = {0, 0};
    
    #pragma unroll
      for (size_t ptr = bcsrRowPtrA[blockRow]; ptr < bcsrRowPtrA[blockRow + 1]; ptr++) {
        size_t i = bcsrColIdxA[ptr] / MMA_K;
        // skip empty block
    
        size_t A_size = MMA_M * MMA_K * sizeof(half);
        size_t B_size = MMA_N * MMA_K * sizeof(half);
    
        cooperative_groups::memcpy_async(group, &A_smem[0][0],
                                         &bcsrValuesA[nnz_index],
                                         A_size);
        cooperative_groups::memcpy_async(group, &B_smem[0][0],
                                         &B[i * MMA_K + warp_col * K],
                                         B_size);
    
        cooperative_groups::wait(group); // Wait for all copies to complete
        group.sync();
    
        uint32_t RA[4];
        uint32_t RB[2];
    
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);
    
        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);
    
        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
    
        group.sync();
        nnz_index += MMA_M * MMA_K;
      }
    
      *((uint32_t *) (&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
      *((uint32_t *) (&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];
    
      __syncthreads();
    
      if (lane_id < MMA_M) {
        *((int4 *) (&C[(warp_row + lane_id) * N + warp_col])) = *((int4 *) (&C_smem[lane_id][0]));
      }
    }
    
    __global__ void convertFp32ToFp16(half *out, float *in, int n) {
      int idx = blockDim.x * blockIdx.x + threadIdx.x;
      if (idx < n) {
        out[idx] = in[idx];
      }
    }
    
    void dense_to_bcsr(float *dense, int rows, int cols, int block_size_row, int block_size_col,
                       half **values, int **row_ptr, int **col_ind,
                       int *nnz) {
      // First pass: count non-zero elements
      int nnz_block_counter = 0;
      for (int i = 0; i < rows; i += block_size_row) {
        for (int j = 0; j < cols; j += block_size_col) {
          bool is_zero_block = true;
          for (int k = i; k < i + block_size_row; k++) {
            for (int l = j; l < j + block_size_col; l++) {
              if (dense[k * cols + l] != 0.0f) {
                is_zero_block = false;
                break;
              }
            }
            if (!is_zero_block) {
              break;
            }
          }
          if (!is_zero_block) {
            nnz_block_counter++;
          }
        }
      }
    
      int nnz_counter = nnz_block_counter * block_size_row * block_size_col;
    
      // Allocate memory
      *values = (half *) malloc(nnz_counter * sizeof(half));
      *col_ind = (int *) malloc(nnz_block_counter * sizeof(int));
      *row_ptr = (int *) malloc((rows / block_size_row + 1) * sizeof(int));
    
      // Second pass: fill CSR arrays
      int b_count = 0;
      nnz_counter = 0;
      (*row_ptr)[0] = 0;
    
      for (int i = 0; i < rows; i += block_size_row) {
        for (int j = 0; j < cols; j += block_size_col) {
          bool is_zero_block = true;
          for (int k = i; k < i + block_size_row; k++) {
            for (int l = j; l < j + block_size_col; l++) {
              if (dense[k * cols + l] != 0.0f) {
                is_zero_block = false;
                break;
              }
            }
            if (!is_zero_block) {
              break;
            }
          }
          if (!is_zero_block) {
            (*col_ind)[b_count] = j;
            for (int k = i; k < i + block_size_row; k++) {
              for (int l = j; l < j + block_size_col; l++) {
                (*values)[nnz_counter] = dense[k * cols + l];
                nnz_counter++;
              }
            }
            b_count++;
          }
        }
        (*row_ptr)[i / block_size_row + 1] = b_count;
      }
    
      //set the value of nnz
      *nnz = nnz_counter;
    
    }
    
    class SpMMTensorCore : public SpMMSerialFloat {
    
      float *x;
      float *dX, *dY;
      half *dY_half, *dX_half;
    
      float alpha = 1.0f;
      float beta = 0.0f;
    
      half *h_values;
      int *h_row_ptr, *h_col_ind;
      int nnz;
    
    
      half *d_values;
      int *d_row_ptr, *d_col_ind;
      // cuBLAS handle
    
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
        dim3 block(WARP_SIZE);
    
        dim3 grid(div_ceil(InTensor->N, MMA_N), div_ceil(InTensor->M, MMA_M));
    
        mmaCBTKernelSparse<<<grid, block>>>(d_values, d_row_ptr, d_col_ind, dX_half, dY_half, InTensor->M, InTensor->Z, InTensor->N);
    
        t1.stopGPU("");
    
        // Copy the result into a hald array then cast that to the real array
        half *h_Y = (half *) malloc(InTensor->M * InTensor->Z * sizeof(half));
        cudaMemcpy(h_Y, dY_half, InTensor->M * InTensor->Z * sizeof(half), cudaMemcpyDeviceToHost);
        for (int i = 0; i < InTensor->M * InTensor->Z; i++) {
          OutTensor->Dx[i] = __half2float(h_Y[i]);
        }
        
        return t1;
      }
    
    public:
      SpMMTensorCore(TensorInputsFloat *In1, swiftware::benchmark::Stats *Stat1)
          : SpMMSerialFloat(In1, Stat1) {
        x = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));
        for (int i = 0; i < In1->ACsr->nnz; i++) {
          x[i] = static_cast<float>(In1->ACsr->x[i]);
        }
    
        // Allocate memory on GPU
        cudaMalloc((void **)&dX,
                   this->InTensor->ACsr->n * InTensor->Z * sizeof(float));
    
        cudaMalloc((void **)&dY,
                   this->InTensor->ACsr->m * this->InTensor->Z * sizeof(float));
    
        // Allocate the memory for the half precision
        cudaMalloc((void **)&dY_half,
                   this->InTensor->ACsr->m * this->InTensor->Z * sizeof(half));
    
        cudaMalloc((void **)&dX_half,
                    this->InTensor->ACsr->n * InTensor->Z * sizeof(half));
    
        // Allocate and copy B (X) matrix
        cudaMemcpy(dX, this->InTensor->Bx,
                   this->InTensor->ACsr->n * InTensor->Z * sizeof(float),
                   cudaMemcpyHostToDevice);
    
        cudaMemset(dY, 0, this->InTensor->ACsr->m * this->InTensor->Z * sizeof(float));
    
        // Convert float dX to half dX_half
        convertFp32ToFp16<<<(this->InTensor->ACsr->n * InTensor->Z + 255) / 256, 256>>>(dX_half, dX, this->InTensor->ACsr->n * InTensor->Z);
    
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
    
        dense_to_bcsr(A_dense_cpu, rows, cols, 16, 8, &h_values, &h_row_ptr, &h_col_ind, &nnz);
    
        cudaMalloc((void **) &d_values, nnz * sizeof(half));
        cudaMalloc((void **) &d_row_ptr, (rows / 16 + 1) * sizeof(int));
        cudaMalloc((void **) &d_col_ind, nnz / (16 * 8) * sizeof(int));
    
        cudaMemcpy(d_values, h_values, nnz * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row_ptr, h_row_ptr, (rows / 16 + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_ind, h_col_ind, nnz / (16 * 8) * sizeof(int), cudaMemcpyHostToDevice);
    
        // convert b to half
    
    
        // Free CPU dense matrix memory
        free(A_dense_cpu);
        // cuBLAS handle creation
      }
    
      ~SpMMTensorCore() {
        free(x);
        cudaFree(dX);
        cudaFree(dY);
      }
    };
    '''

    file_content += codelet_text

    file_content += kernel_text

    file_content += class_text

    file_content += "#endif // COMPRESSED_TENSOR_ALGEBRA_SPMM_DEMO_GPU_UTILS_H\n"

    return file_content
