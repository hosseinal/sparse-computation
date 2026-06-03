import os
import sys
from pattern_utiles.pattern_dictionary import PatternTypeDictionary


def dense_class_file_generator(m, n, tbs):
    thread_bloc_size = tbs * 32

    file_content = ""

    file_content += f"class SpMMMixePatternGeneralCacheLess: public SpMMSerialFloat {{\n"
    file_content += "    protected:\n"
    file_content += "        float *dX, *dY;\n\n"
    file_content += "    float *x;\n"
    file_content += "    float *denseA;\n"
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
    file_content += f"SpMMFullRowPanelGeneralKernel{m}With{n}BlockCacheLess<<<(InT->MixedFormat->cf[0]->nrpp - 1) * {(2 ** m) - 1},\n{thread_bloc_size}>>>(mf, dX, dY,denseA, InTensor->Z, InTensor->N);\n\n"
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
    file_content += "cudaMalloc((void**)&denseA, InTensor->M * InTensor->N * sizeof(float));\n"
    file_content += "cudaMalloc((void **)&dX, InTensor->N * InTensor->Z * sizeof(float));\n"
    file_content += "cudaMalloc((void **)&dY, InTensor->M * InTensor->Z * sizeof(float));\n"
    file_content += "//\n"
    file_content += "cudaMemset(dY, 0, InTensor->M * InTensor->Z * sizeof(float));\n"
    file_content += "cudaMemcpy(dX, this->InTensor->Bx,\n"
    file_content += "InTensor->N * InTensor->Z * sizeof(float),\n"
    file_content += "cudaMemcpyHostToDevice);\n\n"
    file_content += "cudaMemcpy(denseA, In1->MixedFormat->Dense, InTensor->M * InTensor->N * sizeof(float), cudaMemcpyHostToDevice);\n"
    file_content += "mf = allocateAndCopyMixedFormat(In1->MixedFormat);\n"
    file_content += "};\n\n"
    file_content += f"~SpMMMixePatternGeneralCacheLess() {{\n"
    file_content += "//    cudaFree(dX);\n"
    file_content += "//    cudaFree(dY);\n"
    file_content += "}\n"
    file_content += "};\n"

    return file_content


def dense_kernel_generator_nnz_base(m, n, pattern_type_dictionary: PatternTypeDictionary):
    file_content = ""

    file_content += "__global__ void\n"
    file_content += f"SpMMFullRowPanelGeneralKernel{m}With{n}BlockCacheLess(swiftware::compression::MixedFormatGPU<float> *mf,"
    file_content += " float *dX, float *dY, float *Dense, int c_cols , int N) {\n"
    file_content += "  int i = blockIdx.x;\n"
    number_of_patterns = len(pattern_type_dictionary.pattern_bin_to_id)
    file_content += f"  int pattern = i % {number_of_patterns};\n"

    file_content += "  switch (pattern) {\n"
    for i, pattern in pattern_type_dictionary.id_to_patten_bin.items():
        nnz_number = pattern_type_dictionary.pattern_id_to_number_of_nnz(i)
        file_content += f"    case {i}:\n"
        file_content += f"      PatternWith{m}NNZByRowPanel{nnz_number}RowPerThread{n}BlockCacheLess(\n"
        file_content += f"          mf->cf[{i}]->RPP, Dense, mf->cf[{i}]->CB, dX, dY,\n"
        file_content += f"          i / {number_of_patterns}, c_cols"
        for x in range(len(pattern)):
            if '1' == pattern[x]:
                file_content += f" , {x}"
        file_content += ", N);\n"
        file_content += "      break;\n"
    file_content += "  }\n"
    file_content += "}\n"

    return file_content


def dense_codelet_generator_nnz_base(m, n, b_col_iteration, loop_order):
    iterator = b_col_iteration

    file_content = ""

    # part of cache less
    for i in range(1, m + 1):
        file_content += f"__device__ void PatternWith{m}NNZByRowPanel{i}RowPerThread{n}BlockCacheLess(\n"
        file_content += ("    int *RPP, float *NNZ, int *CB,"
                         " float *dx, float *dy,")
        file_content += "int i, int c_cols"
        for j in range(i):
            file_content += f", int offset{j}"

        file_content += ", int M) {\n"

        file_content += "int blockSize = blockDim.x;\n"
        file_content += "int thread_id_in_warp = threadIdx.x % 32;\n"
        file_content += "int warp_id = threadIdx.x / 32;\n"
        file_content += "int number_of_warps = blockSize / 32;\n"

        file_content += "int rowStart_all = RPP[i];\n"
        file_content += "int rowEnd_all = RPP[i + 1];\n"

        file_content += f"int c_col_partition = (c_cols + {iterator * 32 - 1}) / {iterator * 32};\n"
        file_content += f"int c_col_share = ((c_col_partition + number_of_warps - 1) / number_of_warps) * {iterator * 32};\n"
        file_content += f"c_col_share = c_col_share == 0 ? {iterator * 32} : c_col_share;\n"
        file_content += "int c_col_start = c_col_share * warp_id;\n"
        file_content += "int c_col_end = c_col_start + c_col_share;\n"
        file_content += "c_col_end = c_col_end > c_cols ? c_cols : c_col_end;\n"

        file_content += "if (rowStart_all == rowEnd_all) {\n"
        file_content += "   return;\n"
        file_content += "}\n"
        file_content += "if (c_col_start >= c_col_end) {\n"
        file_content += "    return;\n"
        file_content += "}\n"

        file_content += f"int c_row = i * {m};\n"

        if loop_order == "i-j":
            file_content += f"for (int col = c_col_start; col < c_col_end; col += {iterator * 32}) {{\n"
            for x in range(iterator * i):
                file_content += f"float sum_{x} = 0.0f;\n"

            file_content += f"for (int j = rowStart_all; j < rowEnd_all; j += {n}) {{\n"
        else:
            file_content += f"for (int j = rowStart_all; j < rowEnd_all; j += {n}) {{\n"
            file_content += f"  for (int col = c_col_start; col < c_col_end; col += {iterator * 32}) {{\n"
            for x in range(iterator * i):
                file_content += f"  float sum_{x} = 0.0f;\n"

        for x in range(n):
            file_content += f"int sub_b_row_start_{x} = CB[j + {x}];\n"

        for x in range(iterator):
            file_content += f"int sub_b_col_{x} = col + {32 * x} + thread_id_in_warp;\n"

        for y in range(n):
            for x in range(i):
                file_content += f"float nnz{x}_{y} = NNZ[(c_row + offset{x}) * M + sub_b_row_start_{y}];\n"

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

        if loop_order == "i-j":
            file_content += "}\n"
            file_content += f"int c_col = col + thread_id_in_warp;\n"
            for xx in range(iterator):
                for x in range(i):
                    file_content += f"atomicAdd(&dy[(c_row + offset{x}) * c_cols + c_col + {xx * 32}], sum_{x + xx * i});\n"

        else:
            file_content += f"int c_col = col + thread_id_in_warp;\n"
            for xx in range(iterator):
                for x in range(i):
                    file_content += f"atomicAdd(&dy[(c_row + offset{x}) * c_cols + c_col + {xx * 32}], sum_{x + xx * i});\n"
            file_content += "}\n"

        file_content += "}\n"
        file_content += "}\n"
        file_content += "\n\n"

    return file_content


def dense_base_file(codelet_text, kernel_text, class_text):
    """
    the spmm demo gou utiles file text here
    :return:
    """
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

    file_content += "void copyCSR(sym_lib::CSR *A, sym_lib::CSR *ACsr) {\n";
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

    file_content += codelet_text

    file_content += kernel_text

    file_content += class_text

    file_content += "#endif // COMPRESSED_TENSOR_ALGEBRA_SPMM_DEMO_GPU_UTILS_H\n"

    return file_content
