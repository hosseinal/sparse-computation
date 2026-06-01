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
    file_content += "\n\n\n"

    return file_content


def kernel_generator_nnz_base(m, n, pattern_type_dictionary: PatternTypeDictionary):
    file_content = ""

    file_content += "__global__ void\n"
    file_content += f"SpMMFullRowPanelGeneralKernel{m}With{n}Block(MixedFormatGPU<float> *mf,"
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
    file_content += f"SpMMFullRowPanelGeneralKernel{m}With{n}Block(MixedFormatGPU<float> *mf,"
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


def base_file(codelet_text, kernel_text, class_text, m, n):
    file_content = ""
    file_content += "#ifndef COMPRESSED_TENSOR_ALGEBRA_SPMM_SMALL_B_COL_DEMO_GPU_UTILS_H\n"
    file_content += "#define COMPRESSED_TENSOR_ALGEBRA_SPMM_SMALL_B_COL_DEMO_GPU_UTILS_H\n\n"
    file_content += "#include <cublas_v2.h>\n"
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

    file_content += '''
    template <class T> struct CompressedFormatGPU {
          size_t m;    // number of rows of matrix
          size_t n;    // number of columns
          size_t nrpp; // number of row pattern pointer
          size_t nnz;  // number of non zeros
          T *NNZ;      // Array of Non Zero Values
          int *cols;
          int *RPP;       // ROW Pattern Pointer
          int *NPP;       // NNZ Pattern Pointer
          int *PT;        // Pattern Type array
          int PTlen;      // Length of Pattern Type array
          int *CB;        // Column Begin array
          int CBLen;      // Length of Column Begin array
          bool preAlloc;  // if it is already allocated somewhere else
          // CSR format
        
          CompressedFormatGPU() {}
        
          ~CompressedFormatGPU() {}
        };
        
        template <class T> struct CSRFormatGPU {
          size_t m;
          size_t n;
          size_t nnz;
          T *NNZ;  // non zeroes
          int *RP; // row pointer
          int *CI; // column index
        
          CSRFormatGPU(size_t M, size_t N, size_t Nnz) : m(M), n(N), nnz(Nnz){};
        };
        
        template <class T> struct MixedFormatGPU {
        public:
          CompressedFormatGPU<T> **cf; // Array of CompressedFormatGPU pointers
          int cflen;                   // Length of cf array
          CSRFormatGPU<T> *csr;
        
          MixedFormatGPU(){};
        };
    '''

    file_content += codelet_text

    file_content += kernel_text

    # file_content += class_text

    file_content += '''
    // Example host-side caller
    void RunSpMMKernel(int M, int N, int NNZ, int c_cols) {
        // === 1. Allocate and initialize MixedFormatGPU on host ===
        MixedFormatGPU<float> *h_mf = new MixedFormatGPU<float>();
        h_mf->cflen = 1;
        h_mf->cf = new CompressedFormatGPU<float> *[1];
        h_mf->cf[0] = new CompressedFormatGPU<float>();
    
        CompressedFormatGPU<float> *cf0 = h_mf->cf[0];
        cf0->m = M;
        cf0->n = N;
        cf0->nnz = NNZ;
    
        // Allocate GPU memory for matrix values (dummy example)
        cudaMalloc(&cf0->NNZ, sizeof(float) * NNZ);
        cudaMalloc(&cf0->cols, sizeof(int) * NNZ);
        cudaMalloc(&cf0->RPP, sizeof(int) * (M + 1));
        cudaMalloc(&cf0->NPP, sizeof(int) * NNZ);
        cudaMalloc(&cf0->PT, sizeof(int) * M);
        cudaMalloc(&cf0->CB, sizeof(int) * M);
    
        cf0->nrpp = M + 1;
        cf0->PTlen = M;
        cf0->CBLen = M;
    
        // === CSR part ===
        h_mf->csr = new CSRFormatGPU<float>(M, N, NNZ);
        cudaMalloc(&h_mf->csr->NNZ, sizeof(float) * NNZ);
        cudaMalloc(&h_mf->csr->RP, sizeof(int) * (M + 1));
        cudaMalloc(&h_mf->csr->CI, sizeof(int) * NNZ);
    
        // === 2. Copy MixedFormatGPU to device ===
        MixedFormatGPU<float> *d_mf;
        cudaMalloc(&d_mf, sizeof(MixedFormatGPU<float>));
    
        // You must also allocate device-side cf[0] and csr
        // For simplicity here we assume pointers are all valid and device-side
    
        // Normally, you'd need to copy over the inner pointers and arrays as well
        // For now, assign device pointers to d_mf directly
        cudaMemcpy(d_mf, h_mf, sizeof(MixedFormatGPU<float>), cudaMemcpyHostToDevice);
    
        // === 3. Allocate input/output vectors ===
        float *dX, *dY;
        cudaMalloc(&dX, sizeof(float) * N);
        cudaMalloc(&dY, sizeof(float) * M);
    
        // === 4. Launch kernel ===
        int threadsPerBlock = 256;
        int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;
        
       '''

    file_content += f"SpMMFullRowPanelGeneralKernel{m}With{n}Block<<<blocksPerGrid, threadsPerBlock>>>(d_mf, dX, dY, c_cols);\n"

    file_content += '''
        cudaDeviceSynchronize();
    
        // === 5. Cleanup ===
        cudaFree(dX);
        cudaFree(dY);
        cudaFree(cf0->NNZ);
        cudaFree(cf0->cols);
        cudaFree(cf0->RPP);
        cudaFree(cf0->NPP);
        cudaFree(cf0->PT);
        cudaFree(cf0->CB);
        cudaFree(h_mf->csr->NNZ);
        cudaFree(h_mf->csr->RP);
        cudaFree(h_mf->csr->CI);
        cudaFree(d_mf);
    
        delete h_mf->csr;
        delete cf0;
        delete[] h_mf->cf;
        delete h_mf;
    }
    '''

    file_content += "#endif // COMPRESSED_TENSOR_ALGEBRA_SPMM_SMALL_B_COL_DEMO_GPU_UTILS_H\n"

    return file_content
