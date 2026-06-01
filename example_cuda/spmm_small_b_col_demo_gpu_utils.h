#ifndef COMPRESSED_TENSOR_ALGEBRA_SPMM_SMALL_B_COL_DEMO_GPU_UTILS_H
#define COMPRESSED_TENSOR_ALGEBRA_SPMM_SMALL_B_COL_DEMO_GPU_UTILS_H

#include "SWTensorBench.h"
#include "aggregation/def.h"
#include <cublas_v2.h>
#include "compressed_format/CompressedFormat.h"
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#define CUDA_CHECK(err) \
  do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("CUDA error"); \
    } \
  } while (0)

__device__ __half warp_shfl_down_half(__half val, int delta) {
uint16_t val_bits = *reinterpret_cast<uint16_t*>(&val);
uint16_t shuffled = __shfl_down_sync(0xffffffff, val_bits, delta);
return *reinterpret_cast<__half*>(&shuffled);
}

void spmm_csr_float(int m, int n, const int *Ap, const int *Ai, const float *Ax,
                    const float *B, float *C) {
  for (int i = 0; i < m; i++) {
    for (int j = Ap[i]; j < Ap[i + 1]; j++) {
      int col_index = Ai[j];
      float value = Ax[j];
      for (int k = 0; k < n; k++) {
        C[i * n + k] += value * B[col_index * n + k];
      }
    }
  }
}

template <typename T>
swiftware::compression::CompressedFormatGPU<T> *allocateAndCopyCompressedFormat(
    swiftware::compression::CompressedFormat<T> *h_cf) {
  swiftware::compression::CompressedFormatGPU<T> *d_cf;
  cudaMalloc((void **)&d_cf,
             sizeof(swiftware::compression::CompressedFormatGPU<T>));
  cudaMemcpy(d_cf, h_cf, sizeof(swiftware::compression::CompressedFormat<T>),
             cudaMemcpyHostToDevice);
  T *d_NNZ;
  int *d_cols, *d_RPP, *d_NPP;
  int *d_PT, *d_CB, *d_CSRRows;
  cudaMalloc((void **)&d_NNZ, h_cf->nnz * sizeof(T));
  cudaMemcpy(d_NNZ, h_cf->NNZ, h_cf->nnz * sizeof(T), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_RPP, h_cf->nrpp * sizeof(int));
  cudaMemcpy(d_RPP, h_cf->RPP, h_cf->nrpp * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_NPP, h_cf->nrpp * sizeof(int));
  cudaMemcpy(d_NPP, h_cf->NPP, h_cf->nrpp * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_PT, h_cf->PT.size() * sizeof(int));
  cudaMemcpy(d_PT, h_cf->PT.data(), h_cf->PT.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_CB, h_cf->CB.size() * sizeof(int));
  cudaMemcpy(d_CB, h_cf->CB.data(), h_cf->CB.size() * sizeof(int), cudaMemcpyHostToDevice);
  CUDA_CHECK(
      cudaMemcpy(&(d_cf->NNZ), &d_NNZ, sizeof(T *), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(&(d_cf->RPP), &d_RPP, sizeof(int *), cudaMemcpyHostToDevice));
  cudaMemcpy(&(d_cf->NPP), &d_NPP, sizeof(int *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_cf->PT), &d_PT, sizeof(int *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_cf->CB), &d_CB, sizeof(int *), cudaMemcpyHostToDevice);
  return d_cf;
}
void copyCSR(sym_lib::CSR *A, sym_lib::CSR *ACsr) {
  ACsr->m = A->m;
  ACsr->n = A->n;
  ACsr->nnz = A->nnz;
  for (int i = 0; i < A->m + 1; ++i) {
    ACsr->p[i] = A->p[i];
  }
  for (int i = 0; i < A->m; ++i) {
    for (int j = A->p[i]; j < A->p[i + 1]; ++j) {
      ACsr->x[j] = A->x[j];
      ACsr->i[j] = A->i[j];
    }
  }
}
swiftware::compression::CompressedFormatGPU<__half> *
allocateAndCopyCompressedFormatToHalf(
    swiftware::compression::CompressedFormat<float> *h_cf) {

  swiftware::compression::CompressedFormatGPU<__half> *d_cf;
  cudaMalloc((void **)&d_cf,
             sizeof(swiftware::compression::CompressedFormatGPU<__half>));

  // Allocate and convert arrays from float to half before copying
  __half *d_NNZ;
  int *d_RPP, *d_NPP;
  int *d_PT, *d_CB;

  // Allocate temporary array to hold half-precision values
  __half *temp_NNZ = new __half[h_cf->nnz];
  for (size_t i = 0; i < h_cf->nnz; ++i) {
    temp_NNZ[i] = __float2half(h_cf->NNZ[i]);
  }

  // Copy the converted half-precision array to the device
  cudaMalloc((void **)&d_NNZ, h_cf->nnz * sizeof(__half));
  cudaMemcpy(d_NNZ, temp_NNZ, h_cf->nnz * sizeof(__half), cudaMemcpyHostToDevice);
  delete[] temp_NNZ;

  // Copy integer arrays directly
  cudaMalloc((void **)&d_RPP, h_cf->nrpp * sizeof(int));
  cudaMemcpy(d_RPP, h_cf->RPP, h_cf->nrpp * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_NPP, h_cf->nrpp * sizeof(int));
  cudaMemcpy(d_NPP, h_cf->NPP, h_cf->nrpp * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_PT, h_cf->PT.size() * sizeof(int));
  cudaMemcpy(d_PT, h_cf->PT.data(), h_cf->PT.size() * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_CB, h_cf->CB.size() * sizeof(int));
  cudaMemcpy(d_CB, h_cf->CB.data(), h_cf->CB.size() * sizeof(int),
             cudaMemcpyHostToDevice);

  // Copy pointers to the device
  CUDA_CHECK(
      cudaMemcpy(&(d_cf->NNZ), &d_NNZ, sizeof(__half *), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(&(d_cf->RPP), &d_RPP, sizeof(int *), cudaMemcpyHostToDevice));
  cudaMemcpy(&(d_cf->NPP), &d_NPP, sizeof(int *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_cf->PT), &d_PT, sizeof(int *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_cf->CB), &d_CB, sizeof(int *), cudaMemcpyHostToDevice);

  return d_cf;
}

swiftware::compression::MixedFormatGPU<__half> *allocateAndCopyMixedFormatToHalf(
    swiftware::compression::MixedFormat<float> *h_mixed) {

  swiftware::compression::MixedFormatGPU<__half> *d_mixed;
  cudaMalloc((void **)&d_mixed,
             sizeof(swiftware::compression::MixedFormatGPU<__half> *));

  // Allocate and copy the array of CompressedFormat pointers
  swiftware::compression::CompressedFormatGPU<__half> **d_cf;
  cudaMalloc((void **)&d_cf,
             h_mixed->cf.size() *
                 sizeof(swiftware::compression::CompressedFormatGPU<__half> *));

  for (size_t i = 0; i < h_mixed->cf.size(); ++i) {
    swiftware::compression::CompressedFormatGPU<__half> *d_cf_element =
        allocateAndCopyCompressedFormatToHalf(h_mixed->cf[i]);
    cudaMemcpy(&(d_cf[i]), &d_cf_element,
               sizeof(swiftware::compression::CompressedFormatGPU<__half> *),
               cudaMemcpyHostToDevice);
  }

  cudaMemcpy(&(d_mixed->cf), &d_cf,
             sizeof(swiftware::compression::CompressedFormatGPU<__half> *),
             cudaMemcpyHostToDevice);

  return d_mixed;
}
template <typename T>
struct TensorOutputs : public swiftware::benchmark::Outputs<T> {
  int M;
  T *Dx;

  TensorOutputs(int M) : M(M) { Dx = new T[M](); }

  ~TensorOutputs() { delete[] Dx; }

  void printDx() {
    std::cout << "\n Dx:\n";
    std::cout << "\n";
  }

  void reset() { std::fill_n(Dx, M, 0.0); }
};
struct TensorInputsFloat : public swiftware::benchmark::Inputs<float> {
  int M, N, Z;
  sym_lib::CSR *ACsr;

  float *Bx;
  float *CorrectMul;
  bool IsSolProvided;

  TensorInputsFloat(int M1, int N1, int Z1, sym_lib::CSR *A1, int NumThreads1,
                    int NumTrial1, std::string ExpN)
      : swiftware::benchmark::Inputs<float>(NumTrial1, NumThreads1, ExpN) {
    M = M1;
    N = N1;
    Z = Z1;
    if (A1 != nullptr)
      ACsr = new sym_lib::CSR(M, N, A1->nnz);

    copyCSR(A1, ACsr);
    IsSolProvided = false;
    swiftware::benchmark::Inputs<float>::Threshold = 1e-6;
  }

  ~TensorInputsFloat() {
    free(CorrectSol);
    free(CorrectMul);
    free(Bx);
    delete ACsr;
  }
};
template <typename T>
swiftware::compression::MixedFormatGPU<T> *
allocateAndCopyMixedFormat(swiftware::compression::MixedFormat<T> *h_mixed) {

  swiftware::compression::MixedFormatGPU<T> *d_mixed;
  cudaMalloc((void **)&d_mixed,
             sizeof(swiftware::compression::MixedFormatGPU<T> *));

  // Allocate and copy the array of CompressedFormat pointers
  swiftware::compression::CompressedFormatGPU<T> **d_cf;
  cudaMalloc((void **)&d_cf,
             h_mixed->cf.size() *
                 sizeof(swiftware::compression::CompressedFormatGPU<T> *));

  for (size_t i = 0; i < h_mixed->cf.size(); ++i) {
    swiftware::compression::CompressedFormatGPU<T> *d_cf_element =
        allocateAndCopyCompressedFormat(h_mixed->cf[i]);
    cudaMemcpy(&(d_cf[i]), &d_cf_element,
               sizeof(swiftware::compression::CompressedFormatGPU<T> *),
               cudaMemcpyHostToDevice);
  }

  cudaMemcpy(&(d_mixed->cf), &d_cf,
             sizeof(swiftware::compression::CompressedFormatGPU<T> *),
             cudaMemcpyHostToDevice);

  //  int cf_len = h_mixed->cf.size();
  //  cudaMemcpy(&(d_mixed->cflen), &cf_len, sizeof(int *),
  //  cudaMemcpyHostToDevice);

  // Allocate and copy the CSRFormat
  //  swiftware::compression::CSRFormatGPU<T> *d_csr =
  //      allocateAndCopyCSRFormat(h_mixed->csr);
  //  cudaMemcpy(&(d_mixed->csr), &d_csr,
  //             sizeof(swiftware::compression::CSRFormatGPU<T> *),
  //             cudaMemcpyHostToDevice);

  return d_mixed;
}
// Mixed Format
struct TensorInputsMixed : public TensorInputsFloat {

    int number_of_warps;
public:
  swiftware::compression::MixedFormat<float> *MixedFormat;
  TensorInputsMixed(int M1, int N1, int Z1, sym_lib::CSR *A1,
                    swiftware::compression::MixedFormat<float> *Mixed,
                    int NumThreads1, int NumTrial1, std::string ExpN)
      : TensorInputsFloat(M1, N1, Z1, A1, NumThreads1, NumTrial1, ExpN),
        MixedFormat(Mixed){};
};

class SpMMSerialFloat : public swiftware::benchmark::SWTensorBench<float> {
protected:
  TensorInputsFloat *InTensor;
  float *csr_x;
  void setup() override { this->St->OtherStats["NTile"] = {4}; }

  void preExecute() override {}

  swiftware::benchmark::Timer execute() override {
    OutTensor->reset();
    swiftware::benchmark::Timer t1;
    t1.start();

    spmm_csr_float(InTensor->M, InTensor->Z, InTensor->ACsr->p,
                   InTensor->ACsr->i, csr_x, InTensor->Bx, OutTensor->Dx);

    t1.stop();
    return t1;
  }

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

public:
  TensorOutputs<float> *OutTensor;
  SpMMSerialFloat(TensorInputsFloat *In1, swiftware::benchmark::Stats *Stat1)
      : swiftware::benchmark::SWTensorBench<float>(In1, Stat1) {
    OutTensor = new TensorOutputs<float>(In1->M * In1->Z);
    InTensor = In1;
    csr_x =
        static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));
    for (int i = 0; i < In1->ACsr->nnz; i++) {
      csr_x[i] = static_cast<float>(In1->ACsr->x[i]);
    }
  };

  virtual void envelopCalculation() {}

  ~SpMMSerialFloat() {
    delete OutTensor;
    free(csr_x);
  }
};
template <typename T>
__global__ void spmm_complex_kernel(int *d_ptr, int *d_cols, T *d_val,
                                    T *d_matrix, T *d_out, int N, int b_cols) {
  int thread_id_in_warp = threadIdx.x % 32; // Lane index in the vector
  int warp_id = threadIdx.x / 32;           // Vector index in the thread block
  int number_of_warps =
      blockDim.x / 32; // Number of vectors in the thread block

  // Calculate the row range for this block
  int startRow = blockIdx.x;
  int endRow = min(startRow + 1, N);

  int col_share = b_cols / number_of_warps;
  int col_start = col_share * warp_id;
  int col_end = col_start + col_share;

  for (int col = col_start; col < col_end; ++col) {

    // Process rows assigned to this block

    // Loop over all columns of the dense matrix (b_cols)
    T sum = 0;

    int rowStart = d_ptr[startRow];
    int rowEnd = d_ptr[endRow];

    // Compute the dot product for this row and current column
    for (int i = rowStart + thread_id_in_warp; i < rowEnd; i += 32) {
      sum += d_val[i] * d_matrix[d_cols[i] * b_cols + col];
      //      printf("sum: %f\n", sum);
    }

    // Intra-vector reduction
    for (int i = 32 >> 1; i > 0; i >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, i);
    }

    // Save the result for the current column
    if (thread_id_in_warp == 0) {
      //      printf("sum: %f\n", sum);
      d_out[startRow * b_cols + col] += sum;
    }
  }
}
class SpMMComplexKernel : public SpMMSerialFloat {

  float *x;
  int *dA_csrOffsets, *dA_columns;
  float *dA_values, *dX, *dY;

  int *m_gpu, *n_gpu, *nnz_gpu;

  int BlockDim = 256;
  float alpha = 1.0f;

  swiftware::benchmark::Timer execute() override {

    OutTensor->reset();
    swiftware::benchmark::Timer t1;

    t1.startGPU();

    spmm_complex_kernel<float><<<InTensor->ACsr->m + 1, 128>>>(
        dA_csrOffsets, dA_columns, dA_values, dX, dY, InTensor->ACsr->m,
        InTensor->Z);

    t1.stopGPU("");

    cudaMemcpy(this->OutTensor->Dx, dY,
               this->InTensor->ACsr->m * InTensor->Z * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaMemset(dY, 0, this->InTensor->ACsr->m * InTensor->Z * sizeof(float));

    return t1;
  }

public:
  SpMMComplexKernel(TensorInputsFloat *In1, swiftware::benchmark::Stats *Stat1)
      : SpMMSerialFloat(In1, Stat1) {
    x = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));
    for (int i = 0; i < In1->ACsr->nnz; i++) {
      x[i] = static_cast<float>(In1->ACsr->x[i]);
    }

    cudaMalloc((void **)&dA_csrOffsets,
               (this->InTensor->ACsr->m + 1) * sizeof(int));
    cudaMalloc((void **)&dA_columns, this->InTensor->ACsr->nnz * sizeof(int));
    cudaMalloc((void **)&dA_values, this->InTensor->ACsr->nnz * sizeof(float));
    cudaMalloc((void **)&dX,
               this->InTensor->ACsr->n * this->InTensor->Z * sizeof(float));
    cudaMalloc((void **)&dY,
               this->InTensor->ACsr->m * this->InTensor->Z * sizeof(float));

    cudaMemcpy(dA_values, this->x, this->InTensor->ACsr->nnz * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dA_csrOffsets, this->InTensor->ACsr->p,
               (this->InTensor->ACsr->m + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dA_columns, this->InTensor->ACsr->i,
               this->InTensor->ACsr->nnz * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(dX, this->InTensor->Bx,
               this->InTensor->ACsr->n * this->InTensor->Z * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dY, OutTensor->Dx,
               this->InTensor->ACsr->m * this->InTensor->Z * sizeof(float),
               cudaMemcpyHostToDevice);

  };
  ~SpMMComplexKernel() {
    free(x);
    cudaFree(dA_csrOffsets);
    cudaFree(dA_columns);
    cudaFree(dA_values);
    cudaFree(dX);
    cudaFree(dY);
  }
};
class SpMMcuSparse : public SpMMSerialFloat {

  float *x;
  int *dA_csrOffsets, *dA_columns;
  float *dA_values, *dX, *dY;

  // CUSPARSE APIs
  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matX, matY;
  void *dBuffer = NULL;
  size_t bufferSize = 0;

  float alpha = 1.0f;
  float beta = 0.0f;

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
    cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matX, &beta,
                 matY, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
    t1.stopGPU("");

    cudaMemcpy(this->OutTensor->Dx, dY,
               this->InTensor->ACsr->m * InTensor->Z * sizeof(float),
               cudaMemcpyDeviceToHost);

    return t1;
  }

public:
  SpMMcuSparse(TensorInputsFloat *In1, swiftware::benchmark::Stats *Stat1)
      : SpMMSerialFloat(In1, Stat1) {
    x = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));
    for (int i = 0; i < In1->ACsr->nnz; i++) {
      x[i] = static_cast<float>(In1->ACsr->x[i]);
    }

    cudaMalloc((void **)&dA_csrOffsets,
               (this->InTensor->ACsr->m + 1) * sizeof(int));
    cudaMalloc((void **)&dA_columns, this->InTensor->ACsr->nnz * sizeof(int));
    cudaMalloc((void **)&dA_values, this->InTensor->ACsr->nnz * sizeof(float));
    cudaMalloc((void **)&dX,
               this->InTensor->ACsr->n * InTensor->Z * sizeof(float));
    cudaMalloc((void **)&dY,
               this->InTensor->ACsr->m * this->InTensor->Z * sizeof(float));

    cudaMemcpy(dA_values, this->x, this->InTensor->ACsr->nnz * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dA_csrOffsets, this->InTensor->ACsr->p,
               (this->InTensor->ACsr->m + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dA_columns, this->InTensor->ACsr->i,
               this->InTensor->ACsr->nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, this->InTensor->Bx,
               this->InTensor->ACsr->n * InTensor->Z * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dY, OutTensor->Dx,
               this->InTensor->ACsr->m * this->InTensor->Z * sizeof(float),
               cudaMemcpyHostToDevice);

    cusparseCreate(&handle);
    cusparseCreateCsr(&matA, this->InTensor->ACsr->m, this->InTensor->ACsr->n,
                      this->InTensor->ACsr->nnz, dA_csrOffsets, dA_columns,
                      dA_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matX, this->InTensor->ACsr->n, this->InTensor->Z,
                        this->InTensor->Z, dX, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matY, this->InTensor->ACsr->m, this->InTensor->Z,
                        this->InTensor->Z, dY, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                            matX, &beta, matY, CUDA_R_32F,
                            CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
  };
  ~SpMMcuSparse() {

    free(x);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matX);
    cusparseDestroyDnMat(matY);
    cusparseDestroy(handle);
    cudaFree(dBuffer);
    cudaFree(dA_csrOffsets);
    cudaFree(dA_columns);
    cudaFree(dA_values);
    cudaFree(dX);
    cudaFree(dY);
  }
};
class SpMMcuBlas : public SpMMSerialFloat {

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

    // cuBLAS dense matrix multiplication
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                this->InTensor->M,  // Rows of A
                this->InTensor->Z,        // Columns of B (and Y)
                this->InTensor->N,  // Columns of A (and rows of B)
                &alpha, dA_dense, this->InTensor->M,  // A is now dense
                dX, this->InTensor->N,                // B
                &beta, dY, this->InTensor->M);        // Y (output)

    t1.stopGPU("");

    cudaMemcpy(temp_c, dY,
               this->InTensor->ACsr->m * InTensor->Z * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Store transpose of temp_c to OutTensor->Dx
    for (int i = 0; i < this->InTensor->ACsr->m; i++) {
      for (int j = 0; j < this->InTensor->Z; j++) {
        OutTensor->Dx[i * this->InTensor->Z + j] = temp_c[j * this->InTensor->M + i];
      }
    }

    return t1;
  }

public:
  SpMMcuBlas(TensorInputsFloat *In1, swiftware::benchmark::Stats *Stat1)
      : SpMMSerialFloat(In1, Stat1) {
    x = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));
    for (int i = 0; i < In1->ACsr->nnz; i++) {
      x[i] = static_cast<float>(In1->ACsr->x[i]);
    }

    temp_c = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->M * In1->Z));
    // Allocate memory on GPU
    cudaMalloc((void **)&dX, this->InTensor->ACsr->n * InTensor->Z * sizeof(float));

    cudaMalloc((void **)&dY, this->InTensor->ACsr->m * this->InTensor->Z * sizeof(float));

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
    float *A_dense_cpu = static_cast<float *>(aligned_alloc(32, rows * cols * sizeof(float)));

    for (int row = 0; row < rows; ++row) {
      for (int idx = In1->ACsr->p[row]; idx < In1->ACsr->p[row + 1]; ++idx) {
        int col = In1->ACsr->i[idx];
        A_dense_cpu [col * rows + row] = x[idx];
      }
    }

    float *B_trap = static_cast<float *>(aligned_alloc(32, this->InTensor->ACsr->n * InTensor->Z * sizeof(float)));

    // Store transpose B
    for (int i = 0; i < this->InTensor->ACsr->n; i++) {
      for (int j = 0; j < this->InTensor->Z; j++) {
        B_trap[j * this->InTensor->ACsr->n + i] = this->InTensor->Bx[i * this->InTensor->Z + j];
      }
    }

    cudaMemcpy(dX, B_trap,
               this->InTensor->ACsr->n * InTensor->Z * sizeof(float),
               cudaMemcpyHostToDevice);

    // Allocate and copy A (dense matrix) to GPU
    cudaMalloc((void **)&dA_dense, rows * cols * sizeof(float));
    cudaMemcpy(dA_dense, A_dense_cpu, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Free CPU dense matrix memory
    free(A_dense_cpu);
    free(B_trap);
    // cuBLAS handle creation
    cublasCreate(&cublasHandle);
  }

  ~SpMMcuBlas() {
    free(x);
    cublasDestroy(cublasHandle);
    cudaFree(dA_dense);
    cudaFree(dX);
    cudaFree(dY);
 }
 };

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
    __device__ void PatternWith1NNZByRowPanelFull(int *RPP, int *NPP, float *NNZ,
                                              int *CB, float *dx, float *dy, int i , int offset0) {

  int blockSize = blockDim.x;
  int thread_id_in_warp = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;


  int rowStart_all = RPP[i];
  int rowEnd_all = RPP[i + 1];
  int number_of_packs = (rowEnd_all - rowStart_all) / 32;
  if (rowEnd_all - rowStart_all == 0) {
    return;
  }

  number_of_packs = number_of_packs == 0 ? 1 : number_of_packs;
  int number_of_warps = blockSize / 32;
  int pack_share = (number_of_packs + number_of_warps - 1) / number_of_warps;
  int warp_share = pack_share * 32;
  int rowStart = rowStart_all + warp_id * warp_share;
  int rowEnd = rowStart + warp_share;
  rowEnd = rowEnd > rowEnd_all ? rowEnd_all : rowEnd;
  rowEnd = warp_id == number_of_warps - 1 && rowEnd < rowEnd_all ? rowEnd_all : rowEnd;
  int t_nnz = NPP[i] + warp_id * warp_share * 1;
  float sum_all0 = 0.0f;
  int j = rowStart;
  for ( j ; j < rowEnd - 31; j += 32) {
    float nnz = (thread_id_in_warp < 32) ? NNZ[t_nnz + thread_id_in_warp] : 0;
    float B0 = 0.0f;
    B0 = dx[CB[j + thread_id_in_warp / 1] * 1 + 0];

    sum_all0 += nnz * B0;
    t_nnz += 32;
  }
  int tail = rowEnd - j;
  int rest_nnz = 1 * tail; 
  if (thread_id_in_warp < rest_nnz && tail > 0) {
    float nnz = NNZ[t_nnz + thread_id_in_warp];
    float B0 = 0.0f;
    B0 = dx[CB[j + thread_id_in_warp / 1] * 1 + 0];
    sum_all0 += nnz * B0;
  }
  sum_all0 += __shfl_down_sync(0xFFFFFFFF, sum_all0, 1);
  sum_all0 += __shfl_down_sync(0xFFFFFFFF, sum_all0, 2);
  sum_all0 += __shfl_down_sync(0xFFFFFFFF, sum_all0, 4);
  sum_all0 += __shfl_down_sync(0xFFFFFFFF, sum_all0, 8);
  sum_all0 += __shfl_down_sync(0xFFFFFFFF, sum_all0, 16);
  if (thread_id_in_warp == 0) {
    atomicAdd(&dy[(i * 3 + offset0) * 1 + 0], sum_all0);
  }
}
__device__ void PatternWith2NNZByRowPanelFull(int *RPP, int *NPP, float *NNZ,
                                              int *CB, float *dx, float *dy, int i , int offset0, int offset1) {

  int blockSize = blockDim.x;
  int thread_id_in_warp = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;


  int rowStart_all = RPP[i];
  int rowEnd_all = RPP[i + 1];
  int number_of_packs = (rowEnd_all - rowStart_all) / 16;
  if (rowEnd_all - rowStart_all == 0) {
    return;
  }

  number_of_packs = number_of_packs == 0 ? 1 : number_of_packs;
  int number_of_warps = blockSize / 32;
  int pack_share = (number_of_packs + number_of_warps - 1) / number_of_warps;
  int warp_share = pack_share * 16;
  int rowStart = rowStart_all + warp_id * warp_share;
  int rowEnd = rowStart + warp_share;
  rowEnd = rowEnd > rowEnd_all ? rowEnd_all : rowEnd;
  rowEnd = warp_id == number_of_warps - 1 && rowEnd < rowEnd_all ? rowEnd_all : rowEnd;
  int t_nnz = NPP[i] + warp_id * warp_share * 2;
  float sum_all0 = 0.0f;
  int j = rowStart;
  for ( j ; j < rowEnd - 15; j += 16) {
    float nnz = (thread_id_in_warp < 32) ? NNZ[t_nnz + thread_id_in_warp] : 0;
    float B0 = 0.0f;
    B0 = dx[CB[j + thread_id_in_warp / 2] * 1 + 0];

    sum_all0 += nnz * B0;
    t_nnz += 32;
  }
  int tail = rowEnd - j;
  int rest_nnz = 2 * tail; 
  if (thread_id_in_warp < rest_nnz && tail > 0) {
    float nnz = NNZ[t_nnz + thread_id_in_warp];
    float B0 = 0.0f;
    B0 = dx[CB[j + thread_id_in_warp / 2] * 1 + 0];
    sum_all0 += nnz * B0;
  }
  sum_all0 += __shfl_down_sync(0xFFFFFFFF, sum_all0, 2);
  sum_all0 += __shfl_down_sync(0xFFFFFFFF, sum_all0, 4);
  sum_all0 += __shfl_down_sync(0xFFFFFFFF, sum_all0, 8);
  sum_all0 += __shfl_down_sync(0xFFFFFFFF, sum_all0, 16);
  if (thread_id_in_warp == 0) {
    atomicAdd(&dy[(i * 3 + offset0) * 1 + 0], sum_all0);
  }
  else if (thread_id_in_warp == 1){
    atomicAdd(&dy[(i * 3 + offset1) * 1 + 0], sum_all0);
  }
}
__device__ void PatternWith3NNZByRowPanelFull(int *RPP, int *NPP, float *NNZ,
                                              int *CB, float *dx, float *dy, int i , int offset0, int offset1, int offset2) {

  int blockSize = blockDim.x;
  int thread_id_in_warp = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;


  int rowStart_all = RPP[i];
  int rowEnd_all = RPP[i + 1];
  int number_of_packs = (rowEnd_all - rowStart_all) / 10;
  if (rowEnd_all - rowStart_all == 0) {
    return;
  }

  number_of_packs = number_of_packs == 0 ? 1 : number_of_packs;
  int number_of_warps = blockSize / 32;
  int pack_share = (number_of_packs + number_of_warps - 1) / number_of_warps;
  int warp_share = pack_share * 10;
  int rowStart = rowStart_all + warp_id * warp_share;
  int rowEnd = rowStart + warp_share;
  rowEnd = rowEnd > rowEnd_all ? rowEnd_all : rowEnd;
  rowEnd = warp_id == number_of_warps - 1 && rowEnd < rowEnd_all ? rowEnd_all : rowEnd;
  int t_nnz = NPP[i] + warp_id * warp_share * 3;
  float sum_all0 = 0.0f;
  int j = rowStart;
  for ( j ; j < rowEnd - 9; j += 10) {
    float nnz = (thread_id_in_warp < 30) ? NNZ[t_nnz + thread_id_in_warp] : 0;
    float B0 = 0.0f;
    B0 = dx[CB[j + thread_id_in_warp / 3] * 1 + 0];

    sum_all0 += nnz * B0;
    t_nnz += 30;
  }
  int tail = rowEnd - j;
  int rest_nnz = 3 * tail; 
  if (thread_id_in_warp < rest_nnz && tail > 0) {
    float nnz = NNZ[t_nnz + thread_id_in_warp];
    float B0 = 0.0f;
    B0 = dx[CB[j + thread_id_in_warp / 3] * 1 + 0];
    sum_all0 += nnz * B0;
  }
  sum_all0 += __shfl_down_sync(0xFFFFFFFF, sum_all0, 3);
  float temp_sum0_0 = sum_all0;
  sum_all0 += __shfl_down_sync(0xFFFFFFFF, sum_all0, 6);
  sum_all0 += __shfl_down_sync(0xFFFFFFFF, sum_all0, 12);
  sum_all0 += __shfl_down_sync(0xFFFFFFFF, temp_sum0_0, 24);
  if (thread_id_in_warp == 0) {
    atomicAdd(&dy[(i * 3 + offset0) * 1 + 0], sum_all0);
  }
  else if (thread_id_in_warp == 1){
    atomicAdd(&dy[(i * 3 + offset1) * 1 + 0], sum_all0);
  }
  else if (thread_id_in_warp == 2){
    atomicAdd(&dy[(i * 3 + offset2) * 1 + 0], sum_all0);
  }
}


__device__ void PatternWith1NNZByRowPanelFullHALF(int *RPP, int *NPP, __half *NNZ,
                                              int *CB, __half *dx, __half *dy, int i , int offset0) {

  int blockSize = blockDim.x;
  int thread_id_in_warp = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;


  int rowStart_all = RPP[i];
  int rowEnd_all = RPP[i + 1];
  int number_of_packs = (rowEnd_all - rowStart_all) / 32;
  if (rowEnd_all - rowStart_all == 0) {
    return;
  }

  number_of_packs = number_of_packs == 0 ? 1 : number_of_packs;
  int number_of_warps = blockSize / 32;
  int pack_share = (number_of_packs + number_of_warps - 1) / number_of_warps;
  int warp_share = pack_share * 32;
  int rowStart = rowStart_all + warp_id * warp_share;
  int rowEnd = rowStart + warp_share;
  rowEnd = rowEnd > rowEnd_all ? rowEnd_all : rowEnd;
  rowEnd = warp_id == number_of_warps - 1 && rowEnd < rowEnd_all ? rowEnd_all : rowEnd;
  int t_nnz = NPP[i] + warp_id * warp_share * 1;
  __half sum_all0 = 0.0;
  int j = rowStart;
  for ( j ; j < rowEnd - 31; j += 32) {
    __half nnz = (thread_id_in_warp < 32) ? NNZ[t_nnz + thread_id_in_warp] : __float2half_rn(0);
    __half B0 = 0.0;
    B0 = dx[CB[j + thread_id_in_warp / 1] * 1 + 0];

    sum_all0 += __hmul(nnz, B0);
    t_nnz += 32;
  }
  int tail = rowEnd - j;
  int rest_nnz = 1 * tail; 
  if (thread_id_in_warp < rest_nnz && tail > 0) {
    __half nnz = NNZ[t_nnz + thread_id_in_warp];
    __half B0 = 0.0;
    B0 = dx[CB[j + thread_id_in_warp / 1] * 1 + 0];
    sum_all0 += __hmul(nnz , B0);
  }
  sum_all0 += warp_shfl_down_half(sum_all0, 1);
  sum_all0 += warp_shfl_down_half(sum_all0, 2);
  sum_all0 += warp_shfl_down_half(sum_all0, 4);
  sum_all0 += warp_shfl_down_half(sum_all0, 8);
  sum_all0 += warp_shfl_down_half(sum_all0, 16);
  if (thread_id_in_warp == 0) {
    atomicAdd(&dy[(i * 3 + offset0) * 1 + 0], sum_all0);
  }
}
__device__ void PatternWith2NNZByRowPanelFullHALF(int *RPP, int *NPP, __half *NNZ,
                                              int *CB, __half *dx, __half *dy, int i , int offset0, int offset1) {

  int blockSize = blockDim.x;
  int thread_id_in_warp = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;


  int rowStart_all = RPP[i];
  int rowEnd_all = RPP[i + 1];
  int number_of_packs = (rowEnd_all - rowStart_all) / 16;
  if (rowEnd_all - rowStart_all == 0) {
    return;
  }

  number_of_packs = number_of_packs == 0 ? 1 : number_of_packs;
  int number_of_warps = blockSize / 32;
  int pack_share = (number_of_packs + number_of_warps - 1) / number_of_warps;
  int warp_share = pack_share * 16;
  int rowStart = rowStart_all + warp_id * warp_share;
  int rowEnd = rowStart + warp_share;
  rowEnd = rowEnd > rowEnd_all ? rowEnd_all : rowEnd;
  rowEnd = warp_id == number_of_warps - 1 && rowEnd < rowEnd_all ? rowEnd_all : rowEnd;
  int t_nnz = NPP[i] + warp_id * warp_share * 2;
  __half sum_all0 = 0.0;
  int j = rowStart;
  for ( j ; j < rowEnd - 15; j += 16) {
    __half nnz = (thread_id_in_warp < 32) ? NNZ[t_nnz + thread_id_in_warp] : __float2half_rn(0);
    __half B0 = 0.0;
    B0 = dx[CB[j + thread_id_in_warp / 2] * 1 + 0];

    sum_all0 += __hmul(nnz, B0);
    t_nnz += 32;
  }
  int tail = rowEnd - j;
  int rest_nnz = 2 * tail; 
  if (thread_id_in_warp < rest_nnz && tail > 0) {
    __half nnz = NNZ[t_nnz + thread_id_in_warp];
    __half B0 = 0.0;
    B0 = dx[CB[j + thread_id_in_warp / 2] * 1 + 0];
    sum_all0 += __hmul(nnz , B0);
  }
  sum_all0 += warp_shfl_down_half(sum_all0, 2);
  sum_all0 += warp_shfl_down_half(sum_all0, 4);
  sum_all0 += warp_shfl_down_half(sum_all0, 8);
  sum_all0 += warp_shfl_down_half(sum_all0, 16);
  if (thread_id_in_warp == 0) {
    atomicAdd(&dy[(i * 3 + offset0) * 1 + 0], sum_all0);
  }
  else if (thread_id_in_warp == 1){
    atomicAdd(&dy[(i * 3 + offset1) * 1 + 0], sum_all0);
  }
}
__device__ void PatternWith3NNZByRowPanelFullHALF(int *RPP, int *NPP, __half *NNZ,
                                              int *CB, __half *dx, __half *dy, int i , int offset0, int offset1, int offset2) {

  int blockSize = blockDim.x;
  int thread_id_in_warp = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;


  int rowStart_all = RPP[i];
  int rowEnd_all = RPP[i + 1];
  int number_of_packs = (rowEnd_all - rowStart_all) / 10;
  if (rowEnd_all - rowStart_all == 0) {
    return;
  }

  number_of_packs = number_of_packs == 0 ? 1 : number_of_packs;
  int number_of_warps = blockSize / 32;
  int pack_share = (number_of_packs + number_of_warps - 1) / number_of_warps;
  int warp_share = pack_share * 10;
  int rowStart = rowStart_all + warp_id * warp_share;
  int rowEnd = rowStart + warp_share;
  rowEnd = rowEnd > rowEnd_all ? rowEnd_all : rowEnd;
  rowEnd = warp_id == number_of_warps - 1 && rowEnd < rowEnd_all ? rowEnd_all : rowEnd;
  int t_nnz = NPP[i] + warp_id * warp_share * 3;
  __half sum_all0 = 0.0;
  int j = rowStart;
  for ( j ; j < rowEnd - 9; j += 10) {
    __half nnz = (thread_id_in_warp < 30) ? NNZ[t_nnz + thread_id_in_warp] : __float2half_rn(0);
    __half B0 = 0.0;
    B0 = dx[CB[j + thread_id_in_warp / 3] * 1 + 0];

    sum_all0 += __hmul(nnz, B0);
    t_nnz += 30;
  }
  int tail = rowEnd - j;
  int rest_nnz = 3 * tail; 
  if (thread_id_in_warp < rest_nnz && tail > 0) {
    __half nnz = NNZ[t_nnz + thread_id_in_warp];
    __half B0 = 0.0;
    B0 = dx[CB[j + thread_id_in_warp / 3] * 1 + 0];
    sum_all0 += __hmul(nnz , B0);
  }
  sum_all0 += warp_shfl_down_half(sum_all0, 3);
  __half temp_sum0_0 = sum_all0;
  sum_all0 += warp_shfl_down_half(sum_all0, 6);
  sum_all0 += warp_shfl_down_half(sum_all0, 12);
  sum_all0 += warp_shfl_down_half(temp_sum0_0, 24);
  if (thread_id_in_warp == 0) {
    atomicAdd(&dy[(i * 3 + offset0) * 1 + 0], sum_all0);
  }
  else if (thread_id_in_warp == 1){
    atomicAdd(&dy[(i * 3 + offset1) * 1 + 0], sum_all0);
  }
  else if (thread_id_in_warp == 2){
    atomicAdd(&dy[(i * 3 + offset2) * 1 + 0], sum_all0);
  }
}
__global__ void SpmvComplexPatternDecomposeByRowSeperated( 
   swiftware::compression::MixedFormatGPU<float> *mf, float *dX, float *dY) {
  int j = blockIdx.x / 7;
  int i = blockIdx.x % 7;

  switch (i) {
    case 0:
      PatternWith3NNZByRowPanelFull(mf->cf[0]->RPP, mf->cf[0]->NPP, mf->cf[0]->NNZ, mf->cf[0]->CB, dX, dY, j , 0, 1, 2);
      break;
    case 1:
      PatternWith2NNZByRowPanelFull(mf->cf[1]->RPP, mf->cf[1]->NPP, mf->cf[1]->NNZ, mf->cf[1]->CB, dX, dY, j , 0, 1);
      break;
    case 2:
      PatternWith2NNZByRowPanelFull(mf->cf[2]->RPP, mf->cf[2]->NPP, mf->cf[2]->NNZ, mf->cf[2]->CB, dX, dY, j , 0, 2);
      break;
    case 3:
      PatternWith1NNZByRowPanelFull(mf->cf[3]->RPP, mf->cf[3]->NPP, mf->cf[3]->NNZ, mf->cf[3]->CB, dX, dY, j , 0);
      break;
    case 4:
      PatternWith2NNZByRowPanelFull(mf->cf[4]->RPP, mf->cf[4]->NPP, mf->cf[4]->NNZ, mf->cf[4]->CB, dX, dY, j , 1, 2);
      break;
    case 5:
      PatternWith1NNZByRowPanelFull(mf->cf[5]->RPP, mf->cf[5]->NPP, mf->cf[5]->NNZ, mf->cf[5]->CB, dX, dY, j , 1);
      break;
    case 6:
      PatternWith1NNZByRowPanelFull(mf->cf[6]->RPP, mf->cf[6]->NPP, mf->cf[6]->NNZ, mf->cf[6]->CB, dX, dY, j , 2);
      break;
  }
}

__global__ void SpmvComplexPatternDecomposeByRowSeperatedHalf( 
   swiftware::compression::MixedFormatGPU<__half> *mf, __half *dX, __half *dY) {
  int j = blockIdx.x / 7;
  int i = blockIdx.x % 7;

  switch (i) {
    case 0:
      PatternWith3NNZByRowPanelFullHALF(mf->cf[0]->RPP, mf->cf[0]->NPP, mf->cf[0]->NNZ, mf->cf[0]->CB, dX, dY, j , 0, 1, 2);
      break;
    case 1:
      PatternWith2NNZByRowPanelFullHALF(mf->cf[1]->RPP, mf->cf[1]->NPP, mf->cf[1]->NNZ, mf->cf[1]->CB, dX, dY, j , 0, 1);
      break;
    case 2:
      PatternWith2NNZByRowPanelFullHALF(mf->cf[2]->RPP, mf->cf[2]->NPP, mf->cf[2]->NNZ, mf->cf[2]->CB, dX, dY, j , 0, 2);
      break;
    case 3:
      PatternWith1NNZByRowPanelFullHALF(mf->cf[3]->RPP, mf->cf[3]->NPP, mf->cf[3]->NNZ, mf->cf[3]->CB, dX, dY, j , 0);
      break;
    case 4:
      PatternWith2NNZByRowPanelFullHALF(mf->cf[4]->RPP, mf->cf[4]->NPP, mf->cf[4]->NNZ, mf->cf[4]->CB, dX, dY, j , 1, 2);
      break;
    case 5:
      PatternWith1NNZByRowPanelFullHALF(mf->cf[5]->RPP, mf->cf[5]->NPP, mf->cf[5]->NNZ, mf->cf[5]->CB, dX, dY, j , 1);
      break;
    case 6:
      PatternWith1NNZByRowPanelFullHALF(mf->cf[6]->RPP, mf->cf[6]->NPP, mf->cf[6]->NNZ, mf->cf[6]->CB, dX, dY, j , 2);
      break;
  }
}
class SpMVMixePatternDecomposeByRowSeperated : public SpMMSerialFloat {

protected:
  float *x;

  float *dX, *dY, *dY1, *dY2;
  swiftware::compression::MixedFormatGPU<float> *mf;

  float alpha = 1.0f;

  TensorInputsMixed *InT;

  bool verify(double &Error) override {
    bool retValue = true;

    double infNorm = 0;
    for (int i = 0; i < InTensor->M * InTensor->Z - 4; ++i) {
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
    // lunch kernel

    SpmvComplexPatternDecomposeByRowSeperated<<<
        (InT->MixedFormat->cf[0]->nrpp - 1) * 7, InT->number_of_warps * 32>>>(mf, dX, dY);

    t1.stopGPU("");

    cudaMemcpy(this->OutTensor->Dx, dY, this->InTensor->ACsr->m  * InTensor->Z * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemset(dY, 0, this->InTensor->ACsr->m  * InTensor->Z * sizeof(float));

    return t1;
  }

public:
  SpMVMixePatternDecomposeByRowSeperated(TensorInputsMixed *In1,
                                         swiftware::benchmark::Stats *Stat1)
      : SpMMSerialFloat(In1, Stat1), InT(In1) {

    x = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));
    for (int i = 0; i < In1->ACsr->nnz; i++) {
      x[i] = static_cast<float>(In1->ACsr->x[i]);
    }

    cudaMalloc((void **)&dX, InTensor->N * InTensor->Z * sizeof(float));
    cudaMalloc((void **)&dY, InTensor->M * InTensor->Z * sizeof(float));
    cudaMalloc((void **)&dY1, InTensor->M * InTensor->Z * sizeof(float));
    cudaMalloc((void **)&dY2, InTensor->M * InTensor->Z * sizeof(float));
    //
    cudaMemset(dY, 0, InTensor->M * InTensor->Z * sizeof(float));
    cudaMemset(dY1, 0, InTensor->M * InTensor->Z * sizeof(float));
    cudaMemset(dY2, 0, InTensor->M * InTensor->Z * sizeof(float));
    cudaMemcpy(dX, this->InTensor->Bx, InTensor->N * InTensor->Z * sizeof(float),
               cudaMemcpyHostToDevice);

    mf = allocateAndCopyMixedFormat(In1->MixedFormat);
  };

  ~SpMVMixePatternDecomposeByRowSeperated() {
    //    cudaFree(dX);
    //    cudaFree(dY);
  }
};


class SpMVMixePatternDecomposeByRowSeperatedHalf : public SpMMSerialFloat {

protected:

  __half *dX, *dY;
  float *dY1;
  swiftware::compression::MixedFormatGPU<__half> *mf;

  float alpha = 1.0f;

  TensorInputsMixed *InT;

  bool verify(double &Error) override {
    bool retValue = true;

    double infNorm = 0;
    for (int i = 0; i < InTensor->M * InTensor->Z - 4; ++i) {
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
    // lunch kernel

    SpmvComplexPatternDecomposeByRowSeperatedHalf<<<
        (InT->MixedFormat->cf[0]->nrpp - 1) * 7, InT->number_of_warps * 32>>>(mf, dX, dY);

    t1.stopGPU("");

   cudaMemcpy(dY1, dY, this->InTensor->ACsr->m  * InTensor->Z * sizeof(__half), cudaMemcpyDeviceToHost);
   for(int i = 0; i < InTensor->M * InTensor->Z; i++) {
   OutTensor->Dx[i] = __half2float(dY1[i]);
}
   cudaMemset(dY, 0, this->InTensor->ACsr->m  * InTensor->Z * sizeof(float));

    return t1;
  }

public:
  SpMVMixePatternDecomposeByRowSeperatedHalf(TensorInputsMixed *In1,
                                         swiftware::benchmark::Stats *Stat1)
      : SpMMSerialFloat(In1, Stat1), InT(In1) {
    dY1 = static_cast<float *>(aligned_alloc(32, sizeof(float) * InTensor->M * InTensor->Z));
    std::fill(dY1,dY1+InTensor->M*InTensor->Z,0);
    cudaMalloc((void **)&dX, InTensor->N * InTensor->Z * sizeof(__half));
    cudaMalloc((void **)&dY, InTensor->M * InTensor->Z * sizeof(__half));
    //
    cudaMemset(dY, 0, InTensor->M * InTensor->Z * sizeof(__half));
    cudaMemset(dY1, 0, InTensor->M * InTensor->Z * sizeof(float));
    cudaMemcpy(dX, this->InTensor->Bx, InTensor->N * InTensor->Z * sizeof(__half),
               cudaMemcpyHostToDevice);

    mf = allocateAndCopyMixedFormatToHalf(In1->MixedFormat);
  };

  ~SpMVMixePatternDecomposeByRowSeperatedHalf() {
  
  }
};
#endif // COMPRESSED_TENSOR_ALGEBRA_SPMM_SMALL_B_COL_DEMO_GPU_UTILS_H
