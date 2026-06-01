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
#include "cooperative_groups.h"
#include <cooperative_groups/memcpy_async.h>
#define CUDA_CHECK(err) \
  do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("CUDA error"); \
    } \
  } while (0)

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
swiftware::compression::CompressedFormatGPU<half> *
allocateAndCopyCompressedFormatToHalf(
    swiftware::compression::CompressedFormat<float> *h_cf) {

  swiftware::compression::CompressedFormatGPU<half> *d_cf;
  cudaMalloc((void **)&d_cf,
             sizeof(swiftware::compression::CompressedFormatGPU<half>));

  // Allocate and convert arrays from float to half before copying
  half *d_NNZ;
  int *d_RPP, *d_NPP;
  int *d_PT, *d_CB;

  // Allocate temporary array to hold half-precision values
  half *temp_NNZ = new half[h_cf->nnz];
  for (size_t i = 0; i < h_cf->nnz; ++i) {
    temp_NNZ[i] = __float2half(h_cf->NNZ[i]);
  }

  // Copy the converted half-precision array to the device
  cudaMalloc((void **)&d_NNZ, h_cf->nnz * sizeof(half));
  cudaMemcpy(d_NNZ, temp_NNZ, h_cf->nnz * sizeof(half), cudaMemcpyHostToDevice);
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
      cudaMemcpy(&(d_cf->NNZ), &d_NNZ, sizeof(half *), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(&(d_cf->RPP), &d_RPP, sizeof(int *), cudaMemcpyHostToDevice));
  cudaMemcpy(&(d_cf->NPP), &d_NPP, sizeof(int *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_cf->PT), &d_PT, sizeof(int *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_cf->CB), &d_CB, sizeof(int *), cudaMemcpyHostToDevice);

  return d_cf;
}

swiftware::compression::MixedFormatGPU<half> *allocateAndCopyMixedFormatToHalf(
    swiftware::compression::MixedFormat<float> *h_mixed) {

  swiftware::compression::MixedFormatGPU<half> *d_mixed;
  cudaMalloc((void **)&d_mixed,
             sizeof(swiftware::compression::MixedFormatGPU<half> *));

  // Allocate and copy the array of CompressedFormat pointers
  swiftware::compression::CompressedFormatGPU<half> **d_cf;
  cudaMalloc((void **)&d_cf,
             h_mixed->cf.size() *
                 sizeof(swiftware::compression::CompressedFormatGPU<half> *));

  for (size_t i = 0; i < h_mixed->cf.size(); ++i) {
    swiftware::compression::CompressedFormatGPU<half> *d_cf_element =
        allocateAndCopyCompressedFormatToHalf(h_mixed->cf[i]);
    cudaMemcpy(&(d_cf[i]), &d_cf_element,
               sizeof(swiftware::compression::CompressedFormatGPU<half> *),
               cudaMemcpyHostToDevice);
  }

  cudaMemcpy(&(d_mixed->cf), &d_cf,
             sizeof(swiftware::compression::CompressedFormatGPU<half> *),
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


  return d_mixed;
}
// Mixed Format
struct TensorInputsMixed : public TensorInputsFloat {

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
    
    // Adding the BCSR PART
    #define MMA_M 16
    #define MMA_N 8
    #define MMA_K 16
    
    #define WARP_SIZE 32
    
    #define LDMATRIX_X2(R0, R1, addr) \
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];" : "=r"(R0), "=r"(R1) : "r"(addr))
    
    #define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" \
                     : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                     : "r"(addr))
    
    #define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)                                                    \
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};" \
                     : "=r"(RD0), "=r"(RD1)                                                                                \
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
    
    __device__ void PatternWith3NNZByRowPanel3RowPerThread8BlockCacheLess(
    int *RPP, int *NPP, float *NNZ, int *CB, float *dx, float *dy,int i, int c_cols, int offset0, int offset1, int offset2) {
    int blockSize = blockDim.x;
    int thread_id_in_warp = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int number_of_warps = blockSize / 32;
    int rowStart_all = RPP[i];
    int rowEnd_all = RPP[i + 1];
    if (rowStart_all == rowEnd_all) {
      return;
    }
    int number_of_row_panel_packs = (rowEnd_all - rowStart_all + 8 - 1) / 8;
    int row_panel_pack_share = ((number_of_row_panel_packs + gridDim.y - 1) / gridDim.y) * 8;
    int row_panel_start = rowStart_all + blockIdx.y * row_panel_pack_share;
    int row_panel_end = row_panel_start + row_panel_pack_share;
    row_panel_end = row_panel_end > rowEnd_all ? rowEnd_all : row_panel_end;
    if (row_panel_start >= row_panel_end) {
        return;
    }
    int c_col_partition = (c_cols + 31) / 32;
    int c_col_share = ((c_col_partition + number_of_warps - 1) / number_of_warps) * 32;
    c_col_share = c_col_share == 0 ? 32 : c_col_share;
    int c_col_start = c_col_share * warp_id;
    int c_col_end = c_col_start + c_col_share;
    c_col_end = c_col_end > c_cols ? c_cols : c_col_end;
    if (c_col_start >= c_col_end) {
        return;
    }
    for (int col = c_col_start; col < c_col_end; col += 32) {
    int t_nnz = NPP[i] + (row_panel_start - rowStart_all) * 3;
    float sum_0 = 0.0f;
    float sum_1 = 0.0f;
    float sum_2 = 0.0f;
    for (int j = row_panel_start; j < row_panel_end; j += 8) {
    int sub_b_row_start_0 = CB[j + 0];
    int sub_b_row_start_1 = CB[j + 1];
    int sub_b_row_start_2 = CB[j + 2];
    int sub_b_row_start_3 = CB[j + 3];
    int sub_b_row_start_4 = CB[j + 4];
    int sub_b_row_start_5 = CB[j + 5];
    int sub_b_row_start_6 = CB[j + 6];
    int sub_b_row_start_7 = CB[j + 7];
    int sub_b_col_0 = col + 0 + thread_id_in_warp;
    float nnz0_0 = NNZ[t_nnz + 0];
    float nnz1_0 = NNZ[t_nnz + 1];
    float nnz2_0 = NNZ[t_nnz + 2];
    float nnz0_1 = NNZ[t_nnz + 3];
    float nnz1_1 = NNZ[t_nnz + 4];
    float nnz2_1 = NNZ[t_nnz + 5];
    float nnz0_2 = NNZ[t_nnz + 6];
    float nnz1_2 = NNZ[t_nnz + 7];
    float nnz2_2 = NNZ[t_nnz + 8];
    float nnz0_3 = NNZ[t_nnz + 9];
    float nnz1_3 = NNZ[t_nnz + 10];
    float nnz2_3 = NNZ[t_nnz + 11];
    float nnz0_4 = NNZ[t_nnz + 12];
    float nnz1_4 = NNZ[t_nnz + 13];
    float nnz2_4 = NNZ[t_nnz + 14];
    float nnz0_5 = NNZ[t_nnz + 15];
    float nnz1_5 = NNZ[t_nnz + 16];
    float nnz2_5 = NNZ[t_nnz + 17];
    float nnz0_6 = NNZ[t_nnz + 18];
    float nnz1_6 = NNZ[t_nnz + 19];
    float nnz2_6 = NNZ[t_nnz + 20];
    float nnz0_7 = NNZ[t_nnz + 21];
    float nnz1_7 = NNZ[t_nnz + 22];
    float nnz2_7 = NNZ[t_nnz + 23];
    float b0 = dx[sub_b_row_start_0 * c_cols + sub_b_col_0];
    float b1 = dx[sub_b_row_start_1 * c_cols + sub_b_col_0];
    float b2 = dx[sub_b_row_start_2 * c_cols + sub_b_col_0];
    float b3 = dx[sub_b_row_start_3 * c_cols + sub_b_col_0];
    float b4 = dx[sub_b_row_start_4 * c_cols + sub_b_col_0];
    float b5 = dx[sub_b_row_start_5 * c_cols + sub_b_col_0];
    float b6 = dx[sub_b_row_start_6 * c_cols + sub_b_col_0];
    float b7 = dx[sub_b_row_start_7 * c_cols + sub_b_col_0];
    sum_0 += nnz0_0 * b0 + nnz0_1 * b1 + nnz0_2 * b2 + nnz0_3 * b3 + nnz0_4 * b4 + nnz0_5 * b5 + nnz0_6 * b6 + nnz0_7 * b7 ;
    sum_1 += nnz1_0 * b0 + nnz1_1 * b1 + nnz1_2 * b2 + nnz1_3 * b3 + nnz1_4 * b4 + nnz1_5 * b5 + nnz1_6 * b6 + nnz1_7 * b7 ;
    sum_2 += nnz2_0 * b0 + nnz2_1 * b1 + nnz2_2 * b2 + nnz2_3 * b3 + nnz2_4 * b4 + nnz2_5 * b5 + nnz2_6 * b6 + nnz2_7 * b7 ;
    t_nnz += 24;
    }
    int c_row = i * 3;
    int c_col = col + thread_id_in_warp;
    atomicAdd(&dy[(c_row + offset0) * c_cols + c_col + 0], sum_0);
    atomicAdd(&dy[(c_row + offset1) * c_cols + c_col + 0], sum_1);
    atomicAdd(&dy[(c_row + offset2) * c_cols + c_col + 0], sum_2);
    }
    }


    __device__ void PatternWith3NNZByRowPanel2RowPerThread8BlockCacheLess(
        int *RPP, int *NPP, float *NNZ, int *CB, float *dx, float *dy,int i, int c_cols, int offset0, int offset1) {
    int blockSize = blockDim.x;
    int thread_id_in_warp = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int number_of_warps = blockSize / 32;
    int rowStart_all = RPP[i];
    int rowEnd_all = RPP[i + 1];
    if (rowStart_all == rowEnd_all) {
      return;
    }
    int number_of_row_panel_packs = (rowEnd_all - rowStart_all + 8 - 1) / 8;
    int row_panel_pack_share = ((number_of_row_panel_packs + gridDim.y - 1) / gridDim.y) * 8;
    int row_panel_start = rowStart_all + blockIdx.y * row_panel_pack_share;
    int row_panel_end = row_panel_start + row_panel_pack_share;
    row_panel_end = row_panel_end > rowEnd_all ? rowEnd_all : row_panel_end;
    if (row_panel_start >= row_panel_end) {
        return;
    }
    int c_col_partition = (c_cols + 31) / 32;
    int c_col_share = ((c_col_partition + number_of_warps - 1) / number_of_warps) * 32;
    c_col_share = c_col_share == 0 ? 32 : c_col_share;
    int c_col_start = c_col_share * warp_id;
    int c_col_end = c_col_start + c_col_share;
    c_col_end = c_col_end > c_cols ? c_cols : c_col_end;
    if (c_col_start >= c_col_end) {
        return;
    }
    for (int col = c_col_start; col < c_col_end; col += 32) {
    int t_nnz = NPP[i] + (row_panel_start - rowStart_all) * 2;
    float sum_0 = 0.0f;
    float sum_1 = 0.0f;
    for (int j = row_panel_start; j < row_panel_end; j += 8) {
    int sub_b_row_start_0 = CB[j + 0];
    int sub_b_row_start_1 = CB[j + 1];
    int sub_b_row_start_2 = CB[j + 2];
    int sub_b_row_start_3 = CB[j + 3];
    int sub_b_row_start_4 = CB[j + 4];
    int sub_b_row_start_5 = CB[j + 5];
    int sub_b_row_start_6 = CB[j + 6];
    int sub_b_row_start_7 = CB[j + 7];
    int sub_b_col_0 = col + 0 + thread_id_in_warp;
    float nnz0_0 = NNZ[t_nnz + 0];
    float nnz1_0 = NNZ[t_nnz + 1];
    float nnz0_1 = NNZ[t_nnz + 2];
    float nnz1_1 = NNZ[t_nnz + 3];
    float nnz0_2 = NNZ[t_nnz + 4];
    float nnz1_2 = NNZ[t_nnz + 5];
    float nnz0_3 = NNZ[t_nnz + 6];
    float nnz1_3 = NNZ[t_nnz + 7];
    float nnz0_4 = NNZ[t_nnz + 8];
    float nnz1_4 = NNZ[t_nnz + 9];
    float nnz0_5 = NNZ[t_nnz + 10];
    float nnz1_5 = NNZ[t_nnz + 11];
    float nnz0_6 = NNZ[t_nnz + 12];
    float nnz1_6 = NNZ[t_nnz + 13];
    float nnz0_7 = NNZ[t_nnz + 14];
    float nnz1_7 = NNZ[t_nnz + 15];
    float b0 = dx[sub_b_row_start_0 * c_cols + sub_b_col_0];
    float b1 = dx[sub_b_row_start_1 * c_cols + sub_b_col_0];
    float b2 = dx[sub_b_row_start_2 * c_cols + sub_b_col_0];
    float b3 = dx[sub_b_row_start_3 * c_cols + sub_b_col_0];
    float b4 = dx[sub_b_row_start_4 * c_cols + sub_b_col_0];
    float b5 = dx[sub_b_row_start_5 * c_cols + sub_b_col_0];
    float b6 = dx[sub_b_row_start_6 * c_cols + sub_b_col_0];
    float b7 = dx[sub_b_row_start_7 * c_cols + sub_b_col_0];
    sum_0 += nnz0_0 * b0 + nnz0_1 * b1 + nnz0_2 * b2 + nnz0_3 * b3 + nnz0_4 * b4 + nnz0_5 * b5 + nnz0_6 * b6 + nnz0_7 * b7 ;
    sum_1 += nnz1_0 * b0 + nnz1_1 * b1 + nnz1_2 * b2 + nnz1_3 * b3 + nnz1_4 * b4 + nnz1_5 * b5 + nnz1_6 * b6 + nnz1_7 * b7 ;
    t_nnz += 16;
    }
    int c_row = i * 3;
    int c_col = col + thread_id_in_warp;
    atomicAdd(&dy[(c_row + offset0) * c_cols + c_col + 0], sum_0);
    atomicAdd(&dy[(c_row + offset1) * c_cols + c_col + 0], sum_1);
    }
    }


    __device__ void PatternWith3NNZByRowPanel1RowPerThread8BlockCacheLess(
        int *RPP, int *NPP, float *NNZ, int *CB, float *dx, float *dy,int i, int c_cols, int offset0) {
    int blockSize = blockDim.x;
    int thread_id_in_warp = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int number_of_warps = blockSize / 32;
    int rowStart_all = RPP[i];
    int rowEnd_all = RPP[i + 1];
    if (rowStart_all == rowEnd_all) {
      return;
    }
    int number_of_row_panel_packs = (rowEnd_all - rowStart_all + 8 - 1) / 8;
    int row_panel_pack_share = ((number_of_row_panel_packs + gridDim.y - 1) / gridDim.y) * 8;
    int row_panel_start = rowStart_all + blockIdx.y * row_panel_pack_share;
    int row_panel_end = row_panel_start + row_panel_pack_share;
    row_panel_end = row_panel_end > rowEnd_all ? rowEnd_all : row_panel_end;
    if (row_panel_start >= row_panel_end) {
        return;
    }
    int c_col_partition = (c_cols + 31) / 32;
    int c_col_share = ((c_col_partition + number_of_warps - 1) / number_of_warps) * 32;
    c_col_share = c_col_share == 0 ? 32 : c_col_share;
    int c_col_start = c_col_share * warp_id;
    int c_col_end = c_col_start + c_col_share;
    c_col_end = c_col_end > c_cols ? c_cols : c_col_end;
    if (c_col_start >= c_col_end) {
        return;
    }
    for (int col = c_col_start; col < c_col_end; col += 32) {
    int t_nnz = NPP[i] + (row_panel_start - rowStart_all) * 1;
    float sum_0 = 0.0f;
    for (int j = row_panel_start; j < row_panel_end; j += 8) {
    int sub_b_row_start_0 = CB[j + 0];
    int sub_b_row_start_1 = CB[j + 1];
    int sub_b_row_start_2 = CB[j + 2];
    int sub_b_row_start_3 = CB[j + 3];
    int sub_b_row_start_4 = CB[j + 4];
    int sub_b_row_start_5 = CB[j + 5];
    int sub_b_row_start_6 = CB[j + 6];
    int sub_b_row_start_7 = CB[j + 7];
    int sub_b_col_0 = col + 0 + thread_id_in_warp;
    float nnz0_0 = NNZ[t_nnz + 0];
    float nnz0_1 = NNZ[t_nnz + 1];
    float nnz0_2 = NNZ[t_nnz + 2];
    float nnz0_3 = NNZ[t_nnz + 3];
    float nnz0_4 = NNZ[t_nnz + 4];
    float nnz0_5 = NNZ[t_nnz + 5];
    float nnz0_6 = NNZ[t_nnz + 6];
    float nnz0_7 = NNZ[t_nnz + 7];
    float b0 = dx[sub_b_row_start_0 * c_cols + sub_b_col_0];
    float b1 = dx[sub_b_row_start_1 * c_cols + sub_b_col_0];
    float b2 = dx[sub_b_row_start_2 * c_cols + sub_b_col_0];
    float b3 = dx[sub_b_row_start_3 * c_cols + sub_b_col_0];
    float b4 = dx[sub_b_row_start_4 * c_cols + sub_b_col_0];
    float b5 = dx[sub_b_row_start_5 * c_cols + sub_b_col_0];
    float b6 = dx[sub_b_row_start_6 * c_cols + sub_b_col_0];
    float b7 = dx[sub_b_row_start_7 * c_cols + sub_b_col_0];
    sum_0 += nnz0_0 * b0 + nnz0_1 * b1 + nnz0_2 * b2 + nnz0_3 * b3 + nnz0_4 * b4 + nnz0_5 * b5 + nnz0_6 * b6 + nnz0_7 * b7 ;
    t_nnz += 8;
    }
    int c_row = i * 3;
    int c_col = col + thread_id_in_warp;
    atomicAdd(&dy[(c_row + offset0) * c_cols + c_col + 0], sum_0);
  }
}


__global__ void
SpMMFullRowPanelGeneralKernel3With8BlockCacheLess(swiftware::compression::MixedFormatGPU<float> *mf, float *dX, float *dY, int c_cols) {
  int i = blockIdx.x;
  int pattern = i % 7;
  switch (pattern) {
    case 0:
      PatternWith3NNZByRowPanel3RowPerThread8BlockCacheLess(
          mf->cf[0]->RPP, mf->cf[0]->NPP, mf->cf[0]->NNZ, mf->cf[0]->CB, dX, dY,
          i / 7, c_cols , 0 , 1 , 2);
      break;
    case 1:
      PatternWith3NNZByRowPanel2RowPerThread8BlockCacheLess(
          mf->cf[1]->RPP, mf->cf[1]->NPP, mf->cf[1]->NNZ, mf->cf[1]->CB, dX, dY,
          i / 7, c_cols , 0 , 1);
      break;
    case 2:
      PatternWith3NNZByRowPanel2RowPerThread8BlockCacheLess(
          mf->cf[2]->RPP, mf->cf[2]->NPP, mf->cf[2]->NNZ, mf->cf[2]->CB, dX, dY,
          i / 7, c_cols , 0 , 2);
      break;
    case 3:
      PatternWith3NNZByRowPanel1RowPerThread8BlockCacheLess(
          mf->cf[3]->RPP, mf->cf[3]->NPP, mf->cf[3]->NNZ, mf->cf[3]->CB, dX, dY,
          i / 7, c_cols , 0);
      break;
    case 4:
      PatternWith3NNZByRowPanel2RowPerThread8BlockCacheLess(
          mf->cf[4]->RPP, mf->cf[4]->NPP, mf->cf[4]->NNZ, mf->cf[4]->CB, dX, dY,
          i / 7, c_cols , 1 , 2);
      break;
    case 5:
      PatternWith3NNZByRowPanel1RowPerThread8BlockCacheLess(
          mf->cf[5]->RPP, mf->cf[5]->NPP, mf->cf[5]->NNZ, mf->cf[5]->CB, dX, dY,
          i / 7, c_cols , 1);
      break;
    case 6:
      PatternWith3NNZByRowPanel1RowPerThread8BlockCacheLess(
          mf->cf[6]->RPP, mf->cf[6]->NPP, mf->cf[6]->NNZ, mf->cf[6]->CB, dX, dY,
          i / 7, c_cols , 2);
      break;
  }
}


__global__ void
SingleSpMMFullRowPanelGeneralKernel3With8BlockCacheLess(swiftware::compression::MixedFormatGPU<float> *mf, float *dX, float *dY, int c_cols) {
  int i = blockIdx.x;
  int pattern = i % 7;
  PatternWith3NNZByRowPanel3RowPerThread8BlockCacheLess(
          mf->cf[0]->RPP, mf->cf[0]->NPP, mf->cf[0]->NNZ, mf->cf[0]->CB, dX, dY,
          i / 7, c_cols , 0 , 1 , 2);
}
class SpMMMixePatternGeneralCacheLess: public SpMMSerialFloat {
    protected:
        float *dX, *dY;

float *x;
swiftware::compression::MixedFormatGPU<float> *mf;

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

dim3 gridDim((InT->MixedFormat->cf[0]->nrpp - 1) * 7 , 1);
SpMMFullRowPanelGeneralKernel3With8BlockCacheLess<<<gridDim,
64>>>(mf, dX, dY, InTensor->Z);

t1.stopGPU("");

cudaMemcpy(this->OutTensor->Dx, dY,
this->InTensor->ACsr->m * InTensor->Z * sizeof(float),
cudaMemcpyDeviceToHost);
cudaMemset(dY, 0, InTensor->M * InTensor->Z * sizeof(float));

return t1;
}

public:
SpMMMixePatternGeneralCacheLess(TensorInputsMixed *In1,
swiftware::benchmark::Stats *Stat1)
: SpMMSerialFloat(In1, Stat1), InT(In1) {

x = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));
for (int i = 0; i < In1->ACsr->nnz; i++) {
    x[i] = static_cast<float>(In1->ACsr->x[i]);
}

cudaMalloc((void **)&dX, InTensor->N * InTensor->Z * sizeof(float));
cudaMalloc((void **)&dY, InTensor->M * InTensor->Z * sizeof(float));
//
cudaMemset(dY, 0, InTensor->M * InTensor->Z * sizeof(float));
cudaMemcpy(dX, this->InTensor->Bx,
InTensor->N * InTensor->Z * sizeof(float),
cudaMemcpyHostToDevice);

mf = allocateAndCopyMixedFormat(In1->MixedFormat);
};

~SpMMMixePatternGeneralCacheLess() {
//    cudaFree(dX);
//    cudaFree(dY);
}
};



class SingleSpMMMixePatternGeneralCacheLess: public SpMMSerialFloat {
    protected:
        float *dX, *dY;

float *x;
swiftware::compression::MixedFormatGPU<float> *mf;

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

dim3 gridDim((InT->MixedFormat->cf[0]->nrpp - 1), 1);
SingleSpMMFullRowPanelGeneralKernel3With8BlockCacheLess<<<gridDim,
64>>>(mf, dX, dY, InTensor->Z);

t1.stopGPU("");

cudaMemcpy(this->OutTensor->Dx, dY,
this->InTensor->ACsr->m * InTensor->Z * sizeof(float),
cudaMemcpyDeviceToHost);
cudaMemset(dY, 0, InTensor->M * InTensor->Z * sizeof(float));

return t1;
}

public:
SingleSpMMMixePatternGeneralCacheLess(TensorInputsMixed *In1,
swiftware::benchmark::Stats *Stat1)
: SpMMSerialFloat(In1, Stat1), InT(In1) {

x = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));
for (int i = 0; i < In1->ACsr->nnz; i++) {
    x[i] = static_cast<float>(In1->ACsr->x[i]);
}

cudaMalloc((void **)&dX, InTensor->N * InTensor->Z * sizeof(float));
cudaMalloc((void **)&dY, InTensor->M * InTensor->Z * sizeof(float));
//
cudaMemset(dY, 0, InTensor->M * InTensor->Z * sizeof(float));
cudaMemcpy(dX, this->InTensor->Bx,
InTensor->N * InTensor->Z * sizeof(float),
cudaMemcpyHostToDevice);

mf = allocateAndCopyMixedFormat(In1->MixedFormat);
};

~SingleSpMMMixePatternGeneralCacheLess() {
//    cudaFree(dX);
//    cudaFree(dY);
}
};
#endif // COMPRESSED_TENSOR_ALGEBRA_SPMM_SMALL_B_COL_DEMO_GPU_UTILS_H
