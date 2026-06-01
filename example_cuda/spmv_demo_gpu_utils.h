#ifndef COMPRESSED_TENSOR_ALGEBRA_SPMV_DEMO_GPU_UTILS_H
#define COMPRESSED_TENSOR_ALGEBRA_SPMV_DEMO_GPU_UTILS_H

#include "Logger.h"
#include "SWTensorBench.h"
#include "aggregation/def.h"
#include "compressed_format/CompressedFormat.h"
#include "iostream"
#include <cooperative_groups.h>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMV
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE         // EXIT_FAILURE

// CUDA API error checking
#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);       \
      throw std::runtime_error("CUDA error");                                  \
    }                                                                          \
  } while (0)

namespace cg = cooperative_groups;
// cublas API error checking
#define CUBLAS_CHECK(err)                                                      \
  do {                                                                         \
    cublasStatus_t err_ = (err);                                               \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
      throw std::runtime_error("cublas error");                                \
    }                                                                          \
  } while (0)

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

void generateDenseFromCsr(int M, int N, sym_lib::CSR *A, float *X) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      X[i * N + j] = 0.0;
    }
  }
  for (int i = 0; i < M; ++i) {
    for (int j = A->p[i]; j < A->p[i + 1]; ++j) {
      X[i * N + A->i[j]] = A->x[j];
    }
  }
}

template <typename T>
struct TensorOutputs : public swiftware::benchmark::Outputs<T> {
  int M;
  T *Dx;

  TensorOutputs(int M) : M(M) { Dx = new T[M](); }

  ~TensorOutputs() { delete[] Dx; }

  void printDx() {
    std::cout << "\n Dx:\n";
    //    printDense<T>(M, 1, Dx);
    std::cout << "\n";
  }

  void reset() { std::fill_n(Dx, M, 0.0); }
};

void spmv_csr_float(int n, const int *Ap, const int *Ai, const float *Ax,
                    const float *x, float *y) {

  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = Ap[i]; j < Ap[i + 1]; j++) {
      int col_index = Ai[j];
      double value = Ax[j];
      sum += value * x[col_index];
    }
    y[i] += sum;
  }
}

struct TensorInputsFloat : public swiftware::benchmark::Inputs<float> {
  int M, N;
  sym_lib::CSR *ACsr;

  float *Bx;
  float *CorrectMul;
  bool IsSolProvided;

  TensorInputsFloat(int M1, int N1, sym_lib::CSR *A1, int NumThreads1,
                    int NumTrial1, std::string ExpN)
      : swiftware::benchmark::Inputs<float>(NumTrial1, NumThreads1, ExpN) {
    M = M1;
    N = N1;
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
class SpMVSerialFloat : public swiftware::benchmark::SWTensorBench<float> {
protected:
  TensorInputsFloat *InTensor;
  float *csr_x;
  void setup() override { this->St->OtherStats["NTile"] = {4}; }

  void preExecute() override {}

  swiftware::benchmark::Timer execute() override {
    OutTensor->reset();
    swiftware::benchmark::Timer t1;
    t1.start();

    spmv_csr_float(InTensor->M, InTensor->ACsr->p, InTensor->ACsr->i, csr_x,
                   InTensor->Bx, OutTensor->Dx);

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
    for (int i = 0; i < InTensor->M - 1; ++i) {

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
  SpMVSerialFloat(TensorInputsFloat *In1, swiftware::benchmark::Stats *Stat1)
      : swiftware::benchmark::SWTensorBench<float>(In1, Stat1) {
    OutTensor = new TensorOutputs<float>(In1->M);
    InTensor = In1;
    csr_x =
        static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));
    for (int i = 0; i < In1->ACsr->nnz; i++) {
      csr_x[i] = static_cast<float>(In1->ACsr->x[i]);
    }
  };

  virtual void envelopCalculation() {}

  ~SpMVSerialFloat() {
    delete OutTensor;
    free(csr_x);
  }
};

class SpMVcuSparse : public SpMVSerialFloat {
  float *x;
  int *dA_csrOffsets, *dA_columns;
  float *dA_values, *dX, *dY;
  // CUSPARSE APIs
  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;
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
    for (int i = 0; i < InTensor->M - 1; ++i) {
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
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
                 &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    t1.stopGPU("");
    cudaMemcpy(this->OutTensor->Dx, dY, this->InTensor->ACsr->m * sizeof(float),
               cudaMemcpyDeviceToHost);
    return t1;
  }
public:
  SpMVcuSparse(TensorInputsFloat *In1, swiftware::benchmark::Stats *Stat1)
      : SpMVSerialFloat(In1, Stat1) {
    x = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));
    for (int i = 0; i < In1->ACsr->nnz; i++) {
      x[i] = static_cast<float>(In1->ACsr->x[i]);
    }
    cudaMalloc((void **)&dA_csrOffsets,
               (this->InTensor->ACsr->m + 1) * sizeof(int));
    cudaMalloc((void **)&dA_columns, this->InTensor->ACsr->nnz * sizeof(int));
    cudaMalloc((void **)&dA_values, this->InTensor->ACsr->nnz * sizeof(float));
    cudaMalloc((void **)&dX, this->InTensor->ACsr->n * sizeof(float));
    cudaMalloc((void **)&dY, this->InTensor->ACsr->m * sizeof(float));
    cudaMemcpy(dA_values, this->x, this->InTensor->ACsr->nnz * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dA_csrOffsets, this->InTensor->ACsr->p,
               (this->InTensor->ACsr->m + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dA_columns, this->InTensor->ACsr->i,
               this->InTensor->ACsr->nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, this->InTensor->Bx, this->InTensor->ACsr->n * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dY, OutTensor->Dx, this->InTensor->ACsr->m * sizeof(float),
               cudaMemcpyHostToDevice);
    cusparseCreate(&handle);
    cusparseCreateCsr(&matA, this->InTensor->ACsr->m, this->InTensor->ACsr->n,
                      this->InTensor->ACsr->nnz, dA_csrOffsets, dA_columns,
                      dA_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vecX, this->InTensor->ACsr->n, dX, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, this->InTensor->ACsr->m, dY, CUDA_R_32F);
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                            matA, vecX, &beta, vecY, CUDA_R_32F,
                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
  }
  ~SpMVcuSparse() {
    free(x);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    cudaFree(dBuffer);
    cudaFree(dA_csrOffsets);
    cudaFree(dA_columns);
    cudaFree(dA_values);
    cudaFree(dX);
    cudaFree(dY);
  }
};
class SpMVcuBlas : public SpMVSerialFloat {

  float *dA, *dX, *dY;
  float alpha = 1.0f;
  float beta = 0.0f;

  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;
  const int incx = 1;
  const int incy = 1;
  cublasOperation_t transa = CUBLAS_OP_N;

  swiftware::benchmark::Timer execute() override {

    OutTensor->reset();
    swiftware::benchmark::Timer t1;
    t1.startGPU();
    CUBLAS_CHECK(cublasSgemv(cublasH, transa, InTensor->ACsr->m,
                             InTensor->ACsr->n, &alpha, dA, InTensor->ACsr->m,
                             dX, incx, &beta, dY, incy));
    t1.stopGPU("");
    cudaMemcpy(this->OutTensor->Dx, dY, this->InTensor->ACsr->m * sizeof(float),
               cudaMemcpyDeviceToHost);
    return t1;
  }
public:
  SpMVcuBlas(TensorInputsFloat *In1, swiftware::benchmark::Stats *Stat1)
      : SpMVSerialFloat(In1, Stat1) {

    cublasCreate(&cublasH);

    // transpose the matrix
    float *NA = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->m * In1->ACsr->n));
    generateDenseFromCsr(In1->ACsr->m, In1->ACsr->n, In1->ACsr, NA);

    float *Trans_NA = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->m * In1->ACsr->n));
    for (int i = 0; i < In1->ACsr->m; i++) {
      for (int j = 0; j < In1->ACsr->n; j++) {
        Trans_NA[j * In1->ACsr->m + i] = NA[i * In1->ACsr->n + j];
      }
    }

    cudaMalloc(&dA, sizeof(float) * In1->ACsr->m * In1->ACsr->n);
    cudaMalloc((void **)&dX, this->InTensor->ACsr->n * sizeof(float));
    cudaMalloc((void **)&dY, this->InTensor->ACsr->m * sizeof(float));

    cudaMemcpy(dA, Trans_NA, sizeof(float) * In1->ACsr->m * In1->ACsr->n,
               cudaMemcpyHostToDevice);
    cudaMemcpy(dX, this->InTensor->Bx, sizeof(float) * In1->ACsr->n,
               cudaMemcpyHostToDevice);

    free(NA);
    free(Trans_NA);
  };

  ~SpMVcuBlas() {
    /* free resources */
    cudaFree(dA);
    cudaFree(dX);
    cudaFree(dY);
    cublasDestroy(cublasH);

    cudaStreamDestroy(stream);

    cudaDeviceReset();
  }
};
template <typename T, int THREADS_PER_VECTOR>
__global__ void spmv_complex_kernel(int *d_ptr, int *d_cols, T *d_val,
                                    T *d_vector, T *d_out, int N) {
  int laneId = threadIdx.x % THREADS_PER_VECTOR; // Lane index in the vector
  int vectorId = threadIdx.x / THREADS_PER_VECTOR; // Vector index in the thread block
  int warpLaneId = threadIdx.x % 32;    // Lane index in the warp
  int warpVectorId = warpLaneId / THREADS_PER_VECTOR; // Vector index in the warp
  __shared__ volatile int space[THREADS_PER_VECTOR][2];
  int blockRows = (N + gridDim.x - 1) / gridDim.x;
  int startRow = blockIdx.x * blockRows;
  int endRow = min(startRow + blockRows, N);
  for (int row = startRow + warpVectorId; row < endRow;
       row += 32 / THREADS_PER_VECTOR) {
    if (laneId < 2) {
      space[vectorId][laneId] = d_ptr[row + laneId];
    }
    int rowStart = space[vectorId][0];
    int rowEnd = space[vectorId][1];
    T sum = 0;
    for (int i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
      sum += d_val[i] * d_vector[d_cols[i]];
    }
    for (int i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, i);
    }
    if (laneId == 0) {
      d_out[row] += sum;
    }
  }
}
template <typename T>
__global__ void spmv_complex_kernel_v2(int *d_ptr, int *d_cols, T *d_val,
                                    T *d_vector, T *d_out, int N) {
  int THREADS_PER_VECTOR = 32;
  int laneId = threadIdx.x % THREADS_PER_VECTOR; // Lane index in the vector
  int vectorId = threadIdx.x / THREADS_PER_VECTOR; // Vector index in the thread block
  int warpLaneId = threadIdx.x % 32;    // Lane index in the warp
  int warpVectorId = warpLaneId / THREADS_PER_VECTOR; // Vector index in the warp
  __shared__ volatile int space[32][2];
  int blockRows = 1;
  int startRow = blockIdx.x * 3 + vectorId;
  int endRow = min(startRow + blockRows, N);
  for (int row = startRow + warpVectorId; row < endRow;
       row += 32 / THREADS_PER_VECTOR) {
    if (laneId < 2) {
      space[vectorId][laneId] = d_ptr[row + laneId];
    }
    int rowStart = space[vectorId][0];
    int rowEnd = space[vectorId][1];
    T sum = 0;
    for (int i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
      sum += d_val[i] * d_vector[d_cols[i]];
    }
    for (int i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, i);
    }
    if (laneId == 0) {
      d_out[row] += sum;
    }
  }
}

class SpMVComplexKernel : public SpMVSerialFloat {
  float *x;
  int *dA_csrOffsets, *dA_columns;
  float *dA_values, *dX, *dY;
  int *m_gpu, *n_gpu, *nnz_gpu;
  int BlockDim = 256;
  float alpha = 1.0f;
  swiftware::benchmark::Timer execute() override {
    OutTensor->reset();
    swiftware::benchmark::Timer t1;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, spmv_complex_kernel<float, 32>);
    t1.startGPU();
    spmv_complex_kernel<float, 32><<<InTensor->ACsr->m, 32>>>(
        dA_csrOffsets, dA_columns, dA_values, dX, dY, InTensor->ACsr->m);
    t1.stopGPU("");
    cudaMemcpy(this->OutTensor->Dx, dY, this->InTensor->ACsr->m * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemset(m_gpu, 0, 2 * sizeof(int));
    cudaMemset(dY, 0, this->InTensor->ACsr->m * sizeof(float));
    return t1;
  }
public:
  SpMVComplexKernel(TensorInputsFloat *In1, swiftware::benchmark::Stats *Stat1)
      : SpMVSerialFloat(In1, Stat1) {
    x = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));
    for (int i = 0; i < In1->ACsr->nnz; i++) {
      x[i] = static_cast<float>(In1->ACsr->x[i]);
    }
    cudaMalloc((void **)&dA_csrOffsets,
               (this->InTensor->ACsr->m + 1) * sizeof(int));
    cudaMalloc((void **)&dA_columns, this->InTensor->ACsr->nnz * sizeof(int));
    cudaMalloc((void **)&dA_values, this->InTensor->ACsr->nnz * sizeof(float));
    cudaMalloc((void **)&dX, this->InTensor->ACsr->n * sizeof(float));
    cudaMalloc((void **)&dY, this->InTensor->ACsr->m * sizeof(float));
    cudaMemcpy(dA_values, this->x, this->InTensor->ACsr->nnz * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dA_csrOffsets, this->InTensor->ACsr->p,
               (this->InTensor->ACsr->m + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dA_columns, this->InTensor->ACsr->i,
               this->InTensor->ACsr->nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, this->InTensor->Bx, this->InTensor->ACsr->n * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dY, OutTensor->Dx, this->InTensor->ACsr->m * sizeof(float),
               cudaMemcpyHostToDevice);
  };
  ~SpMVComplexKernel() {
    free(x);
    cudaFree(dA_csrOffsets);
    cudaFree(dA_columns);
    cudaFree(dA_values);
    cudaFree(dX);
    cudaFree(dY);
  }
};

class SpMVComplexKernelV2 : public SpMVSerialFloat {

  float *x;
  int *dA_csrOffsets, *dA_columns;
  float *dA_values, *dX, *dY;

  int *m_gpu, *n_gpu, *nnz_gpu;

  int BlockDim = 256;
  float alpha = 1.0f;

  swiftware::benchmark::Timer execute() override {

    OutTensor->reset();
    swiftware::benchmark::Timer t1;

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, spmv_complex_kernel<float, 32>);

    t1.startGPU();

    spmv_complex_kernel_v2<float><<< (InTensor->ACsr->m + 1 )/ 3, 96>>>(
        dA_csrOffsets, dA_columns, dA_values, dX, dY, InTensor->ACsr->m);

    t1.stopGPU("");
    cudaMemcpy(this->OutTensor->Dx, dY, this->InTensor->ACsr->m * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemset(m_gpu, 0, 2 * sizeof(int));
    cudaMemset(dY, 0, this->InTensor->ACsr->m * sizeof(float));

    return t1;
  }

public:
  SpMVComplexKernelV2(TensorInputsFloat *In1, swiftware::benchmark::Stats *Stat1)
      : SpMVSerialFloat(In1, Stat1) {
    x = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));
    for (int i = 0; i < In1->ACsr->nnz; i++) {
      x[i] = static_cast<float>(In1->ACsr->x[i]);
    }

    cudaMalloc((void **)&dA_csrOffsets,
               (this->InTensor->ACsr->m + 1) * sizeof(int));
    cudaMalloc((void **)&dA_columns, this->InTensor->ACsr->nnz * sizeof(int));
    cudaMalloc((void **)&dA_values, this->InTensor->ACsr->nnz * sizeof(float));
    cudaMalloc((void **)&dX, this->InTensor->ACsr->n * sizeof(float));
    cudaMalloc((void **)&dY, this->InTensor->ACsr->m * sizeof(float));

    cudaMemcpy(dA_values, this->x, this->InTensor->ACsr->nnz * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dA_csrOffsets, this->InTensor->ACsr->p,
               (this->InTensor->ACsr->m + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dA_columns, this->InTensor->ACsr->i,
               this->InTensor->ACsr->nnz * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(dX, this->InTensor->Bx, this->InTensor->ACsr->n * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dY, OutTensor->Dx, this->InTensor->ACsr->m * sizeof(float),
               cudaMemcpyHostToDevice);

    //    CHECK_CUDA_ERROR(cudaGetLastError());
  }
  ~SpMVComplexKernelV2() {
    free(x);
    cudaFree(dA_csrOffsets);
    cudaFree(dA_columns);
    cudaFree(dA_values);
    cudaFree(dX);
    cudaFree(dY);
  }
};
// Mixed Format
struct TensorInputsMixed : public TensorInputsFloat {

public:
  swiftware::compression::MixedFormat<float> *MixedFormat;
  TensorInputsMixed(int M1, int N1, sym_lib::CSR *A1,
                    swiftware::compression::MixedFormat<float> *Mixed,
                    int NumThreads1, int NumTrial1, std::string ExpN)
      : TensorInputsFloat(M1, N1, A1, NumThreads1, NumTrial1, ExpN),
        MixedFormat(Mixed){}
};
template <typename T>
swiftware::compression::CompressedFormatGPU<T> *allocateAndCopyCompressedFormat(
    swiftware::compression::CompressedFormat<T> *h_cf) {
  swiftware::compression::CompressedFormatGPU<T> *d_cf;
  cudaMalloc((void **)&d_cf,
             sizeof(swiftware::compression::CompressedFormatGPU<T>));
  cudaMemcpy(d_cf, h_cf, sizeof(swiftware::compression::CompressedFormat<T>),
             cudaMemcpyHostToDevice);

  // Allocate and copy NNZ, cols, RPP, NPP, PT, CB, CSRRows arrays
  T *d_NNZ;
  int *d_cols, *d_RPP, *d_NPP;
  int *d_PT, *d_CB, *d_CSRRows;

  cudaMalloc((void **)&d_NNZ, h_cf->nnz * sizeof(T));
  cudaMemcpy(d_NNZ, h_cf->NNZ, h_cf->nnz * sizeof(T), cudaMemcpyHostToDevice);

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

  // Update pointers in the device CompressedFormat
  CUDA_CHECK(
      cudaMemcpy(&(d_cf->NNZ), &d_NNZ, sizeof(T *), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(&(d_cf->RPP), &d_RPP, sizeof(int *), cudaMemcpyHostToDevice));
  cudaMemcpy(&(d_cf->NPP), &d_NPP, sizeof(int *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_cf->PT), &d_PT, sizeof(int *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_cf->CB), &d_CB, sizeof(int *), cudaMemcpyHostToDevice);

  return d_cf;
}
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
  float sum_all = 0.0f;
  int j = rowStart;
  for ( j ; j < rowEnd - 31; j += 32) {
    float nnz = (thread_id_in_warp < 32) ? NNZ[t_nnz + thread_id_in_warp] : 0;
    float B = 0.0f;
    B = dx[CB[j + thread_id_in_warp / 1]];

    sum_all += nnz * B;
    t_nnz += 32;
  }
  int tail = rowEnd - j;
  int rest_nnz = 1 * tail; 
  if (thread_id_in_warp < rest_nnz && tail > 0) {
    float nnz = NNZ[t_nnz + thread_id_in_warp];
    float B = 0.0f;
    B = dx[CB[j + thread_id_in_warp / 1]];
    sum_all += nnz * B;
  }
  sum_all += __shfl_down_sync(0xFFFFFFFF, sum_all, 1);
  sum_all += __shfl_down_sync(0xFFFFFFFF, sum_all, 2);
  sum_all += __shfl_down_sync(0xFFFFFFFF, sum_all, 4);
  sum_all += __shfl_down_sync(0xFFFFFFFF, sum_all, 8);
  sum_all += __shfl_down_sync(0xFFFFFFFF, sum_all, 16);
  if (thread_id_in_warp == 0) {
      atomicAdd(&dy[i * 4 + offset0], sum_all);
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
  float sum_all = 0.0f;
  int j = rowStart;
  for ( j ; j < rowEnd - 15; j += 16) {
    float nnz = (thread_id_in_warp < 32) ? NNZ[t_nnz + thread_id_in_warp] : 0;
    float B = 0.0f;
    B = dx[CB[j + thread_id_in_warp / 2]];

    sum_all += nnz * B;
    t_nnz += 32;
  }
  int tail = rowEnd - j;
  int rest_nnz = 2 * tail; 
  if (thread_id_in_warp < rest_nnz && tail > 0) {
    float nnz = NNZ[t_nnz + thread_id_in_warp];
    float B = 0.0f;
    B = dx[CB[j + thread_id_in_warp / 2]];
    sum_all += nnz * B;
  }
  sum_all += __shfl_down_sync(0xFFFFFFFF, sum_all, 2);
  sum_all += __shfl_down_sync(0xFFFFFFFF, sum_all, 4);
  sum_all += __shfl_down_sync(0xFFFFFFFF, sum_all, 8);
  sum_all += __shfl_down_sync(0xFFFFFFFF, sum_all, 16);
  if (thread_id_in_warp == 0) {
      atomicAdd(&dy[i * 4 + offset0], sum_all);
  }
  else if (thread_id_in_warp == 1){
    atomicAdd(&dy[i * 4 + offset1], sum_all);
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
  float sum_all = 0.0f;
  int j = rowStart;
  for ( j ; j < rowEnd - 9; j += 10) {
    float nnz = (thread_id_in_warp < 30) ? NNZ[t_nnz + thread_id_in_warp] : 0;
    float B = 0.0f;
    B = dx[CB[j + thread_id_in_warp / 3]];

    sum_all += nnz * B;
    t_nnz += 30;
  }
  int tail = rowEnd - j;
  int rest_nnz = 3 * tail; 
  if (thread_id_in_warp < rest_nnz && tail > 0) {
    float nnz = NNZ[t_nnz + thread_id_in_warp];
    float B = 0.0f;
    B = dx[CB[j + thread_id_in_warp / 3]];
    sum_all += nnz * B;
  }
  sum_all += __shfl_down_sync(0xFFFFFFFF, sum_all, 3);
  float temp_sum0 = sum_all;
  sum_all += __shfl_down_sync(0xFFFFFFFF, sum_all, 6);
  sum_all += __shfl_down_sync(0xFFFFFFFF, sum_all, 12);
  sum_all += __shfl_down_sync(0xFFFFFFFF, temp_sum0, 24);
  if (thread_id_in_warp == 0) {
      atomicAdd(&dy[i * 4 + offset0], sum_all);
  }
  else if (thread_id_in_warp == 1){
    atomicAdd(&dy[i * 4 + offset1], sum_all);
  }
  else if (thread_id_in_warp == 2){
    atomicAdd(&dy[i * 4 + offset2], sum_all);
  }
}
__device__ void PatternWith4NNZByRowPanelFull(int *RPP, int *NPP, float *NNZ,
                                              int *CB, float *dx, float *dy, int i , int offset0, int offset1, int offset2, int offset3) {

  int blockSize = blockDim.x;
  int thread_id_in_warp = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;


  int rowStart_all = RPP[i];
  int rowEnd_all = RPP[i + 1];
  int number_of_packs = (rowEnd_all - rowStart_all) / 8;
  if (rowEnd_all - rowStart_all == 0) {
    return;
  }

  number_of_packs = number_of_packs == 0 ? 1 : number_of_packs;
  int number_of_warps = blockSize / 32;
  int pack_share = (number_of_packs + number_of_warps - 1) / number_of_warps;
  int warp_share = pack_share * 8;
  int rowStart = rowStart_all + warp_id * warp_share;
  int rowEnd = rowStart + warp_share;
  rowEnd = rowEnd > rowEnd_all ? rowEnd_all : rowEnd;
  rowEnd = warp_id == number_of_warps - 1 && rowEnd < rowEnd_all ? rowEnd_all : rowEnd;
  int t_nnz = NPP[i] + warp_id * warp_share * 4;
  float sum_all = 0.0f;
  int j = rowStart;
  for ( j ; j < rowEnd - 7; j += 8) {
    float nnz = (thread_id_in_warp < 32) ? NNZ[t_nnz + thread_id_in_warp] : 0;
    float B = 0.0f;
    B = dx[CB[j + thread_id_in_warp / 4]];

    sum_all += nnz * B;
    t_nnz += 32;
  }
  int tail = rowEnd - j;
  int rest_nnz = 4 * tail; 
  if (thread_id_in_warp < rest_nnz && tail > 0) {
    float nnz = NNZ[t_nnz + thread_id_in_warp];
    float B = 0.0f;
    B = dx[CB[j + thread_id_in_warp / 4]];
    sum_all += nnz * B;
  }
  sum_all += __shfl_down_sync(0xFFFFFFFF, sum_all, 4);
  sum_all += __shfl_down_sync(0xFFFFFFFF, sum_all, 8);
  sum_all += __shfl_down_sync(0xFFFFFFFF, sum_all, 16);
  if (thread_id_in_warp == 0) {
      atomicAdd(&dy[i * 4 + offset0], sum_all);
  }
  else if (thread_id_in_warp == 1){
    atomicAdd(&dy[i * 4 + offset1], sum_all);
  }
  else if (thread_id_in_warp == 2){
    atomicAdd(&dy[i * 4 + offset2], sum_all);
  }
  else if (thread_id_in_warp == 3){
    atomicAdd(&dy[i * 4 + offset3], sum_all);
  }
}

__global__ void SpmvComplexPatternDecomposeByRowSeperated( 
   swiftware::compression::MixedFormatGPU<float> *mf, float *dX, float *dY) {
  int j = blockIdx.x / 15;
  int i = blockIdx.x % 15;

  switch (i) {
    case 0:
      PatternWith4NNZByRowPanelFull(mf->cf[0]->RPP, mf->cf[0]->NPP, mf->cf[0]->NNZ, mf->cf[0]->CB, dX, dY, j , 0, 1, 2, 3);
      break;
    case 1:
      PatternWith3NNZByRowPanelFull(mf->cf[1]->RPP, mf->cf[1]->NPP, mf->cf[1]->NNZ, mf->cf[1]->CB, dX, dY, j , 0, 1, 2);
      break;
    case 2:
      PatternWith3NNZByRowPanelFull(mf->cf[2]->RPP, mf->cf[2]->NPP, mf->cf[2]->NNZ, mf->cf[2]->CB, dX, dY, j , 0, 1, 3);
      break;
    case 3:
      PatternWith2NNZByRowPanelFull(mf->cf[3]->RPP, mf->cf[3]->NPP, mf->cf[3]->NNZ, mf->cf[3]->CB, dX, dY, j , 0, 1);
      break;
    case 4:
      PatternWith3NNZByRowPanelFull(mf->cf[4]->RPP, mf->cf[4]->NPP, mf->cf[4]->NNZ, mf->cf[4]->CB, dX, dY, j , 0, 2, 3);
      break;
    case 5:
      PatternWith2NNZByRowPanelFull(mf->cf[5]->RPP, mf->cf[5]->NPP, mf->cf[5]->NNZ, mf->cf[5]->CB, dX, dY, j , 0, 2);
      break;
    case 6:
      PatternWith2NNZByRowPanelFull(mf->cf[6]->RPP, mf->cf[6]->NPP, mf->cf[6]->NNZ, mf->cf[6]->CB, dX, dY, j , 0, 3);
      break;
    case 7:
      PatternWith1NNZByRowPanelFull(mf->cf[7]->RPP, mf->cf[7]->NPP, mf->cf[7]->NNZ, mf->cf[7]->CB, dX, dY, j , 0);
      break;
    case 8:
      PatternWith3NNZByRowPanelFull(mf->cf[8]->RPP, mf->cf[8]->NPP, mf->cf[8]->NNZ, mf->cf[8]->CB, dX, dY, j , 1, 2, 3);
      break;
    case 9:
      PatternWith2NNZByRowPanelFull(mf->cf[9]->RPP, mf->cf[9]->NPP, mf->cf[9]->NNZ, mf->cf[9]->CB, dX, dY, j , 1, 2);
      break;
    case 10:
      PatternWith2NNZByRowPanelFull(mf->cf[10]->RPP, mf->cf[10]->NPP, mf->cf[10]->NNZ, mf->cf[10]->CB, dX, dY, j , 1, 3);
      break;
    case 11:
      PatternWith1NNZByRowPanelFull(mf->cf[11]->RPP, mf->cf[11]->NPP, mf->cf[11]->NNZ, mf->cf[11]->CB, dX, dY, j , 1);
      break;
    case 12:
      PatternWith2NNZByRowPanelFull(mf->cf[12]->RPP, mf->cf[12]->NPP, mf->cf[12]->NNZ, mf->cf[12]->CB, dX, dY, j , 2, 3);
      break;
    case 13:
      PatternWith1NNZByRowPanelFull(mf->cf[13]->RPP, mf->cf[13]->NPP, mf->cf[13]->NNZ, mf->cf[13]->CB, dX, dY, j , 2);
      break;
    case 14:
      PatternWith1NNZByRowPanelFull(mf->cf[14]->RPP, mf->cf[14]->NPP, mf->cf[14]->NNZ, mf->cf[14]->CB, dX, dY, j , 3);
      break;
  }
}

class SpMVMixePatternDecomposeByRowSeperated : public SpMVSerialFloat {

protected:
  float *x;

  float *dX, *dY, *dY1, *dY2;
  swiftware::compression::MixedFormatGPU<float> *mf;

  float alpha = 1.0f;

  TensorInputsMixed *InT;

  bool verify(double &Error) override {
    bool retValue = true;

    double infNorm = 0;
    for (int i = 0; i < InTensor->M - 4; ++i) {
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
        (InT->MixedFormat->cf[0]->nrpp - 1) * 15, 32>>>(mf, dX, dY);

    t1.stopGPU("");

    cudaMemcpy(this->OutTensor->Dx, dY, this->InTensor->ACsr->m * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemset(dY, 0, this->InTensor->ACsr->m * sizeof(float));

    return t1;
  }

public:
  SpMVMixePatternDecomposeByRowSeperated(TensorInputsMixed *In1,
                                         swiftware::benchmark::Stats *Stat1)
      : SpMVSerialFloat(In1, Stat1), InT(In1) {

    x = static_cast<float *>(aligned_alloc(32, sizeof(float) * In1->ACsr->nnz));
    for (int i = 0; i < In1->ACsr->nnz; i++) {
      x[i] = static_cast<float>(In1->ACsr->x[i]);
    }

    cudaMalloc((void **)&dX, InTensor->N * sizeof(float));
    cudaMalloc((void **)&dY, InTensor->M * sizeof(float));
    cudaMalloc((void **)&dY1, InTensor->M * sizeof(float));
    cudaMalloc((void **)&dY2, InTensor->M * sizeof(float));
    //
    cudaMemset(dY, 0, InTensor->M * sizeof(float));
    cudaMemset(dY1, 0, InTensor->M * sizeof(float));
    cudaMemset(dY2, 0, InTensor->M * sizeof(float));
    cudaMemcpy(dX, this->InTensor->Bx, InTensor->N * sizeof(float),
               cudaMemcpyHostToDevice);

    mf = allocateAndCopyMixedFormat(In1->MixedFormat);
  };

  ~SpMVMixePatternDecomposeByRowSeperated() {
    //    cudaFree(dX);
    //    cudaFree(dY);
  }
};
#endif // COMPRESSED_TENSOR_ALGEBRA_SPMV_DEMO_GPU_UTILS_H
