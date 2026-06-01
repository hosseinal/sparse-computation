//
// Created by albakrih on 13/03/25.
//
//
// Created by albakrih on 26/08/24.
//
#include "InputParser.h"
#include "Sparse_io.h"
#include "Stats.h"
#include "spmm_small_b_col_demo_gpu_utils.h"
#include "test_utils.h"

int main(int argc, char *argv[]) {

  swiftware::compression::TestParameters tp;
  swiftware::compression::ScheduleParameters sp;
  swiftware::benchmark::Stats *stats;

  auto config = swiftware::compression::parseInput(argc, argv);

  auto loaded =
      swiftware::compression::loadCfmtxMatrices(config.matrixPath);
  sym_lib::CSR *A = loaded.csr;
  auto *mixedColumnBasedFloat = loaded.compressed;
  if (A == nullptr || mixedColumnBasedFloat == nullptr) {
    return 1;
  }

  int b_col = config.bMatrixCols;
  int b_unroll_factor = config.b_unroll_factor;
  int number_of_warps = config.number_of_warps;
  tp._matrix_name = config.matrixPath;
  tp._dim1 = A->m;
  tp._dim2 = A->n;
  tp._nnz = A->nnz;
  tp.print_header = config.header;

  int numThread = config.nThread, numTrial = 7;

  std::string expName = "SpMM_Demo_";

  auto *inSpMM =
      new TensorInputsFloat(A->m, A->n, b_col, A, numThread, numTrial, expName);

  auto *inMixedSpMM =
      new TensorInputsMixed(A->m, A->n, b_col, A, mixedColumnBasedFloat,
                            numThread, numTrial, expName);

  int kernel_size = inMixedSpMM->MixedFormat->cf[0]->km;
  int kernel_n = inMixedSpMM->MixedFormat->cf[0]->kn;

  inSpMM->Bx = static_cast<float *>(
      aligned_alloc(32, sizeof(float) * inSpMM->N * b_col));
  std::fill(inSpMM->Bx, inSpMM->Bx + inSpMM->N * b_col, 0.0);

  inMixedSpMM->Bx = static_cast<float *>(
      aligned_alloc(32, sizeof(float) * inMixedSpMM->N * b_col));
  std::fill(inMixedSpMM->Bx, inMixedSpMM->Bx + inMixedSpMM->N * b_col, 0.0);

  for (int i = 0; i < inSpMM->N * b_col; ++i) {
    double r = (double)rand() / RAND_MAX;
    inSpMM->Bx[i] = r;
    inMixedSpMM->Bx[i] = r;
  }

  inSpMM->CorrectMul = static_cast<float *>(
      aligned_alloc(32, sizeof(float) * inSpMM->M * b_col));
  inSpMM->CorrectSol = static_cast<float *>(
      aligned_alloc(32, sizeof(float) * inSpMM->M * b_col));

  inMixedSpMM->CorrectMul = static_cast<float *>(
      aligned_alloc(32, sizeof(float) * inMixedSpMM->M * b_col));
  inMixedSpMM->CorrectSol = static_cast<float *>(
      aligned_alloc(32, sizeof(float) * inMixedSpMM->M * b_col));

  std::fill(inSpMM->CorrectMul, inSpMM->CorrectMul + inSpMM->M, 0.0);
  std::fill(inSpMM->CorrectSol, inSpMM->CorrectSol + inSpMM->M, 0.0);
  std::fill(inMixedSpMM->CorrectMul, inMixedSpMM->CorrectMul + inMixedSpMM->M,
            0.0);
  std::fill(inMixedSpMM->CorrectSol, inMixedSpMM->CorrectSol + inMixedSpMM->M,
            0.0);

  stats = new swiftware::benchmark::Stats("SpMM_Demo", "SpMM CuSparse",
                                          numTrial, tp._matrix_name, numThread);
  auto *cuSparse = new SpMMcuSparse(inSpMM, stats);
  cuSparse->run();
  auto headerStat = cuSparse->printStatsHeader();
  auto cuSparseStat = cuSparse->printStats();

  for (int i = 0; i < inSpMM->M * inSpMM->Z; ++i) {
    inSpMM->CorrectSol[i] = cuSparse->OutTensor->Dx[i];
    inSpMM->CorrectMul[i] = cuSparse->OutTensor->Dx[i];
    inMixedSpMM->CorrectSol[i] = cuSparse->OutTensor->Dx[i];
    inMixedSpMM->CorrectMul[i] = cuSparse->OutTensor->Dx[i];
  }

  inMixedSpMM->number_of_warps = number_of_warps;

  delete cuSparse;
  delete stats;

  // cuBlas
  stats = new swiftware::benchmark::Stats("SpMM_Demo", "SpMM CuBlas", numTrial,
                                          tp._matrix_name, numThread);
  auto *cuBlas = new SpMMcuBlas(inSpMM, stats);
  if (config.baseline) {
    cuBlas->run();
  }
  auto cuBlasHeader = cuBlas->printStatsHeader();
  auto cuBlasStat = cuBlas->printStats();
  delete cuBlas;
  delete stats;

  // cuBlas
  stats = new swiftware::benchmark::Stats("SpMM_Demo", "SpMM CuBlas GemmX",
                                          numTrial, tp._matrix_name, numThread);
  auto *cuBlasGemmX = new SpMMcuBlasGemmX(inSpMM, stats);
  if (config.baseline) {
    cuBlasGemmX->run();
  }
  auto cuBlasGemmXHeader = cuBlasGemmX->printStatsHeader();
  auto cuBlasGemmXStat = cuBlasGemmX->printStats();
  delete cuBlasGemmX;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_Demo",
                                          "SpMM CuBlas GemmX Tensor Float",
                                          numTrial, tp._matrix_name, numThread);
  auto *cuBlasGemmXTensorFloat = new SpMMcuBlasTF32(inSpMM, stats);
  if (config.baseline) {
    cuBlasGemmXTensorFloat->run();
  }
  auto cuBlasGemmXTensorFloatHeader =
      cuBlasGemmXTensorFloat->printStatsHeader();
  auto cuBlasGemmXTensorFloatStat = cuBlasGemmXTensorFloat->printStats();
  delete cuBlasGemmXTensorFloat;
  delete stats;

  stats =
      new swiftware::benchmark::Stats("SpMM_Demo", "SpMM CuBlas Half Tensor",
                                      numTrial, tp._matrix_name, numThread);
  auto *cuBlasTensor = new SpMMcuBlasHalf(inSpMM, stats);
  if (config.baseline) {
    cuBlasTensor->run();
  }
  auto cuBlasTensorHalfHeader = cuBlasTensor->printStatsHeader();
  auto cuBlasHalfTensorStat = cuBlasTensor->printStats();
  delete cuBlasTensor;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMM_Demo",
                                          "SpMM Format Pack Block Cache less",
                                          numTrial, tp._matrix_name, numThread);
  auto *SpMM2NNZ16BlockCacheLess =
      new SpMVMixePatternDecomposeByRowSeperated(inMixedSpMM, stats);
  SpMM2NNZ16BlockCacheLess->run();
  auto SpMMcF16BlocCacheLesskHeader =
      SpMM2NNZ16BlockCacheLess->printStatsHeader();
  auto SpMM2NNZ16CacheLessBlockStat = SpMM2NNZ16BlockCacheLess->printStats();
  delete SpMM2NNZ16BlockCacheLess;
  delete stats;

  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);

  if (tp.print_header) {
    std::cout << headerStat + spHeader + tpHeader << ","
              << "kernel_n"
              << ","
              << "kernel_m,"
              << "b_cols,"
              << "b_unroll_factor,"
              << "number_of_warps" << std::endl;
  }

  if (config.baseline) {
    std::cout << cuBlasStat << spStat + tpStat << "," << kernel_n << ","
              << kernel_size << "," << b_col << "," << b_unroll_factor << ","
              << number_of_warps << std::endl;
    std::cout << cuBlasHalfTensorStat << spStat + tpStat << "," << kernel_n
              << "," << kernel_size << "," << b_col << "," << b_unroll_factor
              << "," << number_of_warps << std::endl;
    std::cout << cuBlasGemmXStat << spStat + tpStat << "," << kernel_n << ","
              << kernel_size << "," << b_col << "," << b_unroll_factor << ","
              << number_of_warps << std::endl;
    std::cout << cuBlasGemmXTensorFloatStat << spStat + tpStat << ","
              << kernel_n << "," << kernel_size << "," << b_col << ","
              << b_unroll_factor << "," << number_of_warps << std::endl;
    std::cout << cuSparseStat << spStat + tpStat << "," << kernel_n << ","
              << kernel_size << "," << b_col << "," << b_unroll_factor << ","
              << number_of_warps << std::endl;
  }

  std::cout << SpMM2NNZ16CacheLessBlockStat << spStat + tpStat << ","
            << kernel_n << "," << kernel_size << "," << b_col << ","
            << b_unroll_factor << "," << number_of_warps << std::endl;

  return 0;
}