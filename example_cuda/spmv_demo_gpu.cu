#include "InputParser.h"
#include "Sparse_io.h"
#include "Stats.h"
#include "spmv_demo_gpu_utils.h"
#include "test_utils.h"
#include <cstdlib>


int main(int argc, char *argv[]) {
  swiftware::compression::TestParameters tp;
  swiftware::compression::ScheduleParameters sp;
  swiftware::benchmark::Stats *stats;

  auto config = swiftware::compression::parseInput(argc, argv);


  int b_col = 1;

  auto loaded =
      swiftware::compression::loadCfmtxMatrices(config.matrixPath);
  sym_lib::CSR *A = loaded.csr;
  auto *mixedColumnBasedFloat = loaded.compressed;
  if (A == nullptr || mixedColumnBasedFloat == nullptr) {
    return 1;
  }

  tp._matrix_name = config.matrixPath;
  tp._b_cols = b_col;
  tp._dim1 = A->m;
  tp._dim2 = A->n;
  tp._nnz = A->nnz;
  tp.print_header = config.header;

  int b_unroll_factor = 0;
  int number_of_warps = config.number_of_warps;

  int numThread = config.nThread, numTrial = 20;

  std::string expName = "SpMV_Demo";

  auto *inSpMV = new TensorInputsFloat(A->m, A->n, A, numThread, numTrial, expName);

  auto *inMixedSpMV = new TensorInputsMixed(A->m, A->n, A, mixedColumnBasedFloat ,numThread, numTrial, expName);

  int kernel_size = inMixedSpMV->MixedFormat->cf[0]->km;
  int kernel_n = inMixedSpMV->MixedFormat->cf[0]->kn;

  tp._matrix_name = config.matrixPath;


  inSpMV->Bx =
      static_cast<float *>(aligned_alloc(32, sizeof(float) * inSpMV->N));
  std::fill(inSpMV->Bx, inSpMV->Bx + inSpMV->N, 0.0);
  inMixedSpMV->Bx =
      static_cast<float *>(aligned_alloc(32, sizeof(float) * inMixedSpMV->N));
  std::fill(inMixedSpMV->Bx, inMixedSpMV->Bx + inMixedSpMV->N, 0.0);

  for (int i = 0; i < tp._dim2; ++i) {
    double r = (double)rand() / RAND_MAX;
    inSpMV->Bx[i] = r;
    inMixedSpMV->Bx[i] = r;
  }

  inMixedSpMV->CorrectMul =
      static_cast<float *>(aligned_alloc(32, sizeof(float) * inMixedSpMV->M));
  inMixedSpMV->CorrectSol =
      static_cast<float *>(aligned_alloc(32, sizeof(float) * inMixedSpMV->M));
  inSpMV->CorrectMul =
      static_cast<float *>(aligned_alloc(32, sizeof(float) * inSpMV->M));
  inSpMV->CorrectSol =
      static_cast<float *>(aligned_alloc(32, sizeof(float) * inSpMV->M));

  std::fill(inMixedSpMV->CorrectMul, inMixedSpMV->CorrectMul + inMixedSpMV->M, 0.0);
  std::fill(inMixedSpMV->CorrectSol, inMixedSpMV->CorrectSol + inMixedSpMV->M, 0.0);
  std::fill(inSpMV->CorrectMul, inSpMV->CorrectMul + inSpMV->M, 0.0);
  std::fill(inSpMV->CorrectSol, inSpMV->CorrectSol + inSpMV->M, 0.0);

  stats = new swiftware::benchmark::Stats("SpMV_Demo", "SpMV Complex GPU V1", numTrial,
                                          tp._matrix_name, numThread);
  auto *complexv1 = new SpMVComplexKernel(inSpMV, stats);
  complexv1->run();

  for (int i = 0; i < inSpMV->M; ++i) {
    inSpMV->CorrectSol[i] = complexv1->OutTensor->Dx[i];
    inSpMV->CorrectMul[i] = complexv1->OutTensor->Dx[i];
    inMixedSpMV->CorrectSol[i] = complexv1->OutTensor->Dx[i];
    inMixedSpMV->CorrectMul[i] = complexv1->OutTensor->Dx[i];
  }
  auto headerStat = complexv1->printStatsHeader();
  auto complexKernelStatV1 = complexv1->printStats();
  delete complexv1;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMV_Demo", "SpMV Complex GPU V2", numTrial,
                                          tp._matrix_name, numThread);
  auto *complex = new SpMVComplexKernelV2(inSpMV, stats);
  if (config.baseline){
    complex->run();
  }
  auto complexKernelStat = complex->printStats();
  delete complex;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMV_Demo", "SpMV CuSparse GPU", numTrial,
                                          tp._matrix_name, numThread);
  auto *cuGpu = new SpMVcuSparse(inSpMV, stats);
  if (config.baseline){
    cuGpu->run();
  }
  auto cuSparseTime = cuGpu->printStats();
  delete cuGpu;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMV_Demo", "SpMV CuBlas GPU", numTrial,
                                          tp._matrix_name, numThread);
  auto *cuBlasGpu = new SpMVcuBlas(inSpMV, stats);
  if (config.baseline){
    cuBlasGpu->run();
  }
  auto cuBlasTime = cuBlasGpu->printStats();
  delete cuBlasGpu;
  delete stats;

  stats = new swiftware::benchmark::Stats("SpMV_Demo", "SpMV Compressed Format", numTrial,
                                          tp._matrix_name, numThread);
  auto *kernelBlockRowSeperated8 = new SpMVMixePatternDecomposeByRowSeperated(inMixedSpMV, stats);
  kernelBlockRowSeperated8->run();
  auto kernelBlockStatRowSeperated = kernelBlockRowSeperated8->printStats();
  delete kernelBlockRowSeperated8;
  delete stats;

  auto csvInfo = sp.print_csv(true);
  std::string spHeader = std::get<0>(csvInfo);
  std::string spStat = std::get<1>(csvInfo);

  auto tpCsv = tp.print_csv(true);
  std::string tpHeader = std::get<0>(tpCsv);
  std::string tpStat = std::get<1>(tpCsv);


  if (tp.print_header){
    std::cout << headerStat + spHeader + tpHeader << ","
              << "kernel_n"
              << ","
              << "kernel_m,"
              << "b_cols,"
              << "b_unroll_factor,"
              << "number_of_warps" << std::endl;
  }

  if (config.baseline){
    std::cout << cuSparseTime << spStat + tpStat << "," << kernel_n << ","
              << kernel_size << "," << b_col << "," << b_unroll_factor << ","
              << number_of_warps << std::endl;

    std::cout << cuBlasTime << spStat + tpStat << "," << kernel_n << ","
              << kernel_size << "," << b_col << "," << b_unroll_factor << ","
              << number_of_warps << std::endl;

    std::cout << complexKernelStatV1 << spStat + tpStat << "," << kernel_n << ","
              << kernel_size << "," << b_col << "," << b_unroll_factor << ","
              << number_of_warps << std::endl;

    std::cout << complexKernelStat << spStat + tpStat << ","<< kernel_n << ","
              << kernel_size << "," << b_col << "," << b_unroll_factor << ","
              << number_of_warps << std::endl;
  }
  std::cout << kernelBlockStatRowSeperated << spStat + tpStat << ","<< kernel_n << ","
            << kernel_size << "," << b_col << "," << b_unroll_factor << ","
            << number_of_warps << std::endl;


  delete inMixedSpMV;
//  delete mixedColumnBasedFloat;
  delete A;
  return 0;

}