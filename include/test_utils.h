//
// Created by kazem on 1/26/24.
//

#ifndef COMPRESSED_TENSOR_ALGEBRA_TEST_UTILS_H
#define COMPRESSED_TENSOR_ALGEBRA_TEST_UTILS_H

#include <string>
#include <vector>


namespace swiftware{
namespace compression{
struct ScheduleParameters {
  int IterPerPartition;   // aggregation params
  int _min_workload_size; // min workload size to run in parallel for
                          // tiledFusedCSCCombined
  int NumThreads;
  int TileM{}, TileN{}, TileK{};

  ScheduleParameters() : NumThreads(1),
                         TileK(1), TileM(1), TileN(1), IterPerPartition(1), _min_workload_size(1){
  }

  explicit ScheduleParameters(int nt) : ScheduleParameters() {
    NumThreads = nt;
  }

  /*
   * Prints header and info in csv format
   */
  std::tuple<std::string, std::string> print_csv(bool header = false) const;
};

/*
 * Holding the test parameters
 */
struct TestParameters {
  std::string _matrix_name{}, _matrix_path{}, _result_matrix_path{};
  std::string expariment_name{};
  std::string _mode{};                //"Random" or "MTX"

  std::string _algorithm_choice{};
  double _density{}, _dim1{}, _dim2{}, _nnz{}; // for random mode
  bool print_header{};
  int _b_cols{};    // in gnn experiments bcols is regarded as feature dimension
  int _embed_dim{}; // embed_dim only is used in gnn experiments and is regarded
                    // as hidden dimension
  TestParameters() { _mode = "Random"; }

  /*
   * Prints header and info in csv format
   */
  std::tuple<std::string, std::string> print_csv(bool header = false) const;
};



template <typename T> T* csrToDense(int m, int n, int *Ap, int *Ai, double *Ax) {
  auto *a = static_cast<T *>(aligned_alloc(32, sizeof(T) * m * n));
  for (int i = 0; i < m; ++i) {
    for (int j = Ap[i]; j < Ap[i + 1]; ++j) {
      a[i * n + Ai[j]] = Ax[j];
    }
  }
  return a;
}
}
}
#endif // COMPRESSED_TENSOR_ALGEBRA_TEST_UTILS_H
