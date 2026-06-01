//
// Created by kazem on 08/06/24.
//

#ifndef COMPRESSED_TENSOR_ALGEBRA_INPUT_H
#define COMPRESSED_TENSOR_ALGEBRA_INPUTPARSER_H

#include <string>

namespace swiftware {
namespace compression {

enum NumericalOperation {
  OP_SPMV,
  OP_SPTRS,
  OP_SPMM
};

enum StorageFormat {
  CSR_SF,
  CSC_SF
};

struct MemoryTrace {
  int** ip;
  int ips;
};

struct Config {
  std::string matrixPath;
  std::string dictionaryPath;
  NumericalOperation op;
  int header;
  int nThread;
  StorageFormat sf;
  int coarsening;
  int b_unroll_factor;
  int number_of_warps;
  int lim;
  int prefetch_distance;
  bool baseline;
  bool analyzeCodelets;
  int mTileSize;
  int nTileSize;
  int bMatrixCols;
};


Config parseInput(int argc, char **argv);


} // namespace compression
} // namespace swiftware

#endif // COMPRESSED_TENSOR_ALGEBRA_INPUT_H
