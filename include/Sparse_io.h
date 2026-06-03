//
// Created by albakrih on 27/09/23.
//

#ifndef COMPRESSED_TENSOR_ALGEBRA_SPARCE_IO_H
#define COMPRESSED_TENSOR_ALGEBRA_SPARCE_IO_H

#include <string>

#include "compressed_format/CompressedFormat.h"
#include "aggregation/sparse_io.h"

namespace swiftware {
namespace compression {

// matrixPath must be a .cfmtx file (e.g. matrix-4-8.cfmtx from code_gen).
struct CfmtxMatrixLoad {
  sym_lib::CSR *csr = nullptr;
  MixedFormat<float> *compressed = nullptr;
};

// Loads companion .mtx (CSR for baselines) and .cfmtx (compressed kernels).
// Pattern count and kernel geometry come from the .cfmtx file header.
CfmtxMatrixLoad loadCfmtxMatrices(const std::string &cfmtxPath);

// Loads only the companion Matrix Market file (matrix.mtx).
sym_lib::CSR *readCSRMatrix(const std::string &cfmtxPath);

// Loads only the .cfmtx compressed format.
MixedFormat<float> *readCompressedFormatMatrix(const std::string &cfmtxPath);

} // namespace compression
} // namespace swiftware

#endif // COMPRESSED_TENSOR_ALGEBRA_SPARCE_IO_H
