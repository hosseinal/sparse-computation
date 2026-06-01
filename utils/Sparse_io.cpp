#include "Sparse_io.h"

#include "aggregation/exceptions.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "compressed_format/CompressedFormat.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace swiftware {
namespace compression {

namespace {

bool endsWith(const std::string &str, const std::string &suffix) {
  if (str.size() < suffix.size()) {
    return false;
  }
  return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// e.g. "matrix-4-8.cfmtx" or "matrix-64-128.cfmtx" -> "matrix.mtx"
// Codegen names files: <stem>-<kernel_m>-<kernel_n>.cfmtx
std::string companionMtxPathFromCFMTX(const std::string &cfmtxPath) {
  constexpr size_t kCfmtxSuffixLen = 6; // ".cfmtx"
  if (cfmtxPath.size() <= kCfmtxSuffixLen) {
    return cfmtxPath;
  }

  std::string stem = cfmtxPath.substr(0, cfmtxPath.size() - kCfmtxSuffixLen);
  const size_t lastDash = stem.rfind('-');
  if (lastDash == std::string::npos || lastDash == 0) {
    return stem + ".mtx";
  }
  const size_t secondLastDash = stem.rfind('-', lastDash - 1);
  if (secondLastDash == std::string::npos) {
    return stem + ".mtx";
  }

  stem.resize(secondLastDash);
  return stem + ".mtx";
}

sym_lib::CSR *readCSRMatrixFromMTX(const std::string &path) {
  std::ifstream in_file(path);

  int n, m;
  int shape, arith, mtx_format;
  size_t nnz;
  std::vector<sym_lib::triplet> triplet_vec;

  sym_lib::read_header(in_file, m, n, nnz, arith, shape, mtx_format);

  if (arith != sym_lib::REAL && arith != sym_lib::INT &&
      arith != sym_lib::PATTERN) {
    throw sym_lib::mtx_arith_error("REAL", sym_lib::type_str(arith));
  }
  if (mtx_format != sym_lib::COORDINATE) {
    throw sym_lib::mtx_format_error("COORDINATE",
                                    sym_lib::format_str(mtx_format));
  }
  bool read_val = true;
  if (arith == sym_lib::PATTERN) {
    read_val = false;
  }
  sym_lib::read_triplets_real(in_file, nnz, triplet_vec, read_val, false);

  sym_lib::CSC *ACsc = new sym_lib::CSC(m, n, nnz);
  sym_lib::compress_triplets_to_csc(triplet_vec, ACsc, false);

  if (shape == sym_lib::LOWER) {
    ACsc->stype = sym_lib::shape2int(shape);
    sym_lib::CSC *ACsc2 = make_full(ACsc);
    return csc_to_csr(ACsc2);
  }

  return csc_to_csr(ACsc);
}

MixedFormat<float> *readCFMTXFile(const std::string &path) {
  std::ifstream file(path);

  if (!file.is_open()) {
    std::cerr << "Unable to open file " << path << std::endl;
    return nullptr;
  }

  int num_rows, num_columns, kernel_rows, num_patterns, vector_length;
  std::vector<CompressedFormat<float> *> AA;
  std::string line;
  if (std::getline(file, line)) {
    std::istringstream iss(line);
    iss >> num_rows >> num_columns >> kernel_rows >> num_patterns >>
        vector_length;
  }

  std::vector<int> pattern_nnz;

  for (int i = 0; i < num_patterns; i++) {
    if (std::getline(file, line)) {
      std::istringstream iss(line);
      std::string pattern;
      int pattern_id = 0;
      iss >> pattern >> pattern_id;
      (void)pattern_id;

      int n = 0;
      for (char c : pattern) {
        if (c == '1') {
          n++;
        }
      }
      pattern_nnz.push_back(n);
    }
  }

  for (int i = 0; i < num_patterns; i++) {
    std::vector<int> rowPanelPointer;
    std::vector<int> nonZeroPatternPointer;
    std::vector<int> indices;
    std::vector<float> values;

    if (std::getline(file, line)) {
      std::istringstream iss(line);
      int value;
      while (iss >> value) {
        rowPanelPointer.push_back(value);
        nonZeroPatternPointer.push_back(value * pattern_nnz[i]);
      }
    }

    if (std::getline(file, line)) {
      std::istringstream iss(line);
      int value;
      while (iss >> value) {
        indices.push_back(value);
      }
    }

    if (std::getline(file, line)) {
      std::istringstream iss(line);
      float value;
      while (iss >> value) {
        values.push_back(value);
      }
    }

    if (vector_length == 0) {
      vector_length = 1;
    }
    CompressedFormat<float> *A = new CompressedFormat<float>(
        static_cast<size_t>(num_rows), static_cast<size_t>(num_columns),
        values.size(), static_cast<size_t>(kernel_rows),
        static_cast<size_t>(vector_length));
    A->real_nnz = values.size();

    for (int j = 0; j < rowPanelPointer.size(); j++) {
      A->RPP[j] = rowPanelPointer[j];
      A->NPP[j] = nonZeroPatternPointer[j];
    }

    for (int j = 0; j < indices.size(); j++) {
      A->CB.push_back(indices[j]);
    }

    for (int j = 0; j < values.size(); j++) {
      A->NNZ[j] = values[j];
    }

    AA.push_back(A);
  }

  auto CSR = new CSRFormat<float>(num_rows, num_columns, 0);
  auto *mixedFormat = new MixedFormat<float>(AA, CSR);

  file.close();
  return mixedFormat;
}

} // namespace

CfmtxMatrixLoad loadCfmtxMatrices(const std::string &cfmtxPath) {
  CfmtxMatrixLoad loaded;
  if (!endsWith(cfmtxPath, ".cfmtx")) {
    std::cerr << "loadCfmtxMatrices expects a .cfmtx path: " << cfmtxPath
              << std::endl;
    return loaded;
  }
  loaded.compressed = readCFMTXFile(cfmtxPath);
  loaded.csr = readCSRMatrixFromMTX(companionMtxPathFromCFMTX(cfmtxPath));
  return loaded;
}

sym_lib::CSR *readCSRMatrix(const std::string &cfmtxPath) {
  if (!endsWith(cfmtxPath, ".cfmtx")) {
    std::cerr << "readCSRMatrix expects a .cfmtx path (companion .mtx is "
                 "loaded for baselines): "
              << cfmtxPath << std::endl;
    return nullptr;
  }
  return readCSRMatrixFromMTX(companionMtxPathFromCFMTX(cfmtxPath));
}

MixedFormat<float> *readCompressedFormatMatrix(const std::string &cfmtxPath) {
  if (!endsWith(cfmtxPath, ".cfmtx")) {
    std::cerr << "readCompressedFormatMatrix expects a .cfmtx path: "
              << cfmtxPath << std::endl;
    return nullptr;
  }
  return readCFMTXFile(cfmtxPath);
}

} // namespace compression
} // namespace swiftware
