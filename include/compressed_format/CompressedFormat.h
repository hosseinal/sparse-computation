//
// Created by albakrih on 27/09/23.
//
#include "Sparse_io.h"
#include "aggregation/exceptions.h"
#include "aggregation/sparse_io.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#ifndef COMPRESSED_TENSOR_ALGEBRA_COMPRESSEDFORMAT_H
#define COMPRESSED_TENSOR_ALGEBRA_COMPRESSEDFORMAT_H

namespace swiftware {
namespace compression {

class PatternTypeDictionary {
private:
  int max_patterns;
  std::unordered_map<int, std::string> id_to_pattern_bin;
  std::unordered_map<std::string, int> pattern_bin_to_id;
  std::vector<int> generated_patterns_ids;

public:
  PatternTypeDictionary() { max_patterns = 20; }

  void add_pattern(int pattern_number, std::string pattern_binary) {
    id_to_pattern_bin[pattern_number] = pattern_binary;
    pattern_bin_to_id[pattern_binary] = pattern_number;
    append_generated_pattern(pattern_number);
  }

  bool is_binary_pattern_exists(const std::string &pattern_binary) {
    return pattern_bin_to_id.find(pattern_binary) != pattern_bin_to_id.end();
  }

  int get_pattern_id(const std::string &pattern_binary) {
    return pattern_bin_to_id.find(pattern_binary)->second;
  }

  void append_generated_pattern(int pattern_number) {
    generated_patterns_ids.push_back(pattern_number);
  }

  bool is_pattern_generated(int pattern_number) {
    for (int id : generated_patterns_ids) {
      if (id == pattern_number)
        return true;
    }
    return false;
  }

  void print() {
    // iterate over unordered map
    for (auto it = id_to_pattern_bin.begin(); it != id_to_pattern_bin.end();
         ++it) {
      std::cout << it->first << " " << it->second << std::endl;
    }
    //    for (auto const& [key, val] : id_to_pattern_bin) {
    //      std::cout << key << " " << val << std::endl;
    //    }
  }

  void import(const std::string &file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
      std::cerr << "Error: Unable to open file " << file_path << std::endl;
      return;
    }

    std::string line;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      int pattern_number;
      std::string pattern_binary;
      if (!(iss >> pattern_number >> pattern_binary)) {
        std::cerr << "Error: Invalid line format: " << line << std::endl;
        continue;
      }
      add_pattern(pattern_number, pattern_binary);
    }

    file.close();
  }
};

template <class T> struct CompressedFormat {
  size_t km;   // number of kernel rows
  size_t kn;   // number of kernel columns
  size_t m;    // number of rows of matrix
  size_t n;    // number of columns
  size_t pm;   // number of padded rows
  size_t pn;   // number of padded columns
  size_t nrpp; // number of row pattern pointer
  size_t npt;  // number of pattern type (length of PT array) which equal to
               // number of Column Begin array
  size_t real_nnz; // real number of non zeros
  size_t nnz;  // number of non zeros (might be padded)
  T *NNZ;      // Array of Non Zero Values
  int *RPP;            // ROW Pattern Pointer
  int *NPP;            // NNZ Pattern Pointer
  std::vector<int> PT; // Pattern Type
  std::vector<int> CB; // Column Begin
  std::vector<int>
      CSRRows;   // rows of patterns that are going be computed as csr
  bool preAlloc; // if it is already allocated somewhere else
  // CSR format
  PatternTypeDictionary *patternTypeDictionary;

  CompressedFormat(size_t M, size_t N, size_t Nnz, size_t KM, size_t KN,
                   PatternTypeDictionary &ptd)
      : m(M), n(N), nnz(Nnz), km(KM), kn(KN) {
    preAlloc = false;
    if (M > 0) {
      pm = M + M % KM;
      pn = N + N % KN;

      nrpp = pm / KM + 1;

      RPP = static_cast<int *>(aligned_alloc(32, sizeof(int) * nrpp));
      NPP = static_cast<int *>(aligned_alloc(32, sizeof(int) * nrpp));
      std::fill(RPP, RPP + nrpp, 0.0);
      std::fill(NPP, NPP + nrpp, 0.0);

    } else {
      RPP = nullptr;
      NPP = nullptr;
    }
    if (Nnz > 0) {
      NNZ = static_cast<T *>(aligned_alloc(32, sizeof(T) * nnz));
      std::fill(NNZ, NNZ + nnz, 0.0);
    } else {
      NNZ = nullptr;
    }

    patternTypeDictionary = &ptd;

    npt = 0;
  }

  CompressedFormat(size_t M, size_t N, size_t Nnz, size_t KM, size_t KN)
      : m(M), n(N), nnz(Nnz), km(KM), kn(KN) {
    preAlloc = false;
    if (M > 0) {
      if (M % KM == 0) {
        pm = M;
      } else{
        pm = M + KM - M % KM;
      }
      pn = N + N % KN;

      nrpp = pm / KM + 1;

      RPP = static_cast<int *>(aligned_alloc(32, sizeof(int) * nrpp));
      NPP = static_cast<int *>(aligned_alloc(32, sizeof(int) * nrpp));
      std::fill(RPP, RPP + nrpp, 0.0);
      std::fill(NPP, NPP + nrpp, 0.0);

    } else {
      RPP = nullptr;
      NPP = nullptr;
    }
    if (Nnz > 0) {
      NNZ = static_cast<T *>(aligned_alloc(32, sizeof(T) * nnz));
      std::fill(NNZ, NNZ + nnz, 0.0);

    } else {
      NNZ = nullptr;
    }

    npt = 0;
  }

  CompressedFormat(size_t M, size_t N, size_t Nnz, size_t KM, size_t KN,
                   size_t NPT)
      : m(m), n(N), nnz(Nnz), npt(NPT), km(KM), kn(KN) {
    preAlloc = false;
    if (M > 0) {
      pm = M + M % KM;
      pn = N + N % KN;
      nrpp = pm / KM + 1;
      RPP = static_cast<int *>(aligned_alloc(32, sizeof(int) * nrpp));
      NPP = static_cast<int *>(aligned_alloc(32, sizeof(int) * nrpp));
    } else {
      RPP = nullptr;
      NPP = nullptr;
    }
    if (Nnz > 0) {
      NNZ = static_cast<T *>(aligned_alloc(32, sizeof(T) * nnz));
      std::fill(NNZ, NNZ + nnz, 0.0);
    } else {
      NNZ = nullptr;
    }
  }

  CompressedFormat(size_t M, size_t N, size_t Nnz, size_t NPT, T *ANNZ,
                   int *ARPP, int *ANPP, std::vector<int> &APT, size_t KM,
                   size_t KN, std::vector<int> &ACB)
      : m(M), n(N), nnz(Nnz), npt(NPT), NNZ(ANNZ), RPP(ARPP), NPP(ANPP),
        PT(APT), CB(ACB), km(KM), kn(KN) {
    preAlloc = true;
    pm = M + M % KM;
    pn = N + N % KN;

    nrpp = pm / KM + 1;
  }

  ~CompressedFormat() {

    if(!preAlloc){
      if (m > 0) {
        free(RPP);
        free(NPP);
      }
      if (nnz > 0) {
        free(NNZ);
      }
    }
  }
};

template <class T> struct CSRFormat {
  size_t m;
  size_t n;
  size_t nnz;
  T *NNZ;  // non zeroes
  int *RP; // row pointer
  int *CI; // column index
  bool preAlloc;

  CSRFormat(size_t M, size_t N, size_t Nnz) : m(M), n(N), nnz(Nnz) {
    preAlloc = false;
    if (M > 0) {
      RP = static_cast<int *>(aligned_alloc(32, sizeof(int) * (M + 1)));
      std::fill(RP, RP + M + 1, 0);
    } else {
      RP = nullptr;
    }
    if (Nnz > 0) {
      NNZ = static_cast<T *>(aligned_alloc(32, sizeof(T) * Nnz));
      std::fill(NNZ, NNZ + Nnz, 0.0);
      CI = static_cast<int *>(aligned_alloc(32, sizeof(int) * Nnz));
      std::fill(CI, CI + Nnz, 0);
    } else {
      NNZ = nullptr;
      CI = nullptr;
    }
  }

  ~CSRFormat() {
    if (m > 0) {
      free(RP);
    }
    if (nnz > 0) {
      free(NNZ);
      free(CI);
    }
  }
};

template <class T> struct MixedFormat {
public:
  std::vector<CompressedFormat<T> *> cf;
  CSRFormat<T> *csr;
  T *Dense;
  MixedFormat(std::vector<CompressedFormat<T> *> AA, CSRFormat<T> *BB)
      : cf(AA), csr(BB){};
  MixedFormat(){};
  ~MixedFormat() {
    delete csr;
    for (auto &c : cf) {
      delete c;
    }
    if (Dense != nullptr) {
      free(Dense);
    }
  }
};

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

} // namespace compression
} // namespace swiftware

#endif // COMPRESSED_TENSOR_ALGEBRA_COMPRESSEDFORMAT_H