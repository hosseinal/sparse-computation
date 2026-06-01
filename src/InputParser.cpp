//
// Created by kazem on 08/06/24.
//

#include <stdlib.h>

#include <iostream>

// https://github.com/jarro2783/cxxopts
#include "cxxopts.hpp"
#include "InputParser.h"


namespace swiftware{
namespace compression{
/**
 * @brief Parses commandline input for the program
 *
 * @param argc
 * @param argv
 */
Config parseInput(int argc, char **argv) {
  cxxopts::Options options("DDT",
                           "Generates vectorized code from memory streams");

  options.add_options()("h,help", "Prints help text")(
      "m,matrix", "Path to matrix market file.",
      cxxopts::value<std::string>())("dict", "Path to Dictionary file.",
                                     cxxopts::value<std::string>())(
      "n,numerical_operation", "Numerical operation being performed on matrix.",
      cxxopts::value<std::string>())(
      "s,storage_format", "Storage format for matrix",
      cxxopts::value<std::string>())("t,threads", "Number of parallel threads",
                                     cxxopts::value<int>()->default_value("1"))(
      "c,coarsening", "coarsening levels",
      cxxopts::value<int>()->default_value("5"))(
      "p,packing", "bin-packing", cxxopts::value<int>()->default_value("1"))(
      "u,tuning", "Tuning enabled", cxxopts::value<int>()->default_value("0"))(
      "iteration_limit", "Max length of periodic iteration space to find",
      cxxopts::value<int>()->default_value("0"))(
      "prefer_fsc", "Keep current codelet as FSC when greater than clt_width",
      cxxopts::value<bool>()->default_value("false"))(
      "number_of_warps", "Max length of periodic iteration space to find",
      cxxopts::value<int>()->default_value("8"))(
      "b_unroll_factor", "Max length of periodic iteration space to find",
      cxxopts::value<int>()->default_value("4"))(
      "m_tile_size", "Row tile size for SpMM",
      cxxopts::value<int>()->default_value("64"))(
      "n_tile_size", "Column tile size for SpMM",
      cxxopts::value<int>()->default_value("64"))(
      "b_matrix_columns", "Number of columns in dense matrix for SpMM",
      cxxopts::value<int>()->default_value("256"))(
      "hint", "Max length of periodic iteration space to find",
      cxxopts::value<int>()->default_value("0"))(
      "prefetch_distance", "Max length of periodic iteration space to find",
      cxxopts::value<int>()->default_value("0"))(
      "baseline",
      "calculate the base line or not",
      cxxopts::value<bool>()->default_value("false"))(
      "analyze_codelets",
      "Return analysis information for individual codelets instead of "
      "computing numerical method",
      cxxopts::value<bool>()->default_value("false"))("d,header",
                                                      "prints header or not.");


  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  if (!result.count("matrix")) {
    std::cout << "'matrix' is manditory argument. Use --help" << std::endl;
    exit(0);
  }
  std::string dictionaryPath = "";
  if (result.count("dict")) {
    dictionaryPath = result["dict"].as<std::string>();
  }


  if (!result.count("numerical_operation")) {
    std::cout << "'numerical_operation' is manditory argument. Use --help"
              << std::endl;
    exit(0);
  }
  if (!result.count("storage_format")) {
    std::cout << "'storage_format' is manditory argument. Use --help"
              << std::endl;
    exit(0);
  }

  auto matrixPath = result["matrix"].as<std::string>();
  auto operation = result["numerical_operation"].as<std::string>();
  auto storageFormat = result["storage_format"].as<std::string>();
  auto nThreads = result["threads"].as<int>();
  auto coarsening = result["coarsening"].as<int>();
  auto bpacking = result["packing"].as<int>();
  auto tuning_en = result["tuning"].as<int>();
  auto lim = result["iteration_limit"].as<int>();
  auto prefer_fsc = result["prefer_fsc"].as<bool>();
  auto number_of_warps = result["number_of_warps"].as<int>();
  auto b_unroll_factor = result["b_unroll_factor"].as<int>();
  auto mTileSize = result["m_tile_size"].as<int>();
  auto nTileSize = result["n_tile_size"].as<int>();
  auto bMatrixCols = result["b_matrix_columns"].as<int>();
  auto hint = result["hint"].as<int>();
  auto prefetch_distance = result["prefetch_distance"].as<int>();
  auto baseline = result["baseline"].as<bool>();
  auto analyzeCodelets = result["analyze_codelets"].as<bool>();




  NumericalOperation op;
  if (operation == "SPMV") {
    op = OP_SPMV;
  } else if (operation == "SPTRS") {
    op = OP_SPTRS;
  } else if (operation == "SPMM") {
    op = OP_SPMM;
  } else {
    std::cout << "'numerical_operation' must be passed in as one of: ['SPMV', "
                 "'SPTRS', 'SPMM']"
              << std::endl;
    exit(0);
  }

  StorageFormat sf;
  if (storageFormat == "CSR") {
    sf = CSR_SF;
  } else if (storageFormat == "CSC") {
    sf = CSC_SF;
  } else {
    std::cout << "'storage_format' must be passed in as one of: ['CSC', 'CSR']"
              << std::endl;
    exit(0);
  }
  int header = 0;
  if (result.count("header")) {
    header = 1;
  }


  return Config{matrixPath,
                dictionaryPath,
                op,
                header,
                nThreads,
                sf,
                coarsening,
                b_unroll_factor,
                number_of_warps,
                lim,
                prefetch_distance,
                baseline,
                analyzeCodelets,
                mTileSize,
                nTileSize,
                bMatrixCols};
}
}
}