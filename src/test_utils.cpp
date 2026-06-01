//
// Created by kazem on 1/26/24.
//

#include "test_utils.h"
#include <iostream>
#include <tuple>


namespace swiftware{
namespace compression{
std::tuple<std::string,std::string> TestParameters::print_csv(bool header) const {
  std::string header_text, row;
  if(header){
    header_text  = "MatrixName,Density,nRows,nCols,NNZ,Mode,Algorithm,bCols";
  }
  row = _matrix_name+","+ std::to_string(_density)+","+ std::to_string(_dim1)+","
        +std::to_string(_dim2)+","+std::to_string(_nnz)+","+_mode
        +","+_algorithm_choice+","
        +std::to_string(_b_cols);
  return std::make_tuple(header_text, row);
}

std::tuple<std::string,std::string> ScheduleParameters::print_csv(bool header) const{
  std::string header_text, row;
  if(header){
    header_text = "nThreads,Iter Per Partition,MTile,NTile,";
  }
  row = std::to_string(NumThreads) + "," + std::to_string(IterPerPartition) +","
        + std::to_string(TileM) + "," + std::to_string(TileN) + ",";
  return std::make_tuple(header_text, row);
}



}
}