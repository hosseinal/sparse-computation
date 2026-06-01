//
// Created by albakrih on 12/05/24.
//

#ifndef COMPRESSED_TENSOR_ALGEBRA_LOGGER_H
#define COMPRESSED_TENSOR_ALGEBRA_LOGGER_H

#include "iostream"
struct Logger {
public:
  int a_scaler_loads = 0;
  int a_scaler_stores = 0;
  int a_vector_loads_type_load1 = 0;
  int a_vector_loads_type_load4 = 0;
  int a_vector_store = 0;

  int b_scaler_loads = 0;
  int b_scaler_stores = 0;
  int b_vector_loads_type_load1 = 0;
  int b_vector_loads_type_load4 = 0;

  int c_scaler_loads = 0;
  int c_scaler_stores = 0;
  int c_vector_store = 0;

  int metadata_scaler_loads = 0;
  int metadata_scaler_stores = 0;

  int extra_scaler_loads = 0;
  int extra_scaler_stores = 0;

  int vector_multiply_add = 0;
  int scaler_multiply_Add = 0;

  std::string printHeader(){
    return "a_scaler_loads,"
           " a_scaler_stores,"
           "a_vector_loads_type_load1,"
           "a_vector_loads_type_load4,"
           "a_vector_store,"
           "b_scaler_loads,"
           "b_scaler_stores,"
           "b_vector_loads_type_load1,"
           "b_vector_loads_type_load4,"
           "c_scaler_loads,"
           "c_scaler_stores,"
           "c_vector_store,"
           "metadata_scaler_loads,"
           "metadata_scaler_stores,"
           "extra_scaler_loads,"
           "extra_scaler_stores,"
           "vector_multiply_add,"
           "scaler_multiply_Add";
  }

  std::string printStat(){
    return std::to_string(a_scaler_loads) + "," +
           std::to_string(a_scaler_stores) + "," +
           std::to_string(a_vector_loads_type_load1) + "," +
           std::to_string(a_vector_loads_type_load4) + "," +
           std::to_string(a_vector_store) + "," +
           std::to_string(b_scaler_loads) + "," +
           std::to_string(b_scaler_stores) + "," +
           std::to_string(b_vector_loads_type_load1) + "," +
           std::to_string(b_vector_loads_type_load4) + "," +
           std::to_string(c_scaler_loads) + "," +
           std::to_string(c_scaler_stores) + "," +
           std::to_string(c_vector_store) + "," +
           std::to_string(metadata_scaler_loads) + "," +
           std::to_string(metadata_scaler_stores) + "," +
           std::to_string(extra_scaler_loads) + "," +
           std::to_string(extra_scaler_stores) + "," +
           std::to_string(vector_multiply_add) + "," +
           std::to_string(scaler_multiply_Add);
  }
};

#endif // COMPRESSED_TENSOR_ALGEBRA_LOGGER_H
