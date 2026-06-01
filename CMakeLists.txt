CMAKE_MINIMUM_REQUIRED(VERSION 3.20)

# Project name
#set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
project(compressed-tensor-algebra LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA OPTIONAL_COMPONENTS)
find_package(CUDAToolkit OPTIONAL_COMPONENTS)

# Directories
add_subdirectory(benchmark)
add_subdirectory(aggregation)

if(SWB_GPU_ENABLED)
    add_definitions(-DSWB_GPU_ENABLED)
endif()
if(PROFILING_WITH_PAPI)
    message(STATUS "Profiling with PAPI")
    message(PAPI_PREFIX)
    add_definitions(-DPROF_WITH_PAPI)
endif()

if(PROFILING_WITH_PAPI)
    include_directories(
            ${PAPI_INCLUDE_DIRS}
    )
endif()

set(LBC_INC "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(SPARSE_UTIL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/utils")
set(LBC_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(TENSOR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor_op")

set(COMPRESSION_SRC_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/test_utils.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/InputParser.cpp")

file(GLOB FILE_FORMAT_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp")

set(FORMAT_SRC_FILES "${FILE_FORMAT_SRC_FILES}")

file(GLOB FILE_SPMV_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/utils/spmv/*.cpp")

set(SPMV_SRC_FILES "${FILE_SPMV_SRC_FILES}")

file(GLOB FILE_GEMM_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/utils/gemm/*.cpp")

set(GEMM_SRC_FILES "${FILE_GEMM_SRC_FILES}")


file(GLOB FILE_GEMV_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/utils/gemv/*.cpp"
        )

set(GEMV_SRC_FILES "${FILE_GEMV_SRC_FILES}")

file(GLOB FILE_Sparse_Header "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")


include(FetchContent)
FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG v3.1.1
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cxxopts)
set(CXXOPTS_INCLUDE_DIR "${cxxopts_SOURCE_DIR}/include")

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/include
        ${CMAKE_CURRENT_SOURCE_DIR}/aggregation/include
        ${LBC_INC}
        ${SPARSE_UTIL_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tensor_op/
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

add_subdirectory(example_cuda)

if (CUDA_FOUND AND CUDAToolkit_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    add_definitions(-D__CUDA__)
    enable_language(CUDA)

else()
    message(STATUS "CUDA not found, proceeding without CUDA support")
    set(PROJECT_LANGUAGES CXX)
endif()
