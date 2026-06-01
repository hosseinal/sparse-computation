project(spmv_demo_gpu LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
# add -std=c++17 flag for host code
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G --ptxas-options=-v")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86)
endif()

find_package(Threads REQUIRED)

#set(CMAKE_CUDA_ARCHITECTURES 50 75)
# TODO: after you had cmake 3.24, you can use the below line so no need to enter the version
#set_property(TARGET tgt PROPERTY CUDA_ARCHITECTURES native)

add_executable(spmv_demo_gpu
        spmv_demo_gpu.cu
        ${COMPRESSION_SRC_FILES}
        ${FORMAT_SRC_FILES}
        spmv_demo_gpu_utils.h
)

set_target_properties(spmv_demo_gpu PROPERTIES CUDA_COMPILE_FLAGS "${CUDA_COMPILE_FLAGS}")

target_include_directories(spmv_demo_gpu
        PRIVATE
        ${PROJECT_SOURCE_DIR}/aggregation/include
        ${PROJECT_SOURCE_DIR}/swbench/include
        ${PROJECT_SOURCE_DIR}/include
        ${CXXOPTS_INCLUDE_DIR}
)

target_link_libraries(spmv_demo_gpu PRIVATE
        cusparse
        cublas
        CUDA::cusparse
        aggregation
        swbench::swbench
)

set_property(TARGET spmv_demo_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)

#spmm_demo_gpu
add_executable(spmm_demo_gpu
        spmm_demo_gpu.cu
        ${COMPRESSION_SRC_FILES}
        ${FORMAT_SRC_FILES}
        spmm_demo_gpu_utils.h
)

set_target_properties(spmm_demo_gpu PROPERTIES CUDA_COMPILE_FLAGS "${CUDA_COMPILE_FLAGS}"
)

target_include_directories(spmm_demo_gpu
        PRIVATE
        ${PROJECT_SOURCE_DIR}/aggregation/include
        ${PROJECT_SOURCE_DIR}/swbench/include
        ${PROJECT_SOURCE_DIR}/include
        ${CXXOPTS_INCLUDE_DIR}
)
target_link_libraries(spmm_demo_gpu PRIVATE
        cusparse
        cublas
        CUDA::cusparse
        aggregation
        swbench::swbench
)

set_property(TARGET spmm_demo_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)


#spmm_small_b_col_demo_gpu
add_executable(spmm_small_b_col_demo_gpu
        spmm_small_b_col_demo_gpu.cu
        ${COMPRESSION_SRC_FILES}
        ${FORMAT_SRC_FILES}
        spmm_small_b_col_demo_gpu_utils.h
)

set_target_properties(spmm_small_b_col_demo_gpu PROPERTIES CUDA_COMPILE_FLAGS "${CUDA_COMPILE_FLAGS}")

target_include_directories(spmm_small_b_col_demo_gpu
        PRIVATE
        ${PROJECT_SOURCE_DIR}/aggregation/include
        ${PROJECT_SOURCE_DIR}/swbench/include
        ${PROJECT_SOURCE_DIR}/include
        ${CXXOPTS_INCLUDE_DIR}
)

target_link_libraries(spmm_small_b_col_demo_gpu PRIVATE
        cusparse
        cublas
        CUDA::cusparse
        aggregation
        swbench::swbench
)

set_property(TARGET spmm_small_b_col_demo_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)





