cmake_minimum_required(VERSION 3.18)

project(cuda_kernel_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")

find_package(CUDAToolkit REQUIRED)

add_executable(matmul0 matmul0.cu)
add_executable(matmul1 matmul1.cu)
add_executable(matmul2 matmul2.cu)
add_executable(matmul3 matmul3.cu)

target_link_libraries(matmul0 PRIVATE CUDA::cublas)
target_link_libraries(matmul1 PRIVATE CUDA::cublas)
target_link_libraries(matmul2 PRIVATE CUDA::cublas)
target_link_libraries(matmul3 PRIVATE CUDA::cublas)
