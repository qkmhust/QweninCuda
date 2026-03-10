cmake_minimum_required(VERSION 3.22)
project(qwen3_cuda_minimal LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(QWEN_MINIMAL_ENABLE_FAST_MATH "Enable CUDA fast math" ON)

if(QWEN_MINIMAL_ENABLE_FAST_MATH)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()

add_executable(qwen_minimal
  src/main.cpp
  src/model.cpp
  src/kernels.cu
)

target_include_directories(qwen_minimal PRIVATE include)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(qwen_minimal PRIVATE CUDA::cudart CUDA::cublas)

set_target_properties(qwen_minimal PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "70;75;80;86;89"
)
