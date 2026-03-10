#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdexcept>
#include <string>

inline void cuda_check(cudaError_t code, const char* expr, const char* file, int line) {
  if (code != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error at ") + file + ":" + std::to_string(line) +
                             " expr=" + expr + " msg=" + cudaGetErrorString(code));
  }
}

inline void cublas_check(cublasStatus_t code, const char* expr, const char* file, int line) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuBLAS error at ") + file + ":" + std::to_string(line) +
                             " expr=" + expr + " status=" + std::to_string(static_cast<int>(code)));
  }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) cublas_check((expr), #expr, __FILE__, __LINE__)
