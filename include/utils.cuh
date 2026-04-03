#pragma once

#include <iostream>
#include <chrono>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _err = (call);                                                  \
    if (_err != cudaSuccess) {                                                  \
      std::cerr << "[CUDA Error] " << cudaGetErrorString(_err)                 \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)


#define CUDA_CHECK_KERNEL()                                                     \
  do {                                                                          \
    cudaError_t _err = cudaGetLastError();                                      \
    if (_err != cudaSuccess) {                                                  \
      std::cerr << "[CUDA Kernel Error] " << cudaGetErrorString(_err)          \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

struct CpuTimer
{
  std::chrono::high_resolution_clock::time_point start_;

  void start()
  {
    start_ = std::chrono::high_resolution_clock::now();
  }

  float stop() const
  {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start_).count();
  }
};

struct GpuTimer
{
  cudaEvent_t start_;
  cudaEvent_t stop_;

  GpuTimer()
  {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }

  ~GpuTimer()
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start(cudaStream_t stream = 0)
  {
    CUDA_CHECK(cudaEventRecord(start_, stream));
  }

  float stop(cudaStream_t stream = 0)
  {
    float ms = 0.f;
    CUDA_CHECK(cudaEventRecord(stop_, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }
};
