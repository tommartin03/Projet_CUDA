#pragma once

#include <cstddef>
#include <cuda_runtime.h>

void gaussian_cpu(
  const unsigned char * input,
  unsigned char * output,
  int width,
  int height,
  int channels
);

void gaussian_gpu(
  const unsigned char * input,
  unsigned char * output,
  int width,
  int height,
  int channels,
  dim3 block
);

void gaussian_gpu_shared(
  const unsigned char * input,
  unsigned char * output,
  int width,
  int height,
  int channels,
  dim3 block
);
