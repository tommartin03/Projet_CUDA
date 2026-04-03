#pragma once

#include <cstddef>
#include <cuda_runtime.h>

void laplace_cpu(
  const unsigned char * input,
  unsigned char * output,
  int width,
  int height,
  int channels
);

void laplace_gpu(
  const unsigned char * input,
  unsigned char * output,
  int width,
  int height,
  int channels,
  dim3 block
);

void laplace_gpu_shared(
  const unsigned char * input,
  unsigned char * output,
  int width,
  int height,
  int channels,
  dim3 block
);
