#pragma once

#include <cstddef>
#include <cuda_runtime.h>

void sobel_cpu(
  const unsigned char * input,
  unsigned char * output,
  int width,
  int height,
  int channels
);

void sobel_gpu(
  const unsigned char * input,
  unsigned char * output,
  int width,
  int height,
  int channels,
  dim3 block
);

void sobel_gpu_streams(
  const unsigned char * input,
  unsigned char * output,
  int width,
  int height,
  int channels,
  dim3 block
);
