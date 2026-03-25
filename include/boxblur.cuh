#pragma once

#include <cstddef>
#include <cuda_runtime.h>

void boxblur_cpu(
  const unsigned char * input,
  unsigned char *       output,
  int                   width,
  int                   height,
  int                   channels,
  int                   radius
);

void boxblur_gpu(
  const unsigned char * input,
  unsigned char *       output,
  int                   width,
  int                   height,
  int                   channels,
  int                   radius,
  dim3                  block
);

void boxblur_gpu_shared(
  const unsigned char * input,
  unsigned char *       output,
  int                   width,
  int                   height,
  int                   channels,
  int                   radius,
  dim3                  block
);
