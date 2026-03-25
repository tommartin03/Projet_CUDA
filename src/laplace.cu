#include "laplace.cuh"
#include "utils.cuh"
#include <algorithm>
#include <cuda_runtime.h>

// VERSION CPU DE RÉFÉRENCE

static const int LAPLACE_KERNEL[3][3] = {
  { 0,  1,  0 },
  { 1, -4,  1 },
  { 0,  1,  0 }
};

void laplace_cpu(const unsigned char * input, unsigned char * output, int width, int height, int channels)
{
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      for (int c = 0; c < channels; ++c)
      {
        int sum = 0;

        for (int ky = -1; ky <= 1; ++ky)
        {
          for (int kx = -1; kx <= 1; ++kx)
          {
            int nx = std::min(std::max(x + kx, 0), width  - 1);
            int ny = std::min(std::max(y + ky, 0), height - 1);

            sum += LAPLACE_KERNEL[ky + 1][kx + 1] * input[(ny * width + nx) * channels + c];
          }
        }

        // Valeur absolue pour visualiser les deux polarités du Laplacien, clampée à [0, 255]
        int val = std::abs(sum);
        output[(y * width + x) * channels + c] = static_cast<unsigned char>(std::min(val, 255));
      }
    }
  }
}

// VERSION GPU — MÉMOIRE GLOBALE

__global__ void laplace_kernel(const unsigned char * input, unsigned char * output, int width, int height, int channels)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height) return;

  const int kernel[3][3] = {
    { 0,  1,  0 },
    { 1, -4,  1 },
    { 0,  1,  0 }
  };

  for (int c = 0; c < channels; ++c)
  {
    int sum = 0;

    for (int ky = -1; ky <= 1; ++ky)
    {
      for (int kx = -1; kx <= 1; ++kx)
      {
        int nx = min(max(x + kx, 0), width  - 1);
        int ny = min(max(y + ky, 0), height - 1);

        sum += kernel[ky + 1][kx + 1] * input[(ny * width + nx) * channels + c];
      }
    }

    int val = abs(sum);
    output[(y * width + x) * channels + c] = static_cast<unsigned char>(min(val, 255));
  }
}

void laplace_gpu( const unsigned char * input, unsigned char * output, int width, int height, int channels, dim3 block)
{
  dim3 grid((width  + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  std::size_t imgSize = width * height * channels * sizeof(unsigned char);

  unsigned char * in_d  = nullptr;
  unsigned char * out_d = nullptr;

  CUDA_CHECK(cudaMalloc(&in_d,  imgSize));
  CUDA_CHECK(cudaMalloc(&out_d, imgSize));
  CUDA_CHECK(cudaMemcpy(in_d, input, imgSize, cudaMemcpyHostToDevice));

  laplace_kernel<<<grid, block>>>(in_d, out_d, width, height, channels);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(output, out_d, imgSize, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(in_d));
  CUDA_CHECK(cudaFree(out_d));
}

// VERSION GPU OPTIMISÉE — MÉMOIRE PARTAGÉE
// Chaque bloc charge sa tuile (bloc + halo 1 pixel) dans la shared memory.
// Avec un noyau 3x3, le halo est de 1 pixel de chaque côté.

__global__ void laplace_kernel_shared( const unsigned char * input, unsigned char * output, int width, int height, int channels)
{
  // Halo de 1 pour un noyau 3x3
  const int HALO = 1;

  int tileW = blockDim.x + 2 * HALO;
  int tileH = blockDim.y + 2 * HALO;

  // Shared memory allouée dynamiquement
  extern __shared__ unsigned char tile[];

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  const int kernel[3][3] = {
    { 0,  1,  0 },
    { 1, -4,  1 },
    { 0,  1,  0 }
  };

  for (int c = 0; c < channels; ++c)
  {
    // Chargement collaboratif de la tuile + halo
    for (int dy = threadIdx.y; dy < tileH; dy += blockDim.y)
    {
      for (int dx = threadIdx.x; dx < tileW; dx += blockDim.x)
      {
        int gx = blockDim.x * blockIdx.x + dx - HALO;
        int gy = blockDim.y * blockIdx.y + dy - HALO;

        gx = min(max(gx, 0), width  - 1);
        gy = min(max(gy, 0), height - 1);

        tile[dy * tileW + dx] = input[(gy * width + gx) * channels + c];
      }
    }
    __syncthreads();

    if (x < width && y < height)
    {
      int sum = 0;

      for (int ky = 0; ky < 3; ++ky)
        for (int kx = 0; kx < 3; ++kx)
          sum += kernel[ky][kx] * tile[(threadIdx.y + ky) * tileW + (threadIdx.x + kx)];

      int val = abs(sum);
      output[(y * width + x) * channels + c] = static_cast<unsigned char>(min(val, 255));
    }
    __syncthreads();
  }
}

void laplace_gpu_shared(const unsigned char * input, unsigned char * output, int width, int height, int channels, dim3 block)
{
  const int HALO = 1;

  dim3 grid((width  + block.x - 1) / block.x,(height + block.y - 1) / block.y);

  std::size_t sharedSize = (block.x + 2 * HALO) * (block.y + 2 * HALO) * sizeof(unsigned char);

  std::size_t imgSize = width * height * channels * sizeof(unsigned char);

  unsigned char * in_d  = nullptr;
  unsigned char * out_d = nullptr;

  CUDA_CHECK(cudaMalloc(&in_d,  imgSize));
  CUDA_CHECK(cudaMalloc(&out_d, imgSize));
  CUDA_CHECK(cudaMemcpy(in_d, input, imgSize, cudaMemcpyHostToDevice));

  laplace_kernel_shared<<<grid, block, sharedSize>>>(in_d, out_d, width, height, channels);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(output, out_d, imgSize, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(in_d));
  CUDA_CHECK(cudaFree(out_d));
}
