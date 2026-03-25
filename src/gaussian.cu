#include "gaussian.cuh"
#include "utils.cuh"
#include <algorithm>
#include <cuda_runtime.h>

// Noyau gaussien 5x5 — normalisé par 256
static const int GAUSS_KERNEL[5][5] = {
  { 1,  4,  6,  4,  1 },
  { 4, 16, 24, 16,  4 },
  { 6, 24, 36, 24,  6 },
  { 4, 16, 24, 16,  4 },
  { 1,  4,  6,  4,  1 }
};
static const int GAUSS_NORM = 256;
static const int GAUSS_RADIUS = 2;

// Noyau stocké en mémoire constante GPU (très rapide, cachée)
__constant__ int d_gauss_kernel[5][5] = {
  { 1,  4,  6,  4,  1 },
  { 4, 16, 24, 16,  4 },
  { 6, 24, 36, 24,  6 },
  { 4, 16, 24, 16,  4 },
  { 1,  4,  6,  4,  1 }
};

// VERSION CPU DE RÉFÉRENCE
void gaussian_cpu(const unsigned char * input, unsigned char * output, int width, int height, int channels)
{
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      for (int c = 0; c < channels; ++c)
      {
        int sum = 0;

        for (int ky = -GAUSS_RADIUS; ky <= GAUSS_RADIUS; ++ky)
          for (int kx = -GAUSS_RADIUS; kx <= GAUSS_RADIUS; ++kx)
          {
            int nx = std::min(std::max(x + kx, 0), width  - 1);
            int ny = std::min(std::max(y + ky, 0), height - 1);
            sum += GAUSS_KERNEL[ky + GAUSS_RADIUS][kx + GAUSS_RADIUS] * input[(ny * width + nx) * channels + c];
          }

        output[(y * width + x) * channels + c] = static_cast<unsigned char>(sum / GAUSS_NORM);
      }
    }
  }
}


// VERSION GPU — MÉMOIRE GLOBALE
__global__ void gaussian_kernel(const unsigned char * input, unsigned char * output, int width, int height, int channels)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height) return;

  const int radius = 2;
  const int norm   = 256;

  for (int c = 0; c < channels; ++c)
  {
    int sum = 0;
    for (int ky = -radius; ky <= radius; ++ky)
      for (int kx = -radius; kx <= radius; ++kx)
      {
        int nx = min(max(x + kx, 0), width  - 1);
        int ny = min(max(y + ky, 0), height - 1);
        sum += d_gauss_kernel[ky + radius][kx + radius] * input[(ny * width + nx) * channels + c];
      }

    output[(y * width + x) * channels + c] = static_cast<unsigned char>(sum / norm);
  }
}

void gaussian_gpu(const unsigned char * input, unsigned char *output, int width, int height, int channels, dim3 block)
{
  dim3 grid((width  + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  std::size_t imgSize = (std::size_t)width * height * channels;

  unsigned char * in_d  = nullptr;
  unsigned char * out_d = nullptr;

  CUDA_CHECK(cudaMalloc(&in_d,  imgSize));
  CUDA_CHECK(cudaMalloc(&out_d, imgSize));
  CUDA_CHECK(cudaMemcpy(in_d, input, imgSize, cudaMemcpyHostToDevice));

  gaussian_kernel<<<grid, block>>>(in_d, out_d, width, height, channels);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(output, out_d, imgSize, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(in_d));
  CUDA_CHECK(cudaFree(out_d));
}


// VERSION GPU OPTIMISÉE — MÉMOIRE PARTAGÉE
// On traite un canal à la fois via le paramètre channel_idx.
// La taille de bloc est plafonnée à 16x16 car le noyau 5x5 avec halo=2 consomme trop de registres pour des blocs 32x32.

__global__ void gaussian_kernel_shared_single(const unsigned char * __restrict__ input, unsigned char * __restrict__ output, int width, int height, int channels, int channel_idx)
{
  const int HALO = 2;
  const int NORM = 256;

  // tileW et tileH calculés depuis blockDim (max 16+4=20)
  int tileW = blockDim.x + 2 * HALO;

  extern __shared__ unsigned char tile[];

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  int tileH = blockDim.y + 2 * HALO;

  // Chargement collaboratif de la tuile + halo
  for (int dy = threadIdx.y; dy < tileH; dy += blockDim.y)
  {
    for (int dx = threadIdx.x; dx < tileW; dx += blockDim.x)
    {
      int gx = blockDim.x * blockIdx.x + dx - HALO;
      int gy = blockDim.y * blockIdx.y + dy - HALO;

      gx = min(max(gx, 0), width  - 1);
      gy = min(max(gy, 0), height - 1);

      tile[dy * tileW + dx] = input[(gy * width + gx) * channels + channel_idx];
    }
  }
  __syncthreads();

  if (x < width && y < height)
  {
    int sum = 0;
    for (int ky = 0; ky < 5; ++ky)
      for (int kx = 0; kx < 5; ++kx)
        sum += d_gauss_kernel[ky][kx] * (int)tile[(threadIdx.y + ky) * tileW + (threadIdx.x + kx)];

    output[(y * width + x) * channels + channel_idx] = static_cast<unsigned char>(min(sum / NORM, 255));
  }
}

void gaussian_gpu_shared(const unsigned char * input, unsigned char * output, int width, int height, int channels, dim3 block)
{
  const int HALO = 2;

  // Plafonnement à 16x16 — au-delà, trop de registres pour ce kernel
  dim3 safeBlock(min(block.x, (unsigned)16), min(block.y, (unsigned)16));

  if (block.x > 16 || block.y > 16)
    std::cout << "  [Info] Gaussian shared : bloc réduit à 16x16 "
              << "(limite registres GPU)\n";

  dim3 grid((width  + safeBlock.x - 1) / safeBlock.x,
            (height + safeBlock.y - 1) / safeBlock.y);

  // Shared memory pour une tuile d'un seul canal (max 20x20 = 400 bytes)
  std::size_t sharedSize = (safeBlock.x + 2 * HALO) * (safeBlock.y + 2 * HALO) * sizeof(unsigned char);

  std::size_t imgSize = (std::size_t)width * height * channels;

  unsigned char * in_d  = nullptr;
  unsigned char * out_d = nullptr;

  CUDA_CHECK(cudaMalloc(&in_d,  imgSize));
  CUDA_CHECK(cudaMalloc(&out_d, imgSize));
  CUDA_CHECK(cudaMemcpy(in_d, input, imgSize, cudaMemcpyHostToDevice));

  // Un lancement de kernel par canal — réduit l'usage des registres
  for (int c = 0; c < channels; ++c)
  {
    gaussian_kernel_shared_single<<<grid, safeBlock, sharedSize>>>(in_d, out_d, width, height, channels, c);
    CUDA_CHECK_KERNEL();
  }

  CUDA_CHECK(cudaMemcpy(output, out_d, imgSize, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(in_d));
  CUDA_CHECK(cudaFree(out_d));
}
