#include "boxblur.cuh"
#include "utils.cuh"
#include <algorithm>
#include <cuda_runtime.h>


// VERSION CPU DE RÉFÉRENCE

void boxblur_cpu(const unsigned char * input, unsigned char * output, int width, int height, int channels, int radius)
{
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      for (int c = 0; c < channels; ++c)
      {
        int sum   = 0;
        int count = 0;

        // Parcours de la fenêtre (2*radius+1) x (2*radius+1)
        for (int ky = -radius; ky <= radius; ++ky)
        {
          for (int kx = -radius; kx <= radius; ++kx)
          {
            int nx = x + kx;
            int ny = y + ky;

            // Clamp aux bords de l'image
            nx = std::max(0, std::min(nx, width  - 1));
            ny = std::max(0, std::min(ny, height - 1));

            sum += input[(ny * width + nx) * channels + c];
            ++count;
          }
        }
        output[(y * width + x) * channels + c] = static_cast<unsigned char>(sum / count);
      }
    }
  }
}

// VERSION GPU — MÉMOIRE GLOBALE

__global__ void boxblur_kernel(const unsigned char * input, unsigned char * output, int width, int height, int channels, int radius)
{
  // Coordonnées globales du pixel traité par ce thread
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height) return;

  for (int c = 0; c < channels; ++c)
  {
    int sum = 0;
    int count = 0;

    for (int ky = -radius; ky <= radius; ++ky)
    {
      for (int kx = -radius; kx <= radius; ++kx)
      {
        int nx = min(max(x + kx, 0), width  - 1);
        int ny = min(max(y + ky, 0), height - 1);

        sum += input[(ny * width + nx) * channels + c];
        ++count;
      }
    }
    output[(y * width + x) * channels + c] =static_cast<unsigned char>(sum / count);
  }
}

void boxblur_gpu(const unsigned char * input, unsigned char * output, int width, int height, int channels, int radius, dim3 block)
{
  // Calcul automatique de la grille en fonction de la taille de l'image
  dim3 grid((width  + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  std::size_t imgSize = width * height * channels * sizeof(unsigned char);

  unsigned char * in_d  = nullptr;
  unsigned char * out_d = nullptr;

  CUDA_CHECK(cudaMalloc(&in_d,  imgSize));
  CUDA_CHECK(cudaMalloc(&out_d, imgSize));

  CUDA_CHECK(cudaMemcpy(in_d, input, imgSize, cudaMemcpyHostToDevice));

  boxblur_kernel<<<grid, block>>>(in_d, out_d, width, height, channels, radius);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(output, out_d, imgSize, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(in_d));
  CUDA_CHECK(cudaFree(out_d));
}

// VERSION GPU OPTIMISÉE — MÉMOIRE PARTAGÉE

// Chaque bloc charge sa tuile + un halo de taille radius dans la shared memory afin de réduire les accès redondants à la mémoire globale.
// Taille maximale de bloc supportée par ce kernel (définie à la compilation)
#define MAX_BLOCK_DIM 32

__global__ void boxblur_kernel_shared(const unsigned char * input, unsigned char * output, int width, int height, int channels, int radius)
{
  // Taille de la tuile en mémoire partagée (bloc + halo des deux côtés)
  // On travaille canal par canal pour rester dans les limites de la shared memory
  extern __shared__ unsigned char tile[];

  int tileW = blockDim.x + 2 * radius;
  int tileH = blockDim.y + 2 * radius;

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  for (int c = 0; c < channels; ++c)
  {
    // Chargement collaboratif du halo dans la shared memory 
    // Chaque thread charge son pixel central + contribue au halo
    for (int dy = threadIdx.y; dy < tileH; dy += blockDim.y)
    {
      for (int dx = threadIdx.x; dx < tileW; dx += blockDim.x)
      {
        int gx = blockDim.x * blockIdx.x + dx - radius;
        int gy = blockDim.y * blockIdx.y + dy - radius;

        // Clamp aux bords
        gx = min(max(gx, 0), width  - 1);
        gy = min(max(gy, 0), height - 1);

        tile[dy * tileW + dx] = input[(gy * width + gx) * channels + c];
      }
    }
    __syncthreads(); // Attendre que tous les threads aient chargé le halo

    if (x < width && y < height)
    {
      int sum   = 0;
      int count = 0;

      for (int ky = 0; ky <= 2 * radius; ++ky)
      {
        for (int kx = 0; kx <= 2 * radius; ++kx)
        {
          sum += tile[(threadIdx.y + ky) * tileW + (threadIdx.x + kx)];
          ++count;
        }
      }
      output[(y * width + x) * channels + c] = static_cast<unsigned char>(sum / count);
    }
    __syncthreads(); // Réinitialiser avant le canal suivant
  }
}

void boxblur_gpu_shared(const unsigned char * input, unsigned char * output, int width, int height, int channels, int radius, dim3 block)
{
  dim3 grid((width  + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  // Taille de la mémoire partagée nécessaire par bloc (1 canal à la fois)
  std::size_t sharedSize =(block.x + 2 * radius) * (block.y + 2 * radius) * sizeof(unsigned char);

  std::size_t imgSize = width * height * channels * sizeof(unsigned char);

  unsigned char * in_d  = nullptr;
  unsigned char * out_d = nullptr;

  CUDA_CHECK(cudaMalloc(&in_d,  imgSize));
  CUDA_CHECK(cudaMalloc(&out_d, imgSize));
  CUDA_CHECK(cudaMemcpy(in_d, input, imgSize, cudaMemcpyHostToDevice));

  boxblur_kernel_shared<<<grid, block, sharedSize>>>(in_d, out_d, width, height, channels, radius);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(output, out_d, imgSize, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(in_d));
  CUDA_CHECK(cudaFree(out_d));
}
