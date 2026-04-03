#include "sobel.cuh"
#include "utils.cuh"
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>


// VERSION CPU DE RÉFÉRENCE

// Noyaux de Sobel
static const int GX[3][3] = { {-1, 0, 1},
                               {-2, 0, 2},
                               {-1, 0, 1} };

static const int GY[3][3] = { {-1, -2, -1},
                               { 0,  0,  0},
                               { 1,  2,  1} };

void sobel_cpu( const unsigned char * input, unsigned char * output, int width, int height, int channels)
{
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      // On travaille en niveaux de gris (moyenne des canaux)
      float gx = 0.f, gy = 0.f;

      for (int ky = -1; ky <= 1; ++ky)
      {
        for (int kx = -1; kx <= 1; ++kx)
        {
          int nx = std::min(std::max(x + kx, 0), width  - 1);
          int ny = std::min(std::max(y + ky, 0), height - 1);

          // Conversion en niveau de gris à la volée
          float gray = 0.f;
          for (int c = 0; c < channels; ++c)
            gray += input[(ny * width + nx) * channels + c];
          gray /= channels;

          gx += GX[ky + 1][kx + 1] * gray;
          gy += GY[ky + 1][kx + 1] * gray;
        }
      }

      // Magnitude du gradient, clampée à [0, 255]
      float mag = std::sqrt(gx * gx + gy * gy);
      unsigned char val = static_cast<unsigned char>(
        std::min(std::max(mag, 0.f), 255.f));

      // Écriture sur tous les canaux de sortie
      for (int c = 0; c < channels; ++c)
        output[(y * width + x) * channels + c] = val;
    }
  }
}

// VERSION GPU — MÉMOIRE GLOBALE
__global__ void sobel_kernel( const unsigned char * input, unsigned char * output, int width, int height, int channels)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height) return;

  // Noyaux de Sobel en mémoire constante du kernel
  const int gx_kernel[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
  const int gy_kernel[3][3] = { {-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1} };

  float gx = 0.f, gy = 0.f;

  for (int ky = -1; ky <= 1; ++ky)
  {
    for (int kx = -1; kx <= 1; ++kx)
    {
      int nx = min(max(x + kx, 0), width  - 1);
      int ny = min(max(y + ky, 0), height - 1);

      float gray = 0.f;
      for (int c = 0; c < channels; ++c)
        gray += input[(ny * width + nx) * channels + c];
      gray /= channels;

      gx += gx_kernel[ky + 1][kx + 1] * gray;
      gy += gy_kernel[ky + 1][kx + 1] * gray;
    }
  }

  float mag = sqrtf(gx * gx + gy * gy);
  unsigned char val = static_cast<unsigned char>(
    fminf(fmaxf(mag, 0.f), 255.f));

  for (int c = 0; c < channels; ++c)
    output[(y * width + x) * channels + c] = val;
}

void sobel_gpu(const unsigned char * input, unsigned char * output, int width, int height, int channels, dim3 block)
{
  dim3 grid((width  + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  std::size_t imgSize = width * height * channels * sizeof(unsigned char);

  unsigned char * in_d  = nullptr;
  unsigned char * out_d = nullptr;

  CUDA_CHECK(cudaMalloc(&in_d,  imgSize));
  CUDA_CHECK(cudaMalloc(&out_d, imgSize));
  CUDA_CHECK(cudaMemcpy(in_d, input, imgSize, cudaMemcpyHostToDevice));

  sobel_kernel<<<grid, block>>>(in_d, out_d, width, height, channels);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(output, out_d, imgSize, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(in_d));
  CUDA_CHECK(cudaFree(out_d));
}


// VERSION GPU OPTIMISÉE — STREAMS CUDA
// L'image est découpée en deux moitiés horizontales.
// Chaque stream gère : transfert H→D / kernel / transfert D→H
// de façon asynchrone et potentiellement en parallèle.

void sobel_gpu_streams(const unsigned char * input, unsigned char * output, int width, int height, int channels, dim3 block)
{
  // On divise l'image en 2 moitiés horizontales
  int const nStreams  = 2;
  int const halfH = height / 2;

  // Tailles en octets pour chaque moitié
  // La seconde moitié peut avoir une ligne de plus si height est impair
  std::size_t sizeHalf0 = (std::size_t)width * halfH * channels;
  std::size_t sizeHalf1 = (std::size_t)width * (height - halfH) * channels;

  std::size_t imgSize = (std::size_t)width * height * channels;

  // La mémoire hôte doit être "pinned" (non-paginable) pour les streams
  unsigned char * in_pinned  = nullptr;
  unsigned char * out_pinned = nullptr;

  CUDA_CHECK(cudaMallocHost(&in_pinned,  imgSize));
  CUDA_CHECK(cudaMallocHost(&out_pinned, imgSize));

  // Copie des données vers la mémoire paginée
  std::copy(input, input + imgSize, in_pinned);

  unsigned char * in_d  = nullptr;
  unsigned char * out_d = nullptr;

  CUDA_CHECK(cudaMalloc(&in_d,  imgSize));
  CUDA_CHECK(cudaMalloc(&out_d, imgSize));

  // Création des streams
  cudaStream_t streams[nStreams];
  for (int s = 0; s < nStreams; ++s)
    CUDA_CHECK(cudaStreamCreate(&streams[s]));

  // Offset en octets pour la seconde moitié
  std::size_t offset0 = 0;
  std::size_t offset1 = sizeHalf0;

  int heights[nStreams] = { halfH, height - halfH };

  //Stream 0 : première moitié 
  CUDA_CHECK(cudaMemcpyAsync(in_d + offset0, in_pinned + offset0, sizeHalf0, cudaMemcpyHostToDevice, streams[0]));

  // Stream 1 : seconde moitié 
  CUDA_CHECK(cudaMemcpyAsync(in_d + offset1, in_pinned + offset1, sizeHalf1, cudaMemcpyHostToDevice, streams[1]));

  // Lancement des kernels dans chaque stream
  for (int s = 0; s < nStreams; ++s)
  {
    int offsetY = s * halfH;
    dim3 grid((width + block.x - 1) / block.x, (heights[s] + block.y - 1) / block.y);

    // Le kernel a besoin du contexte complet de l'image pour les bords, mais n'écrit que dans sa portion de sortie.
    // On passe l'image complète et on décale la sortie.
    sobel_kernel<<<grid, block, 0, streams[s]>>>(
      in_d,
      out_d + s * sizeHalf0,  // Écriture dans la bonne moitié
      width, heights[s], channels);
    CUDA_CHECK_KERNEL();
  }

  // Récupération des résultats
  CUDA_CHECK(cudaMemcpyAsync(out_pinned + offset0, out_d + offset0, sizeHalf0, cudaMemcpyDeviceToHost, streams[0]));

  CUDA_CHECK(cudaMemcpyAsync(out_pinned + offset1, out_d + offset1, sizeHalf1, cudaMemcpyDeviceToHost, streams[1]));

  // Attendre la fin de tous les streams
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copie vers le buffer de sortie
  std::copy(out_pinned, out_pinned + imgSize, output);

  // Nettoyage
  for (int s = 0; s < nStreams; ++s)
    CUDA_CHECK(cudaStreamDestroy(streams[s]));

  CUDA_CHECK(cudaFree(in_d));
  CUDA_CHECK(cudaFree(out_d));
  CUDA_CHECK(cudaFreeHost(in_pinned));
  CUDA_CHECK(cudaFreeHost(out_pinned));
}
