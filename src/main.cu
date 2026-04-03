#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include "utils.cuh"
#include "boxblur.cuh"
#include "sobel.cuh"
#include "laplace.cuh"
#include "gaussian.cuh"


static void print_gpu_info()
{
  int nDevices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&nDevices));
  std::cout << "=== GPU Info ===\n";
  std::cout << "Nombre de GPUs : " << nDevices << "\n";

  for (int i = 0; i < nDevices; ++i)
  {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
    std::cout << "  [" << i << "] " << prop.name << "\n";
    std::cout << "      Threads max/bloc   : " << prop.maxThreadsPerBlock << "\n";
    std::cout << "      Multiprocesseurs   : " << prop.multiProcessorCount  << "\n";
    std::cout << "      Mémoire globale    : "
              << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "      Mémoire partagée/bloc : "
              << prop.sharedMemPerBlock / 1024 << " KB\n";
  }
  std::cout << "================\n\n";
}

static bool load_image(
  const std::string & path,
  cv::Mat & img,
  int & width,
  int & height,
  int & channels)
{
  img = cv::imread(path, cv::IMREAD_COLOR);
  if (img.empty())
  {
    std::cerr << "[Erreur] Impossible de charger l'image : " << path << "\n";
    return false;
  }
  width = img.cols;
  height = img.rows;
  channels = img.channels();
  std::cout << "Image : " << path
            << "  [" << width << "x" << height
            << ", " << channels << " canaux]\n\n";
  return true;
}

static void save_image(
  const std::string & path,
  const unsigned char * data,
  int width,
  int height,
  int channels)
{
  cv::Mat out(height, width, channels == 1 ? CV_8UC1 : CV_8UC3, const_cast<unsigned char *>(data));
  cv::imwrite(path, out);
  std::cout << "  Résultat sauvegardé : " << path << "\n";
}

struct FilterResult
{
  float cpu_ms = 0.f;
  float gpu_ms = 0.f;
  float gpu_opt_ms = 0.f;
};

// BoxBlur
static FilterResult bench_boxblur(const unsigned char * input, int w, int h, int c, int radius, dim3 block, const std::string & prefix)
{
  std::size_t sz = w * h * c;
  std::vector<unsigned char> out_cpu(sz), out_gpu(sz), out_opt(sz);

  FilterResult res;
  CpuTimer cpu_timer;
  GpuTimer gpu_timer;

  std::cout << "--- BoxBlur (radius=" << radius << ") ---\n";

  // CPU
  cpu_timer.start();
  boxblur_cpu(input, out_cpu.data(), w, h, c, radius);
  res.cpu_ms = cpu_timer.stop();
  std::cout << "  CPU          : " << res.cpu_ms << " ms\n";
  save_image(prefix + "_boxblur_cpu.jpg", out_cpu.data(), w, h, c);

  // GPU global
  gpu_timer.start();
  boxblur_gpu(input, out_gpu.data(), w, h, c, radius, block);
  res.gpu_ms = gpu_timer.stop();
  std::cout << "  GPU global   : " << res.gpu_ms << " ms\n";
  save_image(prefix + "_boxblur_gpu.jpg", out_gpu.data(), w, h, c);

  // GPU shared memory
  gpu_timer.start();
  boxblur_gpu_shared(input, out_opt.data(), w, h, c, radius, block);
  res.gpu_opt_ms = gpu_timer.stop();
  std::cout << "  GPU shared   : " << res.gpu_opt_ms << " ms\n";
  save_image(prefix + "_boxblur_gpu_shared.jpg", out_opt.data(), w, h, c);

  std::cout << "  Accélération GPU/CPU : x"
            << res.cpu_ms / res.gpu_ms << "\n\n";
  return res;
}


// Sobel
static FilterResult bench_sobel(const unsigned char * input, int w, int h, int c, dim3 block, const std::string & prefix)
{
  std::size_t sz = w * h * c;
  std::vector<unsigned char> out_cpu(sz), out_gpu(sz), out_stream(sz);

  FilterResult res;
  CpuTimer cpu_timer;
  GpuTimer gpu_timer;

  std::cout << "--- Sobel ---\n";

  // CPU
  cpu_timer.start();
  sobel_cpu(input, out_cpu.data(), w, h, c);
  res.cpu_ms = cpu_timer.stop();
  std::cout << "  CPU          : " << res.cpu_ms << " ms\n";
  save_image(prefix + "_sobel_cpu.jpg", out_cpu.data(), w, h, c);

  // GPU global
  gpu_timer.start();
  sobel_gpu(input, out_gpu.data(), w, h, c, block);
  res.gpu_ms = gpu_timer.stop();
  std::cout << "  GPU global   : " << res.gpu_ms << " ms\n";
  save_image(prefix + "_sobel_gpu.jpg", out_gpu.data(), w, h, c);

  // GPU streams
  gpu_timer.start();
  sobel_gpu_streams(input, out_stream.data(), w, h, c, block);
  res.gpu_opt_ms = gpu_timer.stop();
  std::cout << "  GPU streams  : " << res.gpu_opt_ms << " ms\n";
  save_image(prefix + "_sobel_gpu_streams.jpg", out_stream.data(), w, h, c);

  std::cout << "  Accélération GPU/CPU : x"
            << res.cpu_ms / res.gpu_ms << "\n\n";
  return res;
}


// Laplace
static FilterResult bench_laplace(const unsigned char * input, int w, int h, int c, dim3 block, const std::string & prefix)
{
  std::size_t sz = w * h * c;
  std::vector<unsigned char> out_cpu(sz), out_gpu(sz), out_shared(sz);

  FilterResult res;
  CpuTimer cpu_timer;
  GpuTimer gpu_timer;

  std::cout << "--- Laplace ---\n";

  cpu_timer.start();
  laplace_cpu(input, out_cpu.data(), w, h, c);
  res.cpu_ms = cpu_timer.stop();
  std::cout << "  CPU          : " << res.cpu_ms << " ms\n";
  save_image(prefix + "_laplace_cpu.jpg", out_cpu.data(), w, h, c);

  gpu_timer.start();
  laplace_gpu(input, out_gpu.data(), w, h, c, block);
  res.gpu_ms = gpu_timer.stop();
  std::cout << "  GPU global   : " << res.gpu_ms << " ms\n";
  save_image(prefix + "_laplace_gpu.jpg", out_gpu.data(), w, h, c);

  gpu_timer.start();
  laplace_gpu_shared(input, out_shared.data(), w, h, c, block);
  res.gpu_opt_ms = gpu_timer.stop();
  std::cout << "  GPU shared   : " << res.gpu_opt_ms << " ms\n";
  save_image(prefix + "_laplace_gpu_shared.jpg", out_shared.data(), w, h, c);

  std::cout << "  Accélération GPU/CPU : x"
            << res.cpu_ms / res.gpu_ms << "\n\n";
  return res;
}


// Gaussian Blur
static FilterResult bench_gaussian(const unsigned char * input, int w, int h, int c, dim3 block, const std::string & prefix)
{
  std::size_t sz = w * h * c;
  std::vector<unsigned char> out_cpu(sz), out_gpu(sz), out_shared(sz);

  FilterResult res;
  CpuTimer cpu_timer;
  GpuTimer gpu_timer;

  std::cout << "--- Gaussian Blur ---\n";

  cpu_timer.start();
  gaussian_cpu(input, out_cpu.data(), w, h, c);
  res.cpu_ms = cpu_timer.stop();
  std::cout << "  CPU          : " << res.cpu_ms << " ms\n";
  save_image(prefix + "_gaussian_cpu.jpg", out_cpu.data(), w, h, c);

  gpu_timer.start();
  gaussian_gpu(input, out_gpu.data(), w, h, c, block);
  res.gpu_ms = gpu_timer.stop();
  std::cout << "  GPU global   : " << res.gpu_ms << " ms\n";
  save_image(prefix + "_gaussian_gpu.jpg", out_gpu.data(), w, h, c);

  gpu_timer.start();
  gaussian_gpu_shared(input, out_shared.data(), w, h, c, block);
  res.gpu_opt_ms = gpu_timer.stop();
  std::cout << "  GPU shared   : " << res.gpu_opt_ms << " ms\n";
  save_image(prefix + "_gaussian_gpu_shared.jpg", out_shared.data(), w, h, c);

  std::cout << "  Accélération GPU/CPU : x"
            << res.cpu_ms / res.gpu_ms << "\n\n";
  return res;
}


int main(int argc, char ** argv)
{
  if (argc < 2)
  {
    std::cerr << "Usage : " << argv[0] << " <image.jpg> [block_size=16]\n";
    return EXIT_FAILURE;
  }

  std::string imgPath  = argv[1];
  int blockDim = (argc >= 3) ? std::stoi(argv[2]) : 16;

  // Bloc 2D de threads (optimisation de la décomposition comme vu en TP2)
  dim3 block(blockDim, blockDim);

  print_gpu_info();

  // Chargement de l'image
  cv::Mat img;
  int width, height, channels;
  if (!load_image(imgPath, img, width, height, channels))
    return EXIT_FAILURE;

  const unsigned char * input = img.data;

  // Préfixe pour les fichiers de sortie (nom de l'image sans extension)
  std::string prefix = imgPath.substr(0, imgPath.find_last_of('.'));

  std::cout << "Taille des blocs GPU : " << blockDim << "x" << blockDim << "\n\n";

  // Lancement des benchmarks
  bench_boxblur (input, width, height, channels, /*radius=*/3, block, prefix);
  bench_sobel   (input, width, height, channels, block, prefix);
  bench_laplace (input, width, height, channels, block, prefix);
  bench_gaussian(input, width, height, channels, block, prefix);

  std::cout << "=== Tous les filtres ont été appliqués avec succès. ===\n";
  return EXIT_SUCCESS;
}
