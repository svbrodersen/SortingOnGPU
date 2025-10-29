// bench_cub_like_futhark.cu
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm> 
#include <string>    
#include <cuda_runtime.h>
#include <cub/cub.cuh>

static void ck(cudaError_t e, const char* where){
  if(e != cudaSuccess){
    std::fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e));
    std::exit(1);
  }
}

static std::string with_underscores(size_t n){
  // format e.g. 1000000 -> "1_000_000"
  std::string s = std::to_string(n);
  std::string out; out.reserve(s.size() + s.size()/3);
  int cnt = 0;
  for (int i = (int)s.size()-1; i >= 0; --i){
    out.push_back(s[i]);
    cnt++;
    if (cnt == 3 && i != 0){ out.push_back('_'); cnt = 0; }
  }
  std::reverse(out.begin(), out.end());
  return out;
}

int main(int argc, char** argv){
  if (argc < 2){
    std::fprintf(stderr, "Usage: %s <N> [runs]\n", argv[0]);
    return 1;
  }
  const size_t N = std::strtoull(argv[1], nullptr, 10);
  const int runs = (argc >= 3) ? std::atoi(argv[2]) : 10;

  // Host data (deterministic so results are reproducible)
  std::vector<unsigned int> h(N);
  std::mt19937 rng(123456);
  std::uniform_int_distribution<unsigned int> dist(0, 0xffffffffu);
  for (size_t i = 0; i < N; ++i) h[i] = dist(rng);

  // Device buffers: original input, working input, and output
  unsigned int *d_orig=nullptr, *d_in=nullptr, *d_out=nullptr;
  ck(cudaMalloc(&d_orig, N*sizeof(unsigned int)), "cudaMalloc d_orig");
  ck(cudaMalloc(&d_in,   N*sizeof(unsigned int)), "cudaMalloc d_in");
  ck(cudaMalloc(&d_out,  N*sizeof(unsigned int)), "cudaMalloc d_out");
  ck(cudaMemcpy(d_orig, h.data(), N*sizeof(unsigned int), cudaMemcpyHostToDevice), "HtoD d_orig");

  // CUB temp storage
  void* d_temp = nullptr; size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, N, 0, 32);
  ck(cudaMalloc(&d_temp, temp_bytes), "cudaMalloc temp");

  // Warmup: also ensures kernels are JITed and clocks up
  for (int i = 0; i < 3; ++i){
    ck(cudaMemcpy(d_in, d_orig, N*sizeof(unsigned int), cudaMemcpyDeviceToDevice), "D2D warmup");
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, N, 0, 32);
  }
  ck(cudaDeviceSynchronize(), "sync warmup");

  // Time per-run with CUDA events so we can produce CI
  std::vector<float> times_ms; times_ms.reserve(runs);
  cudaEvent_t start, stop;
  ck(cudaEventCreate(&start), "event start");
  ck(cudaEventCreate(&stop),  "event stop");

  for (int r = 0; r < runs; ++r){
    ck(cudaMemcpy(d_in, d_orig, N*sizeof(unsigned int), cudaMemcpyDeviceToDevice), "D2D reset");
    ck(cudaEventRecord(start), "record start");
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, N, 0, 32);
    ck(cudaEventRecord(stop),  "record stop");
    ck(cudaEventSynchronize(stop), "sync stop");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    times_ms.push_back(ms);
  }

  ck(cudaEventDestroy(start), "destroy start");
  ck(cudaEventDestroy(stop),  "destroy stop");

  // Stats: mean and 95% CI of the mean
  double sum = 0.0;
  for (float v : times_ms) sum += v;
  double mean_ms = sum / runs;

  double var = 0.0;
  for (float v : times_ms){
    double d = v - mean_ms;
    var += d * d;
  }
  var /= (runs > 1 ? (runs - 1) : 1);
  double sd_ms = std::sqrt(var);
  double se_ms = sd_ms / std::sqrt((double)runs);
  // 95% CI using normal approx (for n=10 it’s fine): mean ± 1.96 * SE
  double half95_ms = 1.96 * se_ms;

  // Units like Futhark: microseconds (μs)
  double mean_us   = mean_ms * 1000.0;
  double lo95_us   = (mean_ms - half95_ms) * 1000.0;
  double hi95_us   = (mean_ms + half95_ms) * 1000.0;

  // Throughput (Gkeys/s) for convenience
  double gkeys = (double)N / 1e6 / mean_ms;

  // Print in the same style as futhark bench lines:
  // n=1_000:        383μs (95% CI: [     381.8,      383.9])
  auto label = with_underscores(N);
  std::printf("n=%s: %11.0fμs (95%% CI: [%11.1f, %11.1f])  // %.6f Gkeys/s\n",
              label.c_str(), mean_us, lo95_us, hi95_us, gkeys);

  // Cleanup
  cudaFree(d_temp);
  cudaFree(d_out);
  cudaFree(d_in);
  cudaFree(d_orig);
  return 0;
}
