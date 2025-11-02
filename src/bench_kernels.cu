// bench_kernels.cu — minimal, pretty output, tests K1 / K2 / both
#define WARP 32
#define lgWARP 5

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <string>
#include <algorithm>

#include "kernels.cuh"  // brings in pbb_kernels.cuh which needs WARP/lgWARP

// Tunables
static constexpr uint32_t B   = 256;         // block size
static constexpr uint32_t Q   = 22;          // elems per thread
static constexpr uint32_t LGH = 8;           // bits per pass
static constexpr uint32_t H   = (1u << LGH); // bins

static inline uint32_t ceil_div(uint32_t a, uint32_t b){ return (a + b - 1) / b; }

// Column-wise exclusive scan of hist[block][bin] -> scan[block][bin] (host)
static void host_column_scan(const uint32_t* hist, uint32_t* scan,
                             uint32_t num_blocks, uint32_t Hbins){
  for(uint32_t bin=0; bin<Hbins; ++bin){
    uint64_t acc = 0;
    for(uint32_t b=0; b<num_blocks; ++b){
      const size_t idx = (size_t)b*Hbins + bin;
      scan[idx] = (uint32_t)acc;  // exclusive
      acc += hist[idx];
    }
  }
}

// pretty "1_000_000"
static std::string with_underscores(size_t n){
  std::string s = std::to_string(n), out; out.reserve(s.size()+s.size()/3);
  int cnt=0; for(int i=(int)s.size()-1;i>=0;--i){ out.push_back(s[i]); if(++cnt==3 && i!=0){ out.push_back('_'); cnt=0; } }
  std::reverse(out.begin(), out.end()); return out;
}

// 95% CI on mean (μs) from ms samples
struct CI { double mean_us, lo95_us, hi95_us; };
static CI ci_from_ms(const std::vector<float>& ms){
  double sum=0.0; for(float v: ms) sum += v;
  const double mean_ms = sum / std::max<size_t>(1, ms.size());
  double var=0.0; for(float v: ms){ double d=v-mean_ms; var += d*d; }
  var /= (ms.size()>1 ? (ms.size()-1) : 1);
  const double sd_ms = std::sqrt(var);
  const double se_ms = sd_ms / std::sqrt((double)std::max<size_t>(1, ms.size()));
  const double half95_ms = 1.96 * se_ms;
  return { mean_ms*1000.0, (mean_ms-half95_ms)*1000.0, (mean_ms+half95_ms)*1000.0 };
}

// print exactly like your example
static void print_line(size_t N, const CI& ci){
  auto lbl = with_underscores(N);
  std::printf("n=%s:%11s%6.0fμs (95%% CI: [%12.1f, %12.1f])\n",
              lbl.c_str(), "", ci.mean_us, ci.lo95_us, ci.hi95_us);
}

int main(int argc, char** argv){
  // Usage: ./bench_kernels [--mode k1|k2|both] [--n N] [--runs R]
  const char* mode = "both";
  size_t N = (1u<<20);
  int runs = 10;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--mode") && i+1<argc) mode = argv[++i];
    else if(!strcmp(argv[i],"--n")    && i+1<argc) N    = strtoull(argv[++i], nullptr, 10);
    else if(!strcmp(argv[i],"--runs") && i+1<argc) runs = atoi(argv[++i]);
  }

  const uint32_t num_blocks = ceil_div((uint32_t)N, (uint32_t)(B*Q));
  const size_t   hist_elems = (size_t)num_blocks * H;
  const int      num_passes = (32 + LGH - 1) / LGH;

  // Deterministic host input
  std::vector<uint32_t> h(N);
  std::mt19937 rng(123456);
  std::uniform_int_distribution<uint32_t> dist(0u, 0xffffffffu);
  for(size_t i=0;i<N;++i) h[i] = dist(rng);

  // Device buffers
  uint32_t *d_in=nullptr, *d_out=nullptr, *d_hist=nullptr, *d_scan=nullptr;
  cudaMalloc(&d_in,  N*sizeof(uint32_t));
  cudaMalloc(&d_out, N*sizeof(uint32_t));
  cudaMemcpy(d_in, h.data(), N*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMalloc(&d_hist, hist_elems*sizeof(uint32_t));
  cudaMalloc(&d_scan, hist_elems*sizeof(uint32_t));

  // Host scratch for scan
  std::vector<uint32_t> h_hist(hist_elems), h_sc(hist_elems);

  // Warmup
  initial_kernel<uint32_t, H, LGH, Q><<<num_blocks, B>>>(d_in, d_hist, 0u, (uint32_t)N);
  cudaDeviceSynchronize();

  // Timing
  cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

  if (!strcmp(mode,"k1")){
    std::vector<float> totals_ms; totals_ms.reserve(runs);
    for(int r=0;r<runs;++r){
      float acc_ms = 0.0f;
      for(int pass=0; pass<num_passes; ++pass){
        cudaEventRecord(start);
        initial_kernel<uint32_t, H, LGH, Q><<<num_blocks, B>>>(d_in, d_hist, (uint32_t)pass, (uint32_t)N);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms=0.0f; cudaEventElapsedTime(&ms, start, stop);
        acc_ms += ms;
      }
      totals_ms.push_back(acc_ms);
    }
    print_line(N, ci_from_ms(totals_ms));

  } else if (!strcmp(mode,"k2")){
    uint32_t* d_orig=nullptr; cudaMalloc(&d_orig, N*sizeof(uint32_t));
    cudaMemcpy(d_orig, d_in, N*sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    std::vector<float> totals_ms; totals_ms.reserve(runs);
    for(int r=0;r<runs;++r){
      float acc_ms = 0.0f;
      for(int pass=0; pass<num_passes; ++pass){
        // prep (untimed)
        initial_kernel<uint32_t, H, LGH, Q><<<num_blocks, B>>>(d_in, d_hist, (uint32_t)pass, (uint32_t)N);
        cudaDeviceSynchronize();
        cudaMemcpy(h_hist.data(), d_hist, hist_elems*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        host_column_scan(h_hist.data(), h_sc.data(), num_blocks, H);
        cudaMemcpy(d_scan, h_sc.data(), hist_elems*sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_in, d_orig, N*sizeof(uint32_t), cudaMemcpyDeviceToDevice);

        // time K2
        const size_t shmem = (size_t)(B*Q + 2*H + B) * sizeof(uint32_t);
        cudaEventRecord(start);
        final_kernel<uint32_t, H, LGH, B, Q><<<num_blocks, B, shmem>>>(
          d_in, d_out, d_hist, d_scan, (uint32_t)pass, (uint32_t)N);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms=0.0f; cudaEventElapsedTime(&ms, start, stop);
        acc_ms += ms;
      }
      totals_ms.push_back(acc_ms);
    }
    print_line(N, ci_from_ms(totals_ms));
    cudaFree(d_orig);

  } else { // both
    uint32_t* d_orig=nullptr; cudaMalloc(&d_orig, N*sizeof(uint32_t));
    cudaMemcpy(d_orig, d_in, N*sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    std::vector<float> totals_ms; totals_ms.reserve(runs);
    for(int r=0;r<runs;++r){
      float acc_ms = 0.0f;
      cudaMemcpy(d_in, d_orig, N*sizeof(uint32_t), cudaMemcpyDeviceToDevice);
      for(int pass=0; pass<num_passes; ++pass){
        // K1 (timed)
        cudaEventRecord(start);
        initial_kernel<uint32_t, H, LGH, Q><<<num_blocks, B>>>(d_in, d_hist, (uint32_t)pass, (uint32_t)N);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms=0.0f; cudaEventElapsedTime(&ms, start, stop); acc_ms += ms;

        // scan (untimed)
        cudaMemcpy(h_hist.data(), d_hist, hist_elems*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        host_column_scan(h_hist.data(), h_sc.data(), num_blocks, H);
        cudaMemcpy(d_scan, h_sc.data(), hist_elems*sizeof(uint32_t), cudaMemcpyHostToDevice);

        // K2 (timed)
        const size_t shmem = (size_t)(B*Q + 2*H + B) * sizeof(uint32_t);
        cudaEventRecord(start);
        final_kernel<uint32_t, H, LGH, B, Q><<<num_blocks, B, shmem>>>(
          d_in, d_out, d_hist, d_scan, (uint32_t)pass, (uint32_t)N);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop); acc_ms += ms;

        std::swap(d_in, d_out);
      }
      totals_ms.push_back(acc_ms);
    }
    print_line(N, ci_from_ms(totals_ms));
    cudaFree(d_orig);
  }

  cudaEventDestroy(start); cudaEventDestroy(stop);
  cudaFree(d_scan); cudaFree(d_hist); cudaFree(d_out); cudaFree(d_in);
  return 0;
}
