#include "kernels.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdint>

static inline void ck(cudaError_t e){ if(e){fprintf(stderr,"CUDA error: %s\n", cudaGetErrorString(e)); std::abort();} }

static void cpu_hist(const std::vector<uint32_t>& a, std::vector<uint32_t>& h, uint32_t lgH){
  const uint32_t H = 1u<<lgH, mask = H-1u;
  std::fill(h.begin(), h.end(), 0u);
  for (auto v : a) h[v & mask]++;
}
static void cpu_exclusive_scan(const std::vector<uint32_t>& in, std::vector<uint32_t>& out){
  out.resize(in.size());
  uint32_t acc = 0;
  for (size_t i=0;i<in.size();++i){ out[i] = acc; acc += in[i]; }
}

int main(){
  int dev=0; cudaGetDevice(&dev);
  cudaDeviceProp p{}; cudaGetDeviceProperties(&p, dev);
  printf("Device: %s  SMs:%d  Shared/Block:%d\n", p.name, p.multiProcessorCount, (int)p.sharedMemPerBlock);

  using u32 = uint32_t;
  constexpr u32 lgH = 8, H = 1u<<lgH;
  constexpr u32 B = 64, Q = 4, N = B*Q;     // one block, one tile

  // input with only low 8 bits
  std::vector<u32> h(N);
  for (u32 i=0;i<N;++i) h[i] = (123u*i + 7u) & (H-1u);

  // CPU refs
  std::vector<u32> ref_sorted = h; std::stable_sort(ref_sorted.begin(), ref_sorted.end());
  std::vector<u32> ref_hist(H), ref_local_starts;
  cpu_hist(h, ref_hist, lgH);
  cpu_exclusive_scan(ref_hist, ref_local_starts);

  // device buffers
  u32 *d_in=nullptr, *d_out=nullptr, *d_hist=nullptr, *d_scanned=nullptr;
  u32 *d_tile=nullptr, *d_bins=nullptr, *d_binsG=nullptr;
  ck(cudaMalloc(&d_in,  N*sizeof(u32)));
  ck(cudaMalloc(&d_out, N*sizeof(u32)));
  ck(cudaMalloc(&d_hist,    H*sizeof(u32)));   // [1 x H]
  ck(cudaMalloc(&d_scanned, H*sizeof(u32)));   // [1 x H] bin-major exclusive scan
  ck(cudaMalloc(&d_tile,    N*sizeof(u32)));   // dump local tile (debug)
  ck(cudaMalloc(&d_bins,    H*sizeof(u32)));   // dump local starts (debug)
  ck(cudaMalloc(&d_binsG,   H*sizeof(u32)));   // dump global starts (debug)

  ck(cudaMemcpy(d_in, h.data(), N*sizeof(u32), cudaMemcpyHostToDevice));
  ck(cudaMemset(d_out, 0, N*sizeof(u32)));
  ck(cudaMemset(d_hist, 0, H*sizeof(u32)));

  // 1) histogram (per-block; here only one block)
  initial_kernel<H, lgH><<<1, B>>>(d_in, d_hist, /*current_shift=*/0, /*Q=*/Q);
  ck(cudaDeviceSynchronize());

  // fetch histogram and build bin-major exclusive scan (global offsets for this block)
  std::vector<u32> got_hist(H);
  ck(cudaMemcpy(got_hist.data(), d_hist, H*sizeof(u32), cudaMemcpyDeviceToHost));

  bool ok_hist = (got_hist == ref_hist);
  if (!ok_hist){
    printf("[DBG] histogram mismatch\n");
    for (u32 i=0;i<H;++i){
      if (got_hist[i]!=ref_hist[i]) { printf("  bin %3u: got %u  ref %u\n", i, got_hist[i], ref_hist[i]); break; }
    }
  } else {
    printf("[DBG] histogram OK\n");
  }

  // build scanned_hist (exclusive) for ONE block: prefix across bins
  std::vector<u32> host_scanned(H);
  {
    u32 acc = 0;
    for (u32 j=0;j<H;++j){ host_scanned[j] = acc; acc += got_hist[j]; }
  }
  ck(cudaMemcpy(d_scanned, host_scanned.data(), H*sizeof(u32), cudaMemcpyHostToDevice));

  // 2) debug kernel (local tile + local/global starts + normal scatter)
  // shared memory (debug): B*Q + B + B + H + H + H = B*Q + 2B + 3H
  size_t sh_u32_dbg = (size_t)B*Q + 2*B + 3*H;
  final_kernel_dbg<H, lgH><<<1, B, sh_u32_dbg*sizeof(u32)>>>(d_in, d_out, d_hist, d_scanned, 0, Q, d_tile, d_bins, d_binsG);
  ck(cudaDeviceSynchronize());

  // fetch dumps
  std::vector<u32> got_tile(N), got_bins(H), got_binsG(H), got_out(N);
  ck(cudaMemcpy(got_tile.data(), d_tile, N*sizeof(u32), cudaMemcpyDeviceToHost));
  ck(cudaMemcpy(got_bins.data(), d_bins, H*sizeof(u32), cudaMemcpyDeviceToHost));
  ck(cudaMemcpy(got_binsG.data(), d_binsG, H*sizeof(u32), cudaMemcpyDeviceToHost));
  ck(cudaMemcpy(got_out.data(),  d_out,  N*sizeof(u32), cudaMemcpyDeviceToHost));

  // 3) checks
  bool ok_bins = (got_bins == ref_local_starts);
  if (!ok_bins){
    printf("[DBG] local-starts mismatch\n");
    for (u32 i=0;i<H;++i){
      if (got_bins[i]!=ref_local_starts[i]) { printf("  bin %3u: got %u  ref %u\n", i, got_bins[i], ref_local_starts[i]); break; }
    }
  } else {
    printf("[DBG] local-starts OK\n");
  }

  bool ok_binsG = (got_binsG == host_scanned);
  if (!ok_binsG){
    printf("[DBG] global-starts mismatch\n");
    for (u32 i=0;i<H;++i){
      if (got_binsG[i]!=host_scanned[i]) { printf("  bin %3u: got %u  ref %u\n", i, got_binsG[i], host_scanned[i]); break; }
    }
  } else {
    printf("[DBG] global-starts OK\n");
  }

  bool ok_tile = (got_tile == ref_sorted);
  if (!ok_tile){
    printf("[DBG] local tile (after partition loop) mismatch\n");
    for (u32 i=0;i<16 && i<N;++i){
      if (got_tile[i]!=ref_sorted[i]) { printf("  idx %3u: tile %3u  ref %3u\n", i, got_tile[i], ref_sorted[i]); break; }
    }
  } else {
    printf("[DBG] local tile OK\n");
  }

  bool ok_out = (got_out == ref_sorted);
  printf("Kernel2 one-pass: %s\n", ok_out ? "PASS" : "FAIL");
  if (!ok_out) {  // keep non-zero exit on failure
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_hist); cudaFree(d_scanned);
    cudaFree(d_tile); cudaFree(d_bins); cudaFree(d_binsG);
    return 1;
  }

  // ---------------------------
  // 4) Timing (microseconds, M el/s, GB/s) for final_kernel (non-debug)
  // ---------------------------
  // final_kernel shared memory (no debug buffers): s_data + s_cnt + s_scanB + s_bins + s_binsG
  // -> B*Q + 2*B + 2*H (uint32_t)
  size_t sh_u32 = (size_t)B*Q + 2*B + 2*H;

  // warm-up
  final_kernel<H, lgH><<<1, B, sh_u32*sizeof(u32)>>>(d_in, d_out, d_hist, d_scanned, 0, Q);
  ck(cudaDeviceSynchronize());

  cudaEvent_t e0, e1;
  ck(cudaEventCreate(&e0));
  ck(cudaEventCreate(&e1));

  const int iters = 10000; // large enough to amortize launch overhead
  ck(cudaEventRecord(e0));
  for (int it=0; it<iters; ++it) {
    final_kernel<H, lgH><<<1, B, sh_u32*sizeof(u32)>>>(d_in, d_out, d_hist, d_scanned, 0, Q);
  }
  ck(cudaEventRecord(e1));
  ck(cudaEventSynchronize(e1));
  float ms=0.0f; ck(cudaEventElapsedTime(&ms, e0, e1));
  ck(cudaEventDestroy(e0)); ck(cudaEventDestroy(e1));

  const double us_per_iter = (ms*1000.0) / iters;
  const double elems_per_s = double(N) / (us_per_iter * 1e-6);                  // elements / second
  const double gbps = (double(2ULL)*N*sizeof(u32)) / (us_per_iter*1e-6) / 1e9;   // ~read+write only

  printf("[TIME] final_kernel: %.3f us/iter,  %.2f M el/s,  %.2f GB/s\n",
         us_per_iter, elems_per_s/1e6, gbps);

  // (GFLOPS is not meaningful for integer radix sort; report µs and bandwidth/throughput.)

  cudaFree(d_in); cudaFree(d_out); cudaFree(d_hist); cudaFree(d_scanned);
  cudaFree(d_tile); cudaFree(d_bins); cudaFree(d_binsG);
  return 0;
}
