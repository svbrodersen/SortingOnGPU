#include "host_skel.cuh"
#include "kernels.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <sys/types.h>

template<uint32_t Q, uint32_t B, uint32_t lgH, uint32_t TILE_SIZE>
int radixSort(uint32_t *inp_vals, uint32_t *out_vals, uint32_t N) {
  const uint32_t H = (1 << lgH);
  const uint32_t mem_size = N * sizeof(uint32_t);

  const uint32_t num_blocks = (N + (B * Q) - 1) / (B * Q);
  const uint32_t hist_size = num_blocks * H;
  const uint32_t hist_mem_size = hist_size * sizeof(uint32_t);

  // Dimensions
  const int dimy = (num_blocks + TILE_SIZE - 1) / TILE_SIZE;
  const int dimx = (H + TILE_SIZE - 1) / TILE_SIZE;
  dim3 block(TILE_SIZE, TILE_SIZE, 1);
  dim3 grid_forward(dimx, dimy, 1);
  dim3 grid_backward(dimy, dimx, 1);

  uint32_t *d_inp_vals;
  uint32_t *d_out_vals;
  cudaMalloc((void **)&d_inp_vals, mem_size);
  cudaMemcpy(d_inp_vals, inp_vals, mem_size, cudaMemcpyHostToDevice);

  // Allocate device output
  cudaMalloc((void **)&d_out_vals, mem_size);

  uint32_t *d_hist_scan;
  cudaMalloc((void **)&d_hist_scan, hist_mem_size);

  uint32_t *d_hist_scan_tr_tr;
  cudaMalloc((void **)&d_hist_scan_tr_tr, hist_mem_size);

  uint32_t *d_hist;
  cudaMalloc((void **)&d_hist, hist_mem_size);

  uint32_t *d_tmp_vals;
  cudaMalloc((void **)&d_tmp_vals, hist_size * sizeof(uint32_t));

  // TODO: This is where you should start the test for GFLOPS GB/S. Both have their merit.

  uint32_t largest_shift;
  {
    uint32_t largest_val = 0;
    for (int i = 0; i < N; i++) {
      if (inp_vals[i] > largest_val) {
        largest_val = inp_vals[i];
      }
    }
    uint32_t largest_bit_pos =
        (largest_val == 0) ? 0u : (31 - __builtin_clz(largest_val));
    largest_shift = largest_bit_pos / lgH + 1;
  }

  for (uint32_t current_shift = 0u; current_shift < largest_shift;
       current_shift++) {
    initial_kernel<H, lgH, Q>
        <<<num_blocks, B>>>(d_inp_vals, d_hist, current_shift, N);
    CUDASSERT(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    // d_hist should be size num_blocks X H, want H x num_blocks
    transpose<TILE_SIZE><<<grid_forward, block>>>(d_hist, d_hist_scan, num_blocks, H);
    CUDASSERT(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    // Allocate temporary arrays for scanInc
    scanInc<Add<uint32_t>>(B, hist_size, d_hist_scan, d_hist_scan, d_tmp_vals);
    CUDASSERT(cudaPeekAtLastError());
    cudaDeviceSynchronize();


    transpose<TILE_SIZE><<<grid_backward, block>>>(d_hist_scan, d_hist_scan_tr_tr, H, num_blocks);

    const uint32_t shared_mem_size = (B * Q + H + H + B) * sizeof(uint32_t);
    final_kernel<H, lgH, B, Q><<<num_blocks, B, shared_mem_size>>>(
        d_inp_vals, d_out_vals, d_hist, d_hist_scan_tr_tr, current_shift, N);
    CUDASSERT(cudaPeekAtLastError());
    cudaDeviceSynchronize();

     std::swap(d_inp_vals, d_out_vals);
  }

  // Copy result back to host (assuming this is the only pass for demonstration)
  cudaMemcpy(out_vals, d_inp_vals, mem_size, cudaMemcpyDeviceToHost);

  cudaFree(d_inp_vals);
  cudaFree(d_out_vals);
  cudaFree(d_hist);
  cudaFree(d_tmp_vals);

  return 0;
}
