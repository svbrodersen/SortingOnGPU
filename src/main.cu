#include "constants.cuh"
#include "definitions.cuh"
#include "host_skel.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  initHwd();

  const uint32_t array_length = 1 << 20;
  const uint32_t mem_size = array_length * sizeof(uint32_t);

  uint32_t *inp_vals = (uint32_t *)malloc(mem_size);
  for (int i = 0; i < array_length; i++) {
    inp_vals[i] = rand();
  }

  uint32_t *d_inp_vals;
  cudaMalloc((void **)&d_inp_vals, mem_size);
  cudaMemcpy(d_inp_vals, inp_vals, mem_size, cudaMemcpyHostToDevice);

  const uint32_t num_blocks = (array_length + (B * Q) - 1) / (B * Q);
  const uint32_t hist_size = num_blocks * H * sizeof(uint32_t);
  uint32_t *d_hist;
  cudaMalloc((void **)&d_hist, hist_size);

  initial_kernel<<<num_blocks, B>>>(d_inp_vals, d_hist, 0);
  CUDASSERT(cudaPeekAtLastError());
  cudaDeviceSynchronize();
  printf("Successfully initial_kernel.\n");

  uint32_t *d_hist_tr;
  cudaMalloc((void **)&d_hist_tr, hist_size);

  int n = num_blocks;
  int m = H;

  int dimy = (n + TILE_DIM - 1) / TILE_DIM;
  int dimx = (m + TILE_DIM - 1) / TILE_DIM;
  dim3 block(TILE_DIM, TILE_DIM, 1);
  dim3 grid(dimx, dimy, 1);

  transpose<<<grid, block>>>(d_hist, d_hist_tr, n, m);
  CUDASSERT(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  printf("Successfully transpose.\n");

  // Allocate temporary arrays for scanInc
  uint32_t *d_tmp_vals;
  cudaMalloc((void **)&d_tmp_vals, num_blocks * sizeof(uint32_t));

  // Run scan on each row of transposed histogram
  for (int i = 0; i < H; i++) {
    scanInc<Add<uint32_t>>(B, num_blocks, d_hist_tr + i * num_blocks,
                           d_hist_tr + i * num_blocks, d_tmp_vals);
  }
  CUDASSERT(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  printf("Successfully scanInc.\n");

  // Transpose result back to original layout
  transpose<<<grid, block>>>(d_hist_tr, d_hist, m, n);
  CUDASSERT(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  printf("Successfully scanInc.\n");

  free(inp_vals);
  cudaFree(d_inp_vals);
  cudaFree(d_hist);
  cudaFree(d_hist_tr);
  cudaFree(d_tmp_vals);

  return 0;
}
