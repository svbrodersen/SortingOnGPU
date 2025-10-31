#include "constants.cuh"
#include "host_skel.cuh"
#include "kernels.cuh"
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

void printArray(uint32_t *inp_vals, uint32_t N, const char *name) {
  printf("%s[:%d] = [", name, N);
  for (int i = 0; i < N; i++) {
    if (i == N - 1) {
      printf("%u]\n", inp_vals[i]);
    } else {
      printf("%u, ", inp_vals[i]);
    }
  }
}

void printDeviceArray(uint32_t *inp_vals, int mem_size, uint32_t N,
                      const char *name) {
  uint32_t *d_hist_host = (uint32_t *)malloc(mem_size);
  cudaMemcpy(d_hist_host, inp_vals, mem_size, cudaMemcpyDeviceToHost);
  printArray(d_hist_host, N, name);
}

const uint32_t Q = 22;
const uint32_t B = 256;
const uint32_t lgH = 8;
const uint32_t H = (1 << lgH);
const uint32_t T = 32;

int radixSort(uint32_t *inp_vals, uint32_t *out_vals, uint32_t N) {

  const uint32_t mem_size = N * sizeof(uint32_t);

  const uint32_t num_blocks = (N + (B * Q) - 1) / (B * Q);
  const uint32_t hist_size = num_blocks * H;
  const uint32_t hist_mem_size = hist_size * sizeof(uint32_t);

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

  int dimy = (num_blocks + T - 1) / T;
  int dimx = (H + T - 1) / T;
  dim3 block(T, T, 1);
  dim3 grid_forward(dimx, dimy, 1);
  dim3 grid_backward(dimx, dimy, 1);

  // printf("Largest shift: %d\n", largest_shift);
  for (uint32_t current_shift = 0u; current_shift < largest_shift;
       current_shift++) {
    initial_kernel<H, lgH, Q>
        <<<num_blocks, B>>>(d_inp_vals, d_hist, current_shift, N);
    CUDASSERT(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    // printf("Successfully initial_kernel.\n");


    transpose<T><<<grid_forward, block>>>(d_hist, d_hist_scan, num_blocks, H);
    CUDASSERT(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    // Allocate temporary arrays for scanInc
    scanInc<Add<uint32_t>>(B, hist_size, d_hist_scan, d_hist_scan, d_tmp_vals);
    CUDASSERT(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    // printf("Successfully scanInc.\n");

    transpose<T><<<grid_backward, block>>>(d_hist_scan, d_hist_scan_tr_tr, H, num_blocks);
    // printf("Successfully transpose.\n");
    printDeviceArray(d_hist_scan_tr_tr, hist_mem_size, H, "d_hist_scan_tr_tr");


    const uint32_t shared_mem_size = (B * Q + H + H + B) * sizeof(uint32_t);
    final_kernel<H, lgH, B, Q><<<num_blocks, B, shared_mem_size>>>(
        d_inp_vals, d_out_vals, d_hist, d_hist_scan_tr_tr, current_shift, N);
    CUDASSERT(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    printf("Successfully final_kernel.\n");
    printDeviceArray(d_out_vals, mem_size, 100, "output_pass");


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

int main() {
  initHwd();

  // This works
  const uint32_t N = Q*B;
  // This fails
  // const uint32_t N = Q*B+1;
  const uint32_t mem_size = N * sizeof(uint32_t);

  uint32_t *inp_vals = (uint32_t *)malloc(mem_size);
  for (int i = 0; i < N; i++) {
    inp_vals[i] = N-i;
  }

  // printArray(inp_vals, 10, "inp_vals");

  uint32_t *out_vals = (uint32_t *)malloc(mem_size);

  if (radixSort(inp_vals, out_vals, N) == 0) {
    printArray(out_vals, 100, "out_vals");

    // Simple verification for the first pass (lowest 8 bits)
    bool sorted = true;
    for (uint32_t i = 0; i < N-1; i++) {
      if ((out_vals[i]) > (out_vals[i + 1])) {
        sorted = false;
        printf("Sort failed at index %u: %u > %u \n", i,
               out_vals[i], out_vals[i]);
        break;
      }
    }
    if (sorted) {
      printf("Array is correctly sorted by the lowest 8 bits.\n");
    }
  }

  free(inp_vals);
  free(out_vals);

  return 0;
}
