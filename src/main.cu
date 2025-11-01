#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include "sort.cuh"

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

int main() {
  initHwd();
  cudaSetDevice(1);
  const uint32_t Q = 22;
  const uint32_t B = 256;
  const uint32_t lgH = 8;
  const uint32_t TILE_SIZE = 32;

  // This works
  const uint32_t N = 10000000;
  const uint32_t mem_size = N * sizeof(uint32_t);

  uint32_t *inp_vals = (uint32_t *)malloc(mem_size);
  for (int i = 0; i < N; i++) {
    inp_vals[i] = rand();
  }

  // printArray(inp_vals, 10, "inp_vals");

  uint32_t *out_vals = (uint32_t *)malloc(mem_size);

  if (radixSort<Q, B, lgH, TILE_SIZE>(inp_vals, out_vals, N) == 0) {
    printArray(out_vals, 100, "out_vals");

    // Simple verification that items are sorted
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
