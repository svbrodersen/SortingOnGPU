#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include "sort.cuh"
#include <iostream>

using T = double;

void printArray(T *inp_vals, uint32_t N, const char *name) {
  std::cout << name << "[:" << N << "] = [";
  for (uint32_t i = 0; i < N; i++) {
    std::cout << inp_vals[i];
    if (i < N - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]\n";
}

void printDeviceArray(T *inp_vals, int mem_size, uint32_t N,
                      const char *name) {
  T *d_hist_host = (T *)malloc(mem_size);
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
  const uint32_t mem_size = N * sizeof(T);

  T *inp_vals = (T *)malloc(mem_size);
  for (int i = 0; i < N; i++) {
    inp_vals[i] = (T)N-20.5-i;
  }

  // printArray(inp_vals, 10, "inp_vals");

  T *out_vals = (T *)malloc(mem_size);

  if (radixSort<T, Q, B, lgH, TILE_SIZE>(inp_vals, out_vals, N) == 0) {
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
