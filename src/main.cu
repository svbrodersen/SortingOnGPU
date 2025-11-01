#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include "sort.cuh"
#include <iostream>

#define GPU_RUNS 10

using T = uint32_t;



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

void randomInitNat(uint32_t* data, const uint32_t size, const uint32_t H) {
    for (int i = 0; i < size; ++i) {
        unsigned long int r = rand();
        data[i] = r % H;
    }
}

int main (int argc, char * argv[]) {
  if (argc != 2) {
      printf("Usage: %s <size-of-array>\n", argv[0]);
      exit(1);
  }
  const uint32_t N = (uint32_t) atoi(argv[1]);

  initHwd();
  cudaSetDevice(1);
  const uint32_t Q = 22;
  const uint32_t B = 256;
  const uint32_t lgH = 8;
  const uint32_t H = (1 << lgH);
  const uint32_t TILE_SIZE = 32;

  // This works
  const uint32_t mem_size = N * sizeof(T);

  T *inp_vals = (T *)malloc(mem_size);
  randomInitNat(inp_vals, N, H);

  // printArray(inp_vals, 10, "inp_vals");

  T *out_vals = (T *)malloc(mem_size);
  RadixSorter<T, Q, B, lgH, TILE_SIZE> sorter(N);

  {
    // Dry run
    sorter.sort(inp_vals, out_vals);
  }

  double elapsed;
  struct timeval t_start, t_end, t_diff;
  gettimeofday(&t_start, NULL); 
  for (int i = 0; i < GPU_RUNS; i++) {
    sorter.sort(inp_vals, out_vals);
  }
  gettimeofday(&t_end, NULL);
  timeval_subtract(&t_diff, &t_end, &t_start);
  elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS); 
  printf("Radix sort time for size %d: %.2f microsecs\n", N, elapsed);

  free(inp_vals);
  free(out_vals);

  return 0;
}
