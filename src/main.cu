#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include "sort.cuh"
#include <iostream>

#define GPU_RUNS 400

using T = uint16_t;

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

void randomInitNat(T* data, const uint32_t size, const uint32_t H) {
    for (int i = 0; i < size; ++i) {
        unsigned long int r = rand();
        data[i] = r;
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
  T *sorted_array = (T*) malloc(mem_size); 
  randomInitNat(inp_vals, N, H);

  double elapsed;
  bool sorted = true;
  for (int i =0; i < GPU_RUNS; i++ )  {
    struct timeval t_start, t_end, t_diff;
    RadixSorter<T, Q, B, lgH, TILE_SIZE> sorter(N, inp_vals);
    T *d_inp_vals; 
    cudaMalloc((void **)&d_inp_vals, mem_size);
    cudaMemcpy(d_inp_vals, inp_vals, mem_size, cudaMemcpyHostToDevice);
    gettimeofday(&t_start, NULL); 
    sorter.sort();
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed += (t_diff.tv_sec*1e6+t_diff.tv_usec);

    cudaMemcpy(sorted_array, sorter.d_out_vals_.get(), mem_size, cudaMemcpyDeviceToHost);;
    
    for (int j = 0; j < N-1; j++) {
      if (sorted_array[j] > sorted_array[j+1]) {
        printf("Error: failed sorting");
        sorted = false;
        break;
      }
    }
    cudaFree(d_inp_vals);
  }
  if (sorted) {
  elapsed = elapsed / ((double)GPU_RUNS); 
  printf("Radix sort time for size %d: %.2f microsecs\n", N, elapsed);
  }

  free(inp_vals);
  free(sorted_array);

  return 0;
}
