#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include "sort.cuh"
#include "constants.cuh"  // timeval_subtract is assumed to be here
#include <iostream>

#define GPU_RUNS 400

#define cudaCheckError() {                                              \
    cudaError_t e=cudaGetLastError();                                   \
    if(e!=cudaSuccess) {                                                \
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
      exit(0);                                                          \
    }                                                                   \
}

using T = uint32_t;

void randomInitNat(T* data, const uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        data[i] = static_cast<unsigned long>(rand());
    }
}

void initializeDeviceOnce() {
  static bool initialized = false;
  if (!initialized) {
    cudaSetDevice(1);
    initHwd();

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 1);
    printf("Device name: %s\n", props.name);
    printf("Number of hardware threads: %d\n", props.multiProcessorCount * props.maxThreadsPerMultiProcessor);
    printf("Max block size: %d\n", props.maxThreadsPerBlock);
    printf("Shared memory size: %zu\n", props.sharedMemPerBlock);
    printf("====\n");

    initialized = true;
  }
}

void runBenchmarkForSize(uint32_t N) {
  const uint32_t Q = 22;
  const uint32_t B = 256;
  const uint32_t lgH = 8;
  const uint32_t TILE_SIZE = 32;
  const uint32_t mem_size = N * sizeof(T);

  T *inp_vals = (T *)malloc(mem_size);
  T *sorted_array = (T *)malloc(mem_size);
  randomInitNat(inp_vals, N);

  T *d_inp_vals;
  cudaMalloc((void **)&d_inp_vals, mem_size);
  cudaMemcpy(d_inp_vals, inp_vals, mem_size, cudaMemcpyHostToDevice);

  double elapsed = 0.0;
  bool sorted = true;

  for (int i = 0; i < GPU_RUNS; i++) {
    struct timeval t_start, t_end, t_diff;

    RadixSorter<T, Q, B, lgH, TILE_SIZE> sorter(N, d_inp_vals);
    gettimeofday(&t_start, NULL);
    sorter.sort();
    cudaDeviceSynchronize();
    cudaCheckError();
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed += t_diff.tv_sec * 1e6 + t_diff.tv_usec;

    cudaMemcpy(sorted_array, sorter.d_out_vals_.get(), mem_size, cudaMemcpyDeviceToHost);

    for (uint32_t j = 0; j < N - 1; j++) {
      if (sorted_array[j] > sorted_array[j + 1]) {
        printf("Error: failed sorting\n");
        sorted = false;
        break;
      }
    }
  }

  cudaFree(d_inp_vals);
  free(inp_vals);
  free(sorted_array);

  if (sorted) {
    elapsed /= GPU_RUNS;
    printf("  N = %10u -> Avg time: %8.2f microsecs\n", N, elapsed);
  }
}

int main() {
  initializeDeviceOnce();

  uint32_t sizes[] = {1000, 10000, 100000, 1000000, 2000000, 5000000, 10000000, 100000000};
  printf("Running full benchmark:\n");

  for (uint32_t N : sizes) {
    runBenchmarkForSize(N);
  }

  return 0;
}
