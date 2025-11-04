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
#include <type_traits>
#include "../utils/utils.cuh"

#define cudaCheckError() {                                              \
    cudaError_t e=cudaGetLastError();                                   \
    if(e!=cudaSuccess) {                                                \
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
      exit(0);                                                          \
    }                                                                   \
}


const uint32_t Q = 22;
const uint32_t B = 256;
const uint32_t lgH = 8;
const uint32_t TILE_SIZE = 32;

template<typename T>
void randomInitNat(T* data, const uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        data[i] = randomValue<T>  ();
    }
}

void initializeDeviceOnce() {
  static bool initialized = false;
  if (!initialized) {
    cudaSetDevice(1);
    initHwd();

    initialized = true;
  }
}

template<typename T>
void runBenchmarkForSize(uint32_t N, const char* typeName) {
const uint32_t mem_size = N * sizeof(T);
  T *inp_vals = (T *)malloc(mem_size);
  T *sorted_array = (T *)malloc(mem_size);
  randomInitNat(inp_vals, N);

  T *d_inp_vals;
  cudaMalloc((void **)&d_inp_vals, mem_size);
  cudaMemcpy(d_inp_vals, inp_vals, mem_size, cudaMemcpyHostToDevice);

  T *d_tmp_vals;
  cudaMalloc((void **)&d_tmp_vals, mem_size);

  double elapsed = 0.0;
  for (int i = 0; i < GPU_RUNS; i++) {
    struct timeval t_start, t_end, t_diff;
    cudaMemcpy(d_tmp_vals, d_inp_vals, mem_size, cudaMemcpyDeviceToDevice);

    RadixSorter<T, Q, B, lgH, TILE_SIZE> sorter(N, d_tmp_vals);
    gettimeofday(&t_start, NULL);
    sorter.sort();
    cudaDeviceSynchronize();
    cudaCheckError();
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed += t_diff.tv_sec * 1e6 + t_diff.tv_usec;

    if (i == GPU_RUNS - 1) {
      if constexpr (std::is_unsigned_v<T>) {
        cudaMemcpy(sorted_array, sorter.d_out_vals_.get(), mem_size, cudaMemcpyDeviceToHost);
      } else {
        cudaMemcpy(sorted_array, d_tmp_vals, mem_size, cudaMemcpyDeviceToHost);
      }
    }
  }
  bool valid = validateZ<T>(sorted_array, N);

  cudaFree(d_inp_vals);
  cudaFree(d_tmp_vals);
  free(inp_vals);
  free(sorted_array);

  elapsed /= GPU_RUNS;
  printf("Our sorting (%s) for N=%lu runs in: %.2f us, VALID: %d\n", typeName, N, elapsed, valid);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
      printf("Usage: %s <size-of-array>\n", argv[0]);
      exit(1);
  }
  const uint64_t N = atoi(argv[1]);
  initializeDeviceOnce();
  runBenchmarkForSize<uint32_t>(N, "uint32_t");
  runBenchmarkForSize<int32_t>(N, "int32_t");
  runBenchmarkForSize<float>(N, "float");
  return 0;
}
