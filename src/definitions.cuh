#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cstdint>

#pragma once

#define B 256
#define Q 22
#define lgH 8
#define H (1 << lgH)

__global__ void initial_kernel(uint32_t *inp_vals, volatile uint32_t *hist,
                               uint32_t current_shift) {
  const uint32_t block_start = blockIdx.x * (blockDim.x * Q);

  __shared__ uint32_t s_hist[H];

#pragma unroll
  for (uint32_t i = threadIdx.x; i < H; i += blockDim.x) {
    s_hist[i] = 0u;
  }
  __syncthreads();

  const uint32_t mask = H - 1u;
#pragma unroll
  for (int i = 0; i < Q; i++) {
    uint32_t idx = block_start + i * blockDim.x + threadIdx.x;
    uint32_t val = inp_vals[idx] >> (current_shift * lgH);
    uint32_t bin = val & mask;
    atomicAdd((unsigned int *)&s_hist, 1u);
  }

  __syncthreads();

#pragma unroll
  for (uint32_t i = threadIdx.x; i < H; i += B) {
    hist[blockIdx.x * H + i] = s_hist[i];
  }

  return;
}
