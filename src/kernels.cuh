#include "pbb_kernels.cuh"
#include <cstdint>
#include <sys/types.h>

#pragma once

template <uint32_t H, uint32_t lgH, uint32_t Q>
__global__ void initial_kernel(uint32_t *inp_vals, uint32_t *hist,
                               uint32_t current_shift, uint32_t N) {
  const uint32_t B = blockDim.x;
  const uint32_t block_start = blockIdx.x * (B * Q);

  __shared__ uint32_t s_hist[H];

#pragma unroll
  for (uint32_t i = threadIdx.x; i < H; i += B) {
    s_hist[i] = 0u;
  }
  __syncthreads();

  const uint32_t mask = H - 1u;
#pragma unroll
  for (int i = 0; i < Q; i++) {
    uint32_t idx = block_start + i * B + threadIdx.x;
    if (idx >= N)
      continue;
    uint32_t val = inp_vals[idx] >> (current_shift * lgH);
    uint32_t bin = val & mask;
    atomicAdd((unsigned int *)&s_hist[bin], 1u);
  }

  __syncthreads();

#pragma unroll
  for (uint32_t i = threadIdx.x; i < H; i += B) {
    hist[blockIdx.x * H + i] = s_hist[i];
  }

  return;
}

template <int T>
__global__ void transpose(uint32_t *hist, uint32_t *hist_tr, int N, int M) {
  __shared__ uint32_t tile[T][T + 1];

  int x = blockIdx.x * T + threadIdx.x;
  int y = blockIdx.y * T + threadIdx.y;

  if (x < M && y < N)
    tile[threadIdx.y][threadIdx.x] = hist[y * M + x];

  __syncthreads();

  x = blockIdx.y * T + threadIdx.x;
  y = blockIdx.x * T + threadIdx.y;

  if (x < N && y < M)
    hist_tr[y * N + x] = tile[threadIdx.x][threadIdx.y];
}

template <uint32_t B, uint32_t Q>
__device__ void partition2_by_bit(uint32_t *s_inp, uint32_t reg_mem[Q],
                                  uint32_t current_bit,
                                  uint32_t *s_scan_storage, bool is_last) {
  uint32_t thid = threadIdx.x;
  uint32_t S = 0;

#pragma unroll
  for (int q = 0; q < Q; q++) {
    uint32_t elm = reg_mem[q];
    uint32_t bit_is_0 = 1u - ((elm >> current_bit) & 1u);
    S += bit_is_0;
  }
  s_scan_storage[thid] = S;

  __syncthreads();

  // inclusive scatter
  uint32_t res = scanIncBlock<Add<uint32_t>>(s_scan_storage, thid);
  __syncthreads();
  s_scan_storage[thid] = res;
  __syncthreads();
  uint32_t total_zeros = s_scan_storage[B - 1];


  // Scatter into shared array.
  uint32_t count_zero = 0;
  uint32_t count_one = 0;
  uint32_t inclusive_zero = s_scan_storage[thid];
  uint32_t exclusive_zero = inclusive_zero - S;
#pragma unroll
  for (int q = 0; q < Q; q++) {
    uint32_t elm = reg_mem[q];
    uint32_t bit_is_0 = 1u - ((elm >> current_bit) & 1u);
    if (bit_is_0 == 1) {
      s_inp[exclusive_zero + count_zero] = elm;
      count_zero++;
    } else {
      uint32_t exclusive_one = thid * Q - exclusive_zero;
      s_inp[total_zeros + exclusive_one + count_one] = elm;
      count_one++;
    }
  }
  __syncthreads();

  // Copy the new ordering back into register memory
  for (int q = 0; q < Q; q++) {
    uint32_t loc_pos = thid * Q + q;
    reg_mem[q] = s_inp[loc_pos];
  }
  __syncthreads();
}

template <uint32_t H, uint32_t lgH, uint32_t B, uint32_t Q>
__global__ void final_kernel(uint32_t *inp_vals, uint32_t *out_vals,
                             uint32_t *orig_hist, uint32_t *scanned_hist,
                             uint32_t current_shift, uint32_t N_global) {

  const uint32_t N = B * Q;
  const uint32_t thid = threadIdx.x;
  const uint32_t block_id = blockIdx.x;

  // Shared memory for all 3 steps
  extern __shared__ uint32_t s_mem[];
  uint32_t *s_inp = s_mem;                        // size N
  uint32_t *s_local_hist = s_inp + N;             // size H
  uint32_t *s_local_scanned = s_local_hist + H;   // size H
  uint32_t *s_scan_storage = s_local_scanned + H; // size B (for helpers)

  // --- Step 1: Copy Q*B elements to shared memory  ---
  const uint32_t block_start = block_id * N;
  uint32_t reg_mem[Q];

#pragma unroll
  for (int q = 0; q < Q; q++) {
    uint32_t local_idx = q * B + thid;
    uint32_t global_idx = block_start + local_idx;
    if (global_idx < N_global) {
      s_inp[local_idx] = inp_vals[global_idx];
    } else {
      s_inp[local_idx] = UINT32_MAX;
    }
  }

  __syncthreads();

  for (int q = 0; q < Q; q++) {
    uint32_t local_idx = Q * thid + q;
    reg_mem[q] = s_inp[local_idx];
  }

  // --- Step 2: Loop of size lgH for two-way partitioning  ---
  // (This performs an in-block radix sort)
  for (uint32_t k = 0; k < lgH; k++) {
    uint32_t current_bit = (current_shift * lgH + k);
    bool is_last = k == (lgH - 1);
    // Partition s_data -> s_temp based on bit k
    partition2_by_bit<B, Q>(s_inp, reg_mem, current_bit,
                            s_scan_storage, is_last);
    __syncthreads();
  }

  // if (thid == 0) {
  //   for (uint32_t i = 0; i < Q * B; i++) {
  //     uint32_t val = s_inp[i] & 0xFF;
  //     printf("s_inp[%d] = %d\n", i, val) ;
  //   }
  // }



  // --- Step 3: After the loop  ---

  // 3.1: Copy original and scanned histograms to shared memory
  for (uint32_t i = thid; i < H; i += B) {
    // Load this block's original histogram
    s_local_hist[i] = orig_hist[block_id * H + i];

    // Load this block's global offset for bin 'i'
    s_local_scanned[i] = scanned_hist[block_id * H + i];
  }
  __syncthreads();

  // 3.2: Scan in place the original histogram
  // This gives the *local* offset for each bin. scanIncBlock is inclusive scan.
  uint32_t res = scanIncBlock<Add<uint32_t>>(s_local_hist, thid);
  // s_local_hist[bin] now holds the starting index in s_data[] for 'bin'.
  __syncthreads();
  s_local_hist[thid] = res;
  __syncthreads();

  // 3.3: Write Q elements to their final global positions
  const uint32_t mask = H - 1u;
#pragma unroll
  for (int q = 0; q < Q; q++) {
    uint32_t local_idx = thid * Q + q;
    uint32_t val = s_inp[local_idx]; // Get the locally sorted value
    uint32_t bin = (val >> (current_shift * lgH)) & mask;

    uint32_t final_idx =
        scanned_hist[blockIdx.x * H + bin] - s_local_hist[bin] + local_idx;

    if (final_idx < N_global) {
      out_vals[final_idx] = val;
    }
  }
}
