#include "host_skel.cuh"
#include <cstdint>

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

template <uint32_t B, uint32_t H>
__device__ void block_exclusive_scan(uint32_t *data, uint32_t *temp_storage) {
  uint32_t thid = threadIdx.x;

  // 1. Each thread sequentially reduces its chunk [cite: 41]
  uint32_t sum = 0;
  for (uint32_t i = thid; i < H; i += B) {
    sum += data[i];
  }
  temp_storage[thid] = sum;
  __syncthreads();

  for (uint32_t d = 1; d < B; d *= 2) {
    uint32_t val = (thid >= d) ? temp_storage[thid - d] : 0;
    __syncthreads();
    temp_storage[thid] += val;
    __syncthreads();
  }

  // Shift to convert inclusive scan to exclusive scan
  uint32_t prefix = (thid > 0) ? temp_storage[thid - 1] : 0;
  __syncthreads();
  temp_storage[thid] = prefix;
  __syncthreads();

  // 3. Each thread sequentially scans its chunk using the calculated prefix
  for (uint32_t i = thid; i < H; i += B) {
    uint32_t val = data[i];
    data[i] = temp_storage[thid];
    temp_storage[thid] += val;
  }
  __syncthreads();
}

template <uint32_t B, uint32_t Q>
__device__ void partition2_by_bit(uint32_t *s_data, uint32_t *s_temp,
                                  uint32_t current_offset,
                                  uint32_t *s_scan_storage) {
  uint32_t unset_count = 0;
  uint32_t thid = threadIdx.x;
  for (int q = 0; q < Q; q++) {
    uint32_t elm = s_data[thid + q * blockDim.x];
    if ((elm & (current_offset + q)) == 0) {
      unset_count++;
    }
  }

  s_scan_storage[thid] = unset_count;
  __syncthreads();

  // Perform an inclusive scan on counts
  for (uint32_t d = 1; d < blockDim.x; d *= 2) {
    uint32_t val = (thid >= d) ? s_scan_storage[thid - d] : 0;
    __syncthreads();
    s_scan_storage[thid] += val;
    __syncthreads();
  }

  uint32_t total_unset = s_scan_storage[blockDim.x - 1];
  uint32_t unset_offset = (thid > 0) ? s_scan_storage[thid - 1] : 0;
  __syncthreads();

  uint32_t set_offset = total_unset + (thid * Q - unset_count);

#pragma unroll
  for (int q = 0; q < Q; q++) {
    uint32_t val = s_data[thid + q * blockDim.x];
    if ((val & (current_offset + q)) == 0) {
      s_temp[unset_offset++] = val;
    } else {
      s_temp[set_offset++] = val;
    }
  }
}

template <uint32_t H, uint32_t lgH, uint32_t B, uint32_t Q>
__global__ void final_kernel(uint32_t *inp_vals, uint32_t *out_vals,
                             uint32_t *orig_hist, uint32_t *scanned_hist,
                             uint32_t current_shift, uint32_t N_global) {

  const uint32_t N = B * Q;
  const uint32_t thid = threadIdx.x;
  const uint32_t block_id = blockIdx.x;
  const uint32_t num_blocks = gridDim.x;

  // Shared memory for all 3 steps
  extern __shared__ uint32_t s_mem[];
  uint32_t *s_data = s_mem;                        // size N
  uint32_t *s_temp = s_mem + N;                    // size N
  uint32_t *s_local_hist = s_temp + N;             // size H
  uint32_t *s_global_offsets = s_local_hist + H;   // size H
  uint32_t *s_scan_storage = s_global_offsets + H; // size B (for helpers)

  // --- Step 1: Copy Q*B elements to shared memory  ---
  const uint32_t block_start = block_id * N;
#pragma unroll
  for (int i = 0; i < Q; i++) {
    uint32_t local_idx = i * B + thid;
    uint32_t global_idx = block_start + local_idx;
    if (global_idx < N_global) {
      s_data[local_idx] = inp_vals[global_idx];
    } else {
      s_data[local_idx] = UINT32_MAX;
    }
  }
  __syncthreads();

  // --- Step 2: Loop of size lgH for two-way partitioning  ---
  // (This performs an in-block radix sort)
  for (uint32_t k = 0; k < lgH; k++) {

    // Partition s_data -> s_temp based on bit k
    partition2_by_bit<B, Q>(s_data, s_temp, current_shift, s_scan_storage);
    __syncthreads();

// Copy s_temp -> s_data for the next iteration
#pragma unroll
    for (int i = 0; i < Q; i++) {
      s_data[thid + i * B] = s_temp[thid + i * B];
    }
    __syncthreads();
  }
  // At this point, s_data is locally sorted by the current lgH bits.

  // --- Step 3: After the loop  ---

  // 3.1: Copy original and scanned histograms to shared memory
  for (uint32_t i = thid; i < H; i += B) {
    // Load this block's original histogram
    s_local_hist[i] = orig_hist[block_id * H + i];

    // Load this block's global offset for bin 'i'
    s_global_offsets[i] = scanned_hist[i * num_blocks + block_id];
  }
  __syncthreads();

  // 3.2: Scan in place the original histogram
  // This gives the *local* offset for each bin.
  scanIncBlock<Add<uint32_t>>(s_local_hist, threadIdx.x);
  // s_local_hist[bin] now holds the starting index in s_data[] for 'bin'.
  __syncthreads();

  // 3.3: Write Q elements to their final global positions
  const uint32_t mask = H - 1u;
#pragma unroll
  for (int i = 0; i < Q; i++) {
    uint32_t local_idx = thid + i * B;
    uint32_t orig_global_idx = block_start + local_idx;

    if (orig_global_idx >= N_global) {
      continue;
    }
    uint32_t val = s_data[local_idx]; // Get the locally sorted value
    uint32_t bin = (val >> (current_shift * lgH)) & mask;

    uint32_t final_idx = scanned_hist[blockIdx.x * H + bin] - s_local_hist[bin];

    out_vals[final_idx] = val;
  }
}
