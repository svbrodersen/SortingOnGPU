#include "pbb_kernels.cuh"
#include <cstdint>
#include <sys/types.h>

#pragma once

template <typename UnsignedType, uint32_t H, uint32_t lgH, uint32_t Q>
__global__ void initial_kernel(UnsignedType *inp_vals, uint32_t *hist,
                               uint32_t current_shift, uint32_t N) {
  const uint32_t B = blockDim.x;
  const uint32_t block_start = blockIdx.x * (B * Q);

  __shared__ uint32_t s_hist[H];

#pragma unroll
  for (uint32_t i = threadIdx.x; i < H; i += B) {
    s_hist[i] = 0u;
  }
  __syncthreads();

  const uint64_t mask = H - 1u;
#pragma unroll
  for (int i = 0; i < Q; i++) {
    uint32_t idx = block_start + i * B + threadIdx.x;
    if (idx >= N)
      continue;
    UnsignedType val = inp_vals[idx] >> (current_shift * lgH);
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

template <int TILE_SIZE>
__global__ void transpose(uint32_t *hist, uint32_t *hist_tr, int N, int M) {
  __shared__ uint32_t tile[TILE_SIZE][TILE_SIZE + 1];

  int x = blockIdx.x * TILE_SIZE + threadIdx.x;
  int y = blockIdx.y * TILE_SIZE + threadIdx.y;

  if (x < M && y < N)
    tile[threadIdx.y][threadIdx.x] = hist[y * M + x];

  __syncthreads();

  x = blockIdx.y * TILE_SIZE + threadIdx.x;
  y = blockIdx.x * TILE_SIZE + threadIdx.y;

  if (x < N && y < M)
    hist_tr[y * N + x] = tile[threadIdx.x][threadIdx.y];
}

/* Kernel 2: per-block lgH-bit pass + global scatter */
template<uint32_t H, uint32_t lgH>
__global__ void final_kernel(const uint32_t* __restrict__ inp_vals,
                             uint32_t* __restrict__ out_vals,
                             const uint32_t* __restrict__ hist,
                             const uint32_t* __restrict__ scanned_hist,
                             uint32_t current_shift,
                             uint32_t Q) {
  const uint32_t B = blockDim.x;
  const uint32_t tileStart = blockIdx.x * (B * Q);
  const uint32_t bitBase = current_shift * lgH;

  extern __shared__ uint32_t smem[];
  uint32_t* s_data   = smem;                   // [B*Q]
  uint32_t* s_cnt    = s_data   + B*Q;         // [B]
  uint32_t* s_scanB  = s_cnt    + B;           // [B]    (thread-count scan)
  uint32_t* s_bins   = s_scanB  + B;           // [H]    (original per-bin counts)
  uint32_t* s_scanH  = s_bins   + H;           // [H]    (histogram scan workspace)
  uint32_t* s_binsG  = s_scanH  + H;           // [H]    (global starts from scanned_hist)

  const uint32_t tid  = threadIdx.x;
  const uint32_t base = tileStart + tid;

  uint32_t regs[32];
#pragma unroll
  for (uint32_t i = 0; i < 32; ++i)
    if (i < Q) regs[i] = inp_vals[base + i * B];
  __syncthreads();

#pragma unroll
  for (uint32_t k = 0; k < lgH; ++k) {
    const uint32_t b = bitBase + k;

    uint32_t z = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32; ++i)
      if (i < Q) z += (((regs[i] >> b) & 1u) == 0u);
    s_cnt[tid] = z;
    __syncthreads();

    // scan over B threads
    s_scanB[tid] = s_cnt[tid];
    __syncthreads();
    for (uint32_t off = 1; off < B; off <<= 1) {
      uint32_t add = (tid >= off) ? s_scanB[tid - off] : 0u;
      __syncthreads();
      s_scanB[tid] += add;
      __syncthreads();
    }
    const uint32_t totalZ = s_scanB[B - 1];
    const uint32_t zEx    = (tid == 0) ? 0u : s_scanB[tid - 1];
    const uint32_t oEx    = tid * Q - zEx;

    uint32_t lz = 0, lo = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32; ++i) {
      if (i < Q) {
        const uint32_t v = regs[i];
        if (((v >> b) & 1u) == 0u) s_data[zEx + lz++] = v;
        else                       s_data[totalZ + oEx + lo++] = v;
      }
    }
    __syncthreads();

    const uint32_t slice = tid * Q;
#pragma unroll
    for (uint32_t i = 0; i < 32; ++i)
      if (i < Q) regs[i] = s_data[slice + i];
    __syncthreads();
  }

  // load per-bin counts for this block and global (scanned) starts
  for (uint32_t j = tid; j < H; j += B) {
    s_bins[j]  = hist[blockIdx.x * H + j];
    s_binsG[j] = scanned_hist[blockIdx.x * H + j];
  }
  __syncthreads();

  // exclusive scan over H bins -> local starts (use s_scanH as workspace)
  if (tid == 0) {
    uint32_t acc = 0;
    for (uint32_t j = 0; j < H; ++j) {
      uint32_t c = s_bins[j];
      s_bins[j] = acc;
      acc += c;
    }
  }
  __syncthreads();

  // final scatter
  for (uint32_t idx = tid; idx < B * Q; idx += B) {
    const uint32_t v   = s_data[idx];
    const uint32_t bin = (v >> bitBase) & (H - 1u);
    const uint32_t g0  = s_binsG[bin];
    const uint32_t l0  = s_bins[bin];
    out_vals[g0 + (idx - l0)] = v;
  }
}

/* Kernel 2 (debug): also dump local tile and per-bin starts */
template<uint32_t H, uint32_t lgH>
__global__ void final_kernel_dbg(const uint32_t* __restrict__ inp_vals,
                                 uint32_t* __restrict__ out_vals,
                                 const uint32_t* __restrict__ hist,
                                 const uint32_t* __restrict__ scanned_hist,
                                 uint32_t current_shift,
                                 uint32_t Q,
                                 // debug dumps:
                                 uint32_t* __restrict__ tile_after,   // [B*Q]
                                 uint32_t* __restrict__ bins_local,   // [H]
                                 uint32_t* __restrict__ bins_global)  // [H]
{
  const uint32_t B = blockDim.x;
  const uint32_t tileStart = blockIdx.x * (B * Q);
  const uint32_t bitBase = current_shift * lgH;

  extern __shared__ uint32_t smem[];
  uint32_t* s_data   = smem;                   // [B*Q]
  uint32_t* s_cnt    = s_data   + B*Q;         // [B]
  uint32_t* s_scanB  = s_cnt    + B;           // [B]
  uint32_t* s_bins   = s_scanB  + B;           // [H]
  uint32_t* s_scanH  = s_bins   + H;           // [H]
  uint32_t* s_binsG  = s_scanH  + H;           // [H]

  const uint32_t tid  = threadIdx.x;
  const uint32_t base = tileStart + tid;

  uint32_t regs[32];
#pragma unroll
  for (uint32_t i = 0; i < 32; ++i)
    if (i < Q) regs[i] = inp_vals[base + i * B];
  __syncthreads();

#pragma unroll
  for (uint32_t k = 0; k < lgH; ++k) {
    const uint32_t b = bitBase + k;

    uint32_t z = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32; ++i)
      if (i < Q) z += (((regs[i] >> b) & 1u) == 0u);
    s_cnt[tid] = z;
    __syncthreads();

    s_scanB[tid] = s_cnt[tid];
    __syncthreads();
    for (uint32_t off = 1; off < B; off <<= 1) {
      uint32_t add = (tid >= off) ? s_scanB[tid - off] : 0u;
      __syncthreads();
      s_scanB[tid] += add;
      __syncthreads();
    }
    const uint32_t totalZ = s_scanB[B - 1];
    const uint32_t zEx    = (tid == 0) ? 0u : s_scanB[tid - 1];
    const uint32_t oEx    = tid * Q - zEx;

    uint32_t lz = 0, lo = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32; ++i) {
      if (i < Q) {
        const uint32_t v = regs[i];
        if (((v >> b) & 1u) == 0u) s_data[zEx + lz++] = v;
        else                       s_data[totalZ + oEx + lo++] = v;
      }
    }
    __syncthreads();

    const uint32_t slice = tid * Q;
#pragma unroll
    for (uint32_t i = 0; i < 32; ++i)
      if (i < Q) regs[i] = s_data[slice + i];
    __syncthreads();
  }

  // dump the locally sorted tile
  for (uint32_t idx = tid; idx < B*Q; idx += B) tile_after[idx] = s_data[idx];

  // load bins and global starts
  for (uint32_t j = tid; j < H; j += B) {
    s_bins[j]  = hist[blockIdx.x * H + j];
    s_binsG[j] = scanned_hist[blockIdx.x * H + j];
  }
  __syncthreads();

  // exclusive scan over H bins -> local starts
  if (tid == 0) {
    uint32_t acc = 0;
    for (uint32_t j = 0; j < H; ++j) {
      uint32_t c = s_bins[j];
      s_bins[j] = acc;
      acc += c;
    }
  }
  __syncthreads();

  // dump bins (local starts) and global starts
  for (uint32_t j = tid; j < H; j += B) {
    bins_local[j]  = s_bins[j];
    bins_global[j] = s_binsG[j];
  }
  __syncthreads();

  // normal scatter
  for (uint32_t idx = tid; idx < B * Q; idx += B) {
    const uint32_t v   = s_data[idx];
    const uint32_t bin = (v >> bitBase) & (H - 1u);
    const uint32_t g0  = s_binsG[bin];
    const uint32_t l0  = s_bins[bin];
    out_vals[g0 + (idx - l0)] = v;
  }
}
