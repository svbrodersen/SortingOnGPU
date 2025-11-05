#include "pbb_kernels.cuh"
#include <cstdint>
#include <sys/types.h>

#pragma once

template <typename T, bool IsSigned, bool IsFloat> struct ValueTraitImpl;

template <typename T> struct ValueTraitImpl<T, false, false> {
public:
  using UnsignedType = typename std::make_unsigned<T>::type;

  static __device__ UnsignedType encode(T v) { return static_cast<UnsignedType>(v); }

  static __device__ T decode(UnsignedType v) { return static_cast<T>(v); }

  static bool needsAllBits() { return false; }
};

template <typename T> struct ValueTraitImpl<T, true, false> {
public:
  using UnsignedType = typename std::make_unsigned<T>::type;

  static __device__ UnsignedType encode(T v) {
    UnsignedType u = static_cast<UnsignedType>(v);
    return u ^ (UnsignedType(1) << (sizeof(T) * 8 - 1));
  }

  static __device__ T decode(UnsignedType value) {
    UnsignedType u = value ^ (UnsignedType(1) << (sizeof(T) * 8 - 1));
    return static_cast<T>(u);
  }

  static bool needsAllBits() {
    return true; // Must process all bits for signed bits
  }
};

template <typename T> struct ValueTraitImpl<T, true, true> {
public:
  using UnsignedType = typename std::conditional<
      sizeof(T) == 4, uint32_t,
      typename std::conditional<sizeof(T) == 8, uint64_t, void>::type>::type;

  static __device__ UnsignedType encode(T v) {
    // Reinterpret the floating point bits as unsigned integer
    UnsignedType u;

    // Use union for type punning (safe in CUDA)
    union {
      T f;
      UnsignedType u;
    } converter;

    converter.f = v;

    if (converter.u & (UnsignedType(1) << (sizeof(T) * 8 - 1))) {
      // Negative number: flip all bits
      return ~converter.u;
    } else {
      // Positive number: flip the sign bit
      return converter.u ^ (UnsignedType(1) << (sizeof(T) * 8 - 1));
    }
  }

  static __device__ T decode(UnsignedType value) {
    UnsignedType u;

    // Reverse the transformation
    if (value & (UnsignedType(1) << (sizeof(T) * 8 - 1))) {
      // Was positive: flip the sign bit back
      u = value ^ (UnsignedType(1) << (sizeof(T) * 8 - 1));
    } else {
      // Was negative: flip all bits back
      u = ~value;
    }

    union {
      T f;
      UnsignedType u;
    } converter;

    converter.u = u;
    return converter.f;
  }

  static bool needsAllBits() {
    return true; // Must process all bits for signed bits
  }
};

template <typename T>
struct ValueTraits : ValueTraitImpl<T, std::is_signed<T>::value,
                                    std::is_floating_point<T>::value> {};

template <typename T>
__global__ void encode_kernel(const T *inp_vals, typename ValueTraits<T>::UnsignedType *out_vals, uint32_t N) {
  const int glb_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (glb_idx < N) {
    out_vals[glb_idx] = ValueTraits<T>::encode(inp_vals[glb_idx]);
  }
}

template <typename T>
__global__ void decode_kernel(const typename ValueTraits<T>::UnsignedType *inp_vals, T* out_vals, uint32_t N) {
  const int glb_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (glb_idx < N) {
    out_vals[glb_idx] = ValueTraits<T>::decode(inp_vals[glb_idx]);
  }
}


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

template <typename UnsignedType, uint32_t B, uint32_t Q>
__device__ void partition2_by_bit(UnsignedType *s_inp, UnsignedType reg_mem[Q],
                                  uint32_t current_bit,
                                  uint32_t *s_scan_storage, bool is_last) {
  uint32_t thid = threadIdx.x;
  uint32_t S = 0;

#pragma unroll
  for (int q = 0; q < Q; q++) {
    UnsignedType elm = reg_mem[q];
    uint32_t bit_is_0 = 1u - ((elm >> current_bit) & 1u);
    S += bit_is_0;
  }
  s_scan_storage[thid] = S;

  __syncthreads();

  // inclusive scan
  uint32_t res = scanIncBlock<Add<uint32_t>>(s_scan_storage, thid);
  __syncthreads();
  s_scan_storage[thid] = res;
  __syncthreads();
  uint32_t total_zeros = s_scan_storage[B - 1];

  // Scatter into shared array.
  uint32_t count_zero = 0;
  uint32_t inclusive_zero = s_scan_storage[thid];
  uint32_t zeros_before = inclusive_zero - S;
  uint32_t ones_before = thid * Q - zeros_before;
#pragma unroll
  for (int q = 0; q < Q; q++) {
    UnsignedType elm = reg_mem[q];
    uint32_t bit_is_1 = ((elm >> current_bit) & 1u);
    if (bit_is_1) {
      uint32_t count_one = q - count_zero;
      s_inp[total_zeros + ones_before + count_one] = elm;
    } else {
      s_inp[zeros_before + count_zero] = elm;
      count_zero++;
    }
  }
  __syncthreads();

  if (is_last) {
    for (int q = 0; q < Q; q++) {
      uint32_t loc_pos = q * B + thid;
      reg_mem[q] = s_inp[loc_pos];
    }
  } else {
    for (int q = 0; q < Q; q++) {
      uint32_t loc_pos = thid * Q + q;
      reg_mem[q] = s_inp[loc_pos];
    }
  }
}

template <typename T> __host__ __device__ constexpr T type_max() {
  return static_cast<T>(~T(0));
}

template <typename UnsignedType, uint32_t H, uint32_t lgH, uint32_t B,
          uint32_t Q>
__global__ void final_kernel(UnsignedType *inp_vals, UnsignedType *out_vals,
                             uint32_t *orig_hist, uint32_t *scanned_hist,
                             uint32_t current_shift, uint32_t N_global) {
  const uint32_t N = B * Q;
  const uint32_t thid = threadIdx.x;
  const uint32_t block_id = blockIdx.x;

  // Shared memory for all 3 steps
  extern __shared__ uint32_t s_mem[];
  UnsignedType *s_inp = (UnsignedType *)s_mem;      // size N
  uint32_t *s_local_hist = (uint32_t *)(s_inp + N); // size H
  uint32_t *s_local_scanned = s_local_hist + H;     // size H
  uint32_t *s_scan_storage = s_local_scanned + H;   // size B (for helpers)

  // Step 1: Copy Q*B elements to shared memory  ---
  const uint32_t block_start = block_id * N;
  UnsignedType reg_mem[Q];

#pragma unroll
  for (int q = 0; q < Q; q++) {
    uint32_t local_idx = q * B + thid;
    uint32_t global_idx = block_start + local_idx;
    if (global_idx < N_global) {
      s_inp[local_idx] = inp_vals[global_idx];
    } else {
      s_inp[local_idx] = type_max<UnsignedType>();
    }
  }

  __syncthreads();

  for (int q = 0; q < Q; q++) {
    uint32_t local_idx = Q * thid + q;
    reg_mem[q] = s_inp[local_idx];
  }

  // Step 2: Loop of size lgH for two-way partitioning 
  // (This performs an in-block radix sort)
  for (uint32_t k = 0; k < lgH; k++) {
    uint32_t current_bit = (current_shift * lgH + k);
    bool is_last = k == (lgH - 1);
    // Partition s_data -> s_temp based on bit k
    partition2_by_bit<UnsignedType, B, Q>(s_inp, reg_mem, current_bit,
                                          s_scan_storage, is_last);
    __syncthreads();
  }

  // Step 3: After the loop 

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
    uint32_t local_idx = q * B + thid;
    UnsignedType val = reg_mem[q]; // Get the locally sorted value
    uint32_t bin = (val >> (current_shift * lgH)) & mask;

    uint32_t final_idx = s_local_scanned[bin] - s_local_hist[bin] + local_idx;

    if (final_idx < N_global) {
      out_vals[final_idx] = val;
    }
  }
}
