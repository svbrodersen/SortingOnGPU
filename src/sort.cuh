#include "host_skel.cuh"
#include "kernels.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <sys/types.h>
#include <type_traits>
#include <vector>

template <typename T, bool IsSigned, bool IsFloat> struct ValueTraitImpl;

template <typename T> struct ValueTraitImpl<T, false, false> {
public:
  using UnsignedType = typename std::make_unsigned<T>::type;

  static UnsignedType encode(T v) { return static_cast<UnsignedType>(v); }

  static T decode(UnsignedType v) { return static_cast<T>(v); }

  static bool needsAllBits() { return false; }
};

template <typename T> struct ValueTraitImpl<T, true, false> {
public:
  using UnsignedType = typename std::make_unsigned<T>::type;

  static UnsignedType encode(T v) {
    UnsignedType u = static_cast<UnsignedType>(v);
    return u ^ (UnsignedType(1) << (sizeof(T) * 8 - 1));
  }

  static T decode(UnsignedType value) {
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

  static UnsignedType encode(T v) {
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

  static T decode(UnsignedType value) {
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

template <typename T> class DeviceBuffer {
private:
  T *ptr_;
  size_t size_;

public:
  DeviceBuffer(size_t count) : size_(count) {
    cudaMalloc((void **)&ptr_, count * sizeof(T));
  }

  ~DeviceBuffer() {
    if (ptr_)
      cudaFree(ptr_);
  }

  // No copy
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  // Move semantics
  DeviceBuffer(DeviceBuffer &&other) noexcept
      : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
  }

  T *get() { return ptr_; }
  const T *get() const { return ptr_; }
  size_t size() const { return size_; }

  void copyToDevice(const T *host_data) {
    cudaMemcpy(ptr_, host_data, size_ * sizeof(T), cudaMemcpyHostToDevice);
  }

  void copyToHost(T *host_data) const {
    cudaMemcpy(host_data, ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
  }
};

template <typename T, uint32_t Q, uint32_t B, uint32_t lgH, uint32_t TILE_SIZE>
class RadixSorter {
public:
  using Traits = ValueTraits<T>;
  using UnsignedType = typename Traits::UnsignedType;

  // Check if we're dealing with unsigned integers
  static constexpr bool IsUnsignedInt = std::is_unsigned<T>::value;

private:
  static constexpr uint32_t H = (1 << lgH);
  uint32_t N_;
  uint32_t num_blocks_;
  uint32_t hist_size_;
  uint32_t num_passes_;
  size_t shared_mem_size_;

  // Grid dimensions
  dim3 block_;
  dim3 grid_forward_;
  dim3 grid_backward_;

  // Device buffers
  DeviceBuffer<UnsignedType> d_inp_vals_;
  DeviceBuffer<UnsignedType> d_out_vals_;
  DeviceBuffer<uint32_t> d_hist_;
  DeviceBuffer<uint32_t> d_hist_scan_;
  DeviceBuffer<uint32_t> d_hist_scan_tr_tr_;
  DeviceBuffer<uint32_t> d_tmp_vals_;

  // Host buffers for encoding/decoding
  std::vector<UnsignedType> encoded_data_;

public:
  RadixSorter(uint32_t N)
      : N_(N), num_blocks_((N + (B * Q) - 1) / (B * Q)),
        hist_size_(num_blocks_ * (1 << lgH)),
        shared_mem_size_((B * Q) * sizeof(UnsignedType) +
                         (2 * H + B) * sizeof(uint32_t)),
        d_inp_vals_(N), d_out_vals_(N), d_hist_(hist_size_),
        d_hist_scan_(hist_size_), d_hist_scan_tr_tr_(hist_size_),
        d_tmp_vals_(hist_size_) {
    // Setup grid dimensions
    const int dimy = (num_blocks_ + TILE_SIZE - 1) / TILE_SIZE;
    const int dimx = (H + TILE_SIZE - 1) / TILE_SIZE;
    block_ = dim3(TILE_SIZE, TILE_SIZE, 1);
    grid_forward_ = dim3(dimx, dimy, 1);
    grid_backward_ = dim3(dimy, dimx, 1);

    // Pre-allocate encoding buffer
    if constexpr (!IsUnsignedInt) {
      encoded_data_.resize(N);
    }
  }

  int sort(const T *inp_vals, T *out_vals) {
    if constexpr (IsUnsignedInt) {
      // Direct path for unsigned integers
      d_inp_vals_.copyToDevice(
          reinterpret_cast<const UnsignedType *>(inp_vals));
      num_passes_ = calculateNumPasses(inp_vals);

      UnsignedType *d_current_input = d_inp_vals_.get();
      UnsignedType *d_current_output = d_out_vals_.get();

      for (uint32_t pass = 0; pass < num_passes_; pass++) {
        executeOnePass(d_current_input, d_current_output, pass);
        std::swap(d_current_input, d_current_output);
      }

      d_inp_vals_.copyToHost(reinterpret_cast<UnsignedType *>(out_vals));
    } else {
      // Original path with encoding/decoding
      encodeInput(inp_vals);
      d_inp_vals_.copyToDevice(encoded_data_);
      num_passes_ = calculateNumPasses(inp_vals);

      UnsignedType *d_current_input = d_inp_vals_.get();
      UnsignedType *d_current_output = d_out_vals_.get();

      for (uint32_t pass = 0; pass < num_passes_; pass++) {
        executeOnePass(d_current_input, d_current_output, pass);
        std::swap(d_current_input, d_current_output);
      }

      copyResultAndDecode(out_vals);
    }
    return 0;
  }

private:
  void encodeInput(const T *inp_vals) {
    if constexpr (!IsUnsignedInt) {
    for (uint32_t i = 0; i < N_; i++) {
      encoded_data_[i] = Traits::encode(inp_vals[i]);
    }
    }
  }

  uint32_t calculateNumPasses(const T *inp_vals) {
    // Check if we need all bits or can optimize
    return (sizeof(T) * 8 + lgH - 1) / lgH;
  }

  void __inline__ executeOnePass(UnsignedType *d_input, UnsignedType *d_output,
                                 uint32_t pass) {
    // Step 1: Build histogram
    initial_kernel<UnsignedType, H, lgH, Q>
        <<<num_blocks_, B>>>(d_input, d_hist_.get(), pass, N_);

    // Step 2: Transpose histogram (num_blocks × H -> H × num_blocks)
    transpose<TILE_SIZE><<<grid_forward_, block_>>>(
        d_hist_.get(), d_hist_scan_.get(), num_blocks_, H);

    // Step 3: Scan histogram
    scanInc<Add<uint32_t>>(B, hist_size_, d_hist_scan_.get(),
                           d_hist_scan_.get(), d_tmp_vals_.get());

    // Step 4: Transpose back (H × num_blocks -> num_blocks × H)
    transpose<TILE_SIZE><<<grid_backward_, block_>>>(
        d_hist_scan_.get(), d_hist_scan_tr_tr_.get(), H, num_blocks_);

    // Step 5: Reorder elements
    final_kernel<UnsignedType, H, lgH, B, Q>
        <<<num_blocks_, B, shared_mem_size_>>>(d_input, d_output, d_hist_.get(),
                                               d_hist_scan_tr_tr_.get(), pass,
                                               N_);
    CUDASSERT(cudaPeekAtLastError());
    cudaDeviceSynchronize();
  }

  void copyResultAndDecode(T *out_vals) {
    if constexpr (!IsUnsignedInt) {
      d_inp_vals_.copyToHost(encoded_data_.data());
      for (uint32_t i = 0; i < N_; i++) {
        out_vals[i] = Traits::decode(encoded_data_[i]);
      }
    }
  }
};

template <typename T, uint32_t Q, uint32_t B, uint32_t lgH, uint32_t TILE_SIZE>
int radixSort(T *inp_vals, T *out_vals, uint32_t N) {
  RadixSorter<T, Q, B, lgH, TILE_SIZE> sorter(N);
  return sorter.sort(inp_vals, out_vals);
}
