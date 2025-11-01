#include "host_skel.cuh"
#include "kernels.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <sys/types.h>
#include <type_traits>
#include <vector>

template <typename T, bool IsSigned> struct ValueTraitImpl;

template <typename T> struct ValueTraitImpl<T, false> {
public:
  using UnsignedType = typename std::make_unsigned<T>::type;

  static __host__ __device__ UnsignedType encode(T v) {
    return static_cast<UnsignedType>(v);
  }

  static __host__ __device__ T decode(UnsignedType v) {
    return static_cast<T>(v);
  }

  static __host__ __device__ bool needsAllBits() { return false; }
};

template <typename T> struct ValueTraitImpl<T, true> {
public:
  using UnsignedType = typename std::make_unsigned<T>::type;

  static __host__ __device__ UnsignedType encode(T v) {
    UnsignedType u = static_cast<UnsignedType>(v);
    return u ^ (UnsignedType(1) << (sizeof(T) * 8 - 1));
  }

  static __host__ __device__ T decode(UnsignedType value) {
    UnsignedType u = value ^ (UnsignedType(1) << (sizeof(T) * 8 - 1));
    return static_cast<T>(u);
  }

  static __host__ __device__ T needsAllBits() {
    return true; // Must process all bits for signed bits
  }
};

template <typename T>
struct ValueTraits : ValueTraitImpl<T, std::is_signed<T>::value> {};

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

  // Track which buffer contains final result
  bool result_in_input_buffer_;

public:
  RadixSorter(uint32_t N)
      : N_(N), num_blocks_((N + (B * Q) - 1) / (B * Q)),
        hist_size_(num_blocks_ * (1 << lgH)),
        shared_mem_size_((B*Q) * sizeof(UnsignedType) +
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
    encoded_data_.resize(N);
  }

  int sort(const T *inp_vals, T *out_vals) {
    // Step 1: Encode input
    encodeInput(inp_vals);

    // Step 2: Calculate number of passes
    num_passes_ = calculateNumPasses(inp_vals);

    // Step 3: Copy to device
    d_inp_vals_.copyToDevice(encoded_data_.data());

    // Step 4: Execute sort passes
    executeSortPasses();

    // Step 5: Copy back and decode
    copyResultAndDecode(out_vals);

    return 0;
  }

private:
  void encodeInput(const T *inp_vals) {
    for (uint32_t i = 0; i < N_; i++) {
      encoded_data_[i] = Traits::encode(inp_vals[i]);
    }
  }

  uint32_t calculateNumPasses(const T *inp_vals) {
    // Check if we need all bits or can optimize
    if (Traits::needsAllBits()) {
      return (sizeof(T) * 8 + lgH - 1) / lgH;
    } else {
      // For unsigned, find the largest value
      UnsignedType max_val = 0;
      for (uint32_t i = 0; i < N_; i++) {
        UnsignedType encoded = Traits::encode(inp_vals[i]);
        if (encoded > max_val) {
          max_val = encoded;
        }
      }

      if (max_val == 0)
        return 1;

      // Find highest bit position
      uint32_t highest_bit = sizeof(UnsignedType) * 8 - 1;
      while (highest_bit > 0 && !((max_val >> highest_bit) & 1)) {
        highest_bit--;
      }

      return (highest_bit / lgH) + 1;
    }
  }

  void executeSortPasses() {
    UnsignedType *d_current_input = d_inp_vals_.get();
    UnsignedType *d_current_output = d_out_vals_.get();

    for (uint32_t pass = 0; pass < num_passes_; pass++) {
      executeOnePass(d_current_input, d_current_output, pass);
      // Swap pointers only (not the DeviceBuffer objects)
      std::swap(d_current_input, d_current_output);
    }
  }

  void executeOnePass(UnsignedType *d_input, UnsignedType *d_output,
                      uint32_t pass) {
    // Step 1: Build histogram
    initial_kernel<H, lgH, Q>
        <<<num_blocks_, B>>>(d_input, d_hist_.get(), pass, N_);
    CUDASSERT(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    // Step 2: Transpose histogram (num_blocks × H -> H × num_blocks)
    transpose<TILE_SIZE><<<grid_forward_, block_>>>(
        d_hist_.get(), d_hist_scan_.get(), num_blocks_, H);
    CUDASSERT(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    // Step 3: Scan histogram
    scanInc<Add<uint32_t>>(B, hist_size_, d_hist_scan_.get(),
                           d_hist_scan_.get(), d_tmp_vals_.get());
    CUDASSERT(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    // Step 4: Transpose back (H × num_blocks -> num_blocks × H)
    transpose<TILE_SIZE><<<grid_backward_, block_>>>(
        d_hist_scan_.get(), d_hist_scan_tr_tr_.get(), H, num_blocks_);
    CUDASSERT(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    // Step 5: Reorder elements
    final_kernel<H, lgH, B, Q>
        <<<num_blocks_, B, shared_mem_size_>>>(d_input, d_output, d_hist_.get(),
                                               d_hist_scan_tr_tr_.get(), pass,
                                               N_);
    CUDASSERT(cudaPeekAtLastError());
    cudaDeviceSynchronize();
  }

  void copyResultAndDecode(T *out_vals) {
    d_inp_vals_.copyToHost(encoded_data_.data());
    for (uint32_t i = 0; i < N_; i++) {
      out_vals[i] = Traits::decode(encoded_data_[i]);
    }
  }
};

template <typename T, uint32_t Q, uint32_t B, uint32_t lgH, uint32_t TILE_SIZE>
int radixSort(T *inp_vals, T *out_vals, uint32_t N) {
  RadixSorter<T, Q, B, lgH, TILE_SIZE> sorter(N);
  return sorter.sort(inp_vals, out_vals);
}

