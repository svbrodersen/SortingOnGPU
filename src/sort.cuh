#include "host_skel.cuh"
#include "kernels.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <stdlib.h>
#include <sys/types.h>
#include <system_error>
#include <type_traits>
#include <vector>

template <typename T> class DeviceBuffer {
private:
  T *ptr_;
  size_t size_;

public:
  DeviceBuffer(size_t count) : size_(count) {
    cudaMalloc((void **)&ptr_, count * sizeof(T));
  }

  // No copy, take control
  DeviceBuffer(T *ptr, size_t count) : size_(count) {
    cudaMalloc((void **)&ptr_, count * sizeof(T));
    cudaMemcpy(ptr_, ptr, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
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
    other.size_ = 0;
  }

  void swap(DeviceBuffer &other) noexcept {
    std::swap(ptr_, other.ptr_);
    std::swap(size_, other.size_);
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

  DeviceBuffer<UnsignedType> d_inp_vals_;
  DeviceBuffer<UnsignedType> d_out_vals_;

private:
  static constexpr uint32_t H = (1 << lgH);
  uint32_t N_;
  uint32_t num_blocks_;
  uint32_t hist_size_;
  uint32_t num_passes_;
  size_t shared_mem_size_;
  bool initialized_;

  // Grid dimensions
  dim3 block_;
  dim3 grid_forward_;
  dim3 grid_backward_;

  // Device buffers
  DeviceBuffer<uint32_t> d_hist_;
  DeviceBuffer<uint32_t> d_hist_scan_;
  DeviceBuffer<uint32_t> d_hist_scan_tr_tr_;
  DeviceBuffer<uint32_t> d_tmp_vals_;

  // If non-unsigned, then keep pointer to original
  T *d_original_vals_;

public:
  RadixSorter(uint32_t N)
      : N_(N), num_blocks_((N + (B * Q) - 1) / (B * Q)),
        hist_size_(num_blocks_ * (1 << lgH)),
        shared_mem_size_((B * Q) * sizeof(UnsignedType) +
                         (2 * H + B) * sizeof(uint32_t)),
        d_inp_vals_(N), d_out_vals_(N), d_hist_(hist_size_),
        d_hist_scan_(hist_size_), d_hist_scan_tr_tr_(hist_size_),
        d_tmp_vals_(hist_size_), initialized_(false) {
    // Setup grid dimensions
    const int dimy = (num_blocks_ + TILE_SIZE - 1) / TILE_SIZE;
    const int dimx = (H + TILE_SIZE - 1) / TILE_SIZE;
    block_ = dim3(TILE_SIZE, TILE_SIZE, 1);
    grid_forward_ = dim3(dimx, dimy, 1);
    grid_backward_ = dim3(dimy, dimx, 1);
  }

  RadixSorter(uint32_t N, T *d_inp_vals)
      : RadixSorter(N, d_inp_vals,
                    std::integral_constant<bool, IsUnsignedInt>{}) {}

private:
  // If we are unsigned then we can preallocate the d_inp_vals with copy
  RadixSorter(uint32_t N, T *d_inp_vals, std::true_type)
      : N_(N), num_blocks_((N + (B * Q) - 1) / (B * Q)),
        hist_size_(num_blocks_ * (1 << lgH)),
        shared_mem_size_((B * Q) * sizeof(UnsignedType) +
                         (2 * H + B) * sizeof(uint32_t)),
        d_hist_(hist_size_), d_out_vals_(N), d_hist_scan_(hist_size_),
        d_hist_scan_tr_tr_(hist_size_), d_tmp_vals_(hist_size_),
        initialized_(true),
        d_inp_vals_(reinterpret_cast<UnsignedType *>(d_inp_vals), N) {
    // Setup grid dimensions
    const int dimy = (num_blocks_ + TILE_SIZE - 1) / TILE_SIZE;
    const int dimx = (H + TILE_SIZE - 1) / TILE_SIZE;
    block_ = dim3(TILE_SIZE, TILE_SIZE, 1);
    grid_forward_ = dim3(dimx, dimy, 1);
    grid_backward_ = dim3(dimy, dimx, 1);
  }
  // If we are not unsigned, then we save the pointer to the original data
  RadixSorter(uint32_t N, T *d_inp_vals, std::false_type)
      : N_(N), num_blocks_((N + (B * Q) - 1) / (B * Q)),
        hist_size_(num_blocks_ * (1 << lgH)),
        shared_mem_size_((B * Q) * sizeof(UnsignedType) +
                         (2 * H + B) * sizeof(uint32_t)),
        d_hist_(hist_size_), d_out_vals_(N), d_hist_scan_(hist_size_),
        d_hist_scan_tr_tr_(hist_size_), d_tmp_vals_(hist_size_),
        initialized_(true), d_inp_vals_(N_) {
    // Setup grid dimensions
    const int dimy = (num_blocks_ + TILE_SIZE - 1) / TILE_SIZE;
    const int dimx = (H + TILE_SIZE - 1) / TILE_SIZE;
    block_ = dim3(TILE_SIZE, TILE_SIZE, 1);
    grid_forward_ = dim3(dimx, dimy, 1);
    grid_backward_ = dim3(dimy, dimx, 1);

    d_original_vals_ = d_inp_vals;
  }
public:

  int sort() {
    if (!initialized_) {
      throw std::runtime_error(
          "Not initialized with device buffer, when running sort()");
    }
    encode();
    sortMain();
    decode();
    return 0;
  }

  // This is called from host
  int sort(const T *inp_vals, T *out_vals) {
    if constexpr (!IsUnsignedInt) {
      cudaMalloc(d_original_vals_, N_);
      cudaMemcpy(d_original_vals_, inp_vals, N_, cudaMemcpyHostToDevice);
    }
    encode();
    sortMain();
    decode();
    copyResult(out_vals);
    return 0;
  }

private:
  void __inline__ encode() {
    if constexpr (!IsUnsignedInt) {
      const int numBlocks = (N_ + B - 1) / B;
      UnsignedType *d_out_vals = d_inp_vals_.get();
      encode_kernel<T><<<numBlocks, B>>>(d_original_vals_, d_out_vals, N_);
    }
  }

  void __inline__ decode() {
    if constexpr (!IsUnsignedInt) {
      const int numBlocks = (N_ + B - 1) / B;
      UnsignedType *d_inp_vals = d_out_vals_.get();
      decode_kernel<T><<<numBlocks, B>>>(d_inp_vals, d_original_vals_, N_);
    }
  }

  void copyResult(T *out_vals) {
    cudaMemcpy(out_vals, d_original_vals_, N_, cudaMemcpyDeviceToHost);
  }

  int sortMain() {
    num_passes_ = calculateNumPasses();
    UnsignedType *d_current_input = d_inp_vals_.get();
    UnsignedType *d_current_output = d_out_vals_.get();
    for (uint32_t pass = 0; pass < num_passes_; pass++) {
      executeOnePass(d_current_input, d_current_output, pass);
      std::swap(d_current_input, d_current_output);
    }

    if (num_passes_ % 2 == 0) {
      d_inp_vals_.swap(d_out_vals_);
    }

    return 0;
  }

  uint32_t __inline__ calculateNumPasses() {
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
  }
};

template <typename T, uint32_t Q, uint32_t B, uint32_t lgH, uint32_t TILE_SIZE>
int radixSort(T *inp_vals, T *out_vals, uint32_t N) {
  RadixSorter<T, Q, B, lgH, TILE_SIZE> sorter(N);
  return sorter.sort(inp_vals, out_vals);
}
