#include <random>

#define GPU_RUNS 400

#pragma once
template <typename T> T randomValue() {
  static thread_local std::mt19937_64 rng{42};

  if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
    std::uniform_int_distribution<uint64_t> dist(std::numeric_limits<T>::min(),
                                                 std::numeric_limits<T>::max());
    return static_cast<T>(dist(rng));
  } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
    std::uniform_int_distribution<int64_t> dist(std::numeric_limits<T>::min(),
                                                std::numeric_limits<T>::max());
    return static_cast<T>(dist(rng));
  } else if constexpr (std::is_floating_point_v<T>) {
    // Uniform over a wide range, avoiding infinities
    std::uniform_real_distribution<long double> dist(
        -1.0e308L, 1.0e308L); // covers most of double range safely
    return static_cast<T>(dist(rng));
  } else {
    static_assert(!sizeof(T *), "Unsupported type for randomValue()");
  }
}

template<class Z>
bool validateZ(Z* A, uint32_t sizeAB) {
    for(uint32_t i = 1; i < sizeAB; i++)
      if (A[i-1] > A[i]){
        printf("INVALID RESULT for i:%d, (A[i-1]=%d > A[i]=%d)\n", i, A[i-1], A[i]);
        return false;
      }
    return true;
}
