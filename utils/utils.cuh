#include <map>
#include <random>

#pragma once

#define GPU_RUNS 400
#define SEED 42
#define DEBUG_INFO false
#define DEVICE_NUMBER 0

#define DEBUG_INFO false

#define lgWARP 5
#define WARP (1 << lgWARP)

template <typename T> T randomValue() {
  static thread_local std::mt19937_64 rng{SEED};

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

template <typename T>
bool checkElementPreservation(T *original, T *sorted, uint32_t N) {
  // The map key must be large enough to hold T, or simply use T.
  std::map<T, int> original_counts;
  std::map<T, int> sorted_counts;

  // Count frequencies in original array
  for (uint32_t i = 0; i < N; i++) {
    original_counts[original[i]]++;
  }

  // Count frequencies in sorted array
  for (uint32_t i = 0; i < N; i++) {
    sorted_counts[sorted[i]]++;
  }

  // Compare the maps
  if (original_counts.size() != sorted_counts.size()) {
    printf("ERROR: Frequency map size mismatch. Original unique: %lu, Sorted "
           "unique: %lu\n",
           original_counts.size(), sorted_counts.size());
    return false;
  }

  // Iterate and compare counts
  for (auto const &[key, original_count] : original_counts) {
    if (sorted_counts.find(key) == sorted_counts.end()) {
      // Use appropriate format specifier for the key
      if constexpr (std::is_same_v<T, uint32_t>) {
        printf("ERROR: Element %u from original array is missing in sorted "
               "array.\n",
               key);
      } else {
        printf("ERROR: Element %d from original array is missing in sorted "
               "array.\n",
               key);
      }
      return false;
    }
    int sorted_count = sorted_counts[key];
    if (sorted_count != original_count) {
      if constexpr (std::is_same_v<T, uint32_t>) {
        printf(
            "ERROR: Element count mismatch for %u. Original: %d, Sorted: %d\n",
            key, original_count, sorted_count);
      } else {
        printf(
            "ERROR: Element count mismatch for %d. Original: %d, Sorted: %d\n",
            key, original_count, sorted_count);
      }
      return false;
    }
  }
  return true;
}

uint32_t MAX_HWDTH;
uint32_t MAX_BLOCK;
uint32_t MAX_SHMEM;

cudaDeviceProp prop;

void initHwd() {
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  cudaGetDeviceProperties(&prop, 0);
  MAX_HWDTH = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
  MAX_BLOCK = prop.maxThreadsPerBlock;
  MAX_SHMEM = prop.sharedMemPerBlock;

  if (DEBUG_INFO) {
    printf("Device name: %s\n", prop.name);
    printf("Number of hardware threads: %d\n", MAX_HWDTH);
    printf("Max block size: %d\n", MAX_BLOCK);
    printf("Shared memory size: %d\n", MAX_SHMEM);
    puts("====");
  }
}

void initializeDeviceOnce() {
  static bool initialized = false;
  if (!initialized) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (DEVICE_NUMBER < deviceCount)
    {
      cudaSetDevice(DEVICE_NUMBER);
    }
    else
    {
      printf("error: invalid device choosen\n");
    }
    initHwd();
    initialized = true;
  }
}


int timeval_subtract(struct timeval *result, struct timeval *t2,
                     struct timeval *t1) {
  uint32_t resolution = 1000000;
  int64_t diff = (t2->tv_usec + resolution * t2->tv_sec) -
                 (t1->tv_usec + resolution * t1->tv_sec);
  result->tv_sec = diff / resolution;
  result->tv_usec = diff % resolution;
  return (diff < 0);
}
