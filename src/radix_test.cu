#include <cstdint>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>   // For memcpy
#include <map>        // For frequency check
#include "sort.cuh"
#include "constants.cuh"
#include <cuda_runtime.h>

template <typename T>
void printArray(T *inp_vals, uint32_t N, const char *name) {
    printf("%s[:%u] = [", name, N);
    for (uint32_t i = 0; i < N; i++) {
        if (i == N - 1) {
            // Use appropriate format specifier: %u for unsigned, %d for signed
            if constexpr (std::is_same_v<T, uint32_t>) {
                printf("%u]\n", inp_vals[i]);
            } else {
                printf("%d]\n", inp_vals[i]);
            }
        } else {
            if constexpr (std::is_same_v<T, uint32_t>) {
                printf("%u, ", inp_vals[i]);
            } else {
                printf("%d, ", inp_vals[i]);
            }
        }
    }
}

template <typename T>
bool checkElementPreservation(T* original, T* sorted, uint32_t N) {
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
        printf("ERROR: Frequency map size mismatch. Original unique: %lu, Sorted unique: %lu\n",
                original_counts.size(), sorted_counts.size());
        return false;
    }

    // Iterate and compare counts
    for (auto const& [key, original_count] : original_counts) {
        if (sorted_counts.find(key) == sorted_counts.end()) {
            // Use appropriate format specifier for the key
            if constexpr (std::is_same_v<T, uint32_t>) {
                printf("ERROR: Element %u from original array is missing in sorted array.\n", key);
            } else {
                printf("ERROR: Element %d from original array is missing in sorted array.\n", key);
            }
            return false;
        }
        int sorted_count = sorted_counts[key];
        if (sorted_count != original_count) {
            if constexpr (std::is_same_v<T, uint32_t>) {
                printf("ERROR: Element count mismatch for %u. Original: %d, Sorted: %d\n",
                        key, original_count, sorted_count);
            } else {
                printf("ERROR: Element count mismatch for %d. Original: %d, Sorted: %d\n",
                        key, original_count, sorted_count);
            }
            return false;
        }
    }
    return true;
}

/**
 * @brief Runs a single test case for a given array size N.
 * @param N The number of elements to sort.
 * @return true if the test passed, false otherwise.
 */
template <typename T>
bool runTest(uint32_t N) {
    // Determine the type name for clearer output
    const char* type_name = std::is_same_v<T, uint32_t> ? "uint32_t" : "int32_t";
    
    // 1. Define constants (same for both types, assuming 32-bit key)
    const uint32_t Q = 22;
    const uint32_t B = 256;
    const uint32_t lgH = 8;
    const uint32_t TILE_SIZE = 32;
    const uint32_t mem_size = N * sizeof(T);

    bool success = true;

    // 2. Allocate host memory for type T
    T *inp_vals = (T *)malloc(mem_size);
    T *inp_vals_copy = (T *)malloc(mem_size);
    T *out_vals = (T *)malloc(mem_size);

    if (!inp_vals || !inp_vals_copy || !out_vals) {
        printf("ERROR: Host memory allocation failed!\n");
        if(inp_vals) free(inp_vals);
        if(inp_vals_copy) free(inp_vals_copy);
        if(out_vals) free(out_vals);
        return false;
    }

    // 3. Fill with random data
    for (uint32_t i = 0; i < N; i++) {
        // rand() returns an int, cast it to T
        inp_vals[i] = (T)rand(); 
    }
    
    // Add some edge cases based on the type
    if (N > 10) {
        inp_vals[0] = 0;
        inp_vals[2] = 1;
        inp_vals[4] = 100;
        inp_vals[5] = 100;
        
        if constexpr (std::is_same_v<T, uint32_t>) {
            inp_vals[1] = UINT32_MAX;
            inp_vals[3] = UINT32_MAX;
        } else { // int32_t
            inp_vals[1] = INT32_MIN;
            inp_vals[3] = INT32_MAX;
            // Add a negative number
            if (N > 6) inp_vals[6] = -1;
            if (N > 7) inp_vals[7] = -100;
        }
    }


    // 4. Create copy for verification
    memcpy(inp_vals_copy, inp_vals, mem_size);

    // 5. Run GPU Radix Sort (using template parameter T)
    if (radixSort<T, Q, B, lgH, TILE_SIZE>(inp_vals, out_vals, N) != 0) {
        printf("ERROR: radixSort function returned non-zero exit code.\n");
        success = false;
    } 

    // 6. Verification 1: Check if sorted
    if (success) {
        bool sorted = true;
        for (uint32_t i = 0; i < N - 1; i++) {
            if (out_vals[i] > out_vals[i + 1]) {
                sorted = false;
                
                // Use type-appropriate printing
                if constexpr (std::is_same_v<T, uint32_t>) {
                    printf("ERROR: Sort failed at index %u: %u > %u\n", i,
                            out_vals[i], out_vals[i + 1]);
                } else {
                    printf("ERROR: Sort failed at index %u: %d > %d\n", i,
                            out_vals[i], out_vals[i + 1]);
                }
                
                // Print surrounding elements for context
                printf("... Context ...\n");
                uint32_t start = (i > 5) ? (i - 5) : 0;
                uint32_t end = (i + 5 < N) ? (i + 5) : N;
                printArray(out_vals + start, end - start, "out_vals");
                printf("... End Context ...\n");

                break;
            }
        }
        if (!sorted) {
            success = false;
        }
    }

    // 7. Verification 2: Element Preservation
    if (success) {
        if (!checkElementPreservation(inp_vals_copy, out_vals, N)) {
            printf("ERROR: Element preservation failed! Output is not a permutation of the input.\n");
            success = false;
        }
    }

    // 8. Cleanup
    free(inp_vals);
    free(inp_vals_copy);
    free(out_vals);

    printf("--- Test for N = %u, Type = %s %s ---\n\n", N, type_name, success ? "PASSED" : "FAILED");
    return success;
}

template <typename T>
int runAllTests(const char* type_name, int* test_sizes, int N) {
    // Define a set of test sizes
    int tests_passed = 0;

    for (int i = 0; i < N; i++) {
        if (runTest<T>(test_sizes[i])) {
            tests_passed++;
        }
    }

    return tests_passed;
}

int main() {
    initHwd();
    
    srand(42);  

    int N = 5;
    int test_sizes[] = {10, 100, 1024, 10000, 1000000};
    
    int total_tests = 0;
    int total_passed = 0;
    
    // 1. Run tests for uint32_t
    int uint32_tests_passed = runAllTests<uint32_t>("uint32_t", test_sizes, N);
    
    // 2. Run tests for int32_t
    int int32_tests_passed = runAllTests<int32_t>("int32_t", test_sizes, N);

    total_passed = uint32_tests_passed + int32_tests_passed;
    total_tests = 2*N;


    printf("\n\n====== FINAL TEST SUMMARY (uint32_t + int32_t) ======\n");
    printf("uint32_t tests: Passed %d/%d\n", uint32_tests_passed, N);
    printf("int32_t tests:  Passed %d/%d\n", int32_tests_passed, N);
    printf("------------------------------------------------------\n");
    printf("**Total Tests: Passed %d out of %d tests.**\n", total_passed, total_tests);
    printf("====================================================\n");

    // Return 0 if all tests passed, 1 otherwise
    return (total_passed == total_tests) ? 0 : 1; 
}
