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

void printArray(uint32_t *inp_vals, uint32_t N, const char *name) {
  printf("%s[:%d] = [", name, N);
  for (int i = 0; i < N; i++) {
    if (i == N - 1) {
      printf("%u]\n", inp_vals[i]);
    } else {
      printf("%u, ", inp_vals[i]);
    }
  }
}

bool checkElementPreservation(uint32_t* original, uint32_t* sorted, uint32_t N) {
    std::map<uint32_t, int> original_counts;
    std::map<uint32_t, int> sorted_counts;

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

    for (std::map<uint32_t, int>::iterator it = original_counts.begin(); it != original_counts.end(); ++it) {
        uint32_t key = it->first;
        int original_count = it->second;

        if (sorted_counts.find(key) == sorted_counts.end()) {
            printf("ERROR: Element %u from original array is missing in sorted array.\n", key);
            return false; // Element from original not in sorted
        }
        if (sorted_counts[key] != original_count) {
            printf("ERROR: Element count mismatch for %u. Original: %d, Sorted: %d\n",
                   key, original_count, sorted_counts[key]);
            return false; // Element count mismatch
        }
    }
    return true;
}

/**
 * @brief Runs a single test case for a given array size N.
 * @param N The number of elements to sort.
 * @return true if the test passed, false otherwise.
 */
bool runTest(uint32_t N) {
    printf("--- Starting test for N = %u ---\n", N);
    
    // 1. Define constants from radixSort
    const uint32_t Q = 22;
    const uint32_t B = 256;
    const uint32_t lgH = 8;
    const uint32_t T = 32;
    const uint32_t mem_size = N * sizeof(uint32_t);

    bool success = true;

    // 2. Allocate host memory
    uint32_t *inp_vals = (uint32_t *)malloc(mem_size);
    uint32_t *inp_vals_copy = (uint32_t *)malloc(mem_size);
    uint32_t *out_vals = (uint32_t *)malloc(mem_size);

    if (!inp_vals || !inp_vals_copy || !out_vals) {
        printf("ERROR: Host memory allocation failed!\n");
        if(inp_vals) free(inp_vals);
        if(inp_vals_copy) free(inp_vals_copy);
        if(out_vals) free(out_vals);
        return false;
    }

    // 3. Fill with random data
    for (uint32_t i = 0; i < N; i++) {
        inp_vals[i] = rand();
    }
    // Add some edge cases
    if (N > 10) {
        inp_vals[0] = 0;
        inp_vals[1] = 0xFFFFFFFF;
        inp_vals[2] = 1;
        inp_vals[3] = 0xFFFFFFFF;
        inp_vals[4] = 100;
        inp_vals[5] = 100;
    }


    // 4. Create copy for verification
    memcpy(inp_vals_copy, inp_vals, mem_size);

    // 5. Run GPU Radix Sort
    printf("Running GPU radixSort...\n");
    if (radixSort<Q, B, lgH, T>(inp_vals, out_vals, N) != 0) {
        printf("ERROR: radixSort function returned non-zero exit code.\n");
        success = false;
    } else {
        printf("GPU radixSort finished.\n");
    }

    // 6. Verification 1: Check if sorted
    if (success) {
        printf("Verifying sorted order...\n");
        bool sorted = true;
        for (uint32_t i = 0; i < N - 1; i++) {
            if (out_vals[i] > out_vals[i + 1]) {
                sorted = false;
                printf("ERROR: Sort failed at index %u: %u > %u\n", i,
                       out_vals[i], out_vals[i + 1]); // Fixed print bug
                
                // Print surrounding elements for context
                printf("... Context ...\n");
                uint32_t start = (i > 5) ? (i - 5) : 0;
                uint32_t end = (i + 5 < N) ? (i + 5) : N;
                printArray(out_vals + start, end - start, "out_vals");
                printf("... End Context ...\n");

                break;
            }
        }
        if (sorted) {
            printf("SUCCESS: Array is correctly sorted.\n");
        } else {
            success = false;
        }
    }

    // 7. Verification 2: Element Preservation
    // Only check this if the array claims to be sorted.
    if (success) {
        printf("Verifying element preservation (using frequency map)...\n");
        if (checkElementPreservation(inp_vals_copy, out_vals, N)) {
            printf("SUCCESS: Element preservation confirmed.\n");
        } else {
            printf("ERROR: Element preservation failed! Output is not a permutation of the input.\n");
            success = false;
        }
    }

    // 8. Cleanup
    free(inp_vals);
    free(inp_vals_copy);
    free(out_vals);

    printf("--- Test for N = %u %s ---\n\n", N, success ? "PASSED" : "FAILED");
    return success;
}

// ===================================================================
// == New Main Function
// ===================================================================

int main() {
    initHwd();
    
    // Seed rand with a fixed value for reproducible tests
    srand(42); 

    // Define a set of test sizes
    uint32_t test_sizes[] = {10, 100, 1024, 10000, 1000000, 10000000};
    int num_tests = sizeof(test_sizes) / sizeof(uint32_t);
    int tests_passed = 0;

    for (int i = 0; i < num_tests; i++) {
        if (runTest(test_sizes[i])) {
            tests_passed++;
        }
    }

    printf("====== TEST SUMMARY ======\n");
    printf("Passed %d out of %d tests.\n", tests_passed, num_tests);
    printf("==========================\n");

    // Return 0 if all tests passed, 1 otherwise
    return (tests_passed == num_tests) ? 0 : 1; 
}
