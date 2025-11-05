//#include "../../cub-1.8.0/cub/cub.cuh"   // or equivalently <cub/device/device_histogram.cuh>
#include "cub/cub.cuh"
#include "helper.cu.h"
#include "../utils/utils.cuh"


template<typename T>
void randomInit(T* data, const uint64_t size) {
    for (int i = 0; i < size; ++i) {
        data[i] = randomValue<T>();
    }
}

template<typename T>
int getBitSize() {
    return sizeof(T) * 8;
}

template<typename T>
double sortRedByKeyCUB( T* data_keys_in
                      , T* data_keys_out
                      , const uint64_t N
) {
    int beg_bit = 0;
    int end_bit = getBitSize<T>();
    void * tmp_sort_mem = NULL;
    size_t tmp_sort_len = 0;
    
    { // sort prelude
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
        cudaMalloc(&tmp_sort_mem, tmp_sort_len);
    }
    cudaCheckError();
    
    { // one dry run
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
        cudaDeviceSynchronize();
    }
    cudaCheckError();
    
    // timing
    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);
    
    for(int k=0; k<GPU_RUNS; k++) {
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
    }
    cudaDeviceSynchronize();
    cudaCheckError();
    
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS);
    
    cudaFree(tmp_sort_mem);
    return elapsed;
}

template<typename T>
bool runSortingTest(const uint64_t N, const char* typeName) {
    // Allocate and Initialize Host data with random values
    T* h_keys  = (T*) malloc(N*sizeof(T));
    T* h_keys_res  = (T*) malloc(N*sizeof(T));
    randomInit(h_keys, N);
    
    // Allocate and Initialize Device data
    T* d_keys_in;
    T* d_keys_out;
    cudaSucceeded(cudaMalloc((void**) &d_keys_in,  N * sizeof(T)));
    cudaSucceeded(cudaMemcpy(d_keys_in, h_keys, N * sizeof(T), cudaMemcpyHostToDevice));
    cudaSucceeded(cudaMalloc((void**) &d_keys_out, N * sizeof(T)));
    
    double elapsed = sortRedByKeyCUB( d_keys_in, d_keys_out, N );
    
    cudaMemcpy(h_keys_res, d_keys_out, N*sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaCheckError();
    
    bool success = validateZ(h_keys_res, N);
    printf("CUB Sorting (%s) for N=%lu runs in: %.2f us, VALID: %d\n", 
           typeName, N, elapsed, success);
    
    // Cleanup
    cudaFree(d_keys_in); 
    cudaFree(d_keys_out);
    free(h_keys); 
    free(h_keys_res);
    
    return success;
}

int main (int argc, char * argv[]) {
    initializeDeviceOnce();
    if (argc != 2) {
        printf("Usage: %s <size-of-array>\n", argv[0]);
        exit(1);
    }
    const uint64_t N = atoi(argv[1]);
    
    bool allSuccess = true;
    
    // Test uint32_t
    allSuccess &= runSortingTest<uint32_t>(N, "uint32_t");
    // Test int32_t
    allSuccess &= runSortingTest<int32_t>(N, "int32_t");
    // Test float
    allSuccess &= runSortingTest<float>(N, "float");
    
    return allSuccess ? 0 : 1;
}
