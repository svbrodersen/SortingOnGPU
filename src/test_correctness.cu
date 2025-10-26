#include "constants.cuh"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <vector>
#include <algorithm>

// CPU reference: histogram and scan
void cpu_histogram(const std::vector<uint32_t>& inp, std::vector<uint32_t>& hist, uint32_t lgH) {
    uint32_t H = 1u << lgH;
    std::fill(hist.begin(), hist.end(), 0u);
    for (size_t i = 0; i < inp.size(); ++i) {
        uint32_t bin = inp[i] & (H - 1u);
        hist[bin]++;
    }
}

void cpu_scan(const std::vector<uint32_t>& hist, std::vector<uint32_t>& scan) {
    scan[0] = hist[0];
    for (size_t i = 1; i < hist.size(); ++i) {
        scan[i] = scan[i-1] + hist[i];
    }
}

// Simple deterministic input generator
void make_input(std::vector<uint32_t>& inp, int mode) {
    for (size_t i = 0; i < inp.size(); ++i) {
        if (mode == 0) inp[i] = i; // ascending
        else if (mode == 1) inp[i] = inp.size() - i; // descending
        else inp[i] = (i * 17) % 256; // pseudo-random
    }
}

// GPU pipeline: histogram + scan (single block, small N)
__global__ void gpu_histogram(uint32_t* inp, uint32_t* hist, uint32_t N, uint32_t H) {
    int tid = threadIdx.x;
    for (int i = tid; i < H; i += blockDim.x) hist[i] = 0u;
    __syncthreads();
    for (int i = tid; i < N; i += blockDim.x) {
        uint32_t bin = inp[i] & (H - 1u);
        atomicAdd(&hist[bin], 1u);
    }
}

// Fix: use a single thread for sequential scan (correct for small H)
__global__ void gpu_scan(uint32_t* hist, uint32_t* scan, uint32_t H) {
    if (threadIdx.x == 0) {
        scan[0] = hist[0];
        for (uint32_t i = 1; i < H; ++i) {
            scan[i] = scan[i-1] + hist[i];
        }
    }
}

bool check_equal(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

int main() {
    initHwd();
    struct TestCase { int N, lgH, mode; } cases[] = {
        {1, 4, 0}, {7, 4, 1}, {16, 4, 2}, {64, 6, 0}, {128, 8, 1}, {256, 8, 2}
    };
    int pass = 0, fail = 0;
    for (auto tc : cases) {
        int N = tc.N, lgH = tc.lgH, mode = tc.mode;
        uint32_t H = 1u << lgH;
        std::vector<uint32_t> inp(N), cpu_hist(H), cpu_scan_out(H);
        make_input(inp, mode);
        cpu_histogram(inp, cpu_hist, lgH);
        cpu_scan(cpu_hist, cpu_scan_out);
        // GPU
        uint32_t *d_inp, *d_hist, *d_scan;
        cudaMalloc(&d_inp, N * sizeof(uint32_t));
        cudaMalloc(&d_hist, H * sizeof(uint32_t));
        cudaMalloc(&d_scan, H * sizeof(uint32_t));
        cudaMemcpy(d_inp, inp.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
        gpu_histogram<<<1, 32>>>(d_inp, d_hist, N, H);
        gpu_scan<<<1, 32>>>(d_hist, d_scan, H);
        std::vector<uint32_t> gpu_hist(H), gpu_scan_out(H);
        cudaMemcpy(gpu_hist.data(), d_hist, H * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(gpu_scan_out.data(), d_scan, H * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaFree(d_inp); cudaFree(d_hist); cudaFree(d_scan);
        bool ok_hist = check_equal(cpu_hist, gpu_hist);
        bool ok_scan = check_equal(cpu_scan_out, gpu_scan_out);
        printf("Test N=%d lgH=%d mode=%d: %s\n", N, lgH, mode, (ok_hist && ok_scan) ? "PASS" : "FAIL");
        if (ok_hist && ok_scan) pass++; else fail++;
    }
    printf("Correctness summary: %d PASS, %d FAIL\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
