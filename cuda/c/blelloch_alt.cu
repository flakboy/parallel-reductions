#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 1024
#define WARP_SIZE 32

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Template for warp-level scan using shuffle instructions
template<typename T>
__device__ inline T warpScan(T val, int lane) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        T n = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += n;
    }
    return val;
}

// Template block-level Blelloch scan kernel
template<typename T>
__global__ void blellochScanBlock(const T* d_in, T* d_out, T* d_blockSums, int n) {
    __shared__ T temp[BLOCK_SIZE * 2];
    
    int tid = threadIdx.x;
    int blockStart = blockIdx.x * BLOCK_SIZE * 2;
    int idx1 = blockStart + tid;
    int idx2 = blockStart + tid + BLOCK_SIZE;
    
    // Load data into shared memory
    temp[tid] = (idx1 < n) ? d_in[idx1] : T(0);
    temp[tid + BLOCK_SIZE] = (idx2 < n) ? d_in[idx2] : T(0);
    __syncthreads();
    
    // Up-sweep (reduce) phase
    int offset = 1;
    for (int d = BLOCK_SIZE; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // Save total sum for this block
    if (tid == 0) {
        if (d_blockSums != nullptr) {
            d_blockSums[blockIdx.x] = temp[BLOCK_SIZE * 2 - 1];
        }
        temp[BLOCK_SIZE * 2 - 1] = T(0); // Clear last element for down-sweep
    }
    
    // Down-sweep phase
    for (int d = 1; d < BLOCK_SIZE * 2; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            T t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    
    // Write results
    if (idx1 < n) d_out[idx1] = temp[tid];
    if (idx2 < n) d_out[idx2] = temp[tid + BLOCK_SIZE];
}

// Template kernel to add scanned block sums
template<typename T>
__global__ void addBlockSums(T* d_data, const T* d_blockSums, int n) {
    int idx = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
    
    if (blockIdx.x > 0) {
        T blockSum = d_blockSums[blockIdx.x];
        if (idx < n) {
            d_data[idx] += blockSum;
        }
        if (idx + BLOCK_SIZE < n) {
            d_data[idx + BLOCK_SIZE] += blockSum;
        }
    }
}

// Template host function to perform hierarchical scan
template<typename T>
void prefixSum(const T* d_in, T* d_out, int n) {
    // Number of elements each block processes
    int elemsPerBlock = BLOCK_SIZE * 2;
    int numBlocks = (n + elemsPerBlock - 1) / elemsPerBlock;
    
    if (numBlocks == 1) {
        // Single block can handle it
        blellochScanBlock<<<1, BLOCK_SIZE>>>(d_in, d_out, (T*)nullptr, n);
        return;
    }
    
    // Allocate memory for block sums
    T* d_blockSums;
    T* d_blockSumsScanned;
    CUDA_CHECK(cudaMalloc(&d_blockSums, numBlocks * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_blockSumsScanned, numBlocks * sizeof(T)));
    
    // Step 1: Scan each block and store block sums
    blellochScanBlock<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, d_blockSums, n);
    
    // Step 2: Recursively scan the block sums
    prefixSum(d_blockSums, d_blockSumsScanned, numBlocks);
    
    // Step 3: Add scanned block sums to each block's elements
    addBlockSums<<<numBlocks, BLOCK_SIZE>>>(d_out, d_blockSumsScanned, n);
    
    CUDA_CHECK(cudaFree(d_blockSums));
    CUDA_CHECK(cudaFree(d_blockSumsScanned));
}

// Template verification function
template<typename T>
bool verifyScan(const T* input, const T* output, int n, int checkSize = 1000) {
    T sum = T(0);
    int toCheck = n - 1;
    printf("n: %d", n);

    int miscount = 0;
    // Check beginning
    for (int i = 0; i < toCheck; i++) {
        if (output[i] != sum) {
            printf("Mismatch at %d: expected %.6f, got %.6f\n", 
                   i, (double)sum, (double)output[i]);
            return false;
            miscount++;
        }
        sum += input[i];
    }
    
    return true;
}

// Test function for a specific type
template<typename T>
void testPrefixSum(int n, const char* typeName) {
    size_t size = n * sizeof(T);
    
    printf("\n=== Testing %s ===\n", typeName);
    printf("Elements: %d (%.2f MB)\n", n, size / (1024.0 * 1024.0));
    
    // Allocate host memory
    T* h_input = (T*)malloc(size);
    T* h_output = (T*)malloc(size);
    
    // Initialize input
    for (int i = 0; i < n; i++) {
        h_input[i] = T(1); // Simple test: all ones
    }
    
    // Allocate device memory
    T *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Perform prefix sum
    CUDA_CHECK(cudaEventRecord(start));
    prefixSum(d_input, d_output, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    // Verify
    bool correct = verifyScan(h_input, h_output, n);
    
    if (correct) {
        printf("✓ Prefix sum correct!\n");
        printf("Time: %.3f ms\n", milliseconds);
        printf("Throughput: %.2f GB/s\n", 
               (2.0 * size) / (milliseconds * 1e6)); // Read + Write
    } else {
        printf("✗ Prefix sum verification failed!\n");
    }
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);
}

int main() {
    // Test with different sizes and types
    int testSize = 64 * 1024 * 1024; // 64M elements
    
    // Test int
    testPrefixSum<int>(testSize, "int");
    
    // Test float
    testPrefixSum<float>(testSize, "float");
    
    // Test double
    testPrefixSum<double>(testSize, "double");
    
    // Smaller test to verify correctness more thoroughly
    printf("\n=== Detailed Verification Test ===\n");
    testPrefixSum<int>(10000, "int (small)");
    testPrefixSum<float>(10000, "float (small)");
    testPrefixSum<double>(10000, "double (small)");
    
    return 0;
}