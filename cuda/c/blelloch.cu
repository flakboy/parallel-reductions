#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// amount of threads within single thread block
#define BLOCK_SIZE 1024
#define ELEMS_PER_BLOCK BLOCK_SIZE * 2

// amount of threads handled by single warp
#define WARP_SIZE 32

// Error checking macro
#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// https://www.youtube.com/watch?v=mmYv3Haj6uc
//
// Block-level Blelloch scan kernel
// Each block scans ELEMS_PER_BLOCK elements
// if d_blockSums is a nullptr, then don't write blockSums
template <typename T>
__global__ void blellochScanBlock(const T *d_in, T *d_out, T *d_blockSums, int n)
{
    __shared__ T smem[ELEMS_PER_BLOCK];

    int tid = threadIdx.x;
    int blockStart = blockIdx.x * ELEMS_PER_BLOCK;
    // index of first element processed by thread
    int idx1 = blockStart + tid;
    // index of second element processed by thread
    int idx2 = blockStart + tid + BLOCK_SIZE;

    // Copy data from input into shared memory
    // Check if the indices fit inside the actual buffer size
    smem[tid] = (idx1 < n) ? d_in[idx1] : T(0);
    smem[tid + BLOCK_SIZE] = (idx2 < n) ? d_in[idx2] : T(0);

    __syncthreads();
    // Up-sweep phase
    int stride = 1;
    for (int d = BLOCK_SIZE; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            // left leaf/root of binary tree
            int ai = stride * (2 * tid + 1) - 1;

            // right leaf/root of binary tree
            int bi = stride * (2 * tid + 2) - 1;

            // reduce elements and store in right leaf
            smem[bi] += smem[ai];
        }
        stride *= 2;
    }

    // Save total sum for this block
    if (tid == 0)
    {
        if (d_blockSums != nullptr)
        {
            d_blockSums[blockIdx.x] = smem[ELEMS_PER_BLOCK - 1];
        }
        smem[ELEMS_PER_BLOCK - 1] = 0; // Clear last element for down-sweep
    }

    // Down-sweep phase
    for (int d = 1; d < ELEMS_PER_BLOCK; d *= 2)
    {
        stride /= 2;
        __syncthreads();
        if (tid < d)
        {
            int ai = stride * (2 * tid + 1) - 1;
            int bi = stride * (2 * tid + 2) - 1;
            int t = smem[ai];
            smem[ai] = smem[bi];
            smem[bi] += t;
        }
    }
    __syncthreads();

    // Copy results from shared memory to output
    if (idx1 < n)
        d_out[idx1] = smem[tid];
    if (idx2 < n)
        d_out[idx2] = smem[tid + BLOCK_SIZE];
}

// Add scanned block sums to all elements
template <typename T>
__global__ void addBlockSums(T *d_data, T *d_blockSums, int n)
{
    int idx = blockIdx.x * ELEMS_PER_BLOCK + threadIdx.x;

    if (blockIdx.x > 0)
    {
        T blockSum = d_blockSums[blockIdx.x];
        if (idx < n)
        {
            d_data[idx] += blockSum;
        }
        if (idx + BLOCK_SIZE < n)
        {
            d_data[idx + BLOCK_SIZE] += blockSum;
        }
    }
}

// Host function to perform hierarchical scan
// d_in - device memory for input
// d_out - device memory for output
// n - number of elements (total buffer size)
template <typename T>
void prefixSum(T *d_in, T *d_out, int n)
{
    int elemsPerBlock = ELEMS_PER_BLOCK;

    // numBlocks calculation uses ceil function trick for efficient integer division

    int numBlocks = (n + elemsPerBlock - 1) / elemsPerBlock;

    // Edge case: the size of buffer is smaller than ELEMS_PER_BLOCK
    if (numBlocks == 1)
    {
        blellochScanBlock<<<1, BLOCK_SIZE>>>(d_in, d_out, (T *)nullptr, n);
        return;
    }

    // Allocate memory for block sums
    T *d_blockSums;
    T *d_blockSumsScanned;
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

// // instantiating templates in order to be prevent name mangling
// // and be able to export functions to Python
extern "C"
{
    
    // template __global__ void blellochScanBlock<int>(const int *d_in, int *d_out, int *d_blockSums, int n); 
    // template __global__ void blellochScanBlock<float>(const float *d_in, float *d_out, float *d_blockSums, int n); 
    // template __global__ void blellochScanBlock<double>(const double *d_in, double *d_out, double *d_blockSums, int n); 
 
//     __global__ void blellochScanBlockInt(const int *d_in, int *d_out, int *d_blockSums, int n, int blocks) {
//         blellochScanBlock<<<blocks, BLOCK_SIZE>>>(d_in, d_out, d_blockSums, n);
//     }

    // __global__ void blellochScanBlockFloat(const float *d_in, float *d_out, float *d_blockSums, int n, int blocks) {
    //     blellochScanBlock<<<blocks, BLOCK_SIZE>>>(d_in, d_out, d_blockSums, n);
    // }

    // __global__ void blellochScanBlockDouble(const double *d_in, double *d_out, double *d_blockSums, int n, int blocks) {
    //     blellochScanBlock<<<blocks, BLOCK_SIZE>>>(d_in, d_out, d_blockSums, n);
    // }
    
    // __global__ void blellochScanBlockIntNullptr(const int *d_in, int *d_out, int n, int blocks) {
    //     blellochScanBlock<<<blocks, BLOCK_SIZE>>>(d_in, d_out, (int*)nullptr, n);
    // }

    // __global__ void blellochScanBlockFloatNullptr(const float *d_in, float *d_out, int n, int blocks) {
    //     blellochScanBlock<<<blocks, BLOCK_SIZE>>>(d_in, d_out, (float*)nullptr, n);
    // }

    // __global__ void blellochScanBlockDoubleNullptr(const double *d_in, double *d_out, int n, int blocks) {
    //     blellochScanBlock<<<blocks, BLOCK_SIZE>>>(d_in, d_out, (double*)nullptr, n);
    // }

    // __global__ void addBlockSumsInt(int *d_data, int *d_blockSums, int n, int blocks) {
    //     addBlockSums<<<blocks, BLOCK_SIZE>>>(d_data, d_blockSums, n);
    // }

    // __global__ void addBlockSumsFloat(float *d_data, float *d_blockSums, int n, int blocks) {
    //     addBlockSums<<<blocks, BLOCK_SIZE>>>(d_data, d_blockSums, n);
    // }

    // __global__ void addBlockSumsDouble(double *d_data, double *d_blockSums, int n, int blocks) {
    //     addBlockSums<<<blocks, BLOCK_SIZE>>>(d_data, d_blockSums, n);
    // }
}


#define DEBUG 1
#ifdef DEBUG
// Template verification function
template<typename T>
bool verifyScan(const T* input, const T* output, int n, int checkSize = 1000) {
    T sum = T(0);
    int toCheck = n - 1;
    printf("n: %d\n", n);

    for (int i = 0; i < toCheck; i++) {
        // printf("%d: %lf\n", i, (double)output[i]);
        if (output[i] != sum) {
            printf("Mismatch at %d: expected %.6f, got %.6f\n", 
                   i, (double)sum, (double)output[i]);
            return false;
        }
        sum += input[i];
    }
    
    return true;
}

// Test function for a specific type
template <typename T>
void testPrefixSum(int n, const char *typeName)
{
    size_t size = n * sizeof(T);

    printf("\n=== Testing %s ===\n", typeName);
    printf("Elements: %d (%.2f MB)\n", n, size / (1024.0 * 1024.0));

    // Allocate host memory
    T *h_input = (T *)malloc(size);
    T *h_output = (T *)malloc(size);

    // Initialize input
    for (int i = 0; i < n; i++)
    {
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
    
    if (correct)
    {
        printf("✓ Prefix sum correct!\n");
        printf("Time: %.3f ms\n", milliseconds);
        printf("Throughput: %.2f GB/s\n",
               (2.0 * size) / (milliseconds * 1e6)); // Read + Write
    }
    else
    {
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


int main()
{
    // Test with different sizes and types
    size_t buffer_size = 1024 * sizeof(int);// << 30;
    size_t n = (buffer_size) / sizeof(int);
    // size_t size = n * sizeof(int);

    // Test int
    testPrefixSum<int>(n, "int");

    // Test float
    testPrefixSum<float>(n, "float");

    // Test double
    testPrefixSum<double>(n, "double");

    return 0;
}
#endif