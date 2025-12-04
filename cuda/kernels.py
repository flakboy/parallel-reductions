import math

import cupy as cp
import numpy as np
import numpy.typing as npt

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import DynamicSourceModule

sum_kernel = cp.ReductionKernel(
    "T x",  # input params
    "T y",  # output params
    "x",  # map
    "a + b",  # reduce
    "y = a",  # post-reduction map
    "0",  # identity value
    "buffer_sum",  # kernel name
)

min_kernel = cp.ReductionKernel(
    "T x",  # input params
    "T y",  # output params
    "x",  # map
    "min(a, b)",  # reduce
    "y = a",  # post-reduction map
    # a representation of Infinity number - https://forums.developer.nvidia.com/t/how-to-assign-infinity-to-variables-in-cuda-code/14348
    "CUDART_INF_F",
    "buffer_min",  # kernel name
)

max_kernel = cp.ReductionKernel(
    "T x",  # input params
    "T y",  # output params
    "x",  # map
    "max(a, b)",  # reduce
    "y = a",  # post-reduction map
    # a representation of Infinity number - https://forums.developer.nvidia.com/t/how-to-assign-infinity-to-variables-in-cuda-code/14348
    "-CUDART_INF_F",
    "buffer_max",  # kernel name
)

# Load the CUDA kernel from your .cu file
with open('./c/blelloch.cu', 'r') as f:
    cuda_source = f.read()

__module = DynamicSourceModule(
    cuda_source, 
    no_extern_c=1 # source .cu module should contain explicit extern "C" directives
)

blellochScanBlockInt = __module.get_function('blellochScanBlock<int>')
blellochScanBlockDouble = __module.get_function('blellochScanBlock<float>')
blellochScanBlockFloat = __module.get_function('blellochScanBlock<double>')
blellochScanBlockIntNullptr = __module.get_function('blellochScanBlockIntNullptr')
blellochScanBlockDoubleNullptr = __module.get_function('blellochScanBlockDoubleNullptr')
blellochScanBlockFloatNullptr = __module.get_function('blellochScanBlockFloatNullptr')

addBlockSumsInt = __module.get_function('addBlockSumsInt')
addBlockSumsDouble = __module.get_function('addBlockSumsDouble')
addBlockSumsFloat = __module.get_function('addBlockSumsFloat')

__BLOCK_SIZE = 1024
__ELEMS_PER_BLOCK = __BLOCK_SIZE * 2

# prefixSum(T *d_in, T *d_out, int n)
def prefix_sum(h_in, h_out, n: int):
    dtype = h_in.dtype
    dtype_size = dtype.itemsize
    d_in = cuda.mem_alloc(h_in.nbytes)
    cuda.memcpy_htod(d_in, h_in)
    
    h_out = np.full((n, ), 0, dtype=dtype)
    d_out = cuda.mem_alloc(h_out.nbytes)
    cuda.memcpy_htod(d_out, h_out)

    blellochScanBlock = None
    blellochScanBlockNullptr = None
    addBlocksSums = None

    if h_in.dtype == "int64" and h_out.dtype == "int64":
        blellochScanBlock = blellochScanBlockInt
        blellochScanBlockNullptr = blellochScanBlockIntNullptr
        addBlocksSums = addBlockSumsInt
    elif h_in.dtype == "float32" and h_out.dtype == "float32":
        blellochScanBlock = blellochScanBlockFloat
        blellochScanBlockNullptr = blellochScanBlockFloatNullptr
        addBlocksSums = addBlockSumsFloat
    elif h_in.dtype == "float64" and h_out.dtype == "float64":
        blellochScanBlock = blellochScanBlockDouble
        blellochScanBlockNullptr = blellochScanBlockDoubleNullptr
        addBlocksSums = addBlockSumsDouble


    elemsPerBlock = __ELEMS_PER_BLOCK;
    # numBlocks =  (n + elemsPerBlock - 1) / elemsPerBlock;
    numBlocks = math.ceil(n / elemsPerBlock)


    # Edge case: the size of buffer is smaller than ELEMS_PER_BLOCK
    if numBlocks == 1:
        # print("Edge case")
        blellochScanBlockNullptr(
            d_in, d_out, np.int32(n), np.int32(numBlocks),
            block=(1, 1, 1),
            grid=(numBlocks, 1, 1)
        )
    
        cuda.memcpy_dtoh(h_out, d_out)
        return
        
    raise Exception("AAAAAAAAAAAAAa")
    # # # Allocate memory for block sums
    # # T *d_blockSums;
    # # T *d_blockSumsScanned;
    # # CUDA_CHECK(cudaMalloc(&d_blockSums, numBlocks * sizeof(T)));
    # # CUDA_CHECK(cudaMalloc(&d_blockSumsScanned, numBlocks * sizeof(T)));
    # h_blockSums = np.zeros(numBlocks * h_in.dtype.itemsize, dtype=dtype)
    # d_blockSums = cuda.mem_alloc(h_blockSums.nbytes)
    # h_blockSumsScanned = np.zeros(numBlocks * h_in.dtype.itemsize, dtype=dtype)
    # d_blockSumsScanned = cuda.mem_alloc(h_blockSumsScanned.nbytes)

    # # Step 1: Scan each block and store block sums
    # # blellochScanBlock<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, d_blockSums, n);
    # # print("Step 1")
    # blellochScanBlock(
    #     d_in, d_out, d_blockSums, np.int32(n), np.int32(numBlocks),
    #     block=(1, 1, 1),
    #     grid=(1, 1, 1)
    # )

    # # Step 2: Recursively scan the block sums
    # prefix_sum(h_blockSums, h_blockSumsScanned, numBlocks);

    # #Step 3: Add scanned block sums to each block's elements
    # # addBlockSums<<<numBlocks, BLOCK_SIZE>>>(d_out, d_blockSumsScanned, n);
    # addBlocksSums(
    #     d_out, cuda.Out(h_blockSumsScanned), np.int32(n), np.int32(numBlocks),
    #     block=(__BLOCK_SIZE, 1, 1),
    #     grid=(numBlocks, 1, 1)
    # )

    # # CUDA_CHECK(cudaFree(d_blockSums));
    # # CUDA_CHECK(cudaFree(d_blockSumsScanned));
    # h_result = np.empty_like(h_in)
    
    # cuda.memcpy_dtoh(h_result, d_out)
    # return h_result




# Compile the kernel
# __module = cp.RawModule(code=cuda_source, backend='nvcc')
# __module = cp.RawModule(code=cuda_source, 
#     options=('--std=c++17',),  # Enable C++ features
#     backend='nvcc',
#     )

# __module.compile()

# prefix_sum_int  = __module.get_function("prefixSum<int>")
# blelloch_scan_block_int  = __module.get_function("blellochScanBlock<int>")


# prefix_sum_float = __module.get_function("prefixSumFloat")
# prefix_sum_double = __module.get_function("prefixSumDouble")