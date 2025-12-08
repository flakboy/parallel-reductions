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
with open('./c/blelloch_macro.cu', 'r') as f:
    cuda_source = f.read()

__module = DynamicSourceModule(
    cuda_source,
    # no_extern_c=1 # source .cu module should contain explicit extern "C" directives
)

blellochScanBlockInt = __module.get_function('blellochScanBlock_long')
blellochScanBlockDouble = __module.get_function('blellochScanBlock_double')
blellochScanBlockFloat = __module.get_function('blellochScanBlock_float')
# blellochScanBlockIntNullptr = __module.get_function('blellochScanBlockIntNullptr')
# blellochScanBlockDoubleNullptr = __module.get_function('blellochScanBlockDoubleNullptr')
# blellochScanBlockFloatNullptr = __module.get_function('blellochScanBlockFloatNullptr')

addBlockSumsInt = __module.get_function('addBlockSums_long')
addBlockSumsDouble = __module.get_function('addBlockSums_double')
addBlockSumsFloat = __module.get_function('addBlockSums_float')

__BLOCK_SIZE = 1024
__ELEMS_PER_BLOCK = __BLOCK_SIZE * 2

# prefixSum(T *d_in, T *d_out, int n)
def prefix_sum(h_in, h_out, n: int, dtype: np.dtype) -> np.ndarray:
    print(f"Launching prefix sum... {n}")
    dtype_size = dtype.itemsize

    d_in = cuda.mem_alloc(h_in.nbytes)  # type: ignore
    cuda.memcpy_htod(d_in, h_in)    # type: ignore
    h_out = np.full((n, ), 3, dtype=dtype)
    d_out = cuda.mem_alloc(h_out.nbytes)    # type: ignore
    cuda.memcpy_htod(d_out, h_out)  # type: ignore

    blellochScanBlock = None
    blellochScanBlockNullptr = None
    addBlocksSums = None

    if h_in.dtype == "int64" and h_out.dtype == "int64":
        blellochScanBlock = blellochScanBlockInt
        # blellochScanBlockNullptr = blellochScanBlockIntNullptr
        addBlocksSums = addBlockSumsInt
    elif h_in.dtype == "float32" and h_out.dtype == "float32":
        blellochScanBlock = blellochScanBlockFloat
        # blellochScanBlockNullptr = blellochScanBlockFloatNullptr
        addBlocksSums = addBlockSumsFloat
    elif h_in.dtype == "float64" and h_out.dtype == "float64":
        blellochScanBlock = blellochScanBlockDouble
        # blellochScanBlockNullptr = blellochScanBlockDoubleNullptr
        addBlocksSums = addBlockSumsDouble

    if blellochScanBlock is None or addBlocksSums is None:
        raise Exception(f"Couldn't find function implementation for given type: {dtype}")

    elemsPerBlock = __ELEMS_PER_BLOCK;
    # numBlocks =  (n + elemsPerBlock - 1) / elemsPerBlock;
    numBlocks = math.ceil(n / elemsPerBlock)


    print(f"Passed numbers: {n}")
    # Edge case: the size of buffer is smaller than ELEMS_PER_BLOCK
    if numBlocks == 1:
        print(f"Passed values", h_in[0:15])
        blellochScanBlock(
            d_in, d_out, cuda.In(np.zeros(1, dtype=np.int32)), np.int32(n),
            block=(__BLOCK_SIZE, 1, 1),
            grid=(numBlocks, 1)
        )

        cuda.memcpy_dtoh(h_out, d_out)  # type: ignore
        print("n1 h_outs: ", h_out[:15])
        return h_out
        
    # # Allocate memory for block sums
    # T *d_blockSums;
    # T *d_blockSumsScanned;
    # CUDA_CHECK(cudaMalloc(&d_blockSums, numBlocks * sizeof(T)));
    # CUDA_CHECK(cudaMalloc(&d_blockSumsScanned, numBlocks * sizeof(T)));
    h_blockSums = np.zeros(numBlocks * dtype_size, dtype=dtype)
    d_blockSums = cuda.mem_alloc(h_blockSums.nbytes)    # type: ignore
    h_blockSumsScanned = np.zeros(numBlocks * dtype_size, dtype=dtype)
    d_blockSumsScanned = cuda.mem_alloc(h_blockSumsScanned.nbytes)  # type: ignore

    # Step 1: Scan each block and store block sums
    # blellochScanBlock<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, d_blockSums, n);
    # print("Step 1")
    blellochScanBlock(
        d_in, d_out, d_blockSums, np.int32(n),
        block=(__BLOCK_SIZE, 1, 1),
        grid=(numBlocks, 1)
    )

    # FIXME!!!!!!!!!!!!!!!!
    # BAD ARGUMENTS PASSED
    # Step 2: Recursively scan the block sums
    out = prefix_sum(h_blockSums, h_blockSumsScanned, numBlocks, dtype);
    print("OUT: ", out)

    #Step 3: Add scanned block sums to each block's elements
    # addBlockSums<<<numBlocks, BLOCK_SIZE>>>(d_out, d_blockSumsScanned, n);
    addBlocksSums(
        d_out, cuda.Out(h_blockSumsScanned), np.int32(n),
        block=(__BLOCK_SIZE, 1, 1),
        grid=(numBlocks, 1, 1)
    )

    return h_out




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