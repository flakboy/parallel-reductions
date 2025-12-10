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
    # no_extern_c=1 # source .cu module should contain explicit extern "C" directives
)

blellochScanBlockInt = __module.get_function('blellochScanBlock_long')
blellochScanBlockDouble = __module.get_function('blellochScanBlock_double')
blellochScanBlockFloat = __module.get_function('blellochScanBlock_float')

addBlockSumsInt = __module.get_function('addBlockSums_long')
addBlockSumsDouble = __module.get_function('addBlockSums_double')
addBlockSumsFloat = __module.get_function('addBlockSums_float')

__BLOCK_SIZE = 1024
__ELEMS_PER_BLOCK = __BLOCK_SIZE * 2


def prefix_sum(d_in, d_out, n: int, dtype: np.dtype):
    dtype_size = dtype.itemsize

    blellochScanBlock = None
    addBlocksSums = None

    if dtype == "int64":
        blellochScanBlock = blellochScanBlockInt
        addBlocksSums = addBlockSumsInt
    elif dtype == "float32":
        blellochScanBlock = blellochScanBlockFloat
        addBlocksSums = addBlockSumsFloat
    elif dtype == "float64":
        blellochScanBlock = blellochScanBlockDouble
        addBlocksSums = addBlockSumsDouble

    if blellochScanBlock is None or addBlocksSums is None:
        raise Exception(f"Couldn't find function implementation for given type: {dtype}")

    elemsPerBlock = __ELEMS_PER_BLOCK;
    # numBlocks =  (n + elemsPerBlock - 1) / elemsPerBlock;
    numBlocks = math.ceil(n / elemsPerBlock)


    # Edge case: the size of buffer is smaller than ELEMS_PER_BLOCK
    if numBlocks == 1:
        # alloc memory for only one element, 
        # since the function call will return only sum of a single block
        blellochScanBlock(
            d_in, d_out, cuda.In(np.zeros(1, dtype=dtype)), np.int32(n),
            block=(__BLOCK_SIZE, 1, 1),
            # grid=(numBlocks, 1)
            grid=(1, 1)
        )

        return d_out
        
    # Allocate memory for block sums
    d_blockSums = cuda.mem_alloc(numBlocks * dtype_size)    # type: ignore
    d_blockSumsScanned = cuda.mem_alloc(numBlocks * dtype_size)  # type: ignore

    # Step 1: Scan each block and store block sums
    # print("Step 1")
    blellochScanBlock(
        d_in, d_out, d_blockSums, np.int32(n),
        block=(__BLOCK_SIZE, 1, 1),
        grid=(numBlocks, 1)
    )

    # Step 2: Recursively scan the block sums
    prefix_sum(d_blockSums, d_blockSumsScanned, numBlocks, dtype);

    #Step 3: Add scanned block sums to each block's elements
    addBlocksSums(
        d_out, d_blockSumsScanned, np.int32(n),
        block=(__BLOCK_SIZE, 1, 1),
        grid=(numBlocks, 1, 1)
    )

    d_blockSumsScanned.free()
    d_blockSums.free()

    return d_out