import numpy as np
import cupy as cp

import time
from statistics import median, mean

# BUF_BYTE_SIZE = 1 << 24
SAMPLE_COUNT = 50

print("Compiling kernel...")
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



tested_types: list[np.dtype] = [cp.dtype("float16"), cp.dtype("float32"), cp.dtype("float64"), cp.dtype("int")]
tested_counts: list[int] = [1 << 10, 1 << 20, 1 << 27]
# tested_counts: list[int] = [1 << 10]

# np.set_printoptions(formatter={"float_kind": lambda x: "%.0f" % x}, suppress=True)


with cp.cuda.Device(0):
    for count in tested_counts:
        rng = np.random.default_rng()
        base_buffer = rng.uniform(-100, 100, count)
        for dtype in tested_types:    
            samples = []

            for i in range(SAMPLE_COUNT):
                # buffer = cp.full((count), 1, dtype=dtype)
                gpu_buffer = cp.array(base_buffer, dtype=dtype, copy=True)
                start_time = time.time()
                # result = sum_kernel(buffer, axis=0)
                result = max_kernel(gpu_buffer, axis=0)
                # result = prefix_sum_kernel(gpu_buffer, axis=0)

                # print(np.array2string(result))
                samples.append(time.time() - start_time)
                del gpu_buffer
            print(dtype.name, count, result, "\t", mean(samples), median(samples))