import numpy as np
import cupy as cp

import pycuda.driver as cuda

import time
from statistics import median, mean

from kernels import min_kernel, max_kernel, sum_kernel, \
    prefix_sum
    # prefix_sum_int,prefix_sum_float, prefix_sum_double

# BUF_BYTE_SIZE = 1 << 24
SAMPLE_COUNT = 20

print("Compiling kernel...")


tested_types: list[np.dtype] = [cp.dtype("float32"), cp.dtype("float64"), cp.dtype("int")]
tested_counts: list[int] = [
    1 << 10, 
    # 1 << 20, 
    # 1 << 27
]
# tested_counts: list[int] = [1 << 10]

# np.set_printoptions(formatter={"float_kind": lambda x: "%.0f" % x}, suppress=True)


with cp.cuda.Device(0):
    for count in tested_counts:
        rng = np.random.default_rng()
        base_buffer = rng.uniform(1, 1, count)
        # base_buffer = rng.uniform(-100, 100, count)
        for dtype in tested_types:    
            samples = []

            for i in range(SAMPLE_COUNT):
                # buffer = cp.full((count, ), 1, dtype=dtype)
                buffer = np.full((count, ), 1, dtype=dtype)
                # gpu_buffer = cp.array(base_buffer, dtype=dtype, copy=True)
                gpu_buffer = np.array(base_buffer, dtype=dtype, copy=True)
                start_time = time.time()
                # result = sum_kernel(buffer, axis=0)
                # result = max_kernel(gpu_buffer, axis=0)
                # result = prefix_sum_kernel(gpu_buffer, axis=0)

                h_in = base_buffer
                # print("first elem:", base_buffer[1])
                # print("last elem:", base_buffer[-1])

                h_out = np.full((count, ), 0, dtype=dtype)

                result = prefix_sum(h_in, h_out, count)
                print("returned d_out:", result)
                print(h_out[-1])
                # cuda.memcpy_dtoh(h_out, d_out_after)
                # print("AFTER KERNEL:", result[:10])

                # print(np.array2string(result))
                samples.append(time.time() - start_time)
                del gpu_buffer
            # print(dtype.name, count, result, "\t", mean(samples), median(samples))
            print(dtype.name, count, "\t", mean(samples), median(samples))