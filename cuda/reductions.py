import numpy as np
import cupy as cp
import time
from statistics import median, mean

from kernels import min_kernel, max_kernel, sum_kernel
    
SAMPLE_COUNT = 30
tested_types: list[np.dtype] = [
    cp.dtype("float32"), 
    cp.dtype("float64"), 
    np.dtype("int")
]
tested_counts: list[int] = [
    1 << 10, 
    1 << 20, 
    1 << 27
]

with cp.cuda.Device(0):
    for count in tested_counts:
        n = count
        rng = np.random.default_rng()
        base_buffer = rng.uniform(-100, 100, count)
        for dtype in tested_types:    
            base_buffer_typed = np.array(base_buffer, dtype=dtype, copy=True)
            sum_samples = []
            max_samples = []
            min_samples = []

            for i in range(SAMPLE_COUNT):
                buffer = np.full((count, ), 1, dtype=dtype)
                
                start_time = time.time()
                gpu_buffer = cp.array(base_buffer, dtype=dtype, copy=True)
                result = sum_kernel(gpu_buffer, axis=0)
                sum_samples.append(time.time() - start_time)


                start_time = time.time()
                gpu_buffer = cp.array(base_buffer, dtype=dtype, copy=True)
                result = max_kernel(gpu_buffer, axis=0)
                max_samples.append(time.time() - start_time)

                start_time = time.time()
                gpu_buffer = cp.array(base_buffer, dtype=dtype, copy=True)
                result = min_kernel(gpu_buffer, axis=0)
                min_samples.append(time.time() - start_time)


            print("SUM", dtype.name, count, "\t", mean(sum_samples), median(sum_samples))
            print("MAX", dtype.name, count, "\t", mean(max_samples), median(max_samples))
            print("MIN", dtype.name, count, "\t", mean(min_samples), median(min_samples))
        print("=====================================================================")
