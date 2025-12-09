import numpy as np
import cupy as cp
import pycuda.driver as cuda
import time
from statistics import median, mean

from kernels import prefix_sum
    
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


for count in tested_counts:
    n = count
    rng = np.random.default_rng()
    base_buffer = rng.uniform(-100, 100, count)
    for dtype in tested_types:    
        base_buffer_typed = np.array(base_buffer, dtype=dtype, copy=True)
        samples = []

        for i in range(SAMPLE_COUNT):
            h_in = base_buffer_typed
            start_time = time.time()

            d_in = cuda.mem_alloc(h_in.nbytes)  # type: ignore
            cuda.memcpy_htod(d_in, h_in)    # type: ignore
            d_out = cuda.mem_alloc(dtype.itemsize * n)    # type: ignore
            
            h_out = np.full((n, ), 0, dtype=dtype)
            result = prefix_sum(d_in, d_out, count, dtype)
            
            cuda.memcpy_dtoh(h_out, d_out)  # type: ignore

            samples.append(time.time() - start_time)
        print("PREFIX SUM:", dtype.name, count, "\t", mean(samples), median(samples))
    print("=====================================================================")