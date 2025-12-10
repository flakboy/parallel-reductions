import numpy as np
import pycuda.driver as cuda
import time
from statistics import mean

from kernels import prefix_sum
    
SAMPLE_COUNT = 30
tested_types: list[np.dtype] = [
    np.dtype("float32"), 
    np.dtype("float64"), 
    np.dtype("int")
]
tested_counts: list[int] = [
    1 << 10, 
    1 << 20, 
    1 << 27
]

def sequential_prefix_sum(buffer):
    out = np.zeros_like(buffer)

    for index, elem in enumerate(buffer[:-1]):
        out[index + 1] = out[index] + elem  

    return out

# np.set_printoptions(suppress=True,
#    formatter={'float_kind':'{:f}'.format})
rng = np.random.default_rng()
for dtype in tested_types:    
    print(dtype.name)
    for count in tested_counts:
        n = count
        samples = []
        diffs = []

        # moved outside the loop to speed up the process of benchmarking
        h_out = np.full((n, ), 0, dtype=dtype)
        base_buffer = rng.uniform(-100, 100, count)
        base_buffer_typed = np.array(base_buffer, dtype=dtype, copy=True)
        seq_prefix_sum = sequential_prefix_sum(base_buffer_typed)
        h_in = base_buffer_typed

        for i in range(SAMPLE_COUNT):
            start_time = time.time()

            d_in = cuda.mem_alloc(h_in.nbytes)  # type: ignore
            cuda.memcpy_htod(d_in, h_in)    # type: ignore
            d_out = cuda.mem_alloc(dtype.itemsize * n)    # type: ignore
            
            result = prefix_sum(d_in, d_out, count, dtype)
            
            cuda.memcpy_dtoh(h_out, d_out)  # type: ignore
            diff = seq_prefix_sum[i] - h_out[i]
            diffs.append(diff.item())
            
            d_in.free()
            d_out.free()

            samples.append(time.time() - start_time)
            
        

        print("PREFIX SUM:", count, "\tTIME:", mean(samples), "\tERROR:", mean(diffs))
    print("=====================================================================")