import numpy as np
import cupy as cp
import time
from statistics import median, mean

from kernels import min_kernel, max_kernel, sum_kernel
    
SAMPLE_COUNT = 10
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

def sequential_sum(buffer: np.ndarray):
    acc = 0
    for elem in buffer:
        acc += elem
        
    return acc

def sequential_max(buffer: np.ndarray):
    acc = -np.inf

    for elem in buffer:
        if elem > acc:
            acc = elem

    return acc

def sequential_min(buffer: np.ndarray):
    acc = np.inf

    for elem in buffer:
        if elem < acc:
            acc = elem

    return acc

with cp.cuda.Device(0):
    rng = np.random.default_rng()   
    for dtype in tested_types:
        
        print(dtype.name)
        for count in tested_counts:
            n = count

            sum_samples = []
            max_samples = []
            min_samples = []

            sum_diffs = []
            max_diffs = []
            min_diffs = []

            for i in range(SAMPLE_COUNT):
                base_buffer = rng.uniform(-100, 100, count)
                base_buffer_typed = np.array(base_buffer, dtype=dtype, copy=True)
                
                # seq_sum = sequential_sum(base_buffer_typed)
                # seq_max = sequential_max(base_buffer_typed)
                # seq_min = sequential_min(base_buffer_typed)
                seq_sum = np.sum(base_buffer_typed)
                seq_max = np.max(base_buffer_typed)
                seq_min = np.min(base_buffer_typed)

                start_time = time.time()
                gpu_buffer = cp.array(base_buffer_typed, dtype=dtype, copy=True)
                result = sum_kernel(gpu_buffer, axis=0)
                sum_samples.append(time.time() - start_time)
                sum_diffs.append(result.item() - seq_sum)

                start_time = time.time()
                gpu_buffer = cp.array(base_buffer, dtype=dtype, copy=True)
                result = max_kernel(gpu_buffer, axis=0)
                max_samples.append(time.time() - start_time)
                max_diffs.append(result.item() - seq_max)

                start_time = time.time()
                gpu_buffer = cp.array(base_buffer, dtype=dtype, copy=True)
                result = min_kernel(gpu_buffer, axis=0)
                min_samples.append(time.time() - start_time)
                min_diffs.append(result.item() - seq_min)

            print("SUM", count, "\tTIME:", mean(sum_samples), median(sum_samples), "\tERROR:", mean(sum_diffs))
            print("MAX", count, "\tTIME:", mean(max_samples), median(max_samples), "\tERROR:", mean(max_diffs))
            print("MIN", count, "\tTIME:", mean(min_samples), median(min_samples), "\tERROR:", mean(min_diffs))
        print("=====================================================================")
