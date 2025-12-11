import numpy as np
import cupy as cp
import time
from statistics import median, mean

from kernels import min_kernel, max_kernel, sum_kernel
    
SAMPLE_COUNT = 30
tested_types: list[np.dtype] = [
    # cp.dtype("float32"), 
    cp.dtype("float64"), 
    cp.dtype("int")
]
tested_counts: list[int] = [
    1 << 10, 
    1 << 20, 
    1 << 24
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

WARMUP_LENGTH = 10

with cp.cuda.Device(0):
    rng = np.random.default_rng()   
    for dtype in tested_types:
        
        print(dtype.name, dtype.itemsize)
        for count in tested_counts:
            n = count

            sum_samples = []
            max_samples = []
            min_samples = []

            sum_diffs = []
            max_diffs = []
            min_diffs = []

            for i in range(SAMPLE_COUNT + WARMUP_LENGTH):
                base_buffer = rng.uniform(-100, 100, count)
                base_buffer_typed = np.array(base_buffer, dtype=dtype)

                # seq_sum = sequential_sum(base_buffer_typed)
                # seq_max = sequential_max(base_buffer_typed)
                # seq_min = sequential_min(base_buffer_typed)
                seq_sum = np.sum(base_buffer_typed)
                seq_max = np.max(base_buffer_typed)
                seq_min = np.min(base_buffer_typed)
                
                gpu_buffer = cp.array(base_buffer_typed, copy=True)
                start_time = time.time()
                result = sum_kernel(gpu_buffer, axis=0)

                # the kernel compiles at the first __call__(),
                # and it does it for every dtype and input ndim 
                # so it's necessary to skip the first iteration to prevent skewing the results
                if i > WARMUP_LENGTH:
                    sum_samples.append(time.time() - start_time)
                    sum_diffs.append(result.item() - seq_sum)

                gpu_buffer = cp.array(base_buffer, copy=True)
                start_time = time.time()
                result = max_kernel(gpu_buffer, axis=0)
                # the compiles at the first __call__(),
                # and it does it for every dtype and input ndim 
                # so it's necessary to skip the first iteration to prevent skewing the results
                if i > WARMUP_LENGTH:
                    max_samples.append(time.time() - start_time)
                    max_diffs.append(result.item() - seq_max)

                gpu_buffer = cp.array(base_buffer, copy=True)
                start_time = time.time()
                result = min_kernel(gpu_buffer, axis=0)
                
                # the kernel compiles at the first __call__(),
                # and it does it for every dtype and input ndim 
                # so it's necessary to skip the first iteration to prevent skewing the results
                if i > WARMUP_LENGTH:
                    min_samples.append(time.time() - start_time)
                    min_diffs.append(result.item() - seq_min)

            print("SUM", count, "\tTIME:", mean(sum_samples), "\tERROR:", mean(sum_diffs), "\tBANDWIDTH:", count * dtype.itemsize / mean(sum_samples) / (1000 ** 3))
            print("MAX", count, "\tTIME:", mean(max_samples), "\tERROR:", mean(max_diffs), "\tBANDWIDTH:", count * dtype.itemsize / mean(max_samples) / (1000 ** 3))
            print("MIN", count, "\tTIME:", mean(min_samples), "\tERROR:", mean(min_diffs), "\tBANDWIDTH:", count * dtype.itemsize / mean(min_samples) / (1000 ** 3))
        print("=====================================================================")
