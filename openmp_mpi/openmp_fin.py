import numpy as np
import time
from numba import njit, prange
import csv

open_mpresults = "openmp_results.csv"

def now():
    return time.perf_counter()

@njit(parallel=True)
def psum(a, bs):
    n = len(a)
    nb = (n+bs-1) // bs
    block_sums = np.zeros(nb, a.dtype)
    for b in prange(nb):
        s = b*bs
        e = min(s+bs,n)
        tmp = 0
        for i in range(s, e):
            tmp += a[i]
        block_sums[b] = tmp
    total = 0
    for i in range(nb):
        total += block_sums[i]
    return total

@njit(parallel=True)
def pmin(a, bs):
    n = len(a)
    nb = (n+bs- 1) // bs
    block_mins = np.empty(nb, a.dtype)
    for b in prange(nb):
        s = b*bs
        e = min(s+bs,n)
        local_min = a[s]
        for i in range(s+1,e):
            if a[i] < local_min:
                local_min = a[i]
        block_mins[b] = local_min
    out = block_mins[0]
    for i in range(1, nb):
        if block_mins[i] < out:
            out = block_mins[i]
    return out

@njit(parallel=True)
def pmax(a, bs):
    n = len(a)
    nb = (n+bs-1) // bs
    block_maxs = np.empty(nb, a.dtype)
    for b in prange(nb):
        s = b*bs
        e = min(s+bs,n)
        local_max = a[s]
        for i in range(s+ 1,e):
            if a[i] > local_max:
                local_max = a[i]
        block_maxs[b] = local_max
    out = block_maxs[0]
    for i in range(1, nb):
        if block_maxs[i] > out:
            out = block_maxs[i]
    return out

@njit
def blelloch(x):
    n = len(x)
    m = 1
    while m < n:
        m <<= 1
    t = np.zeros(m, x.dtype)
    t[:n] = x
    d=1
    while d < m:
        for i in range(0,m,2*d):
            t[i+2*d-1] += t[i+d-1]
        d <<= 1
    t[m-1] = 0
    d = m >> 1
    while d:
        for i in range(0,m,2*d):
            v = t[i+d-1]
            t[i+d-1] = t[i+2*d-1]
            t[i+2*d-1] += v
        d >>= 1
    x[:] = t[:n]

@njit(parallel=True)
def scan(a, bs):
    n = len(a)
    orig = a.copy()
    nb = (n+bs-1) // bs
    totals = np.zeros(nb, a.dtype)
    for b in prange(nb):
        s = b*bs
        e = min(s+bs, n)
        if s < e:
            buf = orig[s:e].copy()
            blelloch(buf)
            a[s:e] = buf
            totals[b] = buf[-1] + orig[e - 1]
    offs = np.zeros(nb, a.dtype)
    acc = 0
    for i in range(nb):
        offs[i] = acc
        acc += totals[i]
    for b in prange(nb):
        s = b * bs
        e = min(s+bs, n)
        o = offs[b]
        for i in range(s, e):
            a[i] += o
    for i in prange(n):
        a[i] += orig[i]

def warmup():
    dummy = np.ones(1024, dtype=np.int64)
    psum(dummy, 256)
    pmin(dummy, 256)
    pmax(dummy, 256)
    scan(dummy, 256)

def bench(dtype, n, block_sizes):
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.floating):
        a = rng.uniform(-100, 100, n).astype(dtype)
    else:
        a = rng.integers(-1000, 1000, n, dtype=dtype)
    size = a.dtype.itemsize

    seq_sum = a.sum()
    seq_min = a.min()
    seq_max = a.max()
    seq_scan = np.cumsum(a)

    with open(open_mpresults, "a", newline="") as f:
        w = csv.writer(f)
        
        for bs in block_sizes:
            bs_lbl = f"BS_{bs}"
            
            t0 = now()
            r = psum(a, bs)
            dt = now() - t0
            err = abs(r - seq_sum)
            gbs = (n * size) / (dt * 1e9)
            print(f"{dtype},{n},{bs_lbl},SUM,{r},{seq_sum},{err},{dt},{gbs}")
            w.writerow([str(dtype), n, bs_lbl, "SUM", r, seq_sum, err, dt, gbs])

            t0 = now()
            r = pmin(a, bs)
            dt = now() - t0
            err = abs(r - seq_min)
            gbs = (n * size) / (dt * 1e9)
            print(f"{dtype},{n},{bs_lbl},MIN,{r},{seq_sum},{err},{dt},{gbs}")
            w.writerow([str(dtype), n, bs_lbl, "MIN", r, seq_min, err, dt, gbs])

            t0 = now()
            r = pmax(a, bs)
            dt = now() - t0
            err = abs(r - seq_max)
            gbs = (n * size) / (dt * 1e9)
            print(f"{dtype},{n},{bs_lbl},MAX,{r},{seq_sum},{err},{dt},{gbs}")
            w.writerow([str(dtype), n, bs_lbl, "MAX", r, seq_max, err, dt, gbs])

            b = a.copy()
            t0 = now()
            scan(b, bs)
            dt = now() - t0
            err = np.max(np.abs(b - seq_scan))
            gbs = (2 * n * size) / (dt * 1e9)
            
            par_res = b[-1]
            seq_res = seq_scan[-1]
            
            print(f"{dtype},{n},{bs_lbl},SCAN,{r},{seq_sum},{err},{dt},{gbs}")
            w.writerow([str(dtype), n, bs_lbl, "SCAN", par_res, seq_res, err, dt, gbs])

def main():
    with open(open_mpresults, "w", newline="") as f:
        csv.writer(f).writerow(["dtype", "size", "block_size", "op", "parallel", "seq", "abs_err", "time", "GBs"])
    
    warmup()

    types = [np.int64, np.float64]
    sizes = [1024, 1048576, 16777216] 
    test_block_sizes = [4096, 32768, 262144]

    print("dtype,size,block_size,op,parallel,seq,abs_err,time,GBs")
    for t in types:
        for n in sizes:
            bench(t, n, test_block_sizes)

if __name__ == "__main__":
    main()