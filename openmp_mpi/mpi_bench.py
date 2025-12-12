import numpy as np
import time
import csv
from mpi4py import MPI

mpi_results = "mpi_results.csv"

def now():
    return time.perf_counter()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size_world = comm.Get_size()

def ensure_divisible(n):
    return n % size_world == 0

def local_sum_tiled(a, bs):
    total = 0
    for i in range(0, len(a), bs):
        chunk = a[i:i+bs]
        total += chunk.sum()
    return total

def local_min_tiled(a, bs):
    current_min = a[0]
    for i in range(0, len(a), bs):
        chunk_min = a[i:i+bs].min()
        if chunk_min < current_min:
            current_min = chunk_min
    return current_min

def local_max_tiled(a, bs):
    current_max = a[0]
    for i in range(0, len(a), bs):
        chunk_max = a[i:i+ bs].max()
        if chunk_max > current_max:
            current_max = chunk_max
    return current_max

def local_scan_tiled(a, bs):
    res = np.empty_like(a)
    acc = 0
    for i in range(0, len(a), bs):
        chunk = a[i:i+bs]
        chunk_scan = np.cumsum(chunk)
        res[i:i+ bs] = chunk_scan + acc
        acc += chunk_scan[-1]
    return res

def bench_mpi(dtype, n, block_sizes):
    if rank == 0:
        rng = np.random.default_rng(0)
        if np.issubdtype(dtype, np.floating):
            full = rng.uniform(-100, 100, n).astype(dtype)
        else:
            full = rng.integers(-1000, 1000, n, dtype=dtype)
        seq_sum = full.sum()
        seq_min = full.min()
        seq_max = full.max()
        seq_scan = np.cumsum(full)
    else:
        full = None
        seq_sum = seq_min = seq_max = seq_scan = None

    itemsize = np.dtype(dtype).itemsize
    block_len = n // size_world
    local = np.empty(block_len, dtype=dtype)

    if rank == 0:
        sendcounts = [block_len] * size_world
        displs = [i * block_len for i in range(size_world)]
    else:
        sendcounts = None
        displs = None

    comm.Scatterv([full, sendcounts, displs, MPI._typedict[np.dtype(dtype).char]] if rank == 0 else None, local, root=0)

    if rank == 0:
        f = open(mpi_results, "a", newline="")
        w = csv.writer(f)

    for bs in block_sizes:
        bs_lbl = f"BS_{bs}"
        
        #SUM
        comm.Barrier()
        t0 = now()
        local_val = local_sum_tiled(local, bs)
        global_sum = comm.allreduce(local_val, op=MPI.SUM)
        dt = now() - t0
        if rank == 0:
            err = abs(global_sum - seq_sum)
            gbs = (n * itemsize) / (dt * 1e9) if dt > 0 else 0.0
            print(f"{dtype},{n},{bs_lbl},SUM,{global_sum},{seq_sum},{err},{dt},{gbs}")
            w.writerow([str(dtype), n, bs_lbl, "SUM", global_sum, seq_sum, err, dt, gbs])

        #MIN
        comm.Barrier()
        t0 = now()
        local_val = local_min_tiled(local, bs)
        global_min = comm.allreduce(local_val, op=MPI.MIN)
        dt = now() - t0
        if rank == 0:
            err = abs(global_min - seq_min)
            gbs = (n * itemsize) / (dt * 1e9) if dt > 0 else 0.0
            print(f"{dtype},{n},{bs_lbl},MIN,{global_sum},{seq_sum},{err},{dt},{gbs}")
            w.writerow([str(dtype), n, bs_lbl, "MIN", global_min, seq_min, err, dt, gbs])

        # MAX
        comm.Barrier()
        t0 = now()
        local_val = local_max_tiled(local, bs)
        global_max = comm.allreduce(local_val, op=MPI.MAX)
        dt = now() - t0
        if rank == 0:
            err = abs(global_max - seq_max)
            gbs = (n * itemsize) / (dt * 1e9) if dt > 0 else 0.0
            print(f"{dtype},{n},{bs_lbl},MAX,{global_sum},{seq_sum},{err},{dt},{gbs}")
            w.writerow([str(dtype), n, bs_lbl, "MAX", global_max, seq_max, err, dt, gbs])

        # SCAN
        if ensure_divisible(n):
            comm.Barrier()
            t0 = now()
            
            local_scan = local_scan_tiled(local, bs)
            local_total = local_scan[-1]

            if size_world > 1:
                offset = comm.exscan(local_total)
                if rank == 0 or offset is None:
                    offset = 0
            else:
                offset = 0

            if offset != 0:
                local_scan += offset
            
            local_scan = local_scan.astype(dtype)
            
            dt_compute = now() - t0

            recvbuf = np.empty(n, dtype=dtype) if rank == 0 else None
            recvcounts = [block_len] * size_world
            displs = [i * block_len for i in range(size_world)]

            comm.Gatherv([local_scan, MPI._typedict[np.dtype(dtype).char]],
                         [recvbuf, recvcounts, displs, MPI._typedict[np.dtype(dtype).char]],
                         root=0)

            dt_total = now() - t0
            if rank == 0:
                err = np.max(np.abs(recvbuf - seq_scan))
                gbs = (2 * n * itemsize) / (dt_total * 1e9) if dt_total > 0 else 0.0
                
                par_res = recvbuf[-1]
                seq_res = seq_scan[-1]
                
                print(f"{dtype},{n},{bs_lbl},SCAN,{global_sum},{seq_sum},{err},{dt},{gbs}")
                w.writerow([str(dtype), n, bs_lbl, "SCAN", par_res, seq_res, err, dt, gbs])

    if rank == 0:
        f.close()

def main():
    if rank == 0:
        with open(mpi_results, "w", newline="") as f:
            csv.writer(f).writerow(["dtype", "size", "block_size", "op", "parallel", "seq", "abs_err", "time", "GBs"])
        print("dtype,size,block_size,op,parallel,seq,abs_err,time,GBs")
    
    comm.Barrier()

    types = [np.int64, np.float64]
    sizes = [1024, 1048576, 16777216]
    test_block_sizes = [4096, 32768, 262144]

    for t in types:
        for n in sizes:
            bench_mpi(t, n, test_block_sizes)
            comm.Barrier()

if __name__ == "__main__":
    main()