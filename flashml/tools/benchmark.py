import time
import numpy as np


def stress_gpu(duration_seconds=60, matrix_size=8192):
    import torch

    if not torch.cuda.is_available():
        print("No GPU detected!")
        return

    print(f"Starting GPU stress test for {duration_seconds} seconds...")
    device = torch.device("cuda")

    matrix1 = torch.rand(matrix_size, matrix_size).to(device)
    matrix2 = torch.rand(matrix_size, matrix_size).to(device)

    start_time = time.time()
    operations = 0

    while time.time() - start_time < duration_seconds:
        _ = torch.matmul(matrix1, matrix2)
        operations += 1
        torch.cuda.synchronize()

    print(f"Completed {operations} matrix multiplications")
    print("Stress test finished!")


def _cpu_stress_worker(matrix_size):
    matrix1 = np.random.rand(matrix_size, matrix_size)
    matrix2 = np.random.rand(matrix_size, matrix_size)
    while True:
        _ = np.matmul(matrix1, matrix2)


def stress_cpu(duration_seconds=60, matrix_size=2048):
    import multiprocessing

    print(f"Starting CPU stress test for {duration_seconds} seconds...")

    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores")

    processes = []
    for _ in range(num_cores):
        p = multiprocessing.Process(target=_cpu_stress_worker, args=(matrix_size,))
        processes.append(p)
        p.start()

    time.sleep(duration_seconds)

    for p in processes:
        p.terminate()
        p.join()

    print("CPU stress test finished!")


def benchmark(func: callable, calls: int = 1, verbose: bool = True) -> float:
    """
    Computed time elapsed to run `func` arg. \\
    Returns the time elapsed in seconds
    """
    rng = range(calls)
    start = time.time()
    for i in rng:
        func()
    end = time.time() - start

    if verbose:
        print(f"[{func.__name__} X {calls} times] Time elapsed: {end}s")
