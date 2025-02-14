import time

def benchmark_func(func:callable, verbose:bool=True) -> float:
    '''
    Computed time elapsed to run `func` arg. \\
    Returns the time elapsed in seconds.
    '''
    start = time.time()
    func()
    end = time.time() - start

    if verbose:
        print(f"[{func.__name__}] Time elapsed: {end}s")