import time

def benchmark(func:callable, calls:int=100, verbose:bool=True) -> float:
    '''
    Computed time elapsed to run `func` arg. \\
    Returns the time elapsed in seconds
    '''
    rng = range(calls)
    start = time.time()
    for i in rng:
        func()
    end = time.time() - start

    if verbose:
        print(f"[{func.__name__}] Time elapsed: {end}s")