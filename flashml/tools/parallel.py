### begin of file


def parallel_for(
    fromInclusive: int, toExclusive: int, func: callable, num_workers: int = 8
) -> list:
    from joblib import Parallel, delayed

    if fromInclusive >= toExclusive:
        return []

    total = toExclusive - fromInclusive
    chunk_size = max(1, total // (num_workers * 4))

    def process_chunk(start, end):
        return [func(i) for i in range(start, end)]

    chunks = [
        (i, min(i + chunk_size, toExclusive))
        for i in range(fromInclusive, toExclusive, chunk_size)
    ]

    results = Parallel(n_jobs=num_workers, backend="loky", prefer="processes")(
        delayed(process_chunk)(start, end) for start, end in chunks
    )

    return [item for chunk in results for item in chunk]


def parallel_foreach(lst: list, func: callable, num_workers: int = 8) -> list:
    from joblib import Parallel, delayed

    if not lst:
        return []

    chunk_size = max(1, len(lst) // (num_workers * 4))

    def process_chunk(chunk):
        return [func(item) for item in chunk]

    chunks = [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]
    results = Parallel(n_jobs=num_workers, backend="loky", prefer="processes")(
        delayed(process_chunk)(chunk) for chunk in chunks
    )

    return [item for chunk in results for item in chunk]
