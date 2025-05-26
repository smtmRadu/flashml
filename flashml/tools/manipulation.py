from typing import List, Tuple, Sequence
import random


def sample_from(items: Sequence, num_samples, with_replacement=False):
    """
    Sample n elements from a list with or without replacement.

    Args:
        items (list | zip): The input list to sample from
        n_samples (int): Number of samples to draw
        with_replacement (bool): If True, sample with replacement; if False, sample without replacement

    Returns:
        list: A list containing the sampled elements

    Raises:
        ValueError: If n_samples is greater than list length when sampling without replacement
    """
    assert items is not None and len(items) > 0, "Input list cannot be empty"

    if not with_replacement:
        if num_samples > len(items):
            raise ValueError(
                "Cannot sample more items than available without replacement"
            )
        return random.sample(items, num_samples)
    else:
        return [random.choice(items) for _ in range(num_samples)]


def shuffle_tensor(torch_tensor, axis):
    size = torch_tensor.size(axis)  # Get size along the axis

    # Generate a random permutation using Python's random module
    perm = list(range(size))
    random.shuffle(perm)  # Shuffle in place

    # Convert the permutation to a tensor on the same device as the input tensor
    perm_tensor = torch_tensor.new_tensor(perm).long()

    return torch_tensor.index_select(axis, perm_tensor)


def shuffle_df(df, seed: int | None = None):
    """
    Shuffles dataframe elements (or element-wise shuffling of multiple dataframes of similar length)

    Args:
        df: DataFrame (Polars or Pandas) or list of DataFrames
    Returns:
        Shuffled DataFrame(s)
    """

    if isinstance(df, list):
        if not all(len(d) == len(df[0]) for d in df):
            raise ValueError(
                "All DataFrames in the list must have the same length to maintain row alignment."
            )
        indices = list(range(len(df[0])))
        random.shuffle(indices)

        return [
            d[indices] if hasattr(d, "__getitem__") else d.take(indices) for d in df
        ]

    if hasattr(df, "sample"):
        if "shuffle" in df.sample.__code__.co_varnames:
            return df.sample(fraction=1.0, shuffle=True, seed=seed)
        else:
            return df.sample(frac=1.0, random_state=seed)

    raise TypeError("Input must be a Pandas or Polars DataFrame, or a list of them.")


def generate_batches(
    data_size: int, num_epochs: int, batch_size: int, shuffle: bool = False
) -> List[Tuple]:
    """
    This script computes the indices of the batches for a given dataset, with respect to the number of epochs and batch size.
    You can directly pass through the return as from a dataloader for all epochs.
    Args:
        data_size: int, the length of the dataset.
        num_epochs: int, the number of epochs to iterate over the dataset.
        batch_size: int, the size of each batch.
        shuffle: bool, whether to shuffle the dataset before iterating over it (no doubles will be in the same batch).

    Example:
        >>> print(batch_indices(21, 2, 4))
            [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15), (16, 17, 18, 19), (20, 0, 1, 2), (3, 4, 5, 6), (7, 8, 9, 10), (11, 12, 13, 14), (15, 16, 17, 18)]

    Returns:
        List[List[int]]: a list of lists containing the indices of each batch.
    """
    assert batch_size >= 1, "Batch size must be a positive integer."
    assert num_epochs >= 1, "Number of epochs must be a positive integer."
    assert data_size >= batch_size, (
        "Batch size must be smaller than or equal to the length of the dataset."
    )
    all_batches = []

    for _ in range(num_epochs):
        indices = list(range(data_size))
        if shuffle:
            random.shuffle(indices)
        for i in range(0, data_size, batch_size):
            batch = indices[i : i + batch_size]
            if len(batch) < batch_size:
                batch += indices[: batch_size - len(batch)]
            all_batches.append(tuple(batch))

    return all_batches
