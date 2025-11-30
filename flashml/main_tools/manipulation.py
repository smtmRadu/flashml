from typing import List, Tuple, Any, Optional, Sequence


def sample_elementwise(*sequences: Sequence, num_samples: int, with_replacement: bool = False) -> Tuple[List[Any], ...]:
    """
    Sample indices in parallel from multiple sequences and return results as tuple of lists.

    Args:
        *sequences: Two or more sequences of equal length.
        num_samples (int): Number of samples to draw.
        with_replacement (bool): Sample with replacement or not.

    Returns:
        Tuple[List[Any], ...]: A tuple where each element is a list of sampled elements from each input sequence.
    """
    import random 
    if not sequences:
        raise ValueError("At least one sequence must be provided.")
    seq_len = len(sequences[0])
    if any(len(seq) != seq_len for seq in sequences):
        raise ValueError("All sequences must have the same length.")
    if not with_replacement and num_samples > seq_len:
        raise ValueError("Cannot sample more items than available without replacement.")

    # Sample indices
    if with_replacement:
        indices = [random.randint(0, seq_len - 1) for _ in range(num_samples)]
    else:
        indices = random.sample(range(seq_len), num_samples)

    # Gather sampled elements for each sequence (as separate lists)
    sampled_lists = tuple([seq[i] for i in indices] for seq in sequences)
    return sampled_lists



def sample_from(items: Sequence, num_samples, probs: Sequence = None, with_replacement=False):
    """
    Sample n elements from a list with or without replacement, optionally with probabilities.

    Args:
        items (Sequence): The input list to sample from
        num_samples (int): Number of samples to draw
        probs (Sequence, optional): Probabilities associated with each item. Should sum to 1.
        with_replacement (bool): If True, sample with replacement (a.k.a might appear doubles); if False, sample without replacement (unique values). 

    Returns:
        list: A list containing the sampled elements

    Raises:
        ValueError: If num_samples > list length when sampling without replacement,
                    or if probs is not the same length as items or doesn't sum to 1.
    """
    import random
    assert items is not None and len(items) > 0, "Input list cannot be empty"

    if probs is not None:
        if len(probs) != len(items):
            raise ValueError("Length of probs must match length of items.")
        total_prob = sum(probs)
        if not abs(total_prob - 1.0) < 1e-8:
            raise ValueError("Sum of probabilities must be 1.0, got {:.6f}".format(total_prob))

    if not with_replacement:
        if num_samples > len(items):
            raise ValueError(
                f"Cannot sample {num_samples} samples from a distribution of {len(items)}. Consider sampling with replacement or sample less elements."
            )
        if probs is None:
            # Uniform sampling without replacement
            return random.sample(items, num_samples)
        else:
            # Weighted sampling without replacement
            selected = []
            available_items = list(items)
            available_probs = list(probs)
            for _ in range(num_samples):
                # Normalize the remaining probabilities
                total = sum(available_probs)
                normalized_probs = [p / total for p in available_probs]
                idx = random.choices(range(len(available_items)), weights=normalized_probs, k=1)[0]
                selected.append(available_items.pop(idx))
                available_probs.pop(idx)
            return selected
    else:
        if probs is None:
            # Uniform sampling with replacement
            return [random.choice(items) for _ in range(num_samples)]
        else:
            # Weighted sampling with replacement
            return random.choices(items, weights=probs, k=num_samples)



def shuffle_tensor(torch_tensor, axis:int):
    import random

    size = torch_tensor.size(axis)  # Get size along the axis

    # Generate a random permutation using Python's random module
    perm = list(range(size))
    random.shuffle(perm)  # Shuffle in place

    # Convert the permutation to a tensor on the same device as the input tensor
    perm_tensor = torch_tensor.new_tensor(perm).long()

    return torch_tensor.index_select(axis, perm_tensor)

def reorder_df_columns(
    df,
    columns_to_put_first: Optional[List[str]] = None,
    columns_to_put_last: Optional[List[str]] = None
):
    """
    Reorders the columns of a Polars or Pandas DataFrame.
    - columns_to_put_first: columns to appear at the beginning.
    - columns_to_put_last: columns to appear at the end.
      Ensure no overlap between first and last lists.
    Args:
        df (pl.DataFrame or pd.DataFrame): DataFrame to reorder.
        columns_to_put_first (List[str], optional): Columns to appear first.
        columns_to_put_last (List[str], optional): Columns to appear last.
    Returns:
        DataFrame with reordered columns (same type as input).
    """
    if hasattr(df, 'columns'):
        all_cols = list(df.columns)
    else:
        raise TypeError("The input must be a Polars or Pandas DataFrame.")

    columns_to_put_first = columns_to_put_first or []
    columns_to_put_last = columns_to_put_last or []

    # If no changes requested, return original
    if not columns_to_put_first and not columns_to_put_last:
        return df

    for col in columns_to_put_first:
        if col not in all_cols:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

    for col in columns_to_put_last:
        if col not in all_cols:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

    overlap = set(columns_to_put_first) & set(columns_to_put_last)
    if overlap:
        raise ValueError(f"Columns {overlap} are present in both 'columns_to_put_first' and 'columns_to_put_last'.")

    middle_cols = [
        col for col in all_cols
        if col not in columns_to_put_first and col not in columns_to_put_last
    ]
    reordered_cols = columns_to_put_first + middle_cols + columns_to_put_last

    if hasattr(df, 'select'): 
        return df.select(reordered_cols)
    elif hasattr(df, '__getitem__'):
        return df[reordered_cols]
    else:
        raise TypeError("The input must be a Polars or Pandas DataFrame.")
    
def shuffle_df(df):
    """
    Shuffles dataframe elements (or element-wise shuffling of multiple dataframes of similar length).
    The seed is sampled from random class. (so it remains static if you set random.seed())
    Args:
        df: DataFrame (Polars or Pandas) or list of DataFrames
    Returns:
        Shuffled DataFrame(s)
    """
    import random

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
            return df.sample(
                fraction=1.0, shuffle=True, seed=random.randint(0, 100_000_000)
            )
        else:
            return df.sample(frac=1.0, random_state=random.randint(0, 100_000_000))

    raise TypeError("Input must be a Pandas or Polars DataFrame, or a list of them.")


