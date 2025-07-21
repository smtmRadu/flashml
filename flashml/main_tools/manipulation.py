from typing import List, Literal, Sequence, Tuple, Any, Optional

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

from typing import Sequence
import random

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



def shuffle_tensor(torch_tensor, axis):
    import random

    size = torch_tensor.size(axis)  # Get size along the axis

    # Generate a random permutation using Python's random module
    perm = list(range(size))
    random.shuffle(perm)  # Shuffle in place

    # Convert the permutation to a tensor on the same device as the input tensor
    perm_tensor = torch_tensor.new_tensor(perm).long()

    return torch_tensor.index_select(axis, perm_tensor)

def reorder_columns_df(
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


class Batch:
    """
    Represents a batch in the processing workflow of BatchIterator.

    :ivar (int) id: The id of the batch.
    :ivar (tuple) step: The step information as a tuple (batch_id, num_batches).
    :ivar value: The value associated with the batch.
    :ivar (list) ids: The indices associated with the batch.
    """

    def __init__(self, batch_id, num_batches, batch_value, batch_idcs):
        self.id: int = batch_id
        self.step = (batch_id, num_batches)
        self.value = batch_value
        self.ids = batch_idcs

    def __iter__(self):
        return iter((self.step, self.value))


class BatchIterator:
    """
    Automatically build batch elements from a dataframe for training or testing. Note you can access len(B: BatchIterator) to get the number of steps/batches
    Examples:
    >>> for batch in BatchIterator(df=train_df, num_epochs=10, batch_size=32, mode="train"):
    ...     # or you can just unpack (for step, batch in BatchIterator(...))
    ...     # batch.id is the batch index (int)
    ...     # batch.step is the batch index out of num batches (tuple(current_step, total_steps))
    ...     # batch.value is a df (batch_size,) (or a list with batch_size elements)
    ...     # batch.ids are the indices of the rows in the batch (batch_size,)

    Note you can save the state dict (a.k.a. current step of it)
    Args:
        df: DataFrame (Polars or Pandas) or list/tuple of elements
        num_epochs: int, number of epochs to iterate over the dataset
        batch_size: int, size of each batch
        mode: Literal["train", "test"], mode of operation. If "train", batches are shuffled and partial batches are skipped; if "test", batches are sequential and can be partial.

    """

    @staticmethod
    def generate_batches(
        data_size: int,
        num_epochs: int,
        batch_size: int,
        mode: Literal["train", "test", "eval"],
    ) -> List[Tuple[int, ...]]:
        """
        This script computes the indices of the batches for a given dataset, with respect to the number of epochs and batch size.
        You can directly pass through the return as from a dataloader for all epochs.
        Args:
            data_size: int, the length of the dataset.
            num_epochs: int, the number of epochs to iterate over the dataset.
            batch_size: int, the size of each batch.
            mode: str, "train", "test" or "eval", whether to generate batches for training, testing or evaluation. If "train", partial batches are skipped and everything is shuffled.

        Example:
            >>> print(generate_batches(21, num_epochs=2, batch_size=4, mode="train"))
                [(14, 13, 8, 15), (10, 11, 9, 2), (17, 4, 20, 6), (19, 3, 5, 0), (16, 18, 12, 1), (7, 15, 13, 7), (6, 2, 10, 19), (17, 5, 0, 9), (16, 18, 14, 20), (3, 11, 12, 4), (1, 8, 14, 13)]
            >>> print(generate_batches(21, num_epochs=2, batch_size=4, mode="test"))
                [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15), (16, 17, 18, 19), (20, 0, 1, 2), (3, 4, 5, 6), (7, 8, 9, 10), (11, 12, 13, 14), (15, 16, 17, 18), (19, 20)]
        Returns:
            List[Tuple[int,]]: a list of tuples containing the indices of each element in the batches.
        """
        import random

        assert batch_size >= 1, "Batch size must be a positive integer."
        assert num_epochs >= 1, "Number of epochs must be a positive integer."
        assert data_size >= batch_size, (
            "Batch size must be smaller than or equal to the length of the dataset."
        )

        if mode not in ["train", "test", "eval"]:
            raise ValueError("Mode must be either 'train', 'test', or 'eval'.")

        if mode in ["test", "eval"] and num_epochs != 1:
            raise ValueError("For 'test' mode, num_epochs must be 1.")

        shuffle = True if mode == "train" else False
        skip_partial_batch = True if mode == "train" else False

        stream: List[int] = []
        for _ in range(num_epochs):
            epoch_inds = list(range(data_size))
            if shuffle:
                random.shuffle(epoch_inds)
            stream.extend(epoch_inds)

        all_batches: List[Tuple[int, ...]] = []
        total = len(stream)
        for start in range(0, total, batch_size):
            batch = stream[start : start + batch_size]
            if len(batch) == batch_size:
                all_batches.append(tuple(batch))
            else:
                if skip_partial_batch:
                    pad_needed = batch_size - len(batch)
                    batch.extend(stream[:pad_needed])
                    all_batches.append(tuple(batch))
                else:
                    all_batches.append(tuple(batch))
                break

        return all_batches

    def __init__(
        self,
        df,
        num_epochs: int,
        batch_size: int,
        mode: Literal["train", "test"] = "train",
    ):
        assert batch_size >= 1, "Batch size must be a positive integer."
        assert num_epochs >= 1, "Number of epochs must be a positive integer."
        assert len(df) >= batch_size, (
            "Batch size must be smaller than or equal to the length of the dataset."
        )
        if mode not in ["train", "test"]:
            raise ValueError("Mode must be either 'train' or 'test'.")

        if mode == "test" and num_epochs != 1:
            raise ValueError("For 'test' mode, num_epochs must be 1.")

        self.df = df
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.mode = mode
        self.batch_idcs = self.generate_batches(
            len(df), num_epochs, batch_size, mode=mode
        )
        self.current_step = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_step >= len(self.batch_idcs):
            raise StopIteration

        batch_indices = self.batch_idcs[self.current_step]

        if hasattr(self.df, "iloc"):
            selected_data = self.df.iloc[batch_indices]
        elif hasattr(self.df, "__getitem__") and hasattr(batch_indices, "__iter__"):
            selected_data = (
                self.df[batch_indices]
                if hasattr(self.df, "shape")
                else [self.df[i] for i in batch_indices]
            )
        else:
            selected_data = [self.df[i] for i in batch_indices]

        batch_elem = Batch(
            batch_id=self.current_step,
            num_batches=len(self.batch_idcs),
            batch_value=selected_data,
            batch_idcs=batch_indices,
        )

        self.current_step += 1
        return batch_elem

    def __len__(self):
        return len(self.batch_idcs)

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]
