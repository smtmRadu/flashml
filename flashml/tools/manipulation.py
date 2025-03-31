from typing import List, Tuple
import random
def shuffle_tensor(torch_tensor, axis):
    size = torch_tensor.size(axis)  # Get size along the axis
    
    # Generate a random permutation using Python's random module
    perm = list(range(size))
    random.shuffle(perm)  # Shuffle in place

    # Convert the permutation to a tensor on the same device as the input tensor
    perm_tensor = torch_tensor.new_tensor(perm).long()

    return torch_tensor.index_select(axis, perm_tensor)

def shuffle_df(df, seed: int | None = None):
    '''
    Shuffles dataframe elements (or element-wise shuffling of multiple dataframes of similar length)
    
    Args:
        df: DataFrame (Polars or Pandas) or list of DataFrames
    Returns:
        Shuffled DataFrame(s)
    '''
    
    if isinstance(df, list):
        if not all(len(d) == len(df[0]) for d in df):
            raise ValueError("All DataFrames in the list must have the same length to maintain row alignment.")
        indices = list(range(len(df[0])))
        random.shuffle(indices)

        return [d[indices] if hasattr(d, "__getitem__") else d.take(indices) for d in df]
    
    if hasattr(df, "sample"):  
        if "shuffle" in df.sample.__code__.co_varnames: 
            return df.sample(fraction=1.0, shuffle=True, seed=seed)
        else:
            return df.sample(frac=1.0, random_state=seed)
    
    raise TypeError("Input must be a Pandas or Polars DataFrame, or a list of them.")



def batch_ranges(data_size:int, batch_size:int, discard_partial_batch:bool = True) -> List[Tuple[int, int]]:
    '''
    This script computes the indices of the batches for a given dataset and batch size.
    Args:
        data_size: int, the length of the dataset.
        batch_size: int, the size of each batch.
        discard_partial_batch: bool, whether to discard the last batch if it is not full. Discard is recommended if the dataset is shuffled every epoch.
    Returns:
        batch_ranges: List[Tuple[int, int]], a list of tuples containing the start and end indices of each batch.
    Examples:
        >>> print(batch_ranges(234, 32, discard_partial_batch=True))
            [(0, 32), (32, 64), (64, 96), (96, 128), (128, 160), (160, 192), (192, 224)]
        >>> print(batch_ranges(234, 32, discard_partial_batch=False))
            [(0, 32), (32, 64), (64, 96), (96, 128), (128, 160), (160, 192), (192, 224), (224, 234)]
        >>> print(batch_ranges(256, 64, discard_partial_batch=True))
            [(0, 64), (64, 128), (128, 192), (192, 256)]
        >>> print(batch_ranges(256, 64, discard_partial_batch=False)) # no effect
            [(0, 64), (64, 128), (128, 192), (192, 256)]  
    '''
    assert batch_size >= 1, "Batch size must be a positive integer."
    assert data_size >= batch_size, "Batch size must be smaller than or equal to the length of the dataset."
    num_batches = data_size // batch_size
    if discard_partial_batch or data_size % batch_size == 0:     
        return [(i*batch_size, (i+1)*batch_size) for i in range(num_batches)]
    else:
        ranges = [(i*batch_size, (i+1)*batch_size) for i in range(num_batches)]
        ranges.append((num_batches*batch_size, data_size))
        return ranges
