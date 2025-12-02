from typing import List, Literal, Tuple, Optional, Callable, Any
import concurrent.futures
from collections import deque

class Batch:
    """
    Represents a batch in the processing workflow of BatchIterator.
    """

    def __init__(self, batch_index, num_batches, batch_value, batch_idcs, is_optim_step_time, is_eval_time, is_save_time):
        self.index: int = batch_index
        self.step = (batch_index+1, num_batches)
        self.value = batch_value
        self.ids = batch_idcs
        self.is_optim_step_time = is_optim_step_time
        self.is_eval_time = is_eval_time
        self.is_save_time = is_save_time

    def __iter__(self):
        return iter((self.step, self.value))
    
    def __len__(self):
        return len(self.value)
    
    def size(self):
        """Return the number of samples in the batch. (Batch Size)"""
        return len(self.value)

    def __repr__(self):
        return f"Batch(index={self.index}, is_optim_step_time={self.is_optim_step_time}, eval_time={self.is_eval_time}, save_time={self.is_save_time}, step={self.step}, value_type={type(self.value)})"

class BatchIterator:
    """
    Automatic build batch elements from a dataframe/list for training or testing. Note you can access len(BatchIterator) to get the number of steps/batches
    Examples:
    >>> for batch in BatchIterator(df=train_df, num_epochs=10, batch_size=32, mode="train"):
    ...     # batch.index is the batch index (int), e.g.: 1, 2, 3, 4, ...
    ...     # batch.step is the batch index out of num batches (tuple(step, total_steps)), e.g.: (1,120), (2, 120) ...
    ...     # batch.value is a df (batch_size,) (or a list with batch_size elements), e.g: df[0:32], df[32:64],...
    ...     # batch.ids are the indices of the rows in the batch (batch_size,), e.g: (72, 2817, ... 2182), (2183, 1456, ... 1729), ...
    ...     # batch.is_optim_step_time is True if the batch is used for optimization step, False otherwise. Last batch is always True.
    ...     # batch.is_eval_time is True if the batch is used for evaluation, False otherwise. Last batch is always True.
    ...     # batch.is_save_time is True if the batch is used for saving checkpoints, False otherwise. Last batch is always True.

    Note you can save the state dict (a.k.a. current step of it)
    Args:
        df: DataFrame (Polars or Pandas) or list/tuple of elements
        num_epochs: int, number of epochs to iterate over the dataset
        batch_size: int, size of each batch
        mode: Literal["train", "test"], mode of operation. If "train", batches are shuffled and partial batches are skipped; if "test", batches are sequential and can be partial.
    """

    @staticmethod
    def _generate_batch_ids(
        data_size: int,
        num_epochs: int,
        batch_size: int,
        mode: Literal["train", "test", "eval"]
    ) -> List[Tuple[int, ...]]:
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
        batch_size: int,
        num_epochs: int = 1,
        gradient_accumulation_steps: int = 1,
        mode: Literal["train", "test"] = "train",
        eval_ratio: float = None,
        save_ratio: float = None,
        transform: Optional[Callable[[Any], Any]] = None,
        num_workers: int = 1,
    ):
        """
        Args:
            eval_ratio: (0,1) Ratio of evaluation wrt total batches. (0.2 means 5 evaluations in total during training)
            mode: "train" or "test". If "train", the iterator shuffles the data and skips the last partial batch if incomplete.
            transform: A function that takes a single sample and returns a transformed sample.
                       This is executed on a separate thread.
        """
        assert batch_size >= 1, "Batch size must be a positive integer."
        assert num_epochs >= 1, "Number of epochs must be a positive integer."
        assert len(df) >= batch_size, "Batch size must be smaller than or equal to the length of the dataset."
        assert gradient_accumulation_steps >= 1, "Gradient accumulation steps must be a positive integer."
        
        if mode not in ["train", "test"]:
            raise ValueError("Mode must be either 'train' or 'test'.")
        if mode == "test" and num_epochs != 1:
            raise ValueError("For 'test' mode, num_epochs must be 1.")

        self.df = df
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.mode = mode
        self.eval_ratio = eval_ratio
        self.save_ratio = save_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.transform = transform
        self.num_workers = num_workers
        self.batch_idcs = self._generate_batch_ids(
            len(df), num_epochs, batch_size, mode=mode
        )
        
        self.current_step = 0
        
        self.prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.worker_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        
        self.next_batch_future = None
        
        # Start prefetching batch 0 immediately
        self._schedule_next_batch(0)

    def _process_row(self, idx):
        """Worker method to fetch and transform a single row."""
        raw_sample = self._get_single_item(idx)
        if self.transform:
            return self.transform(raw_sample)
        return raw_sample
    
    def _schedule_next_batch(self, step_index: int):
        if step_index < len(self.batch_idcs):
            # The prefetcher coordinates the batch build
            self.next_batch_future = self.prefetch_executor.submit(self._prepare_batch, step_index)
        else:
            self.next_batch_future = None

    def _get_single_item(self, idx):
        """Helper to safely retrieve a single item from df, polars, or list."""
        if hasattr(self.df, "iloc"):
            return self.df.iloc[idx]
        elif hasattr(self.df, "row"):
            return self.df[idx] 
        else: # List/Tuple
            return self.df[idx]

    def _prepare_batch(self, step_index: int) -> Batch:
        """
        Runs on the prefetch_thread. 
        Distributes the work of processing samples to the worker_pool.
        """
        batch_indices = self.batch_idcs[step_index]

        if self.transform is not None:
            selected_data = list(self.worker_pool.map(self._process_row, batch_indices))
        else:
            if hasattr(self.df, "iloc"):
                selected_data = self.df.iloc[batch_indices]
            elif hasattr(self.df, "get_column"):
                if len(batch_indices) > 2:
                    selected_data = self.df[batch_indices]
                else:
                    import polars as pl
                    selected_data = pl.concat([self.df[i] for i in batch_indices])
            else:
                selected_data = [self.df[i] for i in batch_indices]

        batch_elem = Batch(
            batch_index=step_index,
            num_batches=len(self.batch_idcs),
            batch_value=selected_data,
            batch_idcs=batch_indices,
            is_optim_step_time=self._should_optim_step(step_index),
            is_eval_time=self._should_eval(step_index),
            is_save_time=self._should_save(step_index)
        )
        return batch_elem

    def __next__(self):
        if self.next_batch_future is None and self.current_step >= len(self.batch_idcs):
            raise StopIteration
            
        if self.next_batch_future:
            try:
                batch_elem = self.next_batch_future.result()
            except Exception as e:
                raise e
        else:
            raise StopIteration

        self.current_step += 1
        self._schedule_next_batch(self.current_step)

        return batch_elem

    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.batch_idcs)

    def reset(self):
        self.current_step = 0
        self.next_batch_future = None 
        self._schedule_next_batch(0)
        
    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]
        self._schedule_next_batch(self.current_step)
        
    def size(self):
        """Return the total number of samples in the dataset."""
        return len(self.df)
    
    def get_batch_size(self):
        return self.batch_size

    def _should_eval(self, batch_index: int) -> bool:
        if self.eval_ratio is None or self.eval_ratio <= 0:
            return False
        num_batches = len(self.batch_idcs)
        if batch_index == num_batches - 1:
            return True
        interval = int(num_batches * self.eval_ratio)
        if interval == 0: return False
        return (batch_index + 1) % interval == 0

    def _should_save(self, batch_index: int) -> bool:
        if self.save_ratio is None or self.save_ratio <= 0:
            return False
        num_batches = len(self.batch_idcs)
        if batch_index == num_batches - 1:
            return True
        interval = int(num_batches * self.save_ratio)
        if interval == 0: return False
        return (batch_index + 1) % interval == 0
    
    def _should_optim_step(self, batch_index: int) -> bool:
        if self.gradient_accumulation_steps == 1:
            return True
        num_batches = len(self.batch_idcs)
        if batch_index == num_batches - 1:
            return True
        return (batch_index + 1) % self.gradient_accumulation_steps == 0