from typing import List, Literal,Tuple

class Batch:
    """
    Represents a batch in the processing workflow of BatchIterator.

    :ivar (int) index: The index of the batch.
    :ivar (tuple) step: The step information as a tuple (batch_id, num_batches).
    :ivar value: The value associated with the batch.
    :ivar (list) ids: The indices associated with the batch.
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
        """Returns the number of elements in the batch.
        """
        return len(self.value)
    
    def get_num_samples(self):
        """Returns the number of elements in the batch."""
        return len(self.value)

    def __repr__(self):
        return f"Batch(index={self.index}, is_optim_step_time={self.is_optim_step_time}, eval_time={self.is_eval_time}, save_time={self.is_save_time}, step={self.step}, value={self.value}, ids={self.ids})"

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
        """
        This script computes the indices of the batches for a given dataset, with respect to the number of epochs and batch size.
        You can directly pass through the return as from a dataloader for all epochs.
        Args:
            data_size: int, the length of the dataset.
            num_epochs: int, the number of epochs to iterate over the dataset.
            batch_size: int, the size of each batch.
            mode: str, "train", "test" or "eval", whether to generate batches for training, testing or evaluation. If "train", last batch is skipped (if partial) and everything is shuffled.

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
        gradient_accumulation_steps: int = 1, # default 1
        mode: Literal["train", "test"] = "train",
        eval_ratio: float = None,
        save_ratio: float = None,
    ):
        """
        df: list or DataFrame
        mode: "train" or "test". In train mode, batches are shuffled and partial batches are skipped. In test mode, batches are sequential and can be partial.
        """
        assert batch_size >= 1, "Batch size must be a positive integer."
        assert num_epochs >= 1, "Number of epochs must be a positive integer."
        assert len(df) >= batch_size, (
            "Batch size must be smaller than or equal to the length of the dataset."
        )
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
        self.batch_idcs = self._generate_batch_ids(
            len(df), num_epochs, batch_size, mode=mode
        )
        self.current_step = 0

    

    def __next__(self):
        if self.current_step >= len(self.batch_idcs):
            raise StopIteration

        batch_indices = self.batch_idcs[self.current_step]

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
            batch_index=self.current_step,
            num_batches=len(self.batch_idcs),
            batch_value=selected_data,
            batch_idcs=batch_indices,
            is_optim_step_time=self._should_optim_step(self.current_step),
            is_eval_time=self._should_eval(self.current_step),
            is_save_time=self._should_save(self.current_step)
        )

        self.current_step += 1
        return batch_elem

    def __iter__(self):
            return self
    
    def __len__(self):
        """Returns the number of batches.
        """
        return len(self.batch_idcs)

    def reset(self):
        """Resets the current position of the iterator to the beginning."""
        self.current_step = 0
        
    def state_dict(self):
        ## we only save the current step because the object is already initialized by the same script.
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]
        
    def get_num_samples(self):
        """
        Return the total number of samples.
        """
        return len(self.df)
    
    def get_batch_size(self):
        return self.batch_size

    def _should_eval(self, batch_index: int) -> bool:
        """Check if the current batch should be marked for evaluation."""
        if self.eval_ratio is None or self.eval_ratio <= 0:
            return False
        
        num_batches = len(self.batch_idcs)
        
        # Always eval at the last batch
        if batch_index == num_batches - 1:
            return True
        
        # Calculate interval between evals
        interval = int(num_batches * self.eval_ratio)
        if interval == 0:
            return False
        
        # Mark at indices: interval-1, 2*interval-1, 3*interval-1, etc.
        return (batch_index + 1) % interval == 0

    def _should_save(self, batch_index: int) -> bool:
        """Check if the current batch should be marked for checkpoint saving."""
        if self.save_ratio is None or self.save_ratio <= 0:
            return False
        
        num_batches = len(self.batch_idcs)
        
        # Always save at the last batch
        if batch_index == num_batches - 1:
            return True
        
        # Calculate interval between saves
        interval = int(num_batches * self.save_ratio)
        if interval == 0:
            return False
        
        # Mark at indices: interval-1, 2*interval-1, 3*interval-1, etc.
        return (batch_index + 1) % interval == 0
    
    
    def _should_optim_step(self, batch_index: int) -> bool:
        """Check if the current batch should perform an optimizer step."""
        if self.gradient_accumulation_steps == 1:
            return True
        
        num_batches = len(self.batch_idcs)
        
        # Always step at the last batch
        if batch_index == num_batches - 1:
            return True
        
        # Mark at indices: gradient_accumulation_steps-1, 2*gradient_accumulation_steps-1, etc.
        return (batch_index + 1) % self.gradient_accumulation_steps == 0