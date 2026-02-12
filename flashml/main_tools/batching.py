from typing import List, Literal, Tuple, Optional, Callable, Any
import torch
from torch.utils.data import Dataset, DataLoader


def _is_notebook() -> bool:
    """Detect if running in a Jupyter notebook or similar interactive environment."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except (NameError, ImportError):
        return False  # Probably standard Python interpreter

class Batch:
    """
    Represents a batch in the processing workflow of BatchIterator.
    """

    def __init__(self, batch_index, batch_value, batch_idcs, is_optim_step_time, is_eval_time, is_save_time, optim_step_index, total_optim_steps):
        self.index: int = batch_index
        self.optim_step = (optim_step_index, total_optim_steps) if is_optim_step_time is True else None
        self.value = batch_value
        self.ids = batch_idcs
        self.is_optim_step_time = is_optim_step_time
        self.is_eval_time = is_eval_time
        self.is_save_time = is_save_time

    def __len__(self):
        return len(self.value)
    
    def size(self):
        """Return the number of samples in the batch. (Batch Size)"""
        return len(self.value)
    
    def __iter__(self):
        return iter(self.optim_step)
    
    def __repr__(self):
        return f"Batch(index={self.index}, size={self.size()}, optim_step={self.optim_step}, is_optim_step_time={self.is_optim_step_time}, is_eval_time={self.is_eval_time}, is_save_time={self.is_save_time}, value:{type(self.value).__name__}={len(self.value)} samples)"

    def __str__(self):
        return self.__repr__()

class _InternalDataset(Dataset):
    """Internal Dataset wrapper for PyTorch DataLoader."""
    
    def __init__(self, data, transform=None):
        # Check if it's already a PyTorch Dataset
        if isinstance(data, Dataset):
            self.data = data
            self.data_type = "pytorch_dataset"
            self.is_wrapper = True
        # Convert DataFrame to list for better pickling compatibility
        elif hasattr(data, "to_dicts"):  # Polars - check this FIRST
            self.data = data.to_dicts()
            self.data_type = "dict"
            self.is_wrapper = False
        elif hasattr(data, "to_dict"):  # Pandas
            self.data = data.to_dict('records')
            self.data_type = "dict"
            self.is_wrapper = False
        else:  # Already a list/tuple
            self.data = list(data)
            self.data_type = "list"
            self.is_wrapper = False
        
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single item with optional transform."""
        # If wrapping a PyTorch Dataset, get item from it
        if self.is_wrapper:
            sample = self.data[idx]
        else:
            sample = self.data[idx]
        
        # Apply transform only if not wrapping a PyTorch Dataset (assume dataset handles its own transforms)
        if self.transform and not self.is_wrapper:
            try:
                sample = self.transform(sample)
            except Exception as e:
                raise RuntimeError(
                    f"Transform failed on sample {idx}. "
                    f"Make sure your transform function is defined at module level "
                    f"(not as a lambda or inside a class/function). Error: {e}"
                )
        
        return sample, idx  # Return both sample and index


def _collate_with_indices(batch):
    """Custom collate function that preserves indices."""
    samples, indices = zip(*batch)
    return list(samples), list(indices)


class BatchIterator:
    """
    Automatic build batch elements from data (dataframe/list/PyTorch Dataset) for training or testing.
    Now uses PyTorch DataLoader as backend for true parallel preprocessing.

Examples:
>>> for batch in BatchIterator(data=train_data, num_epochs=10, batch_size=32, mode="train"):
...     # batch.index is the batch index (int), e.g.: 0, 1, 2, 3, ...
...     # batch.optim_step is the optimization step out of total optim steps (tuple(step, total_steps)), e.g.: (1,60), (1,60), (2,60), (2,60) ... (if grad_accum=2)
...     # batch.value is a list with batch_size elements
...     # batch.ids are the indices of the rows in the batch (list), e.g: [72, 2817, ... 2182], [2183, 1456, ... 1729], ...
...     # batch.is_optim_step_time is True if the batch is used for optimization step
...     # batch.is_eval_time is True if the batch is used for evaluation
...     # batch.is_save_time is True if the batch is used for saving checkpoints

    """

    def __init__(
        self,
        data,
        batch_size: int,
        num_epochs: int = 1,
        gradient_accumulation_steps: int = 1,
        mode: Literal["train", "test", "eval", "val"] = "train",
        eval_ratio: float = None,
        save_ratio: float = None,
        transform: Optional[Callable[[Any], Any]] = None,
        num_workers: int = None,  # Auto-detect if None
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = None,
    ):
        """
        Args:
            data: DataFrame (Polars or Pandas), PyTorch Dataset, or list/tuple of elements
            batch_size: Size of each batch
            num_epochs: Number of epochs to iterate over the dataset
            gradient_accumulation_steps: Number of steps to accumulate gradients
            mode: "train" or "test". If "train", shuffles and drops last incomplete batch
            eval_ratio: (0,1) Ratio of evaluation wrt total batches
            save_ratio: (0,1) Ratio of saving checkpoints wrt total batches
            transform: A function that takes a single sample and returns a transformed sample
                       (executed in parallel workers). Must be picklable (defined at module level).
                       Note: Ignored if data is already a PyTorch Dataset (assumes dataset handles transforms)
            num_workers: Number of worker processes for data loading. 
                        - None = auto-detect (0 for notebooks, 4 for scripts)
                        - 0 = single process (safest for notebooks)
                        - >0 = multiprocessing (requires proper setup)
            pin_memory: If True, use pinned memory for faster GPU transfer
            prefetch_factor: Number of batches to prefetch per worker
            persistent_workers: Keep workers alive between epochs (auto-enabled if num_workers > 0)
        """
        assert batch_size >= 1, "Batch size must be a positive integer."
        assert num_epochs >= 1, "Number of epochs must be a positive integer."
        assert len(data) >= batch_size, "Batch size must be smaller than or equal to the length of the dataset."
        assert gradient_accumulation_steps >= 1, "Gradient accumulation steps must be a positive integer."
        
        if mode not in ["train", "test", "eval", "val"]:
            raise ValueError(f"Unknown mode type ({mode}).")
        if mode in ["test", "eval", "val"] and num_epochs != 1:
            raise ValueError("For 'test'/'eval'/'val' modes, num_epochs must be 1.")

        # Auto-detect num_workers based on environment
        if num_workers is None:
            if _is_notebook():
                num_workers = 0
                print("[⚠️ BatchIterator] Detected Jupyter notebook environment. Setting num_workers=0 for compatibility.")
            else:
                num_workers = 0 # Safe default for scripts

        self.data = data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.mode = mode
        self.eval_ratio = eval_ratio
        self.save_ratio = save_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.transform = transform
        self.num_workers = num_workers
        
        # Auto-enable persistent workers if using multiprocessing
        if persistent_workers is None:
            persistent_workers = num_workers > 0
        
        # Create internal dataset (wraps PyTorch datasets or converts data to dataset)
        self.dataset = _InternalDataset(data, transform=transform)
        
        # Calculate total batches per epoch
        drop_last = (mode == "train")
        batches_per_epoch = len(data) // batch_size
        if not drop_last and len(data) % batch_size != 0:
            batches_per_epoch += 1
        self.total_batches = batches_per_epoch * num_epochs
        self.total_optim_steps = self.total_batches // gradient_accumulation_steps
        if self.total_batches % gradient_accumulation_steps != 0:
            self.total_optim_steps += 1
        
        # Create DataLoader with appropriate settings
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(mode == "train"),
            num_workers=num_workers,
            collate_fn=_collate_with_indices,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=drop_last,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )
        
        self.current_batch = 0
        self.dataloader_iter = None
        self._epoch = 0
        
    def _should_eval(self, batch_index: int) -> bool:
        if self.eval_ratio is None or self.eval_ratio <= 0:
            return False
        if batch_index == self.total_batches - 1:
            return True
        interval = int(self.total_batches * self.eval_ratio)
        if interval == 0: 
            return False
        return (batch_index + 1) % interval == 0

    def _should_save(self, batch_index: int) -> bool:
        if self.save_ratio is None or self.save_ratio <= 0:
            return False
        if batch_index == self.total_batches - 1:
            return True
        interval = int(self.total_batches * self.save_ratio)
        if interval == 0: 
            return False
        return (batch_index + 1) % interval == 0
    
    def _should_optim_step(self, batch_index: int) -> bool:
        if self.gradient_accumulation_steps == 1:
            return True
        if batch_index == self.total_batches - 1:
            return True
        return (batch_index + 1) % self.gradient_accumulation_steps == 0

    def __next__(self):
        if self.current_batch >= self.total_batches:
            raise StopIteration
        
        # Initialize iterator if needed
        if self.dataloader_iter is None:
            self.dataloader_iter = iter(self.dataloader)
        
        try:
            batch_value, batch_idcs = next(self.dataloader_iter)
        except StopIteration:
            # Start new epoch
            self._epoch += 1
            if self._epoch >= self.num_epochs:
                raise StopIteration
            self.dataloader_iter = iter(self.dataloader)
            batch_value, batch_idcs = next(self.dataloader_iter)
        
        # Calculate current optimization step
        current_optim_step = (self.current_batch // self.gradient_accumulation_steps) + 1

        # Create Batch object
        batch_elem = Batch(
            batch_index=self.current_batch,
            optim_step_index=current_optim_step,
            total_optim_steps=self.total_optim_steps,
            batch_value=batch_value,
            batch_idcs=batch_idcs,
            is_optim_step_time=self._should_optim_step(self.current_batch),
            is_eval_time=self._should_eval(self.current_batch),
            is_save_time=self._should_save(self.current_batch),
        )
        
        self.current_batch += 1
        return batch_elem

    def __iter__(self):
        return self
    
    def __len__(self):
        """Return the total number of batches across all epochs."""
        return self.total_batches

    def reset(self):
        """Reset iterator to beginning."""
        self.current_batch = 0
        self._epoch = 0
        self.dataloader_iter = None
        
    def state_dict(self):
        """Save current state for resuming training."""
        return {
            "current_step": self.current_batch,
            "epoch": self._epoch
        }

    def load_state_dict(self, state_dict):
        """Load saved state to resume training."""
        self.current_batch = state_dict["current_step"]
        self._epoch = state_dict.get("epoch", 0)
        self.dataloader_iter = None
        
        # Fast-forward to the correct position
        if self.current_batch > 0:
            self.dataloader_iter = iter(self.dataloader)
            steps_in_current_epoch = self.current_batch % (self.total_batches // self.num_epochs)
            for _ in range(steps_in_current_epoch):
                try:
                    next(self.dataloader_iter)
                except StopIteration:
                    break
        
    def size(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)
    
    def get_batch_size(self):
        """Return the batch size."""
        return self.batch_size