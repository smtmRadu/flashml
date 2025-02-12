from typing import Any, Tuple, Union
from tqdm import tqdm

def log_metrics(name: Union[str, Tuple, list],
                value: Union[float, Tuple, list],
                epoch_idx: int | Tuple[int, int],
                batch_idx: Tuple[int, int]) -> None:
    """
    Update the progress bar with the current metrics. Usefull only in casual epoch/batch training.
    Do not call multiple times in a step. For RL, check rl.log_metrics.
    Args:
        `name` (str | tuple[str] | list[str]): Name(s) of the metric(s).
        `value` (float | tuple[float] | list[float]): Value(s) of the metric(s).
        `epoch_idx` (int | tuple[int, int]): current batch | (current_epoch, total_epochs)
        `batch_idx` (tuple[int, int]): (current_batch, total_batches)
    """
    _TrainingLogger.log_metrics(name, value, epoch_idx, batch_idx)

class _TrainingLogger:
    _instance: '_TrainingLogger' = None

    def __new__(cls, num_iters: int = 1, num_bar_steps: int = 1):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, num_iters: int = 1, num_bar_steps: int = 1):
        if hasattr(self, '_initialized'):
            return

        self.num_iterations = num_iters
        self.num_bar_steps = num_bar_steps
        self.display = tqdm(total=num_bar_steps, leave=False)

        self._initialized = True
        self.last_epoch_recorded = -1
        self.history_log: list[Tuple[Any, Any, int, int]] = None # holds out all records
        
    def _update(self, log: Tuple[Any, Any, int, int]) -> None:

        epoch_idx = log[2]
        batch_idx = log[3]

        if epoch_idx == 0  and batch_idx == 0:
            self.display.reset()
            self.display.update(1)
            self.last_epoch_recorded = -1
            self.history_log = []
        

        if epoch_idx > self.last_epoch_recorded:
            self.display.reset()
            self.display.update(1)
            self.last_epoch_recorded = epoch_idx

        self.display.set_description(f"[Epoch {epoch_idx+1}/{self.num_iterations}]" if self.num_iterations!=None else f"[Iter {epoch_idx+1}]")
        self.display.set_postfix({log[0]: log[1]} if isinstance(log[0], str) else {n: v for n, v in zip(log[0], log[1])})
        self.display.update(1)

        self.history_log.append(log)

    @staticmethod
    def log_metrics(name: Union[str, Tuple, list],
                    value: Union[float, Tuple, list],
                    iter_idx: int | Tuple[int, int],
                    bar_idx: Tuple[int, int]):
        logger = _TrainingLogger(num_iters=iter_idx[1] if isinstance(iter_idx, Tuple) else None, num_bar_steps=bar_idx[1])
        logger._update((name, value, iter_idx[0] if isinstance(iter_idx, Tuple) else iter_idx, bar_idx[0]))