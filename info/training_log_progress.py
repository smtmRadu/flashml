from typing import Any, Tuple, Union
from tqdm import tqdm

def log_metrics(name: Union[str, Tuple, list],
                value: Union[float, Tuple, list],
                epoch_idx: Tuple[int, int],
                batch_idx: Tuple[int, int],
                retain_logs:bool = True) -> None:
    """
    Update the progress bar with the current metrics.

    Args:
        name (str or tuple/list): Name(s) of the metric(s).
        value (float or tuple/list): Value(s) of the metric(s).
        curr_epoch (tuple): (current_epoch, total_epochs)
        curr_batch (tuple): (current_batch, total_batches)
        retain_logs (bool): Whether to retain the logs or not. If retaining, the logs can be graphed out later using `display_metrics`. Can set it to false to save memory.
    """
    _TrainingLogger.log_metrics(name, value, epoch_idx, batch_idx, retain_logs)

class _TrainingLogger:
    _instance: '_TrainingLogger' = None

    def __new__(cls, num_epochs: int = 1, num_batches: int = 1):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, num_epochs: int = 1, num_batches: int = 1):
        if hasattr(self, '_initialized'):
            return

        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.display = tqdm(total=num_batches, leave=False)

        self._initialized = True
        self.last_epoch_recorded = -1
        self.history_log: list[Tuple[Any, Any, int, int]] = None # holds out all records
        
    def _update(self, log: Tuple[Any, Any, int, int], retain_logs:bool = False) -> None:

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

        self.display.set_description(f"[Epoch {epoch_idx+1}/{self.num_epochs}]")
        self.display.set_postfix({log[0]: log[1]} if isinstance(log[0], str) else {n: v for n, v in zip(log[0], log[1])})
        self.display.update(1)

        if retain_logs:
            self.history_log.append(log)

    @staticmethod
    def log_metrics(name: Union[str, Tuple, list],
                    value: Union[float, Tuple, list],
                    epoch_idx: Tuple[int, int],
                    batch_idx: Tuple[int, int],
                    retain_logs:bool = False) -> None:
        logger = _TrainingLogger(num_epochs=epoch_idx[1], num_batches=batch_idx[1])
        logger._update((name, value, epoch_idx[0], batch_idx[0]), retain_logs)