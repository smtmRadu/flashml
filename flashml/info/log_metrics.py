from typing import Any, Tuple, Union, overload
from tqdm import tqdm

def log_metrics2(metrics: dict[str, Any], step: Tuple[int, int]) -> None:
    _TrainingLogger_step.log_metrics(metrics,step=step)
    

def log_metrics(metrics: dict[str, Any], epoch_idx: int | Tuple[int, int], batch_idx: Tuple[int, int]) -> None:
    """
    Update the progress bar with the current metrics. Usefull only in casual epoch/batch training.
    Do not call multiple times in a step. For RL, check rl.log_metrics.
    Example to call: \\
    log_metrics({"lr" : lr, "time" : datetime.now()}, epoch_idx = (ep, num_epochs), batch_idx= (b_idx, num_batches))
    Args:
        metrics (dict[str, Any]): Name(s) of the metric(s).
        epoch_idx (int | tuple[int, int]): current batch | (current_epoch, total_epochs)
        batch_idx (tuple[int, int]): (current_batch, total_batches)
    """
    _TrainingLogger_epoch_batch.log_metrics(metrics, epoch_idx, batch_idx)

class _TrainingLogger_step:
    _instance: '_TrainingLogger_step' = None

    def __new__(cls, num_steps:int=1):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, num_steps:int=1):
        if hasattr(self, '_initialized'):
            return

        self.num_steps = num_steps
        self.display = tqdm(total=int(num_steps), leave=False)
        self.display.reset()
        self._initialized = True
        self.last_epoch_recorded = -1
        self.history_log: list[Tuple[dict[str, Any], int]] = [] # holds out all records
        
    def _update(self, log: Tuple[dict[str, Any], int]) -> None:
        self.display.set_description(f"Step")

        self.display.set_postfix(log[0])
        self.display.n = log[1]
        self.display.update(0)
        self.history_log.append(log)

    @staticmethod
    def log_metrics(metrics:dict[str, Any],
                    step: Tuple[int, int]):
        
        assert metrics != None, "You logged no metric"
        assert len(metrics) > 0, "Metric log is empty"
        logger = _TrainingLogger_step(num_steps=step[1])
        logger._update((metrics, step[0]))

class _TrainingLogger_epoch_batch:
    _instance: '_TrainingLogger_epoch_batch' = None

    def __new__(cls, num_iters: int = 1, num_bar_steps: int = 1):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, num_iters: int = 1, num_bar_steps: int = 1):
        if hasattr(self, '_initialized'):
            return

        self.num_iterations = num_iters
        self.num_bar_steps = num_bar_steps
        self.display = tqdm(total=int(num_bar_steps), leave=False)

        self._initialized = True
        self.last_epoch_recorded = -1
        self.history_log: list[Tuple[dict[str, Any], int, int]] = None # holds out all records
        
    def _update(self, log: Tuple[dict[str, Any], int, int]) -> None:

        epoch_idx = log[1]
        batch_idx = log[2]

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

        self.display.set_postfix(log[0])
        self.display.update(1)

        self.history_log.append(log)

    @staticmethod
    def log_metrics(metrics:dict[str, Any],
                    iter_idx: int | Tuple[int, int],
                    bar_idx: Tuple[int, int]):
        
        assert metrics != None, "You logged no metric"
        assert len(metrics) > 0, "Metric log is empty"
        logger = _TrainingLogger_epoch_batch(num_iters=iter_idx[1] if isinstance(iter_idx, Tuple) else None, num_bar_steps=bar_idx[1])
        logger._update((metrics, iter_idx[0] if isinstance(iter_idx, Tuple) else iter_idx, bar_idx[0]))