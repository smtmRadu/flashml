from typing import Any, Tuple, List


def log_metrics(metrics: dict[str, Any], step: Tuple[int, int]) -> None:
    """Update the progress bar with the current metrics. Usefull only in casual epoch/batch training.
    Do not call multiple times in a step, just once per for loop. For RL, check rl.log_metrics.
    Example to call: \\
    for idx, batch in enumerate(batch_indices):
        ... stuff ...
        log_metrics({"loss" : loss}, step=(idx, len(batch_indices)))

    Args:
        metrics (dict[str, Any]): metrics dict
        step (Tuple[int, int]): the current step and total steps
    """
    _TrainingLogger_step.log_metrics(metrics, step=step)


def log_metrics2(
    metrics: dict[str, Any],
    epoch_idx: int | Tuple[int, int],
    batch_idx: Tuple[int, int],
) -> None:
    """
    Update the progress bar with the current metrics. Usefull only in casual epoch/batch training.
    Do not call multiple times in a step. For RL, check rl.log_metrics.
    Example to call: \\
    log_metrics2({"lr" : lr, "time" : datetime.now()}, epoch_idx = (ep, num_epochs), batch_idx= (b_idx, num_batches))
    Args:
        metrics (dict[str, Any]): Name(s) of the metric(s).
        epoch_idx (int | tuple[int, int]): current batch | (current_epoch, total_epochs)
        batch_idx (tuple[int, int]): (current_batch, total_batches)
    """
    _TrainingLogger_epoch_batch.log_metrics(metrics, epoch_idx, batch_idx)


class _TrainingLogger_step:
    _instance: "_TrainingLogger_step" = None

    def __new__(cls, num_steps: int = 1):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, num_steps: int = 1):
        from tqdm import tqdm

        if hasattr(self, "_initialized"):
            return

        self.num_steps = num_steps
        self.display = tqdm(total=int(num_steps), leave=False)
        self.display.reset()
        self._initialized = True
        self.last_epoch_recorded = -1
        self.history_log: list[Tuple[dict[str, Any], int]] = []  # holds out all records

    def _update(self, log: Tuple[dict[str, Any], int]) -> None:
        self.display.set_description(desc="Training Progress")

        self.display.set_postfix(log[0])
        self.display.n = log[1]
        self.display.update(0)
        self.history_log.append(log)

    @staticmethod
    def log_metrics(metrics: dict[str, Any], step: Tuple[int, int]):
        assert metrics is not None, "You logged no metric"
        assert len(metrics) > 0, "Metric log is empty"
        logger = _TrainingLogger_step(num_steps=step[1])
        logger._update((metrics, step[0]))


class _TrainingLogger_epoch_batch:
    _instance: "_TrainingLogger_epoch_batch" = None

    def __new__(cls, num_iters: int = 1, num_bar_steps: int = 1):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, num_iters: int = 1, num_bar_steps: int = 1):
        from tqdm import tqdm

        if hasattr(self, "_initialized"):
            return

        self.num_iterations = num_iters
        self.num_bar_steps = num_bar_steps
        self.display = tqdm(total=int(num_bar_steps), leave=False)

        self._initialized = True
        self.last_epoch_recorded = -1
        self.history_log: list[Tuple[dict[str, Any], int, int]] = (
            None  # holds out all records
        )

    def _update(self, log: Tuple[dict[str, Any], int, int]) -> None:
        epoch_idx = log[1]
        batch_idx = log[2]

        if epoch_idx == 0 and batch_idx == 0:
            self.display.reset()
            self.display.update(1)
            self.last_epoch_recorded = -1
            self.history_log = []

        if epoch_idx > self.last_epoch_recorded:
            self.display.reset()
            self.display.update(1)
            self.last_epoch_recorded = epoch_idx

        self.display.set_description(
            f"[Epoch {epoch_idx + 1}/{self.num_iterations}]"
            if self.num_iterations is not None
            else f"[Iter {epoch_idx + 1}]"
        )

        self.display.set_postfix(log[0])
        self.display.update(1)

        self.history_log.append(log)

    @staticmethod
    def log_metrics(
        metrics: dict[str, Any],
        epoch_idx: int | Tuple[int, int],
        batch_idx: Tuple[int, int],
    ):
        assert metrics is not None, "You logged no metric"
        assert len(metrics) > 0, "Metric log is empty"
        assert isinstance(batch_idx, Tuple), (
            f"LOG_METRICS ERROR:batch_idx should be a tuple(current_batch, total_batches), got type: {type(batch_idx)}"
        )
        logger = _TrainingLogger_epoch_batch(
            num_iters=epoch_idx[1] if isinstance(epoch_idx, Tuple) else None,
            num_bar_steps=batch_idx[1],
        )
        logger._update(
            (
                metrics,
                epoch_idx[0] if isinstance(epoch_idx, Tuple) else epoch_idx,
                batch_idx[0],
            )
        )


def _plot_metrics_on_thread(block: bool, show_epoch_ticks: bool) -> None:
    from matplotlib.widgets import Button
    import matplotlib.pyplot as plt

    """
    Show the collected metrics from `log_metrics` at the end of the training. The graphs can be exported as csv.
    The labels are: `Metric` (metric name), `Value` (metric value), `Epoch` (epoch idx), `Batch Index` (batch idx in epoch), `Step` (optim step)
    """
    if (
        _TrainingLogger_epoch_batch._instance is not None
        and len(_TrainingLogger_epoch_batch._instance.history_log) > 0
    ):
        history: List[Tuple[dict[str, Any], int, int]] = (
            _TrainingLogger_epoch_batch._instance.history_log
        )

        metrics_rec, epochs_rec, rec = zip(*history)  # all elements are tuples
        names = metrics_rec[0].keys()

        batch_num = len(set(rec))
        epoch_idx_scaled_at_step = [(e + 1) * batch_num for e in set(epochs_rec)]

        steps = [e * batch_num + b for e, b in zip(epochs_rec, rec)]
        for idx, name in enumerate(names):
            values_for_name = [
                v if isinstance(v, float) else v[name] for v in metrics_rec
            ]

            fig, ax = plt.subplots(figsize=(8, 5))
            fig.canvas.manager.set_window_title("flashml")
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax.plot(steps, values_for_name, linestyle="-", linewidth=0.75)

            ax.set_xlabel("Step | Epoch" if show_epoch_ticks else "Step")
            ax.set_ylabel(name)
            ax.set_title(
                name + " vs Step/Epoch" if show_epoch_ticks else name + " vs Step"
            )
            ax.grid(True)
            if show_epoch_ticks:
                sec2 = ax.secondary_xaxis("bottom")
                sec2.set_xticks(epoch_idx_scaled_at_step)
                sec2.set_xticklabels([x // batch_num for x in epoch_idx_scaled_at_step])
                sec2.tick_params(axis="x", length=30, width=1)

            plt.tight_layout()

            export_ax = plt.axes([0.85, 0.01, 0.12, 0.05])
            export_button = Button(export_ax, "Export CSV")
            export_button.on_clicked(
                lambda x: _export_values_batch_epoch(history, metric_name=name)
            )
            plt.show(block=block)

    elif (
        _TrainingLogger_step._instance is not None
        and len(_TrainingLogger_step._instance.history_log) > 0
    ):
        history: List[Tuple[dict[str, Any], int]] = (
            _TrainingLogger_step._instance.history_log
        )
        metrics_rec, step_rec = zip(*history)  # all elements are tuples
        names = metrics_rec[0].keys()
        steps = list(step_rec)
        for idx, name in enumerate(names):
            values_for_name = [
                v if isinstance(v, float) else v[name] for v in metrics_rec
            ]

            fig, ax = plt.subplots(figsize=(8, 5))
            fig.canvas.manager.set_window_title("flashml")
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax.plot(steps, values_for_name, linestyle="-", linewidth=0.75)

            ax.set_xlabel("Step")
            ax.set_ylabel(name)
            ax.set_title(name + " vs Step")
            ax.grid(True)
            plt.tight_layout()

            export_ax = plt.axes([0.85, 0.01, 0.12, 0.05])
            export_button = Button(export_ax, "Export CSV")
            export_button.on_clicked(
                lambda x: _export_values_step(history, metric_name=name)
            )
            plt.show(block=block)

    else:
        raise "Training logger was not initialized. You need to `log_metrics` or `log_metrics2` before calling `display_metrics`"


def _export_values_step(history, metric_name):
    import csv
    from tkinter import filedialog

    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )

    if file_path:
        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Metric", "Value", "Step"])

            for idx, (dct, step) in enumerate(history):
                writer.writerow([metric_name, dct[metric_name], step])


def _export_values_batch_epoch(history, metric_name):
    import csv
    from tkinter import filedialog

    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )

    if file_path:
        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Metric", "Value", "Epoch", "Batch", "Step"])

            _, epochs_rec, rec = zip(*history)
            batch_num = len(set(rec))
            steps = [e * batch_num + b for e, b in zip(epochs_rec, rec)]

            for idx, (dct, iter, b_idx) in enumerate(history):
                writer.writerow(
                    [metric_name, dct[metric_name], iter, b_idx, steps[idx]]
                )


def plot_metrics(block=True, show_epoch_ticks: bool = False) -> None:
    """
    Show the current collected metrics from `log_metrics` (can be called multiple times, even if the training didn't end). The graphs can be exported as csv.

    Args:
        block (bool, optional): If True, the function will block the main thread until the plot is closed. Defaults to True.
        show_epoch_ticks (bool, optional): If True, the x-axis will show the epoch index instead of the step index. Defaults to False.
    """
    # assert _TrainingLogger._instance is not None, "Training logger not initialized. You need to log_metrics before calling display_metrics."
    # assert len(_TrainingLogger._instance.history_log) > 0, "No metrics logged. You need to set retain_logs=True in display_metrics."
    #
    # thread = Thread(target=_display_metrics_on_thread, args=(show_epoch_ticks,))
    # thread.start()

    # for now they are disabled
    _plot_metrics_on_thread(block, show_epoch_ticks)
