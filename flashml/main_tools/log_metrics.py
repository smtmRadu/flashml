from typing import Any, Tuple
import time


def log_metrics(
    metrics: dict[str, Any],
    step: Tuple[int, int] | int,
    hyperparams: dict[str, Any] = None,
) -> None:
    """Update the progress bar with the current metrics while logging them in MLFlow. It is not required to be consistent (different metrics, different step)
    Example to call: \\
    for idx, batch in enumerate(batch_indices):\\
        ... stuff ...\\
        log_metrics({"loss" : loss}, step=(idx, len(batch_indices)))\\

    Args:
        metrics (dict[str, Any]): metrics dict
        step (Tuple[int, int]): the current step and total steps
        hyperparams (dict[str, Any]): hyperparameters dict (can contain even objects as values). Only the first pass will be considered.
    """
    _TrainingLogger_step.log_metrics(metrics, step=step, params=hyperparams)


class _TrainingLogger_step:
    _instance: "_TrainingLogger_step" = None

    def __new__(cls, num_steps: int = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, num_steps: int = None):
        if hasattr(self, "_initialized"):
            return

        from tqdm import tqdm
        import mlflow
        import atexit

        self.host = "127.0.0.1"
        self.port = 5000
        self._start_mlflow_ui(self.host, self.port)

        self.num_steps = num_steps
        # mlflow.enable_system_metrics_logging()
        self.mlflow_op = mlflow.start_run(
            tags={"flashml": 0}
        )  # (log_system_metrics=True)
        print(
            f"\033[90mAccess MLflow UI at:\033[0m \033[94mhttp://{self.host}:{self.port}\033[0m \033[95m({self.mlflow_op.info.run_name})\033[0m"
        )

        self.display = tqdm(total=int(num_steps), leave=False)

        atexit.register(self._end_mlflow_run)
        self.display.reset()
        self._initialized = True
        self._hyperparams_logged = False

    def _start_mlflow_ui(self, host: str, port: int) -> None:
        """Check if MLflow UI is running and start it in the Conda environment if not."""
        import requests
        import subprocess
        import sys

        try:
            response = requests.get(f"http://{host}:{port}", timeout=5)
            if response.status_code == 200:
                print("\033[90mMLflow UI is already running.\033[0m", end=" ")
                return
        except requests.ConnectionError:
            pass

        print("\033[90mStarting MLflow UI.\033[0m", end=" ")
        try:
            cmd = [
                sys.executable,
                "-m",
                "mlflow",
                "ui",
                "--host",
                host,
                "--port",
                str(port),
            ]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
            )
            time.sleep(3)  # Wait for server to start
            if process.poll() is None:
                print("\033[90mMLflow UI started.\033[0m", end=" ")
            else:
                stdout, stderr = process.communicate()
                print(f"Failed to start MLflow UI: {stderr}")
                sys.exit(1)
        except Exception as e:
            print(f"Error starting MLflow UI: {e}")
            sys.exit(1)

    def _update(self, log: Tuple[dict[str, Any], int]) -> None:
        # self.display.set_description(desc="Training")
        self.display.set_postfix(log[0])
        self.display.n = log[1]
        self.display.update(0)

    @staticmethod
    def log_metrics(
        metrics: dict[str, Any],
        step: Tuple[int, int] | int,
        params: dict[str, Any] = None,
    ):
        import mlflow

        assert metrics is not None, "You logged no metric"
        assert len(metrics) > 0, "Metric log is empty"
        logger = _TrainingLogger_step(
            num_steps=None if isinstance(step, int) else step[1]
        )
        step = step if isinstance(step, (int, float)) else step[0]
        mlflow.log_metrics(metrics, step=step, synchronous=False)
        if (
            _TrainingLogger_step._instance._hyperparams_logged is False
            and params is not None
        ):
            mlflow.log_params(params, synchronous=False)
            _TrainingLogger_step._instance._hyperparams_logged = True

        logger._update((metrics, step))

    def _end_mlflow_run(self):
        import mlflow

        if mlflow.active_run():
            mlflow.end_run()
