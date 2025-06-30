from typing import Any, Tuple


def log_metrics(
    metrics: dict[str, Any],
    step: Tuple[int, int] | int = None,
    run_name: str = None,  # if default generates a funny name
    experiment_name: str = None,  # let it Default instead of flashml because Default cannot be removed and is selected first..
) -> None:
    """At first call, initializes a new MLFlow run. It is not required to be consistent (different metrics, different step etc.).
    Hyperparams/Config dict (global variable) is logged automatically.

    If you want to start MLFlow without logging, use the following code in a .ipynb cell:
    >>> import subprocess
    >>> p = subprocess.Popen(["mlflow", "ui", "--host", "127.0.0.1", "--port", "5000"])
    Example:
    >>> for idx, batch in enumerate(batch_indices):
    ...    stuff
    ...    log_metrics({"loss" : loss}, step=(idx, len(batch_indices)))

    Args:
        metrics (dict[str, Any]): metrics dict
        step (Tuple[int, int]): the current step and total steps. It is incremented automatically if none
        experiment_name (str): The tab where to log the experiment (even with that, you can merge them inside mlflow when comparing results)
    """
    _TrainingLogger.log_metrics(
        metrics,
        step=step,
        run_name=run_name,
        experiment_name=experiment_name,
    )


def get_metrics_at_step(step: int, run_id: str = "current_run") -> dict:
    from mlflow.tracking import MlflowClient

    if run_id == "current_run":
        run_id = _TrainingLogger.instance.mlflow_op.info.run_id
    client = MlflowClient()
    run = client.get_run(run_id)
    metrics_at_step = {}

    for metric_name in run.data.metrics:
        metric_history = client.get_metric_history(run_id, metric_name)
        for metric_entry in metric_history:
            if metric_entry.step == step:
                metrics_at_step[metric_name] = metric_entry.value
                break

    return metrics_at_step


def log_checkpoint(
    state_dict: dict,
    info: Any = None,
    overwrite_last_checkpoint: bool = False,
    experiment_name: str = None,
) -> None:
    """Makes a checkpoint of the model+optim+scheduler and logs it in the MLFlow session.
    Example:
    >>> log_checkpoint({"model":model.state_dict(), "optim": optimizer.state_dict()})
    Args:
        state_dict (str): _description_
        info (Any): Any info/metrics to add for this checkpoint. The timestamp and step is automatically logged.
        save_over_last_checkpoint (bool): Either or not to increment and write a new checkpoint. If true, the last checkpoint is replaced with the current one.
        ckpt (dict | Any): the checkpoint that is a state_dict or dicts of state_dicts)

    """
    _TrainingLogger.log_checkpoint(
        state_dict=state_dict,
        info=info,
        save_over_last_checkpoint=overwrite_last_checkpoint,
        experiment_name=experiment_name,
    )


def load_checkpoint(run_id: str, version: int, experiment_name: str = None) -> Any:
    """Loads a checkpoint dict/state_dict from a given run in MLFlow.

    Args:
        run_id (str): click on the run and copy to clipboard the run ID
        version (int): the version of the checkpoint artifact. E.g: if checkpoint_v1, set this to 1.
        experiment_name (str, optional): Of which experiment the run belongs to. Defaults to "Default".

    Raises:
        ValueError: _description_

    Returns:
        Any: the state_dict logged
    """
    import os

    import mlflow

    experiment_id = mlflow.get_experiment_by_name(
        "Default" if experiment_name is None else experiment_name
    ).experiment_id

    artifact_path = f"mlruns/{experiment_id}"

    # if want to search by run name. Note that some runs might have similar names, so this will fuck up.

    # with os.scandir(artifact_path) as entries:
    #    runs_ids = [entry.name for entry in entries if entry.is_dir()]

    # run_id = None
    # for i in runs_ids:
    #     run_name_ = mlflow.get_run(i).info.run_name
    #     if run_name_ == run_name:
    #         run_id = i
    #         break
    #
    # if run_id is None:
    #     raise ValueError(
    #         f"The run `{run_name}` doesn't exist, or the correct experiment name was not provided."
    #     )

    artifact_path = f"mlruns/{experiment_id}/{run_id}/artifacts/checkpoint_v{version}"

    if not os.path.exists(artifact_path):
        raise ValueError(
            f"checkpoint_v{version} doesn't exist in the artifacts of run with ID:{run_id}."
        )

    import torch

    return torch.load(artifact_path + "/state_dict.pth")


def log_figure(figure, figure_name: str = None, experiment_name: str = None) -> None:
    """Logs a matplotlib/plotly figure in the MLFlow run.

    Args:
        figure (_type_): A matplotlib/plotly figure
        experiment_name (str, optional): _description_. Defaults to None.
    """
    _TrainingLogger.log_figure(
        figure, figure_name=figure_name, experiment_name=experiment_name
    )


def host_mlflow(host="127.0.0.1", port="5000"):
    """Hosts MLFlow server.

    Args:
        host (str, optional): _description_. Defaults to "127.0.0.1".
        port (str, optional): _description_. Defaults to "5000".

    Returns: the subprocess
    """
    import subprocess

    print(
        f"\033[90mMLFlow hosted at:\033[0m \033[94mhttp://{host}:{port}\033[0m \033[95m\033[0m"
    )
    return subprocess.Popen(["mlflow", "ui", "--host", host, "--port", port])


class _TrainingLogger:
    _instance: "_TrainingLogger" = None

    def __new__(
        cls, num_steps: int = None, run_name: str = None, experiment_name: str = None
    ):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self, num_steps: int = None, run_name: str = None, experiment_name: str = None
    ):
        if hasattr(self, "_initialized"):
            return

        import atexit

        import mlflow
        from tqdm import tqdm
        # from datetime import datetime
        # import torch

        self.internal_step = -1
        self.host = "127.0.0.1"
        self.port = 5000
        self.ckpt_version = 0
        self._start_mlflow_ui(self.host, self.port)

        self.num_steps = num_steps
        # mlflow.enable_system_metrics_logging()

        if experiment_name is not None:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)

        # now = datetime.now()
        self.mlflow_op = mlflow.start_run(
            run_name=run_name,
            # run_name=f"run_{now.day:02d}{now.month:02d}_{now.time()}",
            tags={
                # "random.state": random.getstate(),
                # "numpy.state": numpy.random.get_state(),
                # "torch.state": torch.get_rng_state(),
                # "torch.cuda_all.state": torch.cuda.get_rng_state_all(),
                # "torch.backends.cudnn.benchmark": torch.backends.cudnn.benchmark,
                # "torch.backends.cudnn.deterministic": torch.backends.cudnn.deterministic,
            },  # it doesn't worth to log states because they are modified before logging the first time. Instead the backends are fine.
        )  # (log_system_metrics=True)

        print(
            f"\033[90mAccess MLFlow run at:\033[0m \033[94mhttp://{self.host}:{self.port}\033[0m \033[95m({self.mlflow_op.info.run_name})\033[0m"
        )
        self.display = tqdm(desc="Step", total=num_steps, leave=False)

        atexit.register(self._end_mlflow_ui)
        self.display.reset()
        self._initialized = True
        self._hyperparams_logged = False

    def _start_mlflow_ui(self, host: str, port: int) -> None:
        """Check if MLFlow UI is running and start it in the Conda environment if not."""
        import subprocess
        import sys

        import requests

        try:
            response = requests.get(f"http://{host}:{port}", timeout=5)
            if response.status_code == 200:
                print("\033[90mMLFlow UI is already running.\033[0m", end=" ")
                return
        except requests.ConnectionError:
            pass

        print("\033[90mStarting MLFlow UI.\033[0m", end=" ")
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
            # time.sleep(3)  # Wait for server to start
            if process.poll() is None:
                print("\033[90mMLFlow UI started.\033[0m", end=" ")
            else:
                stdout, stderr = process.communicate()
                print(f"Failed to start MLFlow UI: {stderr}")
                sys.exit(1)
        except Exception as e:
            print(f"Error starting MLFlow UI: {e}")
            sys.exit(1)

    def _end_mlflow_ui(self):
        import mlflow

        if mlflow.active_run():
            mlflow.end_run()

    @staticmethod
    def log_metrics(
        metrics: dict[str, Any],
        step: Tuple[int, int] | int = None,
        run_name: str = None,
        experiment_name: str = None,
    ):
        import mlflow  # the import overhead is minimal (2ms per 100k calls)

        assert metrics is not None, "You logged no metric"
        assert len(metrics) > 0, "Metric log is empty"
        logger = _TrainingLogger(
            num_steps=None if step is None or isinstance(step, int) else step[1],
            run_name=run_name,
            experiment_name=experiment_name,
        )
        if step is None:
            logger.internal_step += 1
            step = logger.internal_step
        step = step if isinstance(step, (int, float)) else step[0]

        mlflow.log_metrics(metrics, step=step, synchronous=False)
        if _TrainingLogger._instance._hyperparams_logged is False:
            for var_name in globals():
                if var_name.lower() in [
                    "hyperparam",
                    "hyperparams",
                    "hyperparameters",
                    "config",
                    "configs",
                    "configuration",
                    "configurations",
                    "hp",
                    "hps",
                    "hparam",
                    "hparams",
                ]:
                    mlflow.log_params(globals()[var_name], synchronous=False)
            _TrainingLogger._instance._hyperparams_logged = True

        logger.display.set_postfix(metrics)
        logger.display.n = step
        logger.display.update(0)
        logger.internal_step = step

    @staticmethod
    def log_checkpoint(
        state_dict: dict,
        info: Any,
        save_over_last_checkpoint: bool = False,
        experiment_name: str = None,
    ):
        from datetime import datetime

        import mlflow

        logger = _TrainingLogger(None, run_name=None, experiment_name=experiment_name)

        if not save_over_last_checkpoint:
            logger.ckpt_version += 1

        mlflow.pytorch.log_state_dict(
            state_dict=state_dict,
            artifact_path=f"checkpoint_v{logger.ckpt_version}",
        )

        __info_dict = {
            "timestamp": datetime.now(),
            "step": logger.internal_step,
        }  # or maybe sometimes +1 idk..

        if info is not None:
            __info_dict["_"] = info

        mlflow.log_dict(
            __info_dict,
            artifact_file=f"checkpoint_v{logger.ckpt_version}/~info.json",
        )

    @staticmethod
    def log_figure(figure, figure_name: str = None, experiment_name: str = None):
        import mlflow

        _ = _TrainingLogger(
            None, run_name=None, experiment_name=experiment_name
        )  # initialize if needed
        mlflow.log_figure(
            figure=figure,
            artifact_file=figure_name
            if figure_name.endswith(".html")
            else f"{figure_name}.html",
        )
