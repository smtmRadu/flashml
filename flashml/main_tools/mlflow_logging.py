from typing import Any, Tuple


def log_metrics(
    metrics: dict[str, Any],
    step: Tuple[int, int] | int = None,
    hyperparams: dict[str, Any] = None,
    log_system_params_interval: int = 1,
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
        hyperparams (dict[str, Any]): hyperparams dict. If not provided, it will be taken from the global variable (if possible)
        run_name (str): If None, it will initialize a new run and will log only to it. You can have multiple runs in a single python call (e.g. for grid search/parallel coords)
        experiment_name (str): The tab where to log the experiment (even with that, you can merge them inside mlflow when comparing results)
    """
    _TrainingLogger.log_metrics(
        metrics,
        step=step,
        hyperparams=hyperparams,
        log_system_params_interval=log_system_params_interval,
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

    # if want to search by run name. Note that some runs might have similar names, so this will mess up.

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
    
    To connect to runpod runtime from local machine, open a wsl terminal and run:\\
    ssh -L 1234:127.0.0.1:5000 -o StrictHostKeyChecking=no [root@... -p <port> a.k.a. SSH over exposed TCP w/o ssh] (e.g. "ssh -L 1234:127.0.0.1:5000 -y root@104.255.9.187 -p 12780")\\
    Then connect in local browser to 127.0.0.1:1234\\
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

def load_run_state(run_id: str, experiment_name: str = None):
    """
    Loads all metrics (including system metrics) from a given MLflow run and logs them to the current active run,
    initializing the TrainingLogger first and using direct mlflow.log_metrics calls.
    Args:
        run_id (str): The MLflow run ID to load from.
        experiment_name (str, optional): The experiment name. Defaults to None.
    """
    raise Exception("Not in use. It is too slow and must be updated in the future with some sort of copy pasting from the source..")
    import mlflow
    from mlflow.tracking import MlflowClient

    # 1. Explicitly initialize the logger (this starts a new run if not already started)
    logger = _TrainingLogger(None, run_name=None, experiment_name=experiment_name)
    client = MlflowClient()

    # 2. Copy scalar metrics (user-logged metrics)
    run = client.get_run(run_id)
    metric_names = list(run.data.metrics.keys())

    for metric_name in metric_names:
        history = client.get_metric_history(run_id, metric_name)
        for metric in history:
            # Use native mlflow log_metrics to log at the exact step
            mlflow.log_metrics({metric_name: metric.value}, step=metric.step, synchronous=False)

    # 3. Copy system metrics (MLflow >=2.5.0)
    try:
        sys_metric_names = getattr(run.data, "system_metrics", {}).keys()
        for sys_metric_name in sys_metric_names:
            sys_history = client.get_metric_history(run_id, sys_metric_name, metric_type="system")
            for sys_metric in sys_history:
                mlflow.log_metrics({sys_metric_name: sys_metric.value}, step=sys_metric.step, synchronous=False)
    except Exception as e:
        print(f"Warning: Could not load system metrics. ({e})")

    print(f"Copied all metrics and system metrics from run {run_id} to current run.")
    
    # this should also increment the internal step counter but fuck off...
class _TrainingLogger:
    _instance: "_TrainingLogger" = None

    def __new__(
        cls, num_steps: int = None, log_system_params_interval:int=1, run_name: str = None, experiment_name: str = None
    ):
        """Ensure that a new instance is created only when the parameters change."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        else:
            if (cls._instance.run_name != run_name or cls._instance.experiment_name != experiment_name):
                if hasattr(cls._instance, "display") and cls._instance.display is not None:
                    cls._instance.display.close()  # close the tqdm bar
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self, num_steps: int = None, log_system_params_interval:int=1,run_name: str = None, experiment_name: str = None
    ):
        """Initialize MLFlow logging, creating or using an existing run."""
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
        self.run_name = run_name
        self.experiment_name = experiment_name  

        if experiment_name is not None:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)

        # now = datetime.now()
        if mlflow.active_run():
            mlflow.end_run()
        self.mlflow_op = mlflow.start_run(
            run_name=run_name,
            log_system_metrics=True if log_system_params_interval>0 else False,
            # run_name=f"run_{now.day:02d}{now.month:02d}_{now.time()}",
            tags={},
        )  # (log_system_metrics=True)
        if log_system_params_interval:
            mlflow.set_system_metrics_sampling_interval(log_system_params_interval)
            mlflow.set_system_metrics_samples_before_logging(log_system_params_interval)
        mlflow.set_tracking_uri(f"http://{self.host}:{self.port}")
        print(
            f"\033[90mAccess MLFlow run at:\033[0m \033[94mhttp://{self.host}:{self.port}\033[0m \033[95m({self.mlflow_op.info.run_name})\033[0m"
        )
        self.display = tqdm(desc="Step", total=num_steps, leave=True)

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

    def _get_run_id_by_name(self, run_name: str) -> str:
        """Retrieve run ID by run name."""
        import mlflow
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
        runs = client.search_runs(experiment_ids=[experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'")
        return runs[0].info.run_id if runs else None

    @staticmethod
    def log_metrics(
        metrics: dict[str, Any],
        step: Tuple[int, int] | int = None,
        hyperparams: dict[str, Any] = None,
        log_system_params_interval: int = 1,
        run_name: str = None,
        experiment_name: str = None,
    ):
        """Log metrics to MLFlow."""
        import mlflow
        assert metrics, "You logged no metric"
        assert len(metrics) > 0, "Metric log is empty"
        logger = _TrainingLogger(
            num_steps=None if step is None or isinstance(step, int) else step[1],
            log_system_params_interval=log_system_params_interval,
            run_name=run_name,
            experiment_name=experiment_name,
        )
        if step is None:
            logger.internal_step += 1
            step = logger.internal_step
        step = step if isinstance(step, (int, float)) else step[0]

        mlflow.log_metrics(metrics, step=step, synchronous=False)
        
        if hyperparams and not _TrainingLogger._instance._hyperparams_logged:
            mlflow.log_params(hyperparams, synchronous=False)
            _TrainingLogger._instance._hyperparams_logged = True
        elif not _TrainingLogger._instance._hyperparams_logged:
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
        """Log checkpoint to MLFlow."""
        import mlflow
        logger = _TrainingLogger(None, run_name=None, experiment_name=experiment_name)

        if not save_over_last_checkpoint:
            logger.ckpt_version += 1

        mlflow.pytorch.log_state_dict(
            state_dict=state_dict,
            artifact_path=f"checkpoint_v{logger.ckpt_version}",
        )
        from datetime import datetime
        __info_dict = {
            "timestamp": datetime.now(),
            "step": logger.internal_step,
        }

        if info:
            __info_dict["_"] = info
        
        mlflow.log_dict(
            __info_dict,
            artifact_file=f"checkpoint_v{logger.ckpt_version}/~info.json",
        )

    @staticmethod
    def log_figure(figure, figure_name: str = None, experiment_name: str = None):
        """Log figure to MLFlow."""
        _ = _TrainingLogger(
            None, run_name=None, experiment_name=experiment_name
        )
        import mlflow
        mlflow.log_figure(
            figure=figure,
            artifact_file=figure_name
            if figure_name.endswith(".html")
            else f"{figure_name}.html",
        )
