import math
import sys
import time
from typing import Any


def log_episode(
    cumulative_reward: float,
    episode_length: int,
    step: tuple[int, int],
    other_metrics: dict[str, Any] = None,
    running_statistics_momentum: float = 0.9,
    experiment_name: str = "Default",
) -> None:
    """
    Records RL training data and logs it in MLFlow.
    Args:
            `reward`(float): Cumulated reward at the end of an episode.
            `length`(int): The length of the episode computed in steps.
            `step`(tuple[int, int]): The current (global step out of max_steps, max_steps).

            `other_metrics`(dict[str, Any]): Other information to log (e.g. GD steps).
            `momentum`(int): RT statistics are computed using running average.
    """
    _RLTrainLogger.log_episode(
        cumulative_reward=cumulative_reward,
        episode_length=episode_length,
        step=step,
        momentum=running_statistics_momentum,
        other=other_metrics,
        experiment_name=experiment_name,
    )


class _RLTrainLogger:
    _instance: "_RLTrainLogger" = None

    def __new__(
        cls,
        total_steps: int = 1e5,
        experiment_name: str = "flashml-rl",
    ):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        total_steps: int = 1e5,
        experiment_name: str = "flashml-rl",
    ):
        if hasattr(self, "_initialized"):
            return

        import atexit

        import mlflow
        from tqdm import tqdm

        self.host = "127.0.0.1"
        self.port = 5000
        self._start_mlflow_ui(self.host, self.port)

        self._initialized = True
        self.t = 0

        self.reward_MEAN = 0
        self.reward_VAR = 0
        self.eplen_MEAN = 0
        self.eplen_VAR = 0
        self.reward_MAX: float = -float("inf")
        self.eplen_MAX: int = 0

        self.max_steps = total_steps
        self.other: dict[str, Any] = None
        self._hyperparams_logged = False
        if experiment_name is not None:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp is None:
                mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        self.mlflow_op = mlflow.start_run(
            tags={"flashml": 0}
        )  # (log_system_metrics=True)
        print(
            f"MLFlow UI is accessible at: \033[94mhttp://{self.host}:{self.port}\033[0m \033[95m({self.mlflow_op.info.run_name})\033[0m"
        )

        self.display = tqdm(total=int(total_steps), leave=False)

        atexit.register(self._end_mlflow_run)

    def _update(self, step: int, momentum: float):
        bias_correction = 1 - momentum**self.t
        postfix_str = (
            f"Cumulated Reward [max: {self.reward_MAX:.3f}] [µ: {self.reward_MEAN / bias_correction:.3f}] [σ: {math.sqrt(self.reward_VAR) / bias_correction:.2f}z]            \n"  # the spaces are for clearning out parathesis if the message shortens
            f"Episode Length   [max: {self.eplen_MAX}] [µ: {self.eplen_MEAN / bias_correction:.3f}] [σ: {math.sqrt(self.eplen_VAR) / bias_correction:.2f}z]              "
        )

        # Add other dict values if it exists
        if self.other is not None:
            # Format each key-value pair and join with newlines
            other_items = []
            for key, value in self.other.items():
                # Format the value based on its type
                if isinstance(value, float):
                    formatted_value = f"{value}"
                else:
                    formatted_value = str(value)
                other_items.append(
                    f"{key}: {formatted_value}                              "
                )

            other_str = "\n".join(other_items)
            postfix_str += f"\n{other_str}"

        self.display.set_description(f"Episodes: {self.t}")
        self.display.n = min(step, self.max_steps)

        # Clear previous output lines
        if hasattr(self, "prev_postfix_lines") and self.prev_postfix_lines > 0:
            sys.stdout.write("\033[F" * self.prev_postfix_lines)
            sys.stdout.write("\033[K" * self.prev_postfix_lines)

        self.display.refresh()

        # Write the full postfix string and count total lines
        sys.stdout.write("\n" + postfix_str + "\n")
        sys.stdout.flush()

        # Update the line count to include all lines (including those from the other dict)
        self.prev_postfix_lines = postfix_str.count("\n") + 2

    @staticmethod
    def log_episode(
        cumulative_reward: float,
        episode_length: int,
        step: tuple[int, int],
        momentum: float = 0.9,
        other: dict[str, Any] = None,
        experiment_name: str = "flashml-rl",
    ) -> None:
        assert isinstance(step, tuple), "Note `step` must be a tuple[int, int]"
        assert episode_length > 0, "Episodes cannot last 0 steps or less."
        assert 0 < momentum < 1, (
            "Please use a high window size in order to get good info"
        )

        import mlflow

        logger = _RLTrainLogger(total_steps=step[1], experiment_name=experiment_name)

        if cumulative_reward > logger.reward_MAX:
            logger.reward_MAX = cumulative_reward

        if episode_length > logger.eplen_MAX:
            logger.eplen_MAX = episode_length

        logger.t += 1

        logger.reward_MEAN = logger.reward_MEAN * momentum + cumulative_reward * (
            1 - momentum
        )
        logger.eplen_MEAN = logger.eplen_MEAN * momentum + episode_length * (
            1 - momentum
        )

        logger.reward_VAR = (
            momentum * logger.reward_VAR
            + (1 - momentum) * (cumulative_reward - logger.reward_MEAN) ** 2
        )

        logger.eplen_VAR = (
            momentum * logger.eplen_VAR
            + (1 - momentum) * (episode_length - logger.eplen_MEAN) ** 2
        )

        logger.other = other

        if _RLTrainLogger._instance._hyperparams_logged is False:
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
            _RLTrainLogger._instance._hyperparams_logged = True

        mlflow.log_metrics(
            {
                "cumulative_reward": cumulative_reward,
                "episode_length": episode_length,
            },
            step=step[0],
            synchronous=False,
        )

        mlflow.log_metrics(other, step=step[0], synchronous=False)

        logger._update(step[0], momentum=momentum)

    def _start_mlflow_ui(self, host: str, port: int) -> None:
        """Check if MLflow UI is running and start it in the Conda environment if not."""
        import subprocess

        import requests

        try:
            response = requests.get(f"http://{host}:{port}", timeout=5)
            if response.status_code == 200:
                print("MLflow UI is already running.", end=" ")
                return
        except requests.ConnectionError:
            pass

        print("Starting MLflow UI...", end=" ")
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
                print("MLflow UI started successfully.", end=" ")
            else:
                stdout, stderr = process.communicate()
                print(f"Failed to start MLflow UI: {stderr}")
                sys.exit(1)
        except Exception as e:
            print(f"Error starting MLflow UI: {e}")
            sys.exit(1)

    def _end_mlflow_run(self):
        import mlflow

        if mlflow.active_run():
            mlflow.end_run()
