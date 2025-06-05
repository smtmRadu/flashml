import sys
import numpy as np
from tqdm import tqdm
from typing import Any, Tuple, List
import matplotlib.pyplot as plt
from tkinter import filedialog
import csv
from matplotlib.widgets import Button


def log_episode(
    cumulative_reward: float,
    episode_length: int,
    step: tuple[int, int],
    window_size: int = 100,
    other_metrics: dict[str, Any] = None,
) -> None:
    """
    Records Reward data in an RL training session. You can call `plot_episodes` to visualize the logged data.
    Args:
            `reward`(float): Cumulated reward at the end of an episode.
            `length`(int): The length of the episode computed in steps.
            `step`(tuple[int, int]): The current (global step out of max_steps, max_steps).
            `window_size`(int): RT statistics are computed using a window of last elements.
            `other`(dict[str, Any]): Other information to log (e.g. GD steps). It must have constant keys along the logging process.
    """
    _RLTrainLogger.log_episode(
        cumulative_reward=cumulative_reward,
        episode_length=episode_length,
        step=step,
        window_size=window_size,
        other=other_metrics,
    )


class _RLTrainLogger:
    _instance: "_RLTrainLogger" = None

    def __new__(cls, total_steps: int = 1e5, window_size=100):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, total_steps: int = 1e5, window_size=100):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self.window_size: int = window_size
        self.reward_window = []
        self.episode_len_window = []
        self.episodes_history = []
        self.other_history = []
        self.max_recorded_reward: float = -float("inf")
        self.max_recorded_ep_len: int = 0
        self.max_steps = total_steps
        self.other: dict[str, Any] = None
        self.display = tqdm(total=int(total_steps), leave=False)

    def _update(self, step: int):
        r_window_mean = np.mean(self.reward_window)
        r_window_std = np.std(self.reward_window)
        ep_len_window_mean = np.mean(self.episode_len_window)
        ep_len_window_std = np.std(self.episode_len_window)

        postfix_str = (
            f"Cumulated Reward [max: {self.max_recorded_reward:.3f}] [µ: {r_window_mean:.3f}] [σ: {r_window_std:.2f}z]            \n"  # the spaces are for clearning out parathesis if the message shortens
            f"Episode Length   [max: {self.max_recorded_ep_len}] [µ: {ep_len_window_mean:.3f}] [σ: {ep_len_window_std:.2f}z]              "
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

        self.display.set_description(f"Episodes: {len(self.episodes_history)}")
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
        window_size: int = 100,
        other: dict[str, Any] = None,
    ) -> None:
        assert isinstance(step, tuple), "Note `step` must be a tuple[int, int]"
        assert episode_length > 0, "Episodes cannot last 0 steps or less."
        assert window_size >= 5, (
            "Please use a high window size in order to get good info"
        )

        logger = _RLTrainLogger(total_steps=step[1], window_size=window_size)

        if cumulative_reward > logger.max_recorded_reward:
            logger.max_recorded_reward = cumulative_reward

        if episode_length > logger.max_recorded_ep_len:
            logger.max_recorded_ep_len = episode_length

        logger.reward_window.append(cumulative_reward)
        logger.episode_len_window.append(episode_length)
        if other is not None:
            logger.other = other
            logger.other_history.append(other)
            logger.episodes_history.append(
                (cumulative_reward, episode_length, step, *[v for v in other.values()])
            )
        else:
            logger.episodes_history.append((cumulative_reward, episode_length, step))

        if len(logger.reward_window) > logger.window_size:
            logger.reward_window.pop(0)
            logger.episode_len_window.pop(0)

        logger._update(step[0])


def _plot_episodes_on_thread(block: bool) -> None:
    assert _RLTrainLogger._instance is not None, (
        "Training logger not initialized. You need to log_metrics before calling display_metrics."
    )
    assert len(_RLTrainLogger._instance.episodes_history) > 0, (
        "No metrics logged. You need to set retain_logs=True in display_metrics."
    )

    history: List[Tuple[float, int, tuple]] = _RLTrainLogger._instance.episodes_history
    other: dict[str, Any] = _RLTrainLogger._instance.other
    tpl = list(zip(*history))
    rewards = tpl[0]
    ep_lens = tpl[1]
    steps_tuple = tpl[2]
    # other we are not interested for now

    steps = [s[0] for s in steps_tuple]
    episodes = list(range(len(steps)))

    fig, axs = plt.subplots(len(tpl) - 1, 2, figsize=(8, 7))
    fig.canvas.manager.set_window_title("flashml")
    plt.subplots_adjust(
        left=0.07,
        right=0.95,
        top=0.92,
        bottom=0.15,
        wspace=0.1 * len(tpl),
        hspace=0.15 * len(tpl),
    )

    ax = axs[0, 0]
    ax.plot(steps, rewards, linestyle="-", linewidth=0.75)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward vs Step")
    ax.grid(True)

    ax = axs[0, 1]
    ax.plot(episodes, rewards, linestyle="-", linewidth=0.75)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Reward vs Episode")
    ax.grid(True)

    ax = axs[1, 0]
    ax.plot(steps, ep_lens, linestyle="-", linewidth=0.75, color="green")
    ax.set_xlabel("Step")
    ax.set_ylabel("Episode Length")
    ax.set_title("Episode Length vs Step")
    ax.grid(True)

    ax = axs[1, 1]
    ax.plot(episodes, ep_lens, linestyle="-", linewidth=0.75, color="green")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.set_title("Episode Length vs Episode")
    ax.grid(True)

    if len(tpl) > 3:
        new_elems_names = list(other.keys())
        for i in range(3, len(tpl)):
            elem = tpl[i]
            elem_name = new_elems_names[i - 3]
            ax = axs[i - 1, 0]
            ax.plot(steps, elem, linestyle="-", linewidth=0.75, color="purple")
            ax.set_xlabel("Step")
            ax.set_ylabel(elem_name)
            ax.set_title(f"{elem_name} vs Step")
            ax.grid(True)

            ax = axs[i - 1, 1]
            ax.plot(episodes, elem, linestyle="-", linewidth=0.75, color="purple")
            ax.set_xlabel("Episode")
            ax.set_ylabel(elem_name)
            ax.set_title(f"{elem_name} vs Episode")
            ax.grid(True)

    export_ax = fig.add_axes([0.85, 0.01, 0.12, 0.05])
    export_button = Button(export_ax, "Export CSV")
    export_button.on_clicked(lambda event: _export_values(history, other))

    plt.show(block=block)


def _export_values(history, other):
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )

    if file_path:
        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Episode",
                    "Reward",
                    "Episode Length",
                    "Step",
                    *([k for k in other.keys()] if other is not None else []),
                ]
            )
            for idx, tuplex in enumerate(history):
                writer.writerow([idx, *tuplex])


def plot_episodes(block=True) -> None:
    """
    Show the collected metrics from `log_episode` (can be called multiple times, even if the training didn't end). The graphs can be exported as csv.
    """

    _plot_episodes_on_thread(block)
