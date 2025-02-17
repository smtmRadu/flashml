from .log_episode import _RLTrainLogger
from typing import Any, Tuple, Union, List
import matplotlib.pyplot as plt
from threading import Thread
from tkinter import filedialog
import csv
from matplotlib.widgets import Button
import deprecation

@DeprecationWarning
def plot_table_with(x_label, y_label, title, x_ticks, values, history):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.canvas.manager.set_window_title("torchex")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.plot(x_ticks, values, linestyle='-', linewidth=0.75)
    ax.set_xlabel(xlabel=x_label)
    ax.set_ylabel(ylabel=y_label)
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    export_ax = plt.axes([0.85, 0.01, 0.12, 0.05])
    export_button = Button(export_ax, 'Export Data')
    export_button.on_clicked(lambda x: _export_values(history))
    plt.show()


def _display_episodes_on_thread() -> None:
    assert _RLTrainLogger._instance is not None, (
        "Training logger not initialized. You need to log_metrics before calling display_metrics."
    )
    assert len(_RLTrainLogger._instance.episodes_history) > 0, (
        "No metrics logged. You need to set retain_logs=True in display_metrics."
    )
    
    history: List[Tuple[float, int, tuple]] = _RLTrainLogger._instance.episodes_history

    rewards, ep_lens, steps_tuple = zip(*history)
    steps = [s[0] for s in steps_tuple]
    episodes = list(range(len(steps)))

    fig, axs = plt.subplots(2, 2, figsize=(8, 7))
    fig.canvas.manager.set_window_title("flashml")
    plt.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.15, wspace=0.3, hspace=0.3)

    ax = axs[0, 0]
    ax.plot(steps, rewards, linestyle='-', linewidth=0.75)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward vs Step")
    ax.grid(True)

    ax = axs[0, 1]
    ax.plot(episodes, rewards, linestyle='-', linewidth=0.75)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Reward vs Episode")
    ax.grid(True)

    ax = axs[1, 0]
    ax.plot(steps, ep_lens, linestyle='-', linewidth=0.75,  color="green")
    ax.set_xlabel("Step")
    ax.set_ylabel("Episode Length")
    ax.set_title("Episode Length vs Step")
    ax.grid(True)

    ax = axs[1, 1]
    ax.plot(episodes, ep_lens, linestyle='-', linewidth=0.75, color="green")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.set_title("Episode Length vs Episode")
    ax.grid(True)

    export_ax = fig.add_axes([0.85, 0.01, 0.12, 0.05])
    export_button = Button(export_ax, 'Export CSV')
    export_button.on_clicked(lambda event: _export_values(history))

    plt.show()

def _export_values(history):
        file_path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Episode', 'Reward', 'Episode Length', 'Step'])
                for idx, (reward, ep_len, step) in enumerate(history):
                    writer.writerow([idx, reward, ep_len, step[0]])

def display_episodes() -> None:
    '''
    Show the collected metrics from `log_episodes` without interrupting the training process. The graphs can be exported as csv.
    '''

    # for now they are disabled
    _display_episodes_on_thread()