from .training_log_progress import _TrainingLogger
from typing import Any, Tuple, Union, List
import matplotlib.pyplot as plt



def display_metrics(show_epoch_ticks:bool = False) -> None:
    # ... (Assertions remain the same)

    history: List[Tuple[Any, Any, int, int]] = _TrainingLogger._instance.history_log

    names, values, epochs_rec, rec = zip(*history)
    names = names[0]
    batch_num = len(set(rec))
    epochs = [e+1 for e in epochs_rec]
    epoch_idx_scaled_at_step = [(e+1)*batch_num for e in set(epochs_rec)]
    
    steps = [e*batch_num + b for e, b in zip(epochs_rec, rec)]
    for idx, name in enumerate(names):
        values_for_name = [v[idx] for v in values]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, values_for_name, linestyle='-', linewidth=0.75)
        # ax.set_xlabel("Step")
        ax.set_ylabel(name)
        ax.set_title(name + " vs Step/Epoch")

        if show_epoch_ticks:
            sec2 = ax.secondary_xaxis('bottom')
            sec2.set_xticks( epoch_idx_scaled_at_step)  
            sec2.set_xticklabels([x//batch_num for x in epoch_idx_scaled_at_step])  
            sec2.tick_params(axis='x', length=30, width=1) 

        # Add secondary x-axis for epochs
        # sec_ax = ax.secondary_xaxis(location=0)
        # sec_ax.set_xticks(epoch_idx_scaled_at_step, labels=[f"Epoch {i+1}" for i in range(len(epoch_idx_scaled_at_step))])
        # sec_ax.tick_params(axis='x', rotation=0, length=0)

        plt.tight_layout()
        plt.show()
