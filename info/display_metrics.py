from .log_metrics import _TrainingLogger
from typing import Any, Tuple, Union, List
import matplotlib.pyplot as plt
from threading import Thread
from tkinter import filedialog
import csv
from matplotlib.widgets import Button


def _display_metrics_on_thread(show_epoch_ticks:bool = False) -> None:
    '''
    Show the collected metrics from `log_metrics` at the end of the training. The graphs can be exported as csv.
    The labels are: `Metric` (metric name), `Value` (metric value), `Epoch` (epoch idx), `Batch Index` (batch idx in epoch), `Step` (optim step)
    '''
    assert _TrainingLogger._instance is not None, "Training logger not initialized. You need to log_metrics before calling display_metrics."
    assert len(_TrainingLogger._instance.history_log) > 0, "No metrics logged. You need to set retain_logs=True in display_metrics."
    
    history: List[Tuple[Any, Any, int, int]] = _TrainingLogger._instance.history_log

    names, values, epochs_rec, rec = zip(*history)
    names = [names[0]] if isinstance(names[0], str) else names[0]
    batch_num = len(set(rec))
    epoch_idx_scaled_at_step = [(e+1)*batch_num for e in set(epochs_rec)]
    
    steps = [e*batch_num + b for e, b in zip(epochs_rec, rec)]
    for idx, name in enumerate(names):
        values_for_name = [v if isinstance(v, float) else v[idx] for v in values]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.canvas.manager.set_window_title("torchex")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.plot(steps, values_for_name, linestyle='-', linewidth=0.75)
        
        ax.set_xlabel("Step | Epoch" if show_epoch_ticks else "Step")
        ax.set_ylabel(name)
        ax.set_title(name + " vs Step/Epoch" if show_epoch_ticks else name + " vs Step")
        ax.grid(True)
        if show_epoch_ticks:
            sec2 = ax.secondary_xaxis('bottom')
            sec2.set_xticks( epoch_idx_scaled_at_step)  
            sec2.set_xticklabels([x//batch_num for x in epoch_idx_scaled_at_step])  
            sec2.tick_params(axis='x', length=30, width=1) 

        plt.tight_layout()
       
        export_ax = plt.axes([0.85, 0.01, 0.12, 0.05])
        export_button = Button(export_ax, 'Export Data')
        export_button.on_clicked(lambda x: _export_values(history))
        plt.show()

def _export_values(history):
        file_path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Metric', 'Value', 'Epoch', 'Batch', 'Step'])
                
                names, values, epochs_rec, rec = zip(*history)
                batch_num = len(set(rec))
                steps = [e*batch_num + b for e, b in zip(epochs_rec, rec)]

                for idx, (name, value, iter, b_idx) in enumerate(history):
                    writer.writerow([name, value, iter, b_idx, steps[idx]])

def display_metrics(show_epoch_ticks:bool = False) -> None:
    '''
    Show the collected metrics from `log_metrics` without interrupting the training process. The graphs can be exported as csv.
    '''
    # assert _TrainingLogger._instance is not None, "Training logger not initialized. You need to log_metrics before calling display_metrics."
    # assert len(_TrainingLogger._instance.history_log) > 0, "No metrics logged. You need to set retain_logs=True in display_metrics."
    # 
    # thread = Thread(target=_display_metrics_on_thread, args=(show_epoch_ticks,))
    # thread.start()

    # for now they are disabled
    _display_metrics_on_thread(show_epoch_ticks)