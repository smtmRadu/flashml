from .log_metrics import _TrainingLogger_epoch_batch, _TrainingLogger_step
from typing import Any, Tuple, List
import matplotlib.pyplot as plt
from tkinter import filedialog
import csv
from matplotlib.widgets import Button


def _display_metrics_on_thread(show_epoch_ticks:bool = False) -> None:
    '''
    Show the collected metrics from `log_metrics` at the end of the training. The graphs can be exported as csv.
    The labels are: `Metric` (metric name), `Value` (metric value), `Epoch` (epoch idx), `Batch Index` (batch idx in epoch), `Step` (optim step)
    '''
    if _TrainingLogger_epoch_batch._instance is not None and len(_TrainingLogger_epoch_batch._instance.history_log) > 0 :
        history: List[Tuple[dict[str, Any], int, int]] = _TrainingLogger_epoch_batch._instance.history_log

        metrics_rec, epochs_rec, rec = zip(*history) #all elements are tuples
        names =  metrics_rec[0].keys()

        batch_num = len(set(rec))
        epoch_idx_scaled_at_step = [(e+1)*batch_num for e in set(epochs_rec)]
        
        steps = [e*batch_num + b for e, b in zip(epochs_rec, rec)]
        for idx, name in enumerate(names):
            values_for_name = [v if isinstance(v, float) else v[name] for v in metrics_rec]
            
            fig, ax = plt.subplots(figsize=(8, 5))
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
            export_button = Button(export_ax, 'Export CSV')
            export_button.on_clicked(lambda x: _export_values_batch_epoch(history, metric_name= name))
            plt.show()
         
    elif _TrainingLogger_step._instance is not None and len(_TrainingLogger_step._instance.history_log) > 0 :
        history: List[Tuple[dict[str, Any], int]] = _TrainingLogger_step._instance.history_log
        metrics_rec, step_rec = zip(*history) #all elements are tuples
        names =  metrics_rec[0].keys()
        steps = list(step_rec)
        for idx, name in enumerate(names):
            values_for_name = [v if isinstance(v, float) else v[name] for v in metrics_rec]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.canvas.manager.set_window_title("torchex")
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax.plot(steps, values_for_name, linestyle='-', linewidth=0.75)
            
            ax.set_xlabel("Step")
            ax.set_ylabel(name)
            ax.set_title(name + " vs Step")
            ax.grid(True)
            plt.tight_layout()
        
            export_ax = plt.axes([0.85, 0.01, 0.12, 0.05])
            export_button = Button(export_ax, 'Export CSV')
            export_button.on_clicked(lambda x: _export_values_step(history, metric_name= name))
            plt.show()

    else:
         raise "Training logger was not initialized. You need to `log_metrics` or `log_metrics2` before calling `display_metrics`"

def _export_values_step(history, metric_name):
    file_path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
    if file_path:
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value', 'Step'])

            for idx, (dct, step) in enumerate(history):
                writer.writerow([metric_name, dct[metric_name], step])

def _export_values_batch_epoch(history, metric_name):
        file_path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Metric', 'Value', 'Epoch', 'Batch', 'Step'])
                
                _, epochs_rec, rec = zip(*history)
                batch_num = len(set(rec))
                steps = [e*batch_num + b for e, b in zip(epochs_rec, rec)]

                for idx, (dct, iter, b_idx) in enumerate(history):
                    writer.writerow([metric_name, dct[metric_name], iter, b_idx, steps[idx]])

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