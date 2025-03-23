import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import csv
from tkinter import filedialog, Tk
from collections.abc import Iterable
import numpy as np
import threading

def _export_values(x_label: str, y_label: str, x_ticks, values):
    # Hide the root Tkinter window
    root = Tk()
    root.withdraw()

    file_path = filedialog.asksaveasfilename(
        defaultextension='.csv',
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if file_path:
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            if isinstance(values[0], Iterable):
                header = [x_label] + [f"{y_label}_{i+1}" for i in range(len(values))]
                writer.writerow(header)
                for i in range(len(x_ticks)):
                    row = [x_ticks[i]] + [series[i] for series in values]
                    writer.writerow(row)
            else:
                writer.writerow([x_label, y_label])
                for x, y in zip(x_ticks, values):
                    writer.writerow([x, y])
                    
        print(f"Data exported to {file_path}")

    
def plot_graph(
        values:list | tuple[list] | list[list], 
        steps:list = None,    
        x_label: str = "X", 
        y_label: str = "Y",
        color:str | list[str] = None,
        linestyle:str | list[str] = '-',
        marker:str | list[str] = '',
        blocking:bool=True):
    """
    Plot single or multiple line graphs with export functionality.
    
    Args:
        values (list|tuple[list]|list[list]): Single list for one line or list of lists for multiple lines
        steps (list): Values for x-axis
        
        x_label (str): Label for x-axis
        y_label (str): Label for y-axis
        color (str|list[str]): Single color or list of colors for multiple lines. If None, uses automatic colors
        linestyle (str|list[str]): Single line style or list of styles for multiple lines
    """
    if steps != None:
        assert len(steps) == len(values) if not isinstance(values[0], Iterable) else len(steps) == len(values[0]), \
            f"Length of teps must be equal to length of values"

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.canvas.manager.set_window_title("flashml")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    if isinstance(values[0], Iterable):
        if steps is None:
            steps = [i for i in range(len(values[0]))]

        n_lines = len(values)
        if color is None:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            if n_lines > len(colors):
                colors = plt.cm.tab20(np.linspace(0, 1, n_lines))
        else:
            colors = [color] * n_lines if isinstance(color, str) else color
            colors = colors[:n_lines] + [plt.cm.tab20(i) for i in np.linspace(0, 1, n_lines - len(colors))]

        styles = [linestyle] * n_lines if isinstance(linestyle, str) else linestyle
        styles = styles[:n_lines] + ['-'] * (n_lines - len(styles))
            
        for i, each in enumerate(values):
            ax.plot(steps, each, 
                   linestyle=styles[i],
                   marker=marker, 
                   linewidth=min(100/len(values), 1), 
                   color=colors[i],
                   label=f'Series {i+1}')
        ax.legend()
    else:
        if steps is None:
            steps = [i for i in range(len(values))]
        ax.plot(steps, values, 
               linestyle=linestyle, 
               marker=marker,
               linewidth=min(100/len(values), 1), 
               color='blue' if color is None else color)
    
    ax.set_xlabel(xlabel=x_label)
    ax.set_ylabel(ylabel=y_label)
    ax.set_title(f"{y_label} vs {x_label}")
    ax.grid(True)
    plt.tight_layout()

    export_ax = plt.axes([0.85, 0.01, 0.12, 0.05])
    export_button = Button(export_ax, 'Export CSV')
    export_button.on_clicked(lambda event: _export_values(x_label, y_label, steps, values))

    plt.show(block=blocking)