import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import csv
from tkinter import filedialog, Tk
from collections.abc import Iterable
import numpy as np

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
            
            # Handle multiple lines case
            if isinstance(values[0], Iterable):
                # Write header with series numbers
                header = [x_label] + [f"{y_label}_{i+1}" for i in range(len(values))]
                writer.writerow(header)
                
                # Write data
                for i in range(len(x_ticks)):
                    row = [x_ticks[i]] + [series[i] for series in values]
                    writer.writerow(row)
            else:
                # Single line case
                writer.writerow([x_label, y_label])
                for x, y in zip(x_ticks, values):
                    writer.writerow([x, y])
                    
        print(f"Data exported to {file_path}")

def plot_graph(
        x_steps:list, 
        values:list | tuple[list] | list[list], 
        x_label: str = "X", 
        y_label: str = "Y",
        color:str | list[str] = None,
        linestyle:str | list[str] = '-'):
    """
    Plot single or multiple line graphs with export functionality.
    
    Args:
        x_steps (list): Values for x-axis
        values (list|tuple[list]|list[list]): Single list for one line or list of lists for multiple lines
        x_label (str): Label for x-axis
        y_label (str): Label for y-axis
        color (str|list[str]): Single color or list of colors for multiple lines. If None, uses automatic colors
        linestyle (str|list[str]): Single line style or list of styles for multiple lines
    """
    assert len(x_steps) == len(values) if not isinstance(values[0], Iterable) else len(x_steps) == len(values[0]), \
        f"Length of x_steps must be equal to length of values"

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.canvas.manager.set_window_title("torchex")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    if isinstance(values[0], Iterable):
        # Handle multiple lines
        n_lines = len(values)
        
        # Auto-generate colors if not provided
        if color is None:
            # Use matplotlib's color cycle
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            # If we have more lines than default colors, use color map
            if n_lines > len(colors):
                colors = plt.cm.tab20(np.linspace(0, 1, n_lines))
        else:
            colors = [color] * n_lines if isinstance(color, str) else color
            colors = colors[:n_lines] + [plt.cm.tab20(i) for i in np.linspace(0, 1, n_lines - len(colors))]
        
        # Handle line styles
        styles = [linestyle] * n_lines if isinstance(linestyle, str) else linestyle
        styles = styles[:n_lines] + ['-'] * (n_lines - len(styles))
            
        # Plot each line
        for i, each in enumerate(values):
            ax.plot(x_steps, each, 
                   linestyle=styles[i], 
                   linewidth=0.75, 
                   color=colors[i],
                   label=f'Series {i+1}')
        ax.legend()
    else:
        # Single line case
        ax.plot(x_steps, values, 
               linestyle=linestyle, 
               linewidth=0.75, 
               color='blue' if color is None else color)
    
    ax.set_xlabel(xlabel=x_label)
    ax.set_ylabel(ylabel=y_label)
    ax.set_title(f"{y_label} vs {x_label}")
    ax.grid(True)
    plt.tight_layout()

    export_ax = plt.axes([0.85, 0.01, 0.12, 0.05])
    export_button = Button(export_ax, 'Export Data')
    export_button.on_clicked(lambda event: _export_values(x_label, y_label, x_steps, values))

    plt.show()