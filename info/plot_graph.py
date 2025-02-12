import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import csv
from tkinter import filedialog, Tk

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
            writer.writerow([x_label, y_label])
            for x, y in zip(x_ticks, values):
                writer.writerow([x, y])
        print(f"Data exported to {file_path}")

def plot_graph(
        x_steps:list, 
        values:list, 
        x_label: str = "X", 
        y_label: str = "Y",
        color:str = "blue",
        linestyle:str = '-'):
    assert len(x_steps) == len(values), f"Length of x_steps must be equal to length of value (received {len(x_steps)} and {len(values)})"
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.canvas.manager.set_window_title("torchex")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.plot(x_steps, values, linestyle=linestyle, linewidth=0.75, color=color)
    ax.set_xlabel(xlabel=x_label)
    ax.set_ylabel(ylabel=y_label)
    ax.set_title(f"{y_label} vs {x_label}")
    ax.grid(True)
    plt.tight_layout()

    export_ax = plt.axes([0.85, 0.01, 0.12, 0.05])
    export_button = Button(export_ax, 'Export Data')
    export_button.on_clicked(lambda event: _export_values(x_label, y_label, x_steps, values))

    plt.show()