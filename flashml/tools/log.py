from typing import Any
import tkinter as tk
from tkinter import scrolledtext, filedialog
from datetime import datetime
from flashml.tools.colors import *

_LOGS = []


def log(msg: Any, color: str = 'white', print_:bool=True) -> None:
    """
    Logs a message in a ledger of logs. Also prints it in the console using ANSI color codes.
    The original color value is saved with the log entry for use in the GUI.
    
    Args:
        msg (Any): The message to log.
        color (str, optional): The color name or hex value (e.g., '#RRGGBB'). Defaults to 'white'.
    """
    global _LOGS
    RESET = '\033[0m'
    
    # Determine terminal color code
    if color.startswith('#'):
        try:
            color_code = hex_to_ansi(color)
        except ValueError:
            color_code = ansi_of('white')
    else:
        color_code = ansi_of(color)
    
    msg_str = str(msg)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Save log entry with the original color value (so the GUI can use it directly)
    _LOGS.append({'timestamp': timestamp, 'message': msg_str, 'color': color})
    if print_:
        print(f"{color_code}{msg_str}{RESET}")

def display_logs() -> None:
    """
    Display all the logs in a GUI window with timestamps and an export option.
    Logs are shown in chronological order, separated like blocks, with scrolling.
    """
    global _LOGS
    root = tk.Tk()
    root.title("Log Viewer")
    root.configure(bg="#212121")  # Set the dark background color

    # Styled text area for log display
    text_area = scrolledtext.ScrolledText(
        root, wrap=tk.WORD, width=80, height=20, 
        bg="#2E2E2E", fg="white", insertbackground="white",
        font=("Consolas", 10), relief=tk.FLAT
    )
    text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    for log in _LOGS:
        timestamp = log['timestamp']
        message = log['message']
        text_area.insert(tk.END, f"[{timestamp}] {message}\n", "log")
        text_area.insert(tk.END, "-" * 50 + "\n", "separator")

    # Style tags for different parts of the logs
    text_area.tag_config("log", foreground="lightgray")
    text_area.tag_config("separator", foreground="gray")
    
    text_area.configure(state='disabled')

    def export_logs():
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, 'w') as f:
                for log in _LOGS:
                    f.write(f"[{log['timestamp']}] {log['message']}\n")

    # Styled export button
    export_button = tk.Button(
        root, text="Export Logs", command=export_logs, 
        bg="#424242", fg="white", activebackground="#616161", 
        activeforeground="white", font=("Arial", 10, "bold"), relief=tk.FLAT
    )
    export_button.pack(pady=5)

    root.mainloop()