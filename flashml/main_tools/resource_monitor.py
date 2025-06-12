import time
import collections
import os
import sys
import psutil
import tempfile
import subprocess
import threading

ALWAYS_ON_TOP = True

UPDATE_INTERVAL_MS = 250
HISTORY_LENGTH = 1000
CHART_HEIGHT = 50
BAR_LENGTH = 100

BG_COLOR = "#24292e"
FG_COLOR = "#c9d1d9"
BORDER_COLOR = "#444c56"
CANVAS_BG_COLOR = "#2d333b"
CANVAS_GRID_COLOR = BORDER_COLOR

CPU_ACCENT_COLOR = "#58a6ff"
CPU_FRAME_BORDER_COLOR = "#2a5487"
CPU_PROGRESS_BAR_BG = "#1e3a5c"

RAM_ACCENT_COLOR = "#f1e05a"
RAM_FRAME_BORDER_COLOR = "#7f742f"
RAM_PROGRESS_BAR_BG = "#4d461c"

GPU_ACCENT_COLOR = "#3fb950"
GPU_FRAME_BORDER_COLOR = "#2b6134"
GPU_PROGRESS_BAR_BG = "#22442a"

IO_ACCENT_COLOR = FG_COLOR
IO_FRAME_BORDER_COLOR = BORDER_COLOR  #


LOCKFILE = os.path.join(tempfile.gettempdir(), "resource_monitor.lock")

last_net_io = None
last_disk_io = None
last_time = None
nvml_handles = []
nvml_initialized = False
_pynvml_imported = False
pynvml = None
nvml_lock = threading.Lock()
_monitor_active_flag = False


def is_monitor_running():
    if os.path.exists(LOCKFILE):
        try:
            with open(LOCKFILE, "r") as f:
                pid = int(f.read().strip())
            # Check if PID exists and if the command line looks like python
            # This is a basic check, might not be foolproof
            if psutil.pid_exists(pid):
                try:
                    proc = psutil.Process(pid)
                    # Simple check if it's a python process, could be refined
                    if "python" in proc.name().lower() or "py" in proc.name().lower():
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass  # Process died between checks or we lack permissions
            print("Removing stale lock file...")
            remove_lockfile()
        except (ValueError, FileNotFoundError, EnvironmentError) as e:
            print(f"Error checking lock file, removing it: {e}")
            remove_lockfile()
        except Exception as e:
            print(f"Unexpected error checking lock file: {e}")
            pass  # Ignore other errors like permission denied
    return False


def write_lockfile():
    try:
        with open(LOCKFILE, "w") as f:
            f.write(str(os.getpid()))
    except EnvironmentError as e:
        print(f"Error writing lock file: {e}")


def remove_lockfile():
    try:
        # Check if the lock file still belongs to the current process before removing
        if os.path.exists(LOCKFILE):
            try:
                with open(LOCKFILE, "r") as f:
                    pid_in_file = int(f.read().strip())
                if pid_in_file == os.getpid():
                    os.remove(LOCKFILE)
                # else:
                #    print(f"Lock file {LOCKFILE} belongs to another process ({pid_in_file}), not removing.")
            except (ValueError, FileNotFoundError):
                # If reading fails, maybe remove anyway or log? For simplicity, try removing.
                try:
                    os.remove(LOCKFILE)
                except EnvironmentError as e_rem:
                    print(
                        f"Warning: Could not remove potentially corrupt lock file {LOCKFILE}: {e_rem}"
                    )
            except EnvironmentError as e_rem:
                print(f"Warning: Could not remove lock file {LOCKFILE}: {e_rem}")

    except Exception as e:
        print(f"Unexpected error removing lock file: {e}")


def _bytes2human(n):
    if not isinstance(n, (int, float)):
        return "N/A"
    symbols = ("B", "KB", "MB", "GB", "TB", "PB")
    prefix = {s: 1 << (i * 10) for i, s in enumerate(symbols)}
    for s in reversed(symbols):
        if n >= prefix[s]:
            try:
                value = float(n) / prefix[s]
                return f"{value:.2f} {s}"
            except ValueError:
                return "Error"
    try:
        return f"{int(n)} B"
    except (ValueError, TypeError):
        return "N/A B"


def initialize_nvml():
    global nvml_initialized, nvml_handles, _pynvml_imported, pynvml
    with nvml_lock:  # Acquire lock
        if nvml_initialized or _pynvml_imported:
            return nvml_initialized
        try:
            # print("Attempting to import pynvml...")
            import pynvml as pynvml_lib  # Import with a different local name

            pynvml = pynvml_lib  # Assign to global name
            # print("pynvml imported successfully.")
            _pynvml_imported = True
            # print("Initializing NVML...")
            pynvml.nvmlInit()
            # print("NVML Initialized.")
            nvml_handles = []
            gpu_count = pynvml.nvmlDeviceGetCount()
            # print(f"Found {gpu_count} GPU(s).")
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                # --- THIS LINE IS CHANGED ---
                gpu_name = pynvml.nvmlDeviceGetName(
                    handle
                )  # Already a string in this version
                # --- END OF CHANGE ---
                nvml_handles.append({"handle": handle, "name": gpu_name})
                # print(f"  GPU {i}: {gpu_name}")
            nvml_initialized = True
            # print("NVML setup complete.")
            return True
        except ImportError:
            print("pynvml library not found. GPU monitoring disabled.")
            _pynvml_imported = True
            return False  # Set _pynvml_imported so we don't try again
        except Exception as e:
            err_str = f"{e}"
            # Ensure pynvml and necessary attributes exist before trying to use them
            if (
                _pynvml_imported
                and pynvml
                and hasattr(pynvml, "NVMLError")
                and isinstance(e, pynvml.NVMLError)
            ):
                # Attempt to get error string only if pynvml and NVMLError are available
                try:
                    # NVMLError often returns the error code in args[0]
                    error_code = e.args[0] if e.args else "N/A"
                    err_str = f"{pynvml.nvmlErrorString(e)} (Code: {error_code})"
                except Exception as e_str:
                    print(
                        f"Could not get NVML error string: {e_str}"
                    )  # Log if getting string fails
                    # Fallback to default error string if nvmlErrorString fails
                    pass  # Keep the original err_str
            print(f"Failed to initialize NVML: {err_str}")
            # Attempt shutdown only if pynvml was imported and initialization *might* have partially succeeded
            if _pynvml_imported and pynvml:
                try:
                    # print("Attempting NVML shutdown after initialization error...")
                    pynvml.nvmlShutdown()
                    # print("NVML shutdown successful.")
                except Exception as sd_e:
                    # Don't print shutdown error if it's 'NVML_ERROR_UNINITIALIZED' - expected here
                    is_uninitialized_error = False
                    if hasattr(pynvml, "NVMLError") and isinstance(
                        sd_e, pynvml.NVMLError
                    ):
                        try:
                            if (
                                sd_e.args
                                and sd_e.args[0] == pynvml.NVML_ERROR_UNINITIALIZED
                            ):
                                is_uninitialized_error = True
                        except Exception:
                            pass
                    if not is_uninitialized_error:
                        print(f"Error during NVML shutdown after init failure: {sd_e}")

            nvml_initialized = False
            nvml_handles = []
            return False


def shutdown_nvml():
    global nvml_initialized
    with nvml_lock:  # Acquire lock
        # print(f"Shutdown called. NVML Initialized: {nvml_initialized}, pynvml imported: {_pynvml_imported}")
        if nvml_initialized and _pynvml_imported and pynvml:
            try:
                # print("Shutting down NVML...");
                pynvml.nvmlShutdown()
            # print("NVML shut down successfully.")
            except Exception as e:
                err_str = f"{e}"
                if hasattr(pynvml, "NVMLError") and isinstance(e, pynvml.NVMLError):
                    try:
                        err_str = f"{pynvml.nvmlErrorString(e)} (Code: {e.args[0] if e.args else 'N/A'})"
                    except Exception:
                        pass
                print(f"Error shutting down NVML: {err_str}")
            finally:
                nvml_initialized = False
                nvml_handles.clear()
                # print("NVML state reset.")
        # else:
        # print("NVML shutdown condition not met, skipping.")


def get_cpu_temperature():
    if not hasattr(psutil, "sensors_temperatures"):
        return None
    return None


def get_gpu_info(gpu_index):
    global nvml_initialized, _pynvml_imported, pynvml  # Ensure pynvml is accessible
    if (
        not nvml_initialized
        or not _pynvml_imported
        or not pynvml
        or gpu_index >= len(nvml_handles)
    ):
        return {
            "temp": None,
            "util": None,
            "vram_used": 0,
            "vram_total": 0,
            "vram_percent": 0.0,
        }

    handle_info = nvml_handles[gpu_index]
    handle = handle_info["handle"]
    temp, util = None, None
    vram_used, vram_total, vram_percent = 0, 0, 0.0

    try:
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except Exception:
        pass  # Ignore errors (might happen temporarily)
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    except Exception:
        pass
    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_used, vram_total = mem_info.used, mem_info.total
        vram_percent = (vram_used / vram_total) * 100 if vram_total else 0
    except Exception:
        pass
    return {
        "temp": temp,
        "util": util,
        "vram_used": vram_used,
        "vram_total": vram_total,
        "vram_percent": vram_percent,
    }


def get_all_stats():
    global last_net_io, last_disk_io, last_time
    current_time = time.time()
    disk_read_speed, disk_write_speed = 0, 0
    net_sent_speed, net_recv_speed = 0, 0

    current_disk_io = psutil.disk_io_counters()
    current_net_io = psutil.net_io_counters()

    if last_time is not None and current_time > last_time:
        time_delta = current_time - last_time
        if current_disk_io and last_disk_io:
            read_bytes = current_disk_io.read_bytes - last_disk_io.read_bytes
            write_bytes = current_disk_io.write_bytes - last_disk_io.write_bytes
            disk_read_speed = max(0, read_bytes) / time_delta
            disk_write_speed = max(0, write_bytes) / time_delta
        if current_net_io and last_net_io:
            sent_bytes = current_net_io.bytes_sent - last_net_io.bytes_sent
            recv_bytes = current_net_io.bytes_recv - last_net_io.bytes_recv
            net_sent_speed = max(0, sent_bytes) / time_delta
            net_recv_speed = max(0, recv_bytes) / time_delta

    # Update lasts for the *next* iteration
    last_disk_io = current_disk_io
    last_net_io = current_net_io
    last_time = current_time

    # Fetch other stats
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_temp = get_cpu_temperature()
    mem = psutil.virtual_memory()
    mem_percent, mem_used, mem_total = mem.percent, mem.used, mem.total

    gpus_info = []
    if nvml_initialized:
        for i in range(len(nvml_handles)):
            gpus_info.append({"name": nvml_handles[i]["name"], **get_gpu_info(i)})

    return {
        "cpu_percent": cpu_percent,
        "cpu_temp": cpu_temp,
        "mem_percent": mem_percent,
        "mem_used": mem_used,
        "mem_total": mem_total,
        "disk_read_speed": disk_read_speed,
        "disk_write_speed": disk_write_speed,
        "net_sent_speed": net_sent_speed,
        "net_recv_speed": net_recv_speed,
        "gpus": gpus_info,
    }


class ResourceMonitorApp:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.root.title("flashml Resource Monitor")
        self.root.configure(bg=BG_COLOR)

        # Determine number of GPUs *after* NVML initialization attempt
        num_gpus = len(nvml_handles) if nvml_initialized else 0

        self.history = {
            "cpu": collections.deque([0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH),
            "mem": collections.deque([0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH),
            # Initialize GPU history lists based on actual detected GPUs
            "gpus_util": [
                collections.deque([0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
                for _ in range(num_gpus)
            ],
            "gpus_vram": [
                collections.deque([0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
                for _ in range(num_gpus)
            ],
        }
        self.gpu_widgets = [{} for _ in range(num_gpus)]
        self.gpu_util_chart_canvases = [None] * num_gpus
        self.gpu_vram_chart_canvases = [None] * num_gpus

        # --- Configure Styles ---
        self.style = ttk.Style()
        available_themes = self.style.theme_names()
        theme_to_use = "clam" if "clam" in available_themes else "default"
        try:
            self.style.theme_use(theme_to_use)
        except tk.TclError:
            pass

        # --- Base Styles ---
        self.style.configure(
            ".", background=BG_COLOR, foreground=FG_COLOR, font=("Segoe UI", 9)
        )
        self.style.configure("TFrame", background=BG_COLOR)
        # Default Label Style (for "Utilization:", "History:", etc.)
        self.style.configure("BG.TLabel", background=BG_COLOR, foreground=FG_COLOR)
        self.style.configure(
            "BGUnit.TLabel",
            background=BG_COLOR,
            foreground=FG_COLOR,
            font=("Segoe UI", 8),
        )
        self.style.configure(
            "Error.TLabel",
            background=BG_COLOR,
            foreground="#ffcc00",
            font=("Segoe UI", 9),
        )

        # --- Section Specific Styles ---
        for section, accent, border, prog_bg in [
            ("CPU", CPU_ACCENT_COLOR, CPU_FRAME_BORDER_COLOR, CPU_PROGRESS_BAR_BG),
            ("RAM", RAM_ACCENT_COLOR, RAM_FRAME_BORDER_COLOR, RAM_PROGRESS_BAR_BG),
            ("GPU", GPU_ACCENT_COLOR, GPU_FRAME_BORDER_COLOR, GPU_PROGRESS_BAR_BG),
            (
                "IO",
                IO_ACCENT_COLOR,
                IO_FRAME_BORDER_COLOR,
                None,
            ),  # IO has no progress bar
        ]:
            # Data Label Style (colored)
            self.style.configure(
                f"{section}.Data.TLabel",
                background=BG_COLOR,
                foreground=accent,
                font=("Segoe UI", 9, "bold"),
            )
            # LabelFrame Style (tinted border)
            self.style.configure(
                f"{section}.TLabelframe",
                background=BG_COLOR,
                bordercolor=border,
                relief="groove",
                borderwidth=1,
            )
            self.style.configure(
                f"{section}.TLabelframe.Label",
                background=BG_COLOR,
                foreground=FG_COLOR,
                font=("Segoe UI", 10, "bold"),
            )
            # Progress Bar Style
            if prog_bg:
                self.style.configure(
                    f"{section}.Horizontal.TProgressbar",
                    troughcolor=prog_bg,
                    background=accent,
                    bordercolor=border,
                    lightcolor=accent,
                    darkcolor=accent,
                )

        # --- Main Frame ---
        main_frame = ttk.Frame(self.root, padding="10", style="TFrame")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(
            0, weight=1
        )  # Allow content to expand horizontally

        # --- Create Sections (passing section type for styling) ---
        self._create_cpu_section(main_frame, 0)
        self._create_mem_section(main_frame, 1)
        # _create_gpu_sections now correctly uses the initialized nvml_handles
        next_row = self._create_gpu_sections(main_frame, 2)

        io_container_frame = ttk.Frame(main_frame, style="TFrame")
        # Make IO container span the full width
        io_container_frame.grid(row=next_row, column=0, sticky="ew", padx=0, pady=5)
        # Configure columns inside the IO container to share space
        io_container_frame.grid_columnconfigure(0, weight=1)
        io_container_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(
            next_row, weight=0
        )  # IO section shouldn't expand vertically

        self._create_disk_section(io_container_frame)  # Uses "IO" styles implicitly
        self._create_network_section(io_container_frame)  # Uses "IO" styles implicitly

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._update_running = False
        self._after_id = None  # Store the id of the scheduled 'after' job
        self.root.after(100, self.start_update_loop)  # Start the loop safely

    def start_update_loop(self):
        """Safely starts the periodic update."""
        if not self._update_running:
            self.update_gui()

    # --- Modified Helper Functions to Accept Styles ---
    def _create_section_frame(self, parent, title, section_prefix):
        from tkinter import ttk

        """Creates a LabelFrame using the section-specific style."""
        frame = ttk.LabelFrame(
            parent, text=title, style=f"{section_prefix}.TLabelframe", padding="5"
        )
        frame.grid_columnconfigure(1, weight=1)  # Allow value column to expand
        return frame, 0

    def _add_info_row(self, frame, row_idx, label_text, section_prefix):
        from tkinter import ttk

        """Adds an info row using section-specific styles for data/unit."""
        # Standard label uses the default background style
        label = ttk.Label(frame, text=label_text, style="BG.TLabel")
        label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=1)
        # Data value label uses the section-specific colored style
        # Use anchor "e" (east) for right alignment within its grid cell
        value_label = ttk.Label(
            frame,
            text="N/A",
            style=f"{section_prefix}.Data.TLabel",
            anchor="e",
            width=15,
        )
        value_label.grid(
            row=row_idx, column=1, sticky="ew", padx=5, pady=1
        )  # sticky="ew" makes it expand
        # Unit label uses the default background style (or a specific Unit style if defined)
        unit_label = ttk.Label(
            frame, text="", style="BGUnit.TLabel", anchor="w"
        )  # anchor="w" keeps it left-aligned
        unit_label.grid(row=row_idx, column=2, sticky="w", padx=5, pady=1)
        return value_label, unit_label

    def _add_progress_bar(self, frame, row_idx, section_prefix):
        """Adds a progress bar using the section-specific style."""
        from tkinter import ttk

        pb = ttk.Progressbar(
            frame,
            orient="horizontal",
            length=BAR_LENGTH,
            mode="determinate",
            style=f"{section_prefix}.Horizontal.TProgressbar",
        )
        pb.grid(row=row_idx, column=0, columnspan=3, sticky="ew", padx=5, pady=2)
        return pb

    def _add_chart(self, frame, row_idx, data_key, gpu_index=None, label=True):
        import tkinter as tk
        from tkinter import ttk

        if label:
            chart_label = ttk.Label(frame, text="History:", style="BGUnit.TLabel")
            chart_label.grid(row=row_idx, column=0, sticky="w", padx=5, pady=(5, 0))
        canvas = tk.Canvas(
            frame,
            height=CHART_HEIGHT,
            width=BAR_LENGTH,
            bg=CANVAS_BG_COLOR,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
        )
        canvas.grid(
            row=row_idx + (1 if label else 0),
            column=0,
            columnspan=3,
            sticky="ew",
            padx=5,
            pady=(0, 5),
        )

        if gpu_index is None:
            if data_key == "cpu":
                self.cpu_chart_canvas = canvas
            elif data_key == "mem":
                self.mem_chart_canvas = canvas
        else:
            if data_key == "util" and gpu_index < len(self.gpu_util_chart_canvases):
                self.gpu_util_chart_canvases[gpu_index] = canvas
            elif data_key == "vram" and gpu_index < len(self.gpu_vram_chart_canvases):
                self.gpu_vram_chart_canvases[gpu_index] = canvas
        return row_idx + (2 if label else 1)

    # --- Section Creation Methods (using prefixes) ---
    def _create_cpu_section(self, parent, row_index):
        import tkinter as tk
        from tkinter import ttk

        prefix = "CPU"
        frame, next_row = self._create_section_frame(parent, "CPU", prefix)
        frame.grid(row=row_index, column=0, sticky="ew", padx=5, pady=5)
        self.cpu_util_val, self.cpu_util_unit = self._add_info_row(
            frame, next_row, "Utilization:", prefix
        )
        self.cpu_util_unit.configure(text="%")
        self.cpu_pb = self._add_progress_bar(frame, next_row + 1, prefix)
        next_row += 2

        # Create a sub-frame for the chart and temperature
        chart_frame = ttk.Frame(frame, style="TFrame")
        chart_frame.grid(
            row=next_row, column=0, columnspan=3, sticky="ew", padx=5, pady=(5, 0)
        )

        # Add chart canvas with reduced width
        self.cpu_chart_canvas = tk.Canvas(
            chart_frame,
            height=CHART_HEIGHT,
            width=BAR_LENGTH // 2,
            bg=CANVAS_BG_COLOR,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
        )
        self.cpu_chart_canvas.pack(side="left", fill="x", expand=True)

        # Add temperature label to the right of the chart
        self.cpu_temp_val = ttk.Label(
            chart_frame,
            text="N/A",
            style=f"{prefix}.Data.TLabel",
            font=("Segoe UI", 14, "bold"),
        )
        self.cpu_temp_val.pack(side="right", padx=(5, 0))

        # Update the temperature unit label (not needed as we're displaying temperature directly)
        # self.cpu_temp_unit = ttk.Label(chart_frame, text="°C", style='BGUnit.TLabel')
        # self.cpu_temp_unit.pack(side="right")

        return

    def _create_mem_section(self, parent, row_index):
        prefix = "RAM"
        frame, next_row = self._create_section_frame(parent, "RAM", prefix)
        frame.grid(row=row_index, column=0, sticky="ew", padx=5, pady=5)
        self.mem_usage_val, self.mem_usage_unit = self._add_info_row(
            frame, next_row, "Usage:", prefix
        )
        self.mem_percent_val, self.mem_percent_unit = self._add_info_row(
            frame, next_row + 1, "Percent:", prefix
        )
        self.mem_percent_unit.configure(text="%")
        self.mem_pb = self._add_progress_bar(frame, next_row + 2, prefix)
        next_row += 3
        self._add_chart(frame, next_row, "mem", label=False)

    def _create_gpu_sections(self, parent, start_row_index):
        import tkinter as tk
        from tkinter import ttk

        prefix = "GPU"
        if not nvml_initialized:
            frame, _ = self._create_section_frame(parent, "GPU", "IO")
            frame.grid(row=start_row_index, column=0, sticky="ew", padx=5, pady=5)
            no_gpu_label = ttk.Label(
                frame,
                text="NVIDIA GPU / NVML not detected or failed.",
                style="Error.TLabel",
            )
            no_gpu_label.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky="w")
            return start_row_index + 1

        current_row = start_row_index
        for i, gpu_handle_info in enumerate(nvml_handles):
            if i >= len(self.gpu_widgets):
                continue

            widgets = self.gpu_widgets[i]
            gpu_name = gpu_handle_info.get("name", f"GPU {i}")

            frame, next_row_in_frame = self._create_section_frame(
                parent, f"{gpu_name}", prefix
            )
            frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5)
            widgets["frame"] = frame

            widgets["util_val"], widgets["util_unit"] = self._add_info_row(
                frame, next_row_in_frame, "Utilization:", prefix
            )
            widgets["util_unit"].configure(text="%")
            widgets["util_pb"] = self._add_progress_bar(
                frame, next_row_in_frame + 1, prefix
            )
            next_row_in_frame += 2

            # Create a sub-frame for the utilization chart and temperature
            util_chart_frame = ttk.Frame(frame, style="TFrame")
            util_chart_frame.grid(
                row=next_row_in_frame,
                column=0,
                columnspan=3,
                sticky="ew",
                padx=5,
                pady=(5, 0),
            )

            # Add utilization chart canvas with reduced width
            util_canvas = tk.Canvas(
                util_chart_frame,
                height=CHART_HEIGHT,
                width=BAR_LENGTH // 2,
                bg=CANVAS_BG_COLOR,
                highlightthickness=1,
                highlightbackground=BORDER_COLOR,
            )
            util_canvas.pack(side="left", fill="x", expand=True)
            self.gpu_util_chart_canvases[i] = util_canvas

            # Add temperature label to the right of the utilization chart
            widgets["temp_val"] = ttk.Label(
                util_chart_frame,
                text="N/A",
                style=f"{prefix}.Data.TLabel",
                font=("Segoe UI", 14, "bold"),
            )
            widgets["temp_val"].pack(side="right", padx=(5, 0))

            next_row_in_frame += 1

            widgets["vram_val"], widgets["vram_unit"] = self._add_info_row(
                frame, next_row_in_frame, "VRAM Usage:", prefix
            )
            widgets["vram_perc_val"], widgets["vram_perc_unit"] = self._add_info_row(
                frame, next_row_in_frame + 1, "VRAM Percent:", prefix
            )
            widgets["vram_perc_unit"].configure(text="%")
            widgets["vram_pb"] = self._add_progress_bar(
                frame, next_row_in_frame + 2, prefix
            )
            next_row_in_frame += 3

            # Create a sub-frame for the VRAM chart
            vram_chart_frame = ttk.Frame(frame, style="TFrame")
            vram_chart_frame.grid(
                row=next_row_in_frame,
                column=0,
                columnspan=3,
                sticky="ew",
                padx=5,
                pady=(5, 0),
            )

            # Add VRAM chart canvas with reduced width
            vram_canvas = tk.Canvas(
                vram_chart_frame,
                height=CHART_HEIGHT,
                width=BAR_LENGTH // 2,
                bg=CANVAS_BG_COLOR,
                highlightthickness=1,
                highlightbackground=BORDER_COLOR,
            )
            vram_canvas.pack(side="left", fill="x", expand=True)
            self.gpu_vram_chart_canvases[i] = vram_canvas

            next_row_in_frame += 1

            current_row += 1
        return current_row

    def _create_disk_section(self, parent_container):
        prefix = "IO"  # Use IO prefix for neutral styling
        frame, next_row = self._create_section_frame(
            parent_container, "Disk I/O", prefix
        )
        # Place in the first column (0) of the container, sticky="nsew" to fill its cell
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2), pady=0)
        self.disk_read_val, self.disk_read_unit = self._add_info_row(
            frame, next_row, "Read:", prefix
        )
        self.disk_write_val, self.disk_write_unit = self._add_info_row(
            frame, next_row + 1, "Write:", prefix
        )

    def _create_network_section(self, parent_container):
        prefix = "IO"  # Use IO prefix for neutral styling
        frame, next_row = self._create_section_frame(
            parent_container, "Network I/O", prefix
        )
        # Place in the second column (1) of the container, sticky="nsew" to fill its cell
        frame.grid(row=0, column=1, sticky="nsew", padx=(2, 0), pady=0)
        self.net_recv_val, self.net_recv_unit = self._add_info_row(
            frame, next_row, "Down:", prefix
        )
        self.net_sent_val, self.net_sent_unit = self._add_info_row(
            frame, next_row + 1, "Up:", prefix
        )

    # --- Chart Update Method ---
    def _update_chart(
        self, canvas, data_deque, line_color, max_value=100, overlay_text=None
    ):
        import tkinter as tk

        if not canvas or not canvas.winfo_exists() or not canvas.winfo_viewable():
            return
        try:
            canvas.delete("all")
            width = canvas.winfo_width()
            height = canvas.winfo_height()
        except tk.TclError:
            return
        except Exception:
            return

        if (
            not width
            or not height
            or width < 2
            or height < 2
            or len(data_deque) < 2
            or max_value <= 0
        ):
            return

        y_range = height - 2
        for perc in [0.25, 0.5, 0.75]:
            y_line = 1 + y_range - (perc * y_range)
            canvas.create_line(
                1, y_line, width - 1, y_line, fill=CANVAS_GRID_COLOR, dash=(2, 4)
            )

        points = []
        x_step = (
            (width - 2) / (HISTORY_LENGTH - 1) if HISTORY_LENGTH > 1 else (width - 2)
        )

        for i, value in enumerate(data_deque):
            clamped_value = max(
                0.0, min(float(value if value is not None else 0.0), float(max_value))
            )
            normalized_y = (clamped_value / max_value) * y_range
            y = 1 + y_range - normalized_y
            x = 1 + (i * x_step)
            points.extend([x, y])

        if len(points) >= 4:
            try:
                canvas.create_line(points, fill=line_color, width=1.5)
                if overlay_text is not None:
                    canvas.create_text(
                        width // 2,
                        height // 2,
                        text=overlay_text,
                        fill=line_color,
                        font=("Segoe UI", 10),
                    )
            except tk.TclError:
                pass
            except Exception:
                pass

    def update_gui(self):
        import tkinter as tk

        # Simple lock to prevent re-entry if update takes longer than interval
        if self._update_running:
            return
        self._update_running = True

        try:
            # Check if root window exists before doing anything else
            if not self.root or not self.root.winfo_exists():
                # print("Root window does not exist. Stopping update loop.")
                self._update_running = False
                # Do not reschedule 'after' if the window is gone
                return

            stats = get_all_stats()

            # --- Update History Deques ---
            self.history["cpu"].append(stats.get("cpu_percent", 0))
            self.history["mem"].append(stats.get("mem_percent", 0))

            # Ensure GPU history update matches the number of actual GPUs found
            num_gpus_detected = len(stats.get("gpus", []))
            for i in range(min(num_gpus_detected, len(self.history["gpus_util"]))):
                gpu_info = stats["gpus"][i]
                self.history["gpus_util"][i].append(
                    gpu_info.get("util", 0) if gpu_info.get("util") is not None else 0
                )
            for i in range(min(num_gpus_detected, len(self.history["gpus_vram"]))):
                gpu_info = stats["gpus"][i]
                self.history["gpus_vram"][i].append(gpu_info.get("vram_percent", 0))

            # --- Update Widgets (inside try/except blocks for robustness) ---
            # CPU
            try:
                self.cpu_util_val.configure(text=f"{stats.get('cpu_percent', 0):.1f}")
            except tk.TclError:
                pass
            try:
                self.cpu_pb["value"] = stats.get("cpu_percent", 0)
            except tk.TclError:
                pass
            try:
                cpu_temp = stats.get("cpu_temp")
                self.cpu_temp_val.configure(
                    text=f"{cpu_temp:.0f}" if cpu_temp is not None else "N/A"
                )
            except tk.TclError:
                pass

            # Memory
            try:
                self.mem_usage_val.configure(
                    text=f"{_bytes2human(stats.get('mem_used', 0))} / {_bytes2human(stats.get('mem_total', 0))}"
                )
            except tk.TclError:
                pass
            try:
                self.mem_percent_val.configure(
                    text=f"{stats.get('mem_percent', 0):.1f}"
                )
            except tk.TclError:
                pass
            try:
                self.mem_pb["value"] = stats.get("mem_percent", 0)
            except tk.TclError:
                pass

            # Disk IO
            try:
                self.disk_read_val.configure(
                    text=f"{_bytes2human(stats.get('disk_read_speed', 0))}"
                )
                self.disk_read_unit.configure(text="/s")
            except tk.TclError:
                pass
            try:
                self.disk_write_val.configure(
                    text=f"{_bytes2human(stats.get('disk_write_speed', 0))}"
                )
                self.disk_write_unit.configure(text="/s")
            except tk.TclError:
                pass

            # Network IO
            try:
                self.net_recv_val.configure(
                    text=f"{_bytes2human(stats.get('net_recv_speed', 0))}"
                )
                self.net_recv_unit.configure(text="/s")
            except tk.TclError:
                pass
            try:
                self.net_sent_val.configure(
                    text=f"{_bytes2human(stats.get('net_sent_speed', 0))}"
                )
                self.net_sent_unit.configure(text="/s")
            except tk.TclError:
                pass

            # GPUs
            for i, gpu_info in enumerate(stats.get("gpus", [])):
                if i < len(self.gpu_widgets):
                    widgets = self.gpu_widgets[i]
                    util = gpu_info.get("util")
                    temp = gpu_info.get("temp")
                    vram_perc = gpu_info.get("vram_percent", 0)
                    vram_used = gpu_info.get("vram_used", 0)
                    vram_total = gpu_info.get("vram_total", 0)

                    try:
                        widgets["util_val"].configure(
                            text=f"{util:.0f}" if util is not None else "N/A"
                        )
                        widgets["util_pb"]["value"] = util if util is not None else 0
                        if self.gpu_util_chart_canvases[i]:
                            self._update_chart(
                                self.gpu_util_chart_canvases[i],
                                self.history["gpus_util"][i],
                                GPU_ACCENT_COLOR,
                                overlay_text=f"{temp:.0f}°C"
                                if temp is not None
                                else "N/A",
                            )
                    except tk.TclError:
                        pass
                    try:
                        widgets["temp_val"].configure(
                            text=f"{temp:.0f}°C" if temp is not None else "N/A"
                        )
                    except tk.TclError:
                        pass
                    try:
                        widgets["vram_val"].configure(
                            text=f"{_bytes2human(vram_used)} / {_bytes2human(vram_total)}"
                        )
                        widgets["vram_perc_val"].configure(text=f"{vram_perc:.1f}")
                        widgets["vram_pb"]["value"] = vram_perc
                        if self.gpu_vram_chart_canvases[i]:
                            self._update_chart(
                                self.gpu_vram_chart_canvases[i],
                                self.history["gpus_vram"][i],
                                GPU_ACCENT_COLOR,
                            )
                    except tk.TclError:
                        pass

            # --- Update Charts ---
            # Wrapped in try/except to catch errors if widgets disappear
            try:
                if hasattr(self, "cpu_chart_canvas"):
                    self._update_chart(
                        self.cpu_chart_canvas, self.history["cpu"], CPU_ACCENT_COLOR
                    )
            except Exception as e:
                print(f"Error updating CPU chart: {e}")
            try:
                if hasattr(self, "mem_chart_canvas"):
                    self._update_chart(
                        self.mem_chart_canvas, self.history["mem"], RAM_ACCENT_COLOR
                    )
            except Exception as e:
                print(f"Error updating Mem chart: {e}")

            # Use the correct list of canvases
            try:
                for i, canvas in enumerate(self.gpu_util_chart_canvases):
                    if canvas and i < len(self.history["gpus_util"]):
                        self._update_chart(
                            canvas, self.history["gpus_util"][i], GPU_ACCENT_COLOR
                        )
            except Exception as e:
                print(f"Error updating GPU Util chart: {e}")
            try:
                for i, canvas in enumerate(self.gpu_vram_chart_canvases):
                    if canvas and i < len(self.history["gpus_vram"]):
                        self._update_chart(
                            canvas, self.history["gpus_vram"][i], GPU_ACCENT_COLOR
                        )
            except Exception as e:
                print(f"Error updating GPU VRAM chart index: {e}")

        except Exception as e:
            print(f"Error during GUI update cycle: {e}")
            # import traceback
            # traceback.print_exc() # Uncomment for detailed debugging
        finally:
            self._update_running = False
            # Reschedule the next update only if the root window still exists
            try:
                if self.root and self.root.winfo_exists():
                    # Cancel previous 'after' job if it exists, before scheduling a new one
                    if self._after_id:
                        self.root.after_cancel(self._after_id)
                    self._after_id = self.root.after(
                        UPDATE_INTERVAL_MS, self.update_gui
                    )
                # else:
                # print("Root window gone, update loop stopped naturally.")
            except Exception as e:  # Catch potential errors during rescheduling
                print(f"Error rescheduling GUI update: {e}")
                self._after_id = None  # Ensure we don't hold a stale ID

    def on_closing(self):
        import tkinter as tk

        # print("WM_DELETE_WINDOW triggered (on_closing)")
        try:
            # Cancel any pending update job
            if self._after_id:
                # print(f"Cancelling pending update job: {self._after_id}")
                try:
                    self.root.after_cancel(self._after_id)
                except tk.TclError:
                    pass  # Ignore error if root is already gone
                self._after_id = None

            # print("Destroying root window...")
            if self.root and self.root.winfo_exists():
                self.root.destroy()
                # print("Root window destroyed.")
            # else:
            # print("Root window already destroyed or invalid.")

        except tk.TclError as e:
            print(
                f"TclError during on_closing: {e}"
            )  # Should not happen often if winfo_exists check works
        except Exception as e:
            print(f"Unexpected error during on_closing: {e}")


def launch_monitor_gui():
    import tkinter as tk

    global _monitor_active_flag
    _monitor_active_flag = True
    root = None  # Define root here so it's accessible in finally block

    # Signal handling should ideally be managed by the main thread or carefully in the GUI thread
    # For simplicity here, keep basic signal handling but note Tkinter might have its own loop handling
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum} in GUI thread, initiating shutdown...")
        if root and root.winfo_exists():
            # Use `after` to schedule the destroy from the Tkinter thread itself
            # This is generally safer than calling destroy directly from a signal handler
            try:
                root.after(0, root.destroy)
            except tk.TclError:  # Window might already be closing
                pass
        # Cleanup will happen in the finally block after mainloop exits

    # original_sigint = signal.getsignal(signal.SIGINT)
    # original_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        # Register signal handlers specific to this thread if needed,
        # but be cautious as signals are often delivered to the main thread.
        # It might be better to rely on the main thread catching signals and signaling the GUI thread to close.
        # For now, let's assume WM_DELETE_WINDOW and the finally block handle most cases.
        # signal.signal(signal.SIGTERM, signal_handler) # Be careful with this in threads
        # signal.signal(signal.SIGINT, signal_handler) # Be careful with this in threads

        # --- Initialization within the thread ---
        write_lockfile()
        initialize_nvml()  # Initialize NVML within the GUI thread

        # print("Initializing Tkinter...");
        root = tk.Tk()
        # Set window position (optional, e.g., top-left)
        root.geometry("+0+0")
        root.withdraw()  # Hide window initially until ready

        # print("Creating ResourceMonitorApp...");
        ResourceMonitorApp(root)  # Pass the root window

        # Make window appear after setup
        root.deiconify()
        root.lift()

        if ALWAYS_ON_TOP:
            root.attributes("-topmost", True)
        else:
            root.attributes("-topmost", True)
            root.after_idle(lambda: root.attributes("-topmost", False))
        root.mainloop()

    except Exception as e:
        print(f"An error occurred during GUI execution: {e}")
        import traceback

        traceback.print_exc()
    finally:
        shutdown_nvml()
        remove_lockfile()
        _monitor_active_flag = False


def is_notebook():
    try:
        from IPython import get_ipython

        return "IPKernelApp" in get_ipython().config
    except:
        return False


def resource_monitor():
    """
    Launches the resource monitor GUI as a completely detached process.
    Prevents launching if already running (checks lock file).
    The monitor will continue running independently even after the parent process exits.
    """
    if is_notebook():
        # Run directly in notebook mode
        launch_monitor_gui()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--resource-monitor-standalone":
        launch_monitor_gui()
        return

    # Check if another process is running the monitor
    if is_monitor_running():
        print(
            "\033[93mResource Monitor GUI appears to be already running (found lock file for another process).\033[0m"
        )
        return

    # Get the current script's path
    script_path = os.path.abspath(sys.argv[0])

    # Create a new Python process that will run independently
    try:
        # Use subprocess.Popen with appropriate flags to detach the process
        if os.name == "nt":  # Windows
            # creationflags=subprocess.DETACHED_PROCESS makes it independent
            subprocess.Popen(
                [sys.executable, script_path, "--resource-monitor-standalone"],
                creationflags=subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP,
                close_fds=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:  # Linux/macOS
            # Start process in a new session, redirect output to /dev/null
            subprocess.Popen(
                [sys.executable, script_path, "--resource-monitor-standalone"],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )

        return

    except Exception as e:
        print(f"Failed to start resource monitor: {e}")
        return


if __name__ == "__main__":
    resource_monitor()
