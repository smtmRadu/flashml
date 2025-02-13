import os
import sys
import psutil
import subprocess
import curses
import signal

try:
    import pynvml
    pynvml.nvmlInit()
    use_nvml = True
except Exception as e:
    use_nvml = False

_monitor_launched = False

def is_monitor_running():
    """
    Check if any process is running this script with the '--run-monitor' flag.
    """
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = proc.info['cmdline']
            if cmd and "--run-monitor" in cmd and proc.pid != os.getpid():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def _bytes2human(n):
    symbols = ('B', 'KB', 'MB', 'GB', 'TB', 'PB')
    prefix = {s: 1 << (i * 10) for i, s in enumerate(symbols)}
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return f'{value:.2f} {s}'
    return f"{n} B"

def get_resource_usage_lines():
    """Gathers resource usage info and returns a list of strings."""
    lines = []
   
    # proc = psutil.Process(os.getpid())
    virtual_mem = psutil.virtual_memory()
    total_mem = virtual_mem.total
    used_mem = virtual_mem.used
    mem_usage_percent = virtual_mem.percent
    cpu_usage = psutil.cpu_percent(interval=None)
    
    lines.append("CPU:")
    lines.append(f"  Utilization: {cpu_usage:.1f}%")
    lines.append(f"  RAM: {_bytes2human(used_mem)} used / {_bytes2human(total_mem)} total ({mem_usage_percent:.1f}%)")
    
    if use_nvml:
        try:
            gpu_count = pynvml.nvmlDeviceGetCount()
            lines.append("")
            lines.append("GPU(s):")
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                raw_name = pynvml.nvmlDeviceGetName(handle)
                gpu_name = raw_name.decode('utf-8') if isinstance(raw_name, bytes) else raw_name
                mem_info_gpu = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_gpu = mem_info_gpu.used
                total_gpu = mem_info_gpu.total
                mem_percent = (used_gpu / total_gpu) * 100 if total_gpu else 0
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                lines.append(f"  Device {i}: {gpu_name}")
                lines.append(f"    Utilization: {gpu_util}%")
                lines.append(f"    VRAM: {_bytes2human(used_gpu)} used / {_bytes2human(total_gpu)} total ({mem_percent:.1f}%)")
        except Exception as e:
            lines.append("  Error fetching GPU data via NVML.")
    else:
        lines.append("")
        lines.append("GPU: Not Available")
    
    return lines

def get_resource_usage_box():
    """Wraps the resource info in a box made with '+', '-', and '|' characters."""
    lines = get_resource_usage_lines()
    max_width = max(len(line) for line in lines)
    border = "+" + "-" * (max_width + 2) + "+"
    box_lines = [border]
    for line in lines:
        box_lines.append("| " + line.ljust(max_width) + " |")
    box_lines.append(border)
    return box_lines

def monitor_loop_curses(stdscr, nap_ms):
    """
    Uses curses to update the display in place with minimal flicker.
    The window will exit promptly if the OS sends a termination signal
    (for example, when the user clicks the window's X).
    """
    curses.curs_set(0)
    stdscr.nodelay(True)
    try:
        while True:
            stdscr.erase()
            box_lines = get_resource_usage_box()
            for i, line in enumerate(box_lines):
                try:
                    stdscr.addstr(i, 0, line)
                except curses.error:
                    pass
            stdscr.refresh()
            curses.napms(nap_ms)
    except KeyboardInterrupt:
        pass
    finally:
        if use_nvml:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

def monitor_loop(nap_ms):
    """Wrapper to start the curses monitor loop with the given nap delay."""
    curses.wrapper(lambda stdscr: monitor_loop_curses(stdscr, nap_ms))

def display_usage(nap_ms: int = 50):
    """
    Spawns a new process that opens a separate window displaying live CPU,
    memory, and GPU usage in a boxed layout. The refresh delay (in milliseconds)
    is given by `nap_ms`.

    If a monitor is already running, no new window is created so is safe to double call it.
    """
    global _monitor_launched
    if is_monitor_running():
        print("\033[93mResource Monitor is already running!\033[0m")
        return

    _monitor_launched = True
    args = [sys.executable, __file__, "--run-monitor", f"--nap={nap_ms}"]
    creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    subprocess.Popen(args, creationflags=creationflags)

def setup_signal_handlers():
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

if __name__ == '__main__':
    if "--run-monitor" in sys.argv:
        setup_signal_handlers()
        nap_ms = 100
        for arg in sys.argv:
            if arg.startswith("--nap="):
                try:
                    nap_ms = int(arg.split("=")[1])
                except ValueError:
                    pass
        monitor_loop(nap_ms)
