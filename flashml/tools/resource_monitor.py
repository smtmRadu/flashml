import os
import sys
import psutil
import subprocess
import curses
import signal
import tempfile
import pynvml

LOCKFILE = os.path.join(tempfile.gettempdir(), "resource_monitor.lock")

def is_monitor_running():
    """
    Check for the existence of the lock file and verify that the process ID inside is still running.
    """
    if os.path.exists(LOCKFILE):
        try:
            with open(LOCKFILE, "r") as f:
                pid = int(f.read().strip())
            if psutil.pid_exists(pid):
                return True
        except Exception:
            pass
    return False

def write_lockfile():
    """
    Write the current process ID to the lock file.
    """
    with open(LOCKFILE, "w") as f:
        f.write(str(os.getpid()))

def remove_lockfile():
    """
    Remove the lock file.
    """
    try:
        os.remove(LOCKFILE)
    except OSError:
        pass

def _bytes2human(n):
    symbols = ('B', 'KB', 'MB', 'GB', 'TB', 'PB')
    prefix = {s: 1 << (i * 10) for i, s in enumerate(symbols)}
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return f'{value:.2f} {s}'
    return f"{n} B"

def get_cpu_temperature():
    """Returns CPU temperature in Celsius if available."""
    try:
        temps = psutil.sensors_temperatures()
        if "coretemp" in temps:
            return temps["coretemp"][0].current
    except Exception:
        pass
    return None 

def get_gpu_temperature(handle):
    """Returns GPU temperature in Celsius."""
    try:
        return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except Exception:
        return None
    
def get_resource_usage_lines():
    """Gathers resource usage info and returns a list of strings."""
    lines = []
    virtual_mem = psutil.virtual_memory()
    total_mem = virtual_mem.total
    used_mem = virtual_mem.used
    mem_usage_percent = virtual_mem.percent
    cpu_usage = psutil.cpu_percent(interval=None)
    cpu_temp = get_cpu_temperature()

    lines.append("CPU:")
    lines.append(f"  Temperature: {cpu_temp:.0f}°C" if cpu_temp else "  Temperature: N/A")
    lines.append(f"  Utilization: {cpu_usage:.0f}%")
    lines.append(f"  RAM: {_bytes2human(used_mem)} used / {_bytes2human(total_mem)} total ({mem_usage_percent:.1f}%)")
    lines.append(f"    |{"█" * int(mem_usage_percent/2)}{" " * int(50 - mem_usage_percent/2)}|")
    try:
        import pynvml
        if not getattr(get_resource_usage_lines, "nvml_initialized", False):
            pynvml.nvmlInit()
            get_resource_usage_lines.nvml_initialized = True
        
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
            gpu_temp = get_gpu_temperature(handle)

            lines.append(f"  Device {i}: {gpu_name}")
            lines.append(f"    Temperature: {gpu_temp:.0f}°C" if gpu_temp else "    Temperature: N/A")
            lines.append(f"    Utilization: {gpu_util}%")
            lines.append(f"    VRAM: {_bytes2human(used_gpu)} used / {_bytes2human(total_gpu)} total ({mem_percent:.1f}%)")
            lines.append(f"    |{"█" * int(mem_percent/2)}{" " * int(50 - mem_percent/2)}|")
    except Exception:
        lines.append("")
        lines.append("GPU: Not Available")
    
    return lines

def get_resource_usage_box():
    lines = get_resource_usage_lines()
    max_width = max(len(line) for line in lines)
    box_lines = ["█" + "▀" * (max_width + 2) + "█"]
    for line in lines:
        box_lines.append("█ " + line.ljust(max_width) + " █")
    box_lines.append("█" + "▄" * (max_width + 2) + "█")
    return box_lines

def monitor_loop_curses(stdscr, nap_ms):
    """
    Uses curses to update the display in place with minimal flicker.
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
        remove_lockfile()
        try:
            import pynvml
            pynvml.nvmlShutdown()
        except Exception:
            pass

def monitor_loop(nap_ms):
    """Starts the curses monitor loop with the given refresh delay."""
    curses.wrapper(lambda stdscr: monitor_loop_curses(stdscr, nap_ms))

def resource_monitor(nap_ms: int = 50):
    """
    Spawns a new process that opens a separate window displaying live CPU,
    memory, and GPU usage. Uses a lock file to prevent multiple monitors.
    """
    if is_monitor_running():
        print("\033[93mResource Monitor is already running!\033[0m")
        return

    args = [sys.executable, __file__, "--run-monitor", f"--nap={nap_ms}"]
    creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    subprocess.Popen(args, creationflags=creationflags)

def setup_signal_handlers():
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

if __name__ == '__main__':
    if "--run-monitor" in sys.argv:
        write_lockfile() 
        setup_signal_handlers()
        nap_ms = 50  
        for arg in sys.argv:
            if arg.startswith("--nap="):
                try:
                    nap_ms = int(arg.split("=")[1])
                except ValueError:
                    pass
        monitor_loop(nap_ms)
