import torch
import os
import psutil

def _bytes2human(n):
    symbols = ('B', 'KB', 'MB', 'GB', 'TB', 'PB')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i * 10)
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return f'{value:.2f} {s}'
    return f"{n} B"

def _print_box(lines):
    max_width = max(len(line) for line in lines)
    horizontal_border = "+" + "-" * (max_width + 2) + "+"
    print(horizontal_border)
    for line in lines:
        print(f"| {line.ljust(max_width)} |")
    print(horizontal_border)

def show_memory_usage():
    proc = psutil.Process(os.getpid())
    mem_info = proc.memory_info()
    process_rss = mem_info.rss  
    process_vms = mem_info.vms  
    virtual_mem = psutil.virtual_memory()
    total_mem = virtual_mem.total
    used_mem = virtual_mem.used
    mem_usage_percent = virtual_mem.percent


    output_lines = []
    output_lines.append("CPU:")
    output_lines.append(f"Process Memory (RSS): {_bytes2human(process_rss)} (VMS: {_bytes2human(process_vms)})")
    output_lines.append(f"RAM: {_bytes2human(used_mem)} used / {_bytes2human(total_mem)} total ({mem_usage_percent:.1f}%)")
    

    if torch.cuda.is_available():
        output_lines.append("")  
        output_lines.append("GPU (cuda):")
        num_devices = torch.cuda.device_count()
        for device in range(num_devices):
            device_name = torch.cuda.get_device_name(device)
            props = torch.cuda.get_device_properties(device)
            total_gpu_mem = props.total_memory 
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            allocated_percent = (allocated / total_gpu_mem) * 100
            reserved_percent = (reserved / total_gpu_mem) * 100

            output_lines.append(f"Device {device}: {device_name}")
            output_lines.append(f"  VRAM:  {_bytes2human(allocated)} allocated / {_bytes2human(total_gpu_mem)} total ({allocated_percent:.1f}%)")
            output_lines.append(f"  Reserved: {_bytes2human(reserved)} ({reserved_percent:.1f}%)")

            if device < num_devices - 1:
                output_lines.append("-" * 40)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        output_lines.append("")
        output_lines.append("GPU (mps):")
        num_devices = torch.mps.device_count()
        for device in range(num_devices):
            allocated = torch.mps.current_allocated_memory()
            output_lines.append(f"Device {device}")
            output_lines.append(f"  VRAM:  {_bytes2human(allocated)} allocated / - total")

            if device < num_devices - 1:
                output_lines.append("-" * 40)
    else:
        output_lines.append("")
        output_lines.append("GPU: Not Available")
    
    _print_box(output_lines)

