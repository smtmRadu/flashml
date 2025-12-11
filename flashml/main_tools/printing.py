import os
import inspect
from datetime import datetime 

def _get_caller_info():
    """Get caller information, handling both regular scripts and notebooks."""
    frame = inspect.currentframe().f_back.f_back  # Go up 2 frames to get actual caller
    filepath = os.path.abspath(frame.f_code.co_filename)
    filename = os.path.basename(filepath)
    lineno = frame.f_lineno  # Get the line number
    
    # Check if we're in a Jupyter notebook (temp file with numeric name)
    is_notebook = 'ipykernel' in filepath or filename.isdigit() or filename.replace('.py', '').isdigit()
    
    if is_notebook:
        # For notebooks, just return a simple label without hyperlink
        return "Notebook"
    else:
        # For regular scripts, create clickable link
        # The URL doesn't include line number, but the display text does
        filepath_url = filepath.replace("\\", "/")
        clickable_link = f"\033]8;;file:///{filepath_url}\033\\{filename}:{lineno}\033]8;;\033\\"
        return clickable_link


def print_info(message: str) -> None:
    caller_info = _get_caller_info()
    print(f"\033[34m[ℹ️  {caller_info}] [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\033[0m")
    
    
def print_warning(message: str) -> None:
    caller_info = _get_caller_info()
    print(f"\033[38;5;214m[⚠️  {caller_info}] [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\033[0m")

def print_error(message: str) -> None:
    caller_info = _get_caller_info()
    print(f"\033[31m[❌  {caller_info}] [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\033[0m")
