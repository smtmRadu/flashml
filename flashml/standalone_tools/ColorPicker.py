import tkinter as tk
from pynput import mouse, keyboard
import pyautogui
import pyperclip
import time

# Dark mode color scheme
BG_COLOR = "#1e1e1e"
FG_COLOR = "#ffffff"
BORDER_COLOR = "#3c3f41"
last_update_time = 0
UPDATE_INTERVAL = 0.05
ctrl_pressed = False  # Track if Control key is pressed

def update_preview(rgb, hex_color):
    """Update the live preview color display"""
    preview_canvas.config(bg=hex_color)
    preview_label.config(text=f"PREVIEW\nRGB: {rgb}\nHEX: {hex_color}")

def update_selected(rgb, hex_color):
    """Update selected color, copy to clipboard, and show confirmation"""
    selected_canvas.config(bg=hex_color)
    selected_label.config(text=f"SELECTED\nRGB: {rgb}\nHEX: {hex_color}\n✓ Copied!")
    pyperclip.copy(hex_color)

def on_move(x, y):
    """Track mouse movement and update preview in real-time"""
    global last_update_time
    
    # Rate limiting to prevent lag
    current_time = time.time()
    if current_time - last_update_time < UPDATE_INTERVAL:
        return
    
    # Ignore if mouse is over the picker window
    if root.winfo_containing(x, y):
        return
        
    try:
        rgb = pyautogui.pixel(x, y)
        hex_color = '#%02x%02x%02x' % rgb
        root.after(0, update_preview, rgb, hex_color)
        last_update_time = current_time
    except Exception as e:
        print("Error reading pixel:", e)

def on_click(x, y, button, pressed):
    """Select color ONLY on Ctrl+Click and auto-copy to clipboard"""
    global ctrl_pressed
    
    if pressed and ctrl_pressed:  # Only act if Ctrl is held down
        # Ignore clicks on the picker window itself
        if root.winfo_containing(x, y):
            return 

        try:
            rgb = pyautogui.pixel(x, y)
        except Exception as e:
            print("Error reading pixel:", e)
            return

        hex_color = '#%02x%02x%02x' % rgb
        root.after(0, update_selected, rgb, hex_color)
        root.after(0, root.attributes, '-topmost', True)

def on_key_press(key):
    """Track when Control key is pressed"""
    global ctrl_pressed
    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        ctrl_pressed = True

def on_key_release(key):
    """Track when Control key is released"""
    global ctrl_pressed
    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        ctrl_pressed = False

# Create main window
root = tk.Tk()
root.title("Color Picker")
root.geometry("300x280")
root.resizable(False, False)
root.config(bg=BG_COLOR)

# Preview section (live mouse tracking)
preview_frame = tk.Frame(root, bg=BG_COLOR)
preview_frame.pack(pady=10)
preview_label = tk.Label(preview_frame, text="PREVIEW\nRGB: ( , , )\nHEX: ", 
                         font=("Arial", 10), bg=BG_COLOR, fg=FG_COLOR)
preview_label.pack()
preview_canvas = tk.Canvas(preview_frame, width=40, height=40, bg=BG_COLOR, 
                          highlightthickness=1, highlightbackground=BORDER_COLOR)
preview_canvas.pack()

# Selected section (Ctrl+Click to lock color)
selected_frame = tk.Frame(root, bg=BG_COLOR)
selected_frame.pack(pady=10)
selected_label = tk.Label(selected_frame, text="SELECTED\nRGB: ( , , )\nHEX: ", 
                          font=("Arial", 10), bg=BG_COLOR, fg=FG_COLOR)
selected_label.pack()
selected_canvas = tk.Canvas(selected_frame, width=50, height=50, bg=BG_COLOR, 
                           highlightthickness=1, highlightbackground=BORDER_COLOR)
selected_canvas.pack()

# Instructions
instruction_label = tk.Label(root, text="Move mouse to preview\nCtrl+Click to select & copy to clipboard", 
                            font=("Arial", 9), fg="#aaaaaa", bg=BG_COLOR)
instruction_label.pack(pady=5)

# Start listeners
mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
mouse_listener.start()

keyboard_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
keyboard_listener.start()

# Run application
try:
    root.mainloop()
finally:
    mouse_listener.stop()
    keyboard_listener.stop()