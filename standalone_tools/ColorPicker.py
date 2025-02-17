import tkinter as tk
from pynput import mouse
import pyautogui
import pyperclip

def update_color(rgb, hex_color):
    color_canvas.config(bg=hex_color)
    color_label.config(text=f"RGB: {rgb}\nHEX: {hex_color}")
    copy_button.config(state=tk.NORMAL) 
    copy_button.hex_color = hex_color 

def copy_hex_to_clipboard():
    pyperclip.copy(copy_button.hex_color)

def on_click(x, y, button, pressed):
    if pressed:
        if root.winfo_containing(x, y):
            return 

        try:
            rgb = pyautogui.pixel(x, y)
        except Exception as e:
            print("Error reading pixel:", e)
            return

        hex_color = '#%02x%02x%02x' % rgb
        root.after(0, update_color, rgb, hex_color)
        root.after(0, root.attributes, '-topmost', True)

root = tk.Tk()
root.title("Color Picker")
root.geometry("300x180")
color_canvas = tk.Canvas(root, width=50, height=50, bg="white", highlightthickness=1, highlightbackground="black")
color_canvas.pack(pady=10)
color_label = tk.Label(root, text="RGB: ( , , )\nHEX: ", font=("Arial", 12))
color_label.pack()
copy_button = tk.Button(root, text="Copy HEX", command=copy_hex_to_clipboard, state=tk.DISABLED)
copy_button.pack(pady=5)
listener = mouse.Listener(on_click=on_click)
listener.start()
try:
    root.mainloop()
finally:
    listener.stop()
