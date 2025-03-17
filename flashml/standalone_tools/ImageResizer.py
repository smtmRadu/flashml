import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image
import os
from tkinterdnd2 import DND_FILES, TkinterDnD
import subprocess
from concurrent.futures import ThreadPoolExecutor
from queue import Queue


class ImageResizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Resizer")
        self.root.geometry("305x520")
        self.folder_path = os.path.join(os.path.expanduser("~"), "Downloads")
        self.loaded_files = []
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.gif', '.webp']

        self.processed_count = 0
        self.error_queue = Queue()

        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)

        # Drop zone
        self.drop_frame = ttk.LabelFrame(main_frame, text="Drop Images Here", padding="20")
        self.drop_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        self.drop_frame.columnconfigure(0, weight=1)

        self.drop_label = ttk.Label(self.drop_frame, text="Drag and drop images here")
        self.drop_label.grid(row=0, column=0, pady=50)
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.drop)

        # Remove Duplicates button
        self.remove_duplicates_button = ttk.Button(self.drop_frame, text="Remove Duplicates", command=self.remove_duplicates)
        self.remove_duplicates_button.grid(row=1, column=1, pady=10)
        self.remove_duplicates_button.grid_remove()

        # Clear button
        self.clear_button = ttk.Button(self.drop_frame, text="Clear", command=self.clear_loaded_files)
        self.clear_button.grid(row=1, column=0, pady=10)
        self.clear_button.grid_remove()

        # Resize options frame
        resize_frame = ttk.LabelFrame(main_frame, text="Resize Options", padding="10")
        resize_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        # Resize method
        ttk.Label(resize_frame, text="Resize Method:").grid(row=0, column=0, pady=5)
        self.resize_method = tk.StringVar(value="percentage")
        methods = ttk.Frame(resize_frame)
        methods.grid(row=0, column=1, pady=5)
        ttk.Radiobutton(methods, text="Percentage", variable=self.resize_method, 
                       value="percentage", command=self.toggle_resize_inputs).pack(side=tk.LEFT)
        ttk.Radiobutton(methods, text="Dimensions", variable=self.resize_method, 
                       value="dimensions", command=self.toggle_resize_inputs).pack(side=tk.LEFT)

        # Percentage frame
        self.percentage_frame = ttk.Frame(resize_frame)
        self.percentage_frame.grid(row=1, column=0, columnspan=2, pady=5)
        ttk.Label(self.percentage_frame, text="Scale (%):").pack(side=tk.LEFT)
        self.scale_var = tk.StringVar(value="100")
        self.scale_var.trace('w', self.on_entry_change)  # Add trace to entry
        self.scale_entry = ttk.Entry(self.percentage_frame, textvariable=self.scale_var, width=5)
        self.scale_entry.pack(side=tk.LEFT, padx=5)

        self.scale_slider = ttk.Scale(
        self.percentage_frame,
            from_=1,
            to=200,
            orient='horizontal',
            length=150,
            command=self.on_scale_change
        )
        self.scale_slider.set(100)
        self.scale_slider.pack(side=tk.LEFT, padx=5)

        # Dimensions frame
        self.dimensions_frame = ttk.Frame(resize_frame)
        self.dimensions_frame.grid(row=2, column=0, columnspan=2, pady=5)
        ttk.Label(self.dimensions_frame, text="Width:").pack(side=tk.LEFT)
        self.width_var = tk.StringVar()
        self.width_entry = ttk.Entry(self.dimensions_frame, textvariable=self.width_var, width=8)
        self.width_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.dimensions_frame, text="Height:").pack(side=tk.LEFT)
        self.height_var = tk.StringVar()
        self.height_entry = ttk.Entry(self.dimensions_frame, textvariable=self.height_var, width=8)
        self.height_entry.pack(side=tk.LEFT, padx=5)
        self.dimensions_frame.grid_remove()

        # Maintain aspect ratio
        self.maintain_aspect = tk.BooleanVar(value=True)
        self.aspect_check = ttk.Checkbutton(resize_frame, text="Maintain aspect ratio", 
                                          variable=self.maintain_aspect)
        self.aspect_check.grid(row=3, column=0, columnspan=2, pady=5)

        # Progress bar
        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate')
        self.progress_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        self.progress_bar.grid_remove()

        # Status label
        self.status_label = ttk.Label(main_frame, text="", font=("Arial", 7, "bold"))
        self.status_label.grid(row=4, column=0, columnspan=2, pady=5)

        # Buttons
        self.resize_button = ttk.Button(main_frame, text="Resize", command=self.resize_images)
        self.resize_button.grid(row=5, column=0, columnspan=2, pady=10)

        self.open_folder_button = ttk.Button(main_frame, text="See Results", 
                                           command=self.open_output_folder)
        self.open_folder_button.grid(row=6, column=0, columnspan=2, pady=10)
        self.open_folder_button.grid_remove()

    def toggle_resize_inputs(self):
        if self.resize_method.get() == "percentage":
            self.dimensions_frame.grid_remove()
            self.percentage_frame.grid()
        else:
            self.percentage_frame.grid_remove()
            self.dimensions_frame.grid()

    def drop(self, event):
        files = self.root.tk.splitlist(event.data)
        valid_files = [f for f in files if os.path.splitext(f)[1].lower() in self.supported_formats]
        
        if len(files) != len(valid_files):
            messagebox.showwarning("Invalid Files", 
                                 "Some files were not in a supported format and were excluded.")
        
        self.loaded_files.extend(valid_files)
        
        if len(self.loaded_files) > 0:
            self.clear_button.grid()
            
        if len(set(self.loaded_files)) < len(self.loaded_files):
            self.remove_duplicates_button.grid()
            
        self.drop_label.configure(text=f"{len(self.loaded_files)} image(s) loaded")

    def remove_duplicates(self):
        self.loaded_files = list(set(self.loaded_files))
        self.remove_duplicates_button.grid_remove()
        self.drop_label.configure(text=f"{len(self.loaded_files)} image(s) loaded")

    def clear_loaded_files(self):
        self.loaded_files = []
        self.clear_button.grid_remove()
        self.remove_duplicates_button.grid_remove()
        self.clear_button.configure(text="Clear")
        self.drop_label.configure(text="Drag and drop images here")
        self.status_label.configure(text="")
        self.open_folder_button.grid_remove()
        self.resize_button.grid()
        self.progress_bar.grid_remove()

    def calculate_new_dimensions(self, original_width, original_height):
        if self.resize_method.get() == "percentage":
            try:
                scale = float(self.scale_var.get()) / 100
                return int(original_width * scale), int(original_height * scale)
            except ValueError:
                raise ValueError("Please enter a valid percentage")
        else:
            try:
                new_width = int(self.width_var.get()) if self.width_var.get() else None
                new_height = int(self.height_var.get()) if self.height_var.get() else None
                
                if not new_width and not new_height:
                    raise ValueError("Please enter at least one dimension")
                
                if self.maintain_aspect.get():
                    if new_width and new_height:
                        return new_width, new_height
                    elif new_width:
                        ratio = new_width / original_width
                        return new_width, int(original_height * ratio)
                    else:
                        ratio = new_height / original_height
                        return int(original_width * ratio), new_height
                else:
                    return (new_width if new_width else original_width,
                           new_height if new_height else original_height)
            except ValueError:
                raise ValueError("Please enter valid dimensions")

    def process_single_image(self, file_info):
        file_path, output_dir = file_info
        try:
            with Image.open(file_path) as img:
                # Calculate new dimensions
                new_width, new_height = self.calculate_new_dimensions(img.width, img.height)
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Create new filename with dimensions
                filename, ext = os.path.splitext(os.path.basename(file_path))
                new_filename = f"{filename}_{new_width}x{new_height}{ext}"
                output_path = os.path.join(output_dir, new_filename)
                
                resized_img.save(output_path)
                
                return True
        except Exception as e:
            self.error_queue.put((file_path, str(e)))
            return False

    def update_progress(self):
        self.processed_count += 1
        self.progress_bar['value'] = self.processed_count
        
        # Update status label with current file count
        self.status_label['text'] = f"Processed {self.processed_count} of {len(self.loaded_files)} images..."
        
        # Check if all files have been processed
        if self.processed_count >= len(self.loaded_files):
            self.finalize_resize()

    def process_complete(self, future):
        self.root.after(0, self.update_progress)

    def finalize_resize(self):
        while not self.error_queue.empty():
            file_path, error_msg = self.error_queue.get()
            messagebox.showerror("Error", f"Error resizing {os.path.basename(file_path)}: {error_msg}")
        
        self.status_label['text'] = "Resizing complete!"
        self.resize_button.grid()
        self.open_folder_button.grid()
        self.executor.shutdown(wait=False)
        self.error_queue = Queue()

    def resize_images(self):
        if not self.loaded_files:
            messagebox.showwarning("No Images", "Please load images first.")
            return

        output_dir = os.path.join(self.folder_path, "resized_images")
        os.makedirs(output_dir, exist_ok=True)

        # Initialize processing variables
        self.processed_count = 0
        self.error_queue = Queue()
        self.progress_bar['maximum'] = len(self.loaded_files)
        self.progress_bar.grid()
        self.resize_button.grid_remove()

        # Create a thread pool executor
        self.executor = ThreadPoolExecutor(max_workers=min(4, len(self.loaded_files)))

        # Submit all files for processing
        for file_path in self.loaded_files:
            future = self.executor.submit(self.process_single_image, (file_path, output_dir))
            future.add_done_callback(self.process_complete)
        
    def resize_images(self):
        if not self.loaded_files:
            messagebox.showwarning("No Images", "Please load images first.")
            return

        output_dir = os.path.join(self.folder_path, "resized_images")
        os.makedirs(output_dir, exist_ok=True)

        total_files = len(self.loaded_files)
        self.progress_bar['maximum'] = total_files
        self.progress_bar.grid()

        for i, file_path in enumerate(self.loaded_files, 1):
            try:
                self.status_label['text'] = f"Resizing {os.path.basename(file_path)}..."
                self.root.update_idletasks()

                with Image.open(file_path) as img:
                    # Calculate new dimensions
                    new_width, new_height = self.calculate_new_dimensions(img.width, img.height)
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Create new filename with dimensions
                    filename, ext = os.path.splitext(os.path.basename(file_path))
                    new_filename = f"{filename}_{new_width}x{new_height}{ext}"
                    output_path = os.path.join(output_dir, new_filename)
                    resized_img.save(output_path)

                self.progress_bar['value'] = i
                self.root.update_idletasks()

            except Exception as e:
                messagebox.showerror("Error", f"Error resizing {os.path.basename(file_path)}: {str(e)}")

        self.clear_button.configure(text="Restart Resizer")
        self.status_label['text'] = "Resizing complete!"
        self.resize_button.grid_remove()
        self.open_folder_button.grid()

    def open_output_folder(self):
        output_dir = os.path.join(self.folder_path, "resized_images")
        try:
            if os.name == 'nt':  # Windows
                os.startfile(output_dir)
            elif os.name == 'posix':  # macOS or Linux
                subprocess.Popen(['xdg-open', output_dir])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open output folder: {str(e)}")

    def on_scale_change(self, value):
        self.scale_var.set(str(int(float(value))))

    def on_entry_change(self, *args):
        try:
            value = float(self.scale_var.get())
            if value < 1:
                value = 1
            # Only update slider if value is within its range
            if value <= 200:
                self.scale_slider.set(value)
        except ValueError:
            pass


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ImageResizerApp(root)
    root.mainloop()