import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageOps, ImageEnhance
import os
from tkinterdnd2 import DND_FILES, TkinterDnD
import subprocess
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("220x540")  # Increased height for new buttons
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

        # Processing buttons frame
        process_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding="10")
        process_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        process_frame.columnconfigure(0, weight=1)

        # Add processing buttons
        self.inverse_button = ttk.Button(process_frame, text="Invert Colors", command=lambda: self.process_images("invert"))
        self.inverse_button.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))

        self.grayscale_button = ttk.Button(process_frame, text="Grayscale", command=lambda: self.process_images("grayscale"))
        self.grayscale_button.grid(row=0, column=1, pady=5, sticky=(tk.W, tk.E))

        self.bw_button = ttk.Button(process_frame, text="Black & White", command=lambda: self.process_images("bw"))
        self.bw_button.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))

        self.blur_button = ttk.Button(process_frame, text="Blur", command=lambda: self.process_images("blur"))
        self.blur_button.grid(row=1, column=1, pady=5, sticky=(tk.W, tk.E))

        self.enhance_button = ttk.Button(process_frame, text="Enhance Colors", command=lambda: self.process_images("enhance"))
        self.enhance_button.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))

        self.sepia_button = ttk.Button(process_frame, text="Sepia Tone", command=lambda: self.process_images("sepia"))
        self.sepia_button.grid(row=2, column=1, pady=5, sticky=(tk.W, tk.E))

        # Progress bar
        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate')
        self.progress_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        self.progress_bar.grid_remove()

        # Status label
        self.status_label = ttk.Label(main_frame, text="", font=("Arial", 7, "bold"))
        self.status_label.grid(row=4, column=0, columnspan=2, pady=5)

        # Open folder button
        self.open_folder_button = ttk.Button(main_frame, text="See Results", command=self.open_output_folder)
        self.open_folder_button.grid(row=5, column=0, columnspan=2, pady=10)
        self.open_folder_button.grid_remove()

    def process_single_image(self, file_info):
        file_path, output_dir, effect = file_info
        try:
            with Image.open(file_path) as img:
                if img.mode == 'RGBA':
                    # Convert RGBA to RGB while preserving transparency
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background

                if effect == "invert":
                    processed_img = ImageOps.invert(img)
                    suffix = "inverted"
                elif effect == "grayscale":
                    processed_img = ImageOps.grayscale(img)
                    suffix = "grayscale"
                elif effect == "bw":
                    processed_img = img.convert('L')
                    suffix = "bw"
                elif effect == "blur":
                    from PIL import ImageFilter
                    processed_img = img.filter(ImageFilter.GaussianBlur(radius=2))
                    suffix = "blur"
                elif effect == "enhance":
                    enhancer = ImageEnhance.Color(img)
                    processed_img = enhancer.enhance(1.5)
                    suffix = "enhanced"
                elif effect == "sepia":
                    width, height = img.size
                    pixels = img.load()
                    processed_img = img.copy()
                    for x in range(width):
                        for y in range(height):
                            r, g, b = img.getpixel((x, y))[:3]
                            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                            tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                            processed_img.putpixel((x, y), (min(tr, 255), min(tg, 255), min(tb, 255)))
                    suffix = "sepia"
                
                filename, ext = os.path.splitext(os.path.basename(file_path))
                new_filename = f"{filename}_{suffix}{ext}"
                output_path = os.path.join(output_dir, new_filename)
                processed_img.save(output_path)
                return True
        except Exception as e:
            self.error_queue.put((file_path, str(e)))
            return False

    def process_images(self, effect):
        if not self.loaded_files:
            messagebox.showwarning("No Images", "Please load images first.")
            return

        output_dir = os.path.join(self.folder_path, "processed_images")
        os.makedirs(output_dir, exist_ok=True)

        # Initialize processing variables
        self.processed_count = 0
        self.error_queue = Queue()
        self.progress_bar['maximum'] = len(self.loaded_files)
        self.progress_bar.grid()

        # Disable all processing buttons during processing
        self.inverse_button.state(['disabled'])
        self.bw_button.state(['disabled'])
        self.blur_button.state(['disabled'])
        self.enhance_button.state(['disabled'])
        self.sepia_button.state(['disabled'])

        # Create a thread pool executor
        self.executor = ThreadPoolExecutor(max_workers=min(4, len(self.loaded_files)))

        # Submit all files for processing
        for file_path in self.loaded_files:
            future = self.executor.submit(self.process_single_image, (file_path, output_dir, effect))
            future.add_done_callback(self.process_complete)

    def process_complete(self, future):
        self.root.after(0, self.update_progress)

    def update_progress(self):
        self.processed_count += 1
        self.progress_bar['value'] = self.processed_count
        self.status_label['text'] = f"Processed {self.processed_count} of {len(self.loaded_files)} images..."
        
        if self.processed_count >= len(self.loaded_files):
            self.finalize_processing()

    def finalize_processing(self):
        while not self.error_queue.empty():
            file_path, error_msg = self.error_queue.get()
            messagebox.showerror("Error", f"Error processing {os.path.basename(file_path)}: {error_msg}")
        
        self.status_label['text'] = "Processing complete!"
        self.open_folder_button.grid()
        
        # Re-enable all processing buttons
        self.inverse_button.state(['!disabled'])
        self.bw_button.state(['!disabled'])
        self.blur_button.state(['!disabled'])
        self.enhance_button.state(['!disabled'])
        self.sepia_button.state(['!disabled'])
        
        self.executor.shutdown(wait=False)
        self.error_queue = Queue()

    # Keep existing methods (drop, remove_duplicates, clear_loaded_files, open_output_folder)
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
        self.drop_label.configure(text="Drag and drop images here")
        self.status_label.configure(text="")
        self.open_folder_button.grid_remove()
        self.progress_bar.grid_remove()

    def open_output_folder(self):
        output_dir = os.path.join(self.folder_path, "processed_images")
        try:
            if os.name == 'nt':  # Windows
                os.startfile(output_dir)
            elif os.name == 'posix':  # macOS or Linux
                subprocess.Popen(['xdg-open', output_dir])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open output folder: {str(e)}")

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()