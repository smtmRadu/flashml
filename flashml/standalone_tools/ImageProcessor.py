import tkinter as tk
from tkinter import ttk, messagebox, Scale, filedialog
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import os
from tkinterdnd2 import DND_FILES, TkinterDnD
import subprocess
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import hashlib


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("550x750")  # Wider and taller for new features

        # Initialize threshold value
        self.threshold_value = tk.IntVar(value=128)

        # Setup folder path - using Downloads folder as default
        self.folder_path = os.path.join(os.path.expanduser("~"), "Downloads")
        self.output_folder = os.path.join(self.folder_path, "processed_images")

        self.loaded_files = []
        self.supported_formats = [
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".webp",
            ".bmp",
            ".tiff",
            ".tif",
        ]
        self.processed_count = 0
        self.error_queue = Queue()
        self.image_hashes = {}  # For better duplicate detection

        # Configure root window to be resizable
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.setup_ui()

        # Set up theme
        self.style = ttk.Style()
        self.style.configure("TButton", padding=5, relief="flat")
        self.style.configure("TFrame", background="#f5f5f5")
        self.style.configure("TLabelFrame", background="#f5f5f5")

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)

        # File Operation Frame
        file_frame = ttk.LabelFrame(main_frame, text="File Operations", padding="10")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        file_frame.columnconfigure(0, weight=1)

        # Drop zone
        self.drop_frame = ttk.LabelFrame(
            file_frame, text="Drop Images Here", padding="20"
        )
        self.drop_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        self.drop_frame.columnconfigure(0, weight=1)

        self.drop_label = ttk.Label(self.drop_frame, text="Drag and drop images here")
        self.drop_label.grid(row=0, column=0, pady=30, padx=30)
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind("<<Drop>>", self.drop)

        # Browse button
        self.browse_button = ttk.Button(
            file_frame, text="Browse Files", command=self.browse_files
        )
        self.browse_button.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        # File operation buttons
        file_buttons_frame = ttk.Frame(file_frame)
        file_buttons_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        file_buttons_frame.columnconfigure(0, weight=1)
        file_buttons_frame.columnconfigure(1, weight=1)

        # Clear button
        self.clear_button = ttk.Button(
            file_buttons_frame, text="Clear Files", command=self.clear_loaded_files
        )
        self.clear_button.grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E))

        # Remove Duplicates button
        self.remove_duplicates_button = ttk.Button(
            file_buttons_frame, text="Remove Duplicates", command=self.remove_duplicates
        )
        self.remove_duplicates_button.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))

        # Output folder selection
        output_frame = ttk.Frame(file_frame)
        output_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        output_frame.columnconfigure(1, weight=1)

        output_label = ttk.Label(output_frame, text="Output:")
        output_label.grid(row=0, column=0, padx=5)

        self.output_path_var = tk.StringVar(value=self.output_folder)
        output_entry = ttk.Entry(
            output_frame, textvariable=self.output_path_var, state="readonly"
        )
        output_entry.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))

        self.output_browse_button = ttk.Button(
            output_frame, text="...", width=3, command=self.browse_output_folder
        )
        self.output_browse_button.grid(row=0, column=2, padx=5)

        # Advanced options frame
        advanced_frame = ttk.LabelFrame(
            main_frame, text="Advanced Options", padding="10"
        )
        advanced_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        advanced_frame.columnconfigure(0, weight=1)

        # Threshold slider
        threshold_frame = ttk.Frame(advanced_frame)
        threshold_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        threshold_frame.columnconfigure(1, weight=1)

        self.threshold_label = ttk.Label(threshold_frame, text="Threshold:")
        self.threshold_label.grid(row=0, column=0, padx=5)

        self.threshold_slider = Scale(
            threshold_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.threshold_value,
            resolution=1,
        )
        self.threshold_slider.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))

        self.threshold_value_label = ttk.Label(threshold_frame, text="128")
        self.threshold_value_label.grid(row=0, column=2, padx=5)

        # Update the label when slider changes
        self.threshold_slider.config(
            command=lambda val: self.threshold_value_label.config(
                text=str(int(float(val)))
            )
        )

        # Processing buttons frame
        process_frame = ttk.LabelFrame(
            main_frame, text="Processing Options", padding="10"
        )
        process_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        process_frame.columnconfigure(0, weight=1)
        process_frame.columnconfigure(1, weight=1)
        process_frame.columnconfigure(2, weight=1)

        # Add processing buttons - 3 columns now
        self.inverse_button = ttk.Button(
            process_frame,
            text="Invert Colors",
            command=lambda: self.process_images("invert"),
        )
        self.inverse_button.grid(row=0, column=0, pady=5, padx=5, sticky=(tk.W, tk.E))

        self.grayscale_button = ttk.Button(
            process_frame,
            text="Grayscale",
            command=lambda: self.process_images("grayscale"),
        )
        self.grayscale_button.grid(row=0, column=1, pady=5, padx=5, sticky=(tk.W, tk.E))

        self.threshold_button = ttk.Button(
            process_frame,
            text="Threshold",
            command=lambda: self.process_images("threshold"),
        )
        self.threshold_button.grid(row=0, column=2, pady=5, padx=5, sticky=(tk.W, tk.E))

        self.blur_button = ttk.Button(
            process_frame, text="Blur", command=lambda: self.process_images("blur")
        )
        self.blur_button.grid(row=1, column=0, pady=5, padx=5, sticky=(tk.W, tk.E))

        self.enhance_button = ttk.Button(
            process_frame,
            text="Enhance Colors",
            command=lambda: self.process_images("enhance"),
        )
        self.enhance_button.grid(row=1, column=1, pady=5, padx=5, sticky=(tk.W, tk.E))

        self.sepia_button = ttk.Button(
            process_frame,
            text="Sepia Tone",
            command=lambda: self.process_images("sepia"),
        )
        self.sepia_button.grid(row=1, column=2, pady=5, padx=5, sticky=(tk.W, tk.E))

        self.sharpen_button = ttk.Button(
            process_frame,
            text="Sharpen",
            command=lambda: self.process_images("sharpen"),
        )
        self.sharpen_button.grid(row=2, column=0, pady=5, padx=5, sticky=(tk.W, tk.E))

        self.edge_button = ttk.Button(
            process_frame,
            text="Edge Detection",
            command=lambda: self.process_images("edge"),
        )
        self.edge_button.grid(row=2, column=1, pady=5, padx=5, sticky=(tk.W, tk.E))

        self.emboss_button = ttk.Button(
            process_frame, text="Emboss", command=lambda: self.process_images("emboss")
        )
        self.emboss_button.grid(row=2, column=2, pady=5, padx=5, sticky=(tk.W, tk.E))

        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=10)
        status_frame.columnconfigure(0, weight=1)

        # Progress bar
        self.progress_bar = ttk.Progressbar(status_frame, mode="determinate")
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        # Status label
        self.status_label = ttk.Label(status_frame, text="Ready", font=("Arial", 8))
        self.status_label.grid(row=1, column=0, pady=5)

        # Open folder button
        self.open_folder_button = ttk.Button(
            main_frame, text="Open Output Folder", command=self.open_output_folder
        )
        self.open_folder_button.grid(row=4, column=0, pady=10, sticky=(tk.W, tk.E))

        # Initially hide some elements
        self.progress_bar["value"] = 0

        # Set initial states for all buttons
        self.update_ui_state()

    def update_ui_state(self):
        """Update UI elements based on current state"""
        has_files = len(self.loaded_files) > 0
        has_duplicates = len(set(self.loaded_files)) < len(self.loaded_files)

        # Update file-related buttons
        self.clear_button["state"] = "normal" if has_files else "disabled"
        self.remove_duplicates_button["state"] = (
            "normal" if has_duplicates else "disabled"
        )

        # Update processing buttons
        processing_buttons = [
            self.inverse_button,
            self.grayscale_button,
            self.threshold_button,
            self.blur_button,
            self.enhance_button,
            self.sepia_button,
            self.sharpen_button,
            self.edge_button,
            self.emboss_button,
        ]

        for button in processing_buttons:
            button["state"] = "normal" if has_files else "disabled"

        # Update status display
        if has_files:
            self.drop_label.configure(text=f"{len(self.loaded_files)} image(s) loaded")
        else:
            self.drop_label.configure(text="Drag and drop images here")
            self.status_label.configure(text="Ready")

    def browse_files(self):
        """Allow user to browse and select files"""
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[
                ("Image files", " ".join(f"*{ext}" for ext in self.supported_formats))
            ],
        )

        if files:
            self.loaded_files.extend(files)
            self.update_ui_state()
            self.calculate_image_hashes()

    def browse_output_folder(self):
        """Select output folder"""
        folder = filedialog.askdirectory(
            title="Select Output Folder", initialdir=self.output_folder
        )

        if folder:
            self.output_folder = folder
            self.output_path_var.set(folder)

    def calculate_image_hashes(self):
        """Calculate hashes for all images to better detect duplicates"""
        for file_path in self.loaded_files:
            if file_path not in self.image_hashes:
                try:
                    with open(file_path, "rb") as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                        self.image_hashes[file_path] = file_hash
                except Exception:
                    # If we can't hash it, just use the path as a fallback
                    self.image_hashes[file_path] = file_path

        # Check for duplicates by hash
        hash_count = {}
        for h in self.image_hashes.values():
            hash_count[h] = hash_count.get(h, 0) + 1

        has_duplicates = any(count > 1 for count in hash_count.values())
        if has_duplicates:
            self.remove_duplicates_button["state"] = "normal"
        else:
            self.remove_duplicates_button["state"] = "disabled"

    def process_single_image(self, file_info):
        file_path, output_dir, effect = file_info
        try:
            with Image.open(file_path) as img:
                # Ensure we're working with RGB or RGBA
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")

                # Handle transparency for effects that require RGB
                has_alpha = img.mode == "RGBA"
                if has_alpha and effect not in ("grayscale",):
                    # Save alpha channel for later
                    alpha = img.split()[3]
                    # Convert to RGB for processing
                    img = img.convert("RGB")

                # Apply the selected effect
                if effect == "invert":
                    processed_img = ImageOps.invert(img)
                    suffix = "inverted"
                elif effect == "grayscale":
                    processed_img = ImageOps.grayscale(img)
                    suffix = "grayscale"
                elif effect == "threshold":
                    # Convert to grayscale first
                    gray_img = ImageOps.grayscale(img)
                    # Apply threshold
                    threshold = self.threshold_value.get()
                    processed_img = gray_img.point(
                        lambda p: 255 if p > threshold else 0
                    )
                    suffix = f"threshold_{threshold}"
                elif effect == "blur":
                    processed_img = img.filter(ImageFilter.GaussianBlur(radius=2))
                    suffix = "blur"
                elif effect == "enhance":
                    enhancer = ImageEnhance.Color(img)
                    processed_img = enhancer.enhance(1.5)
                    suffix = "enhanced"
                elif effect == "sepia":
                    width, height = img.size
                    processed_img = img.copy()
                    pixels = processed_img.load()

                    for x in range(width):
                        for y in range(height):
                            r, g, b = img.getpixel((x, y))[:3]
                            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                            tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                            processed_img.putpixel(
                                (x, y), (min(tr, 255), min(tg, 255), min(tb, 255))
                            )
                    suffix = "sepia"
                elif effect == "sharpen":
                    processed_img = img.filter(ImageFilter.SHARPEN)
                    suffix = "sharpen"
                elif effect == "edge":
                    processed_img = img.filter(ImageFilter.FIND_EDGES)
                    suffix = "edge"
                elif effect == "emboss":
                    processed_img = img.filter(ImageFilter.EMBOSS)
                    suffix = "emboss"

                # Restore alpha channel if needed
                if has_alpha and effect not in ("grayscale",):
                    if processed_img.mode != "RGBA":
                        r, g, b = processed_img.split()
                        processed_img = Image.merge("RGBA", (r, g, b, alpha))

                # Save the processed image
                filename, ext = os.path.splitext(os.path.basename(file_path))
                new_filename = f"{filename}_{suffix}{ext}"
                output_path = os.path.join(output_dir, new_filename)

                # Make sure output path doesn't exist already
                counter = 1
                while os.path.exists(output_path):
                    new_filename = f"{filename}_{suffix}_{counter}{ext}"
                    output_path = os.path.join(output_dir, new_filename)
                    counter += 1

                processed_img.save(output_path)
                return True
        except Exception as e:
            self.error_queue.put((file_path, str(e)))
            return False

    def process_images(self, effect):
        if not self.loaded_files:
            messagebox.showwarning("No Images", "Please load images first.")
            return

        # Ensure output directory exists
        os.makedirs(self.output_folder, exist_ok=True)

        # Initialize processing variables
        self.processed_count = 0
        self.error_queue = Queue()
        self.progress_bar["maximum"] = len(self.loaded_files)
        self.progress_bar["value"] = 0

        # Disable all processing buttons during processing
        processing_buttons = [
            self.inverse_button,
            self.grayscale_button,
            self.threshold_button,
            self.blur_button,
            self.enhance_button,
            self.sepia_button,
            self.sharpen_button,
            self.edge_button,
            self.emboss_button,
        ]

        for button in processing_buttons:
            button["state"] = "disabled"

        self.status_label["text"] = "Processing..."

        # Create a thread pool executor
        self.executor = ThreadPoolExecutor(
            max_workers=min(os.cpu_count() or 4, len(self.loaded_files))
        )

        # Submit all files for processing
        for file_path in self.loaded_files:
            future = self.executor.submit(
                self.process_single_image, (file_path, self.output_folder, effect)
            )
            future.add_done_callback(self.process_complete)

    def process_complete(self, future):
        self.root.after(0, self.update_progress)

    def update_progress(self):
        self.processed_count += 1
        self.progress_bar["value"] = self.processed_count
        self.status_label["text"] = (
            f"Processed {self.processed_count} of {len(self.loaded_files)} images..."
        )

        if self.processed_count >= len(self.loaded_files):
            self.finalize_processing()

    def finalize_processing(self):
        # Check for errors
        errors = []
        while not self.error_queue.empty():
            file_path, error_msg = self.error_queue.get()
            errors.append(f"{os.path.basename(file_path)}: {error_msg}")

        if errors:
            error_text = "\n".join(errors[:5])
            if len(errors) > 5:
                error_text += f"\n...and {len(errors) - 5} more errors"
            messagebox.showerror("Processing Errors", f"Errors occurred:\n{error_text}")

        self.status_label["text"] = (
            f"Processing complete! {len(self.loaded_files) - len(errors)} images processed successfully."
        )

        # Re-enable all processing buttons
        processing_buttons = [
            self.inverse_button,
            self.grayscale_button,
            self.threshold_button,
            self.blur_button,
            self.enhance_button,
            self.sepia_button,
            self.sharpen_button,
            self.edge_button,
            self.emboss_button,
        ]

        for button in processing_buttons:
            button["state"] = "normal"

        self.executor.shutdown(wait=False)

    def drop(self, event):
        files = self.root.tk.splitlist(event.data)
        valid_files = [
            f for f in files if os.path.splitext(f)[1].lower() in self.supported_formats
        ]

        if not valid_files:
            messagebox.showwarning("Invalid Files", "No valid image files were found.")
            return

        if len(files) != len(valid_files):
            messagebox.showwarning(
                "Invalid Files",
                "Some files were not in a supported format and were excluded.",
            )

        self.loaded_files.extend(valid_files)
        self.update_ui_state()
        self.calculate_image_hashes()

    def remove_duplicates(self):
        """Remove duplicate files based on content hash"""
        if not self.image_hashes:
            self.calculate_image_hashes()

        # Keep track of hashes we've seen
        seen_hashes = set()
        unique_files = []

        for file_path in self.loaded_files:
            file_hash = self.image_hashes.get(file_path, file_path)
            if file_hash not in seen_hashes:
                seen_hashes.add(file_hash)
                unique_files.append(file_path)

        removed_count = len(self.loaded_files) - len(unique_files)
        self.loaded_files = unique_files

        messagebox.showinfo(
            "Duplicates Removed", f"Removed {removed_count} duplicate image(s)."
        )

        self.update_ui_state()

    def clear_loaded_files(self):
        self.loaded_files = []
        self.image_hashes = {}
        self.update_ui_state()
        self.progress_bar["value"] = 0

    def open_output_folder(self):
        try:
            if os.path.exists(self.output_folder):
                if os.name == "nt":  # Windows
                    os.startfile(self.output_folder)
                elif os.name == "posix":  # macOS or Linux
                    if os.path.exists("/usr/bin/open"):  # macOS
                        subprocess.Popen(["open", self.output_folder])
                    else:  # Linux
                        subprocess.Popen(["xdg-open", self.output_folder])
            else:
                os.makedirs(self.output_folder, exist_ok=True)
                messagebox.showinfo(
                    "Folder Created", f"Output folder created at:\n{self.output_folder}"
                )
                self.open_output_folder()  # Try again
        except Exception as e:
            messagebox.showerror("Error", f"Could not open output folder: {str(e)}")


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
