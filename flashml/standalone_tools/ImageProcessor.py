import tkinter as tk
from tkinter import ttk, messagebox, Scale, filedialog
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageTk
import os
from tkinterdnd2 import DND_FILES, TkinterDnD
import subprocess
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import hashlib
import numpy as np
import cv2  # Using OpenCV now
import sys  # For platform specific open_output_folder


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("1100x750")

        self.threshold_value = tk.IntVar(value=128)
        self.sauvola_window_var = tk.IntVar(value=25)
        self.sauvola_k_var = tk.DoubleVar(value=0.2)
        self.threshold_method_var = tk.StringVar(value="global")

        # Check for cv2.ximgproc availability (for Sauvola)
        self.sauvola_available = False
        try:
            if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "niBlackThreshold"):
                self.sauvola_available = True
        except AttributeError:
            self.sauvola_available = False

        if not self.sauvola_available and self.threshold_method_var.get() == "sauvola":
            self.threshold_method_var.set("global")  # Fallback if Sauvola was default

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
        self.image_hashes = {}

        self.preview_pil_image = None
        self.processed_preview_pil_image = None
        self.preview_tk_image = None

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.setup_ui()

        self.style = ttk.Style()
        self.style.configure("TButton", padding=5, relief="flat")
        self.style.configure("TFrame", background="#f5f5f5")
        self.style.configure("TLabelFrame", background="#f5f5f5")

        self._update_preview_area_state()
        if not self.sauvola_available:
            messagebox.showinfo(
                "Optional Feature",
                "Sauvola thresholding requires 'opencv-contrib-python'.\n"
                "The Sauvola option will be disabled.\n"
                "Install with: pip install opencv-contrib-python",
            )

    def setup_ui(self):
        container_frame = ttk.Frame(self.root)
        container_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        left_controls_frame = ttk.Frame(container_frame, padding="10")
        left_controls_frame.grid(
            row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E), padx=(0, 5)
        )
        container_frame.grid_columnconfigure(0, weight=1)
        container_frame.grid_rowconfigure(0, weight=1)

        file_frame = ttk.LabelFrame(
            left_controls_frame, text="File Operations", padding="10"
        )
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        file_frame.columnconfigure(0, weight=1)

        self.drop_frame = ttk.LabelFrame(
            file_frame, text="Drop Images Here", padding="20"
        )
        self.drop_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        self.drop_frame.columnconfigure(0, weight=1)
        self.drop_label = ttk.Label(self.drop_frame, text="Drag and drop images here")
        self.drop_label.grid(row=0, column=0, pady=30, padx=30)
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind("<<Drop>>", self.drop)

        self.browse_button = ttk.Button(
            file_frame, text="Browse Files", command=self.browse_files
        )
        self.browse_button.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        file_buttons_frame = ttk.Frame(file_frame)
        file_buttons_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        file_buttons_frame.columnconfigure(0, weight=1)
        file_buttons_frame.columnconfigure(1, weight=1)
        self.clear_button = ttk.Button(
            file_buttons_frame, text="Clear Files", command=self.clear_loaded_files
        )
        self.clear_button.grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E))
        self.remove_duplicates_button = ttk.Button(
            file_buttons_frame, text="Remove Duplicates", command=self.remove_duplicates
        )
        self.remove_duplicates_button.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))

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

        threshold_methods_frame = ttk.LabelFrame(
            left_controls_frame, text="Thresholding Methods", padding="10"
        )
        threshold_methods_frame.grid(
            row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5
        )
        threshold_methods_frame.columnconfigure(0, weight=1)

        ttk.Radiobutton(
            threshold_methods_frame,
            text="Global",
            variable=self.threshold_method_var,
            value="global",
            command=self._on_threshold_method_change,
        ).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(
            threshold_methods_frame,
            text="Otsu (OpenCV)",
            variable=self.threshold_method_var,
            value="otsu",
            command=self._on_threshold_method_change,
        ).grid(row=1, column=0, sticky=tk.W, pady=2)

        self.sauvola_radiobutton = ttk.Radiobutton(
            threshold_methods_frame,
            text="Sauvola (OpenCV)",
            variable=self.threshold_method_var,
            value="sauvola",
            command=self._on_threshold_method_change,
        )
        self.sauvola_radiobutton.grid(row=2, column=0, sticky=tk.W, pady=2)
        if not self.sauvola_available:
            self.sauvola_radiobutton.config(state=tk.DISABLED)

        process_frame = ttk.LabelFrame(
            left_controls_frame, text="Batch Processing Operations", padding="10"
        )
        process_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        process_frame.columnconfigure(0, weight=1)
        process_frame.columnconfigure(1, weight=1)
        process_frame.columnconfigure(2, weight=1)

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
            text="Apply Threshold (Batch)",
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

        status_frame = ttk.Frame(left_controls_frame)
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=10)
        status_frame.columnconfigure(0, weight=1)
        self.progress_bar = ttk.Progressbar(status_frame, mode="determinate")
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        self.status_label = ttk.Label(status_frame, text="Ready", font=("Arial", 8))
        self.status_label.grid(row=1, column=0, pady=5)

        self.open_folder_button = ttk.Button(
            left_controls_frame,
            text="Open Output Folder",
            command=self.open_output_folder,
        )
        self.open_folder_button.grid(row=4, column=0, pady=10, sticky=(tk.W, tk.E))

        right_panel_frame = ttk.Frame(container_frame, padding="10")
        right_panel_frame.grid(
            row=0, column=1, sticky=(tk.N, tk.S, tk.W, tk.E), padx=(5, 0)
        )
        container_frame.grid_columnconfigure(1, weight=3)

        preview_frame = ttk.LabelFrame(
            right_panel_frame, text="Image Preview (First Image)", padding="10"
        )
        preview_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        right_panel_frame.rowconfigure(0, weight=1)

        self.preview_label = ttk.Label(
            preview_frame, background="lightgrey", anchor=tk.CENTER
        )
        self.preview_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        # self.preview_label.configure(minwidth=350, minheight=250)

        self.tuners_frame = ttk.LabelFrame(
            right_panel_frame, text="Parameter Tuners", padding="10"
        )
        self.tuners_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        self.tuners_frame.columnconfigure(1, weight=1)
        right_panel_frame.rowconfigure(1, weight=0)

        self.global_thresh_label = ttk.Label(
            self.tuners_frame, text="Threshold (0-255):"
        )
        self.global_thresh_slider_val_label = ttk.Label(
            self.tuners_frame, text=str(self.threshold_value.get()), width=4
        )
        self.global_thresh_slider = Scale(
            self.tuners_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.threshold_value,
            resolution=1,
            command=lambda val: self._update_slider_label_and_preview(
                val, self.global_thresh_slider_val_label, "global"
            ),
        )

        self.sauvola_window_label = ttk.Label(
            self.tuners_frame, text="Window Size (odd, >=3):"
        )
        self.sauvola_window_slider_val_label = ttk.Label(
            self.tuners_frame, text=str(self.sauvola_window_var.get()), width=4
        )
        self.sauvola_window_slider = Scale(
            self.tuners_frame,
            from_=3,
            to=151,
            orient=tk.HORIZONTAL,
            variable=self.sauvola_window_var,
            resolution=2,
            command=lambda val: self._update_slider_label_and_preview(
                val, self.sauvola_window_slider_val_label, "sauvola_window"
            ),
        )

        self.sauvola_k_label = ttk.Label(self.tuners_frame, text="K (0.01-1.0):")
        self.sauvola_k_slider_val_label = ttk.Label(
            self.tuners_frame, text=f"{self.sauvola_k_var.get():.2f}", width=4
        )
        self.sauvola_k_slider = Scale(
            self.tuners_frame,
            from_=0.01,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.sauvola_k_var,
            resolution=0.01,
            command=lambda val: self._update_slider_label_and_preview(
                val, self.sauvola_k_slider_val_label, "sauvola_k", "%.2f"
            ),
        )

        self.save_preview_button = ttk.Button(
            right_panel_frame,
            text="Save Previewed Image",
            command=self.save_current_preview,
        )
        self.save_preview_button.grid(
            row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0)
        )

        self._on_threshold_method_change()
        self.update_ui_state()

    def _update_slider_label_and_preview(
        self, val_str, label_widget, param_type, format_str="%d"
    ):
        if param_type == "sauvola_window":
            val = int(float(val_str))
            if val % 2 == 0:
                val = max(
                    3,
                    val + 1 if val < self.sauvola_window_slider.cget("to") else val - 1,
                )
                max_val = int(self.sauvola_window_slider.cget("to"))
                if val > max_val:
                    val = max_val if max_val % 2 != 0 else max_val - 1

                if self.sauvola_window_var.get() != val:
                    self.sauvola_window_var.set(val)
                    return
            label_widget.config(text=str(val))
        elif param_type == "sauvola_k":
            val = float(val_str)
            label_widget.config(text=format_str % val)
        else:
            val = int(float(val_str))
            label_widget.config(text=str(val))

        # Only update preview if the relevant method is selected (or for global always if it's the param)
        current_method = self.threshold_method_var.get()
        if (param_type == "global" and current_method == "global") or (
            param_type in ["sauvola_window", "sauvola_k"]
            and current_method == "sauvola"
        ):
            self._update_preview_image()

    def _on_threshold_method_change(self):
        method = self.threshold_method_var.get()

        self.global_thresh_label.grid_remove()
        self.global_thresh_slider.grid_remove()
        self.global_thresh_slider_val_label.grid_remove()
        self.sauvola_window_label.grid_remove()
        self.sauvola_window_slider.grid_remove()
        self.sauvola_window_slider_val_label.grid_remove()
        self.sauvola_k_label.grid_remove()
        self.sauvola_k_slider.grid_remove()
        self.sauvola_k_slider_val_label.grid_remove()

        if method == "global":
            self.global_thresh_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.global_thresh_slider.grid(
                row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=2
            )
            self.global_thresh_slider_val_label.grid(
                row=0, column=2, sticky=tk.W, padx=5, pady=2
            )
        elif method == "sauvola":
            if self.sauvola_available:
                self.sauvola_window_label.grid(
                    row=0, column=0, sticky=tk.W, padx=5, pady=2
                )
                self.sauvola_window_slider.grid(
                    row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=2
                )
                self.sauvola_window_slider_val_label.grid(
                    row=0, column=2, sticky=tk.W, padx=5, pady=2
                )
                self.sauvola_k_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
                self.sauvola_k_slider.grid(
                    row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=2
                )
                self.sauvola_k_slider_val_label.grid(
                    row=1, column=2, sticky=tk.W, padx=5, pady=2
                )

                current_win_val = self.sauvola_window_var.get()
                if current_win_val % 2 == 0:
                    new_odd_val = max(
                        3,
                        current_win_val + 1
                        if current_win_val < self.sauvola_window_slider.cget("to")
                        else current_win_val - 1,
                    )
                    self.sauvola_window_var.set(new_odd_val)
                    self.sauvola_window_slider_val_label.config(text=str(new_odd_val))
            else:  # Should not happen if radio button is disabled, but defensive
                self.status_label.config(text="Sauvola (OpenCV) not available.")

        self._update_preview_image()

    def _update_preview_image(self):
        if not self.loaded_files:
            self.preview_label.config(image="", text="No image loaded")
            self.preview_pil_image = None
            self.processed_preview_pil_image = None
            if hasattr(self.preview_label, "image"):
                self.preview_label.image = None
            self._update_preview_area_state()
            return

        try:
            if (
                self.preview_pil_image is None
                or getattr(self.preview_pil_image, "_filepath", "")
                != self.loaded_files[0]
            ):
                try:
                    self.preview_pil_image = Image.open(self.loaded_files[0])
                    setattr(self.preview_pil_image, "_filepath", self.loaded_files[0])
                except FileNotFoundError:
                    self.preview_label.config(
                        image="", text="Error: Image file not found."
                    )
                    self.preview_pil_image = None
                    self.processed_preview_pil_image = None
                    return
                except Exception as e:
                    self.preview_label.config(
                        image="", text=f"Error loading image: {e}"
                    )
                    self.preview_pil_image = None
                    self.processed_preview_pil_image = None
                    return

            img_to_process_pil = self.preview_pil_image.copy()

            if img_to_process_pil.mode == "RGBA":
                background = Image.new("RGB", img_to_process_pil.size, (255, 255, 255))
                background.paste(img_to_process_pil, mask=img_to_process_pil.split()[3])
                img_to_process_pil = background
            elif img_to_process_pil.mode != "RGB" and img_to_process_pil.mode != "L":
                img_to_process_pil = img_to_process_pil.convert("RGB")

            gray_img_pil = img_to_process_pil.convert("L")
            processed_pil_img = gray_img_pil  # Default

            # Convert PIL grayscale to OpenCV format (NumPy array)
            img_np_gray = np.array(gray_img_pil)

            method = self.threshold_method_var.get()

            if method == "global":
                threshold = self.threshold_value.get()
                _, processed_np_img = cv2.threshold(
                    img_np_gray, threshold, 255, cv2.THRESH_BINARY
                )
                processed_pil_img = Image.fromarray(processed_np_img)
            elif method == "otsu":
                _, processed_np_img = cv2.threshold(
                    img_np_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                processed_pil_img = Image.fromarray(processed_np_img)
            elif method == "sauvola" and self.sauvola_available:
                window_size = self.sauvola_window_var.get()
                if window_size % 2 == 0:
                    window_size = max(3, window_size + 1)
                k_val = self.sauvola_k_var.get()

                try:
                    # cv2.ximgproc.niBlackThreshold requires blockSize > 1, k > 0
                    if window_size <= 1:
                        window_size = 3  # Ensure valid
                    if k_val <= 0:
                        k_val = 0.01  # Ensure valid

                    processed_np_img = cv2.ximgproc.niBlackThreshold(
                        img_np_gray,
                        255,
                        cv2.ximgproc.BINARIZATION_SAUVOLA,
                        window_size,
                        k_val,
                        r=128,
                    )
                    processed_pil_img = Image.fromarray(processed_np_img)
                except cv2.error as cv_err:  # Catch OpenCV specific errors
                    self.status_label.config(text=f"Sauvola Error (CV2): {cv_err}")
                    processed_pil_img = gray_img_pil  # Fallback
                except (
                    Exception
                ) as e:  # Catch other errors like missing ximgproc if check failed
                    self.status_label.config(text=f"Sauvola Error: {e}")
                    processed_pil_img = gray_img_pil  # Fallback
            elif method == "sauvola" and not self.sauvola_available:
                self.status_label.config(text="Sauvola unavailable. Using Global.")
                threshold = self.threshold_value.get()
                _, processed_np_img = cv2.threshold(
                    img_np_gray, threshold, 255, cv2.THRESH_BINARY
                )
                processed_pil_img = Image.fromarray(processed_np_img)

            self.processed_preview_pil_image = (
                processed_pil_img.convert("RGB")
                if processed_pil_img.mode != "RGB"
                else processed_pil_img
            )

            max_w, max_h = (
                self.preview_label.winfo_width(),
                self.preview_label.winfo_height(),
            )
            if max_w <= 1 or max_h <= 1:
                max_w, max_h = 350, 250

            display_img = self.processed_preview_pil_image.copy()
            display_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

            self.preview_tk_image = ImageTk.PhotoImage(display_img)
            self.preview_label.config(image=self.preview_tk_image, text="")
            self.preview_label.image = self.preview_tk_image

        except Exception as e:
            self.status_label.config(text=f"Preview error: {e}")
            # print(f"Full preview error: {e}", traceback.format_exc()) # For deeper debugging
            self.preview_label.config(image="", text="Error updating preview.")
            self.processed_preview_pil_image = None
        finally:
            self._update_preview_area_state()

    def _update_preview_area_state(self):
        has_preview_image = (
            self.processed_preview_pil_image is not None and self.loaded_files
        )
        self.save_preview_button["state"] = (
            "normal" if has_preview_image else "disabled"
        )

        slider_state = tk.NORMAL if self.loaded_files else tk.DISABLED
        try:
            self.global_thresh_slider.config(state=slider_state)
            # Only enable Sauvola sliders if method is Sauvola AND it's available
            sauvola_slider_state = tk.DISABLED
            if (
                self.loaded_files
                and self.sauvola_available
                and self.threshold_method_var.get() == "sauvola"
            ):
                sauvola_slider_state = tk.NORMAL

            self.sauvola_window_slider.config(state=sauvola_slider_state)
            self.sauvola_k_slider.config(state=sauvola_slider_state)

        except tk.TclError:
            pass

    def save_current_preview(self):
        if not self.processed_preview_pil_image or not self.loaded_files:
            messagebox.showwarning("No Image", "No preview image to save.")
            return

        original_filename = os.path.basename(self.loaded_files[0])
        original_name, original_ext = os.path.splitext(original_filename)

        method = self.threshold_method_var.get()
        suffix = f"_preview_CV_{method}"
        if method == "global":
            suffix += f"_{self.threshold_value.get()}"
        elif method == "sauvola" and self.sauvola_available:
            suffix += (
                f"_w{self.sauvola_window_var.get()}_k{self.sauvola_k_var.get():.2f}"
            )

        suggested_filename = f"{original_name}{suffix}{original_ext}"

        save_path = filedialog.asksaveasfilename(
            initialdir=self.output_folder,
            initialfile=suggested_filename,
            defaultextension=original_ext,
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("BMP files", "*.bmp"),
                ("TIFF files", "*.tiff"),
                ("All files", "*.*"),
            ],
        )

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_image = self.processed_preview_pil_image
                # Processed images are binary 'L' or '1' mode from thresholding.
                # Convert '1' to 'L' for better compatibility with PIL save, though it often handles it.
                if save_image.mode == "1":
                    save_image = save_image.convert("L")
                elif (
                    save_image.mode != "L"
                    and save_image.mode != "RGB"
                    and save_image.mode != "RGBA"
                ):
                    # If some other mode, try converting to RGB.
                    save_image = save_image.convert("RGB")

                save_image.save(save_path)
                messagebox.showinfo(
                    "Image Saved", f"Previewed image saved to:\n{save_path}"
                )
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save image: {e}")

    def update_ui_state(self):
        has_files = len(self.loaded_files) > 0

        has_duplicates = False
        if has_files and self.image_hashes:
            hash_values = [
                self.image_hashes.get(f)
                for f in self.loaded_files
                if self.image_hashes.get(f)
            ]
            if hash_values:
                counts = {h: hash_values.count(h) for h in set(hash_values)}
                has_duplicates = any(c > 1 for c in counts.values())

        self.clear_button["state"] = "normal" if has_files else "disabled"
        self.remove_duplicates_button["state"] = (
            "normal" if has_duplicates else "disabled"
        )

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

        if has_files:
            self.drop_label.configure(text=f"{len(self.loaded_files)} image(s) loaded")
            self.root.after(50, self._update_preview_image)
        else:
            self.drop_label.configure(text="Drag and drop images here")
            self.status_label.configure(text="Ready")
            self._update_preview_image()

        self._update_preview_area_state()

    def browse_files(self):
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[
                ("Image files", " ".join(f"*{ext}" for ext in self.supported_formats))
            ],
        )
        if files:
            new_files = [f for f in files if f not in self.loaded_files]
            self.loaded_files.extend(new_files)
            if new_files:
                self.calculate_image_hashes(specific_files=new_files)
            self.update_ui_state()

    def browse_output_folder(self):
        folder = filedialog.askdirectory(
            title="Select Output Folder", initialdir=self.output_folder
        )
        if folder:
            self.output_folder = folder
            self.output_path_var.set(folder)

    def calculate_image_hashes(self, specific_files=None):
        files_to_hash = (
            specific_files if specific_files is not None else self.loaded_files
        )

        for file_path in files_to_hash:
            if file_path not in self.image_hashes:
                try:
                    with open(file_path, "rb") as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                        self.image_hashes[file_path] = file_hash
                except Exception:
                    self.image_hashes[file_path] = file_path

        self.update_ui_state()  # Update UI which includes duplicate button state

    def process_single_image(self, file_info):
        file_path, output_dir, effect = file_info
        try:
            with Image.open(file_path) as pil_img:
                alpha_channel = None
                original_mode_is_rgba = pil_img.mode == "RGBA"

                # Prepare image for processing (convert to RGB or L, handle alpha)
                if original_mode_is_rgba and effect not in ("grayscale", "threshold"):
                    alpha_channel = pil_img.split()[-1]
                    img_for_processing_pil = pil_img.convert("RGB")
                elif pil_img.mode not in ("RGB", "L"):
                    img_for_processing_pil = pil_img.convert("RGB")
                else:
                    img_for_processing_pil = pil_img

                processed_pil_img = None
                suffix = effect

                if effect == "invert":
                    if img_for_processing_pil.mode == "L":
                        processed_pil_img = ImageOps.invert(img_for_processing_pil)
                    else:
                        processed_pil_img = ImageOps.invert(
                            img_for_processing_pil.convert("RGB")
                        )
                elif effect == "grayscale":
                    processed_pil_img = ImageOps.grayscale(pil_img)
                    alpha_channel = None
                elif effect == "threshold":
                    gray_pil = pil_img.convert(
                        "L"
                    )  # Start with original for grayscale conversion
                    img_np_gray = np.array(gray_pil)

                    method = self.threshold_method_var.get()
                    if method == "global":
                        threshold_val = self.threshold_value.get()
                        _, processed_np = cv2.threshold(
                            img_np_gray, threshold_val, 255, cv2.THRESH_BINARY
                        )
                        suffix = f"threshold_CV_global_{threshold_val}"
                    elif method == "otsu":
                        _, processed_np = cv2.threshold(
                            img_np_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                        )
                        suffix = "threshold_CV_otsu"
                    elif method == "sauvola" and self.sauvola_available:
                        window_size = self.sauvola_window_var.get()
                        if window_size % 2 == 0:
                            window_size = max(3, window_size + 1)
                        k_val = self.sauvola_k_var.get()
                        if window_size <= 1:
                            window_size = 3
                        if k_val <= 0:
                            k_val = 0.01
                        processed_np = cv2.ximgproc.niBlackThreshold(
                            img_np_gray,
                            255,
                            cv2.ximgproc.BINARIZATION_SAUVOLA,
                            window_size,
                            k_val,
                            r=128,
                        )
                        suffix = f"threshold_CV_sauvola_w{window_size}_k{k_val:.2f}"
                    else:  # Fallback if Sauvola not available or method unknown
                        threshold_val = self.threshold_value.get()
                        _, processed_np = cv2.threshold(
                            img_np_gray, threshold_val, 255, cv2.THRESH_BINARY
                        )
                        suffix = f"threshold_CV_fallback_{threshold_val}"

                    processed_pil_img = Image.fromarray(processed_np)
                    alpha_channel = None

                # PIL-based effects
                elif effect == "blur":
                    processed_pil_img = img_for_processing_pil.filter(
                        ImageFilter.GaussianBlur(radius=2)
                    )
                elif effect == "enhance":
                    enhancer = ImageEnhance.Color(img_for_processing_pil)
                    processed_pil_img = enhancer.enhance(1.5)
                elif effect == "sepia":
                    img_sepia_pil = img_for_processing_pil.convert("RGB").copy()
                    width, height = img_sepia_pil.size
                    pixels = img_sepia_pil.load()
                    for x_coord in range(width):
                        for y_coord in range(height):
                            r, g, b = img_sepia_pil.getpixel((x_coord, y_coord))[:3]
                            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                            tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                            pixels[x_coord, y_coord] = (
                                min(tr, 255),
                                min(tg, 255),
                                min(tb, 255),
                            )
                    processed_pil_img = img_sepia_pil
                elif effect == "sharpen":
                    processed_pil_img = img_for_processing_pil.filter(
                        ImageFilter.SHARPEN
                    )
                elif effect == "edge":
                    processed_pil_img = img_for_processing_pil.filter(
                        ImageFilter.FIND_EDGES
                    )
                elif effect == "emboss":
                    processed_pil_img = img_for_processing_pil.filter(
                        ImageFilter.EMBOSS
                    )

                if processed_pil_img is None:
                    self.error_queue.put((file_path, "Effect produced no image."))
                    return False

                if alpha_channel and processed_pil_img.mode == "RGB":
                    processed_pil_img.putalpha(alpha_channel)
                elif (
                    original_mode_is_rgba
                    and processed_pil_img.mode != "RGBA"
                    and alpha_channel
                ):
                    if processed_pil_img.mode in ("L", "P", "1"):
                        # If image became L/P/1 and had alpha, convert to RGBA to reapply.
                        # Note: This might not be visually perfect if colors under alpha changed significantly.
                        temp_rgba = processed_pil_img.convert("RGBA")
                        r_c, g_c, b_c, _ = temp_rgba.split()
                        processed_pil_img = Image.merge(
                            "RGBA", (r_c, g_c, b_c, alpha_channel)
                        )

                filename, ext = os.path.splitext(os.path.basename(file_path))
                new_filename = f"{filename}_{suffix}{ext}"
                output_path = os.path.join(output_dir, new_filename)
                counter = 1
                while os.path.exists(output_path):
                    new_filename = f"{filename}_{suffix}_{counter}{ext}"
                    output_path = os.path.join(output_dir, new_filename)
                    counter += 1

                save_kwargs = {}
                if (
                    ext.lower() in [".jpg", ".jpeg"]
                    and processed_pil_img.mode == "RGBA"
                ):
                    processed_pil_img = processed_pil_img.convert("RGB")

                if (
                    processed_pil_img.mode == "1"
                ):  # Ensure '1' mode is saved as 'L' for wider compatibility
                    processed_pil_img = processed_pil_img.convert("L")

                processed_pil_img.save(output_path, **save_kwargs)
                return True
        except Exception as e:
            # import traceback # For debugging
            # print(f"Error processing {file_path}: {e}\n{traceback.format_exc()}")
            self.error_queue.put((file_path, str(e)))
            return False

    def process_images(self, effect):
        if not self.loaded_files:
            messagebox.showwarning("No Images", "Please load images first.")
            return

        os.makedirs(self.output_folder, exist_ok=True)
        self.processed_count = 0
        self.error_queue = Queue()
        self.progress_bar["maximum"] = len(self.loaded_files)
        self.progress_bar["value"] = 0

        all_proc_buttons = [
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
        for button in all_proc_buttons:
            button["state"] = "disabled"
        self.status_label["text"] = f"Processing with '{effect}'..."

        self.executor = ThreadPoolExecutor(
            max_workers=min(os.cpu_count() or 4, len(self.loaded_files))
        )
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
        errors = []
        while not self.error_queue.empty():
            file_path, error_msg = self.error_queue.get()
            errors.append(f"{os.path.basename(file_path)}: {error_msg}")

        if errors:
            error_text = "\n".join(errors[:10])
            if len(errors) > 10:
                error_text += f"\n...and {len(errors) - 10} more errors"
            messagebox.showerror("Processing Errors", f"Errors occurred:\n{error_text}")

        successful_count = len(self.loaded_files) - len(errors)
        self.status_label["text"] = (
            f"Processing complete! {successful_count} of {len(self.loaded_files)} images processed."
        )

        all_proc_buttons = [
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
        for button in all_proc_buttons:
            button["state"] = "normal" if self.loaded_files else "disabled"

        self.update_ui_state()
        if hasattr(self, "executor") and self.executor:
            self.executor.shutdown(wait=False)

    def drop(self, event):
        try:
            files = self.root.tk.splitlist(event.data)
            valid_files_new = []
            invalid_or_duplicate_count = 0

            for f_path in files:
                if os.path.splitext(f_path)[1].lower() in self.supported_formats:
                    if f_path not in self.loaded_files:
                        valid_files_new.append(f_path)
                    else:
                        invalid_or_duplicate_count += 1
                else:
                    invalid_or_duplicate_count += 1

            if (
                not valid_files_new
                and invalid_or_duplicate_count > 0
                and not self.loaded_files
            ):
                initial_load_failed = True
                for f_check in files:
                    if f_check in self.loaded_files:
                        initial_load_failed = False
                        break
                if initial_load_failed:
                    messagebox.showwarning(
                        "Invalid Files",
                        "No new valid image files were found in the drop.",
                    )
                    return

            if invalid_or_duplicate_count > 0 and valid_files_new:
                messagebox.showwarning(
                    "Files Excluded",
                    f"{invalid_or_duplicate_count} file(s) were unsupported, already loaded, or invalid and were excluded.",
                )

            if valid_files_new:
                self.loaded_files.extend(valid_files_new)
                self.calculate_image_hashes(
                    specific_files=valid_files_new
                )  # Hash only new files

            self.update_ui_state()  # This will update preview etc.
        except Exception as e:
            messagebox.showerror(
                "Drop Error", f"An error occurred during drag and drop: {e}"
            )
            self.update_ui_state()

    def remove_duplicates(self):
        if not self.loaded_files:
            return

        current_files_to_recheck_hash = [
            f for f in self.loaded_files if f not in self.image_hashes
        ]
        if current_files_to_recheck_hash:
            self.calculate_image_hashes(specific_files=current_files_to_recheck_hash)

        seen_hashes = set()
        unique_files = []

        for file_path in self.loaded_files:
            file_hash = self.image_hashes.get(file_path, file_path)
            if file_hash not in seen_hashes:
                seen_hashes.add(file_hash)
                unique_files.append(file_path)

        removed_count = len(self.loaded_files) - len(unique_files)
        self.loaded_files = unique_files
        self.image_hashes = {
            k: v for k, v in self.image_hashes.items() if k in self.loaded_files
        }

        if removed_count > 0:
            messagebox.showinfo(
                "Duplicates Removed", f"Removed {removed_count} duplicate image(s)."
            )
        else:
            messagebox.showinfo("No Duplicates", "No duplicate images found to remove.")

        self.update_ui_state()

    def clear_loaded_files(self):
        self.loaded_files = []
        self.image_hashes = {}
        self.preview_pil_image = None
        self.processed_preview_pil_image = None
        if (
            hasattr(self.preview_label, "image")
            and self.preview_label.image is not None
        ):
            self.preview_label.config(image="")
            self.preview_label.image = None
            self.preview_tk_image = None

        self.update_ui_state()
        self.progress_bar["value"] = 0
        self.status_label.config(text="Ready. Files cleared.")

    def open_output_folder(self):
        try:
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder, exist_ok=True)
                messagebox.showinfo(
                    "Folder Created", f"Output folder created at:\n{self.output_folder}"
                )

            if os.name == "nt":
                os.startfile(self.output_folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", self.output_folder])
            else:
                subprocess.Popen(["xdg-open", self.output_folder])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open output folder: {str(e)}")


if __name__ == "__main__":
    # import traceback # For debugging preview errors
    root = TkinterDnD.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
