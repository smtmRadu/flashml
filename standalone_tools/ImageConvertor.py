import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image
import os
from tkinterdnd2 import DND_FILES, TkinterDnD
import subprocess

class ImageConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Converter")
        self.root.geometry("270x410")
        self.folder_path = os.path.join(os.path.expanduser("~"), "Downloads")
        self.loaded_files = []
        self.formats = ['png', 'jpg', 'jpeg', 'gif', 'webp']

        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)


        # Drop zone
        self.drop_frame = ttk.LabelFrame(main_frame, text="Drop Images Here", padding="20")
        self.drop_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        self.drop_frame.columnconfigure(0, weight=1)
        self.drop_frame.rowconfigure(0, weight=1)

        self.drop_label = ttk.Label(self.drop_frame, text="Drag and drop images here")
        self.drop_label.grid(row=0, column=0, pady=50, sticky="n")
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.drop)


        # Remove Duplicates button, initially hidden
        self.remove_duplicates_button = ttk.Button(self.drop_frame, text="Remove Duplicates", command=self.remove_duplicates)
        self.remove_duplicates_button.grid(row=1, column=1, pady=10, columnspan=2)
        self.remove_duplicates_button.grid_remove()  # Hide by default

        # clear images
        self.clear_loaded_files_button = ttk.Button(self.drop_frame, text="Clear", command=self.clear_loaded_files)
        self.clear_loaded_files_button.grid(row=1, column=0, pady=10, columnspan=1)
        self.clear_loaded_files_button.grid_remove()  # Hide by default




        # Format selection
        ttk.Label(main_frame, text="Convert to:").grid(row=1, column=0, pady=10)
        self.target_format = tk.StringVar(value='png')
        format_menu = ttk.OptionMenu(main_frame, self.target_format, 'png', *self.formats)
        format_menu.grid(row=1, column=1, pady=10)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate')
        self.progress_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        self.progress_bar.grid_remove()
        
        self.convert_button = ttk.Button(main_frame, text="Convert", command=self.convert_images)
        self.convert_button.grid(row=3, column=0, columnspan=2, pady=10)
        
        # go to explorer
        self.go_to_explorer_button = ttk.Button(main_frame, text="See Results", command=self.open_conversion_folder)
        self.go_to_explorer_button.grid(row=4, column=0, pady=10, columnspan=2)
        self.go_to_explorer_button.grid_remove()  # Hide by default


        self.status_label = ttk.Label(main_frame, text="...", font=("Arial", 7, "bold"))
        self.status_label.grid(row=3, column=0, columnspan=2, pady=5)
        self.status_label.grid_remove()  # Hide by default


    def drop(self, event):
        files = self.root.tk.splitlist(event.data)
        valid_files = [f for f in files if os.path.splitext(f)[1].lower()[1:] in self.formats]

        if len(files) != len(valid_files):
            messagebox.showwarning("No Valid Files", "Some files were not in a valid format and they were excluded.")
            return

        self.loaded_files.extend(valid_files)
        unique_files = list(set(self.loaded_files))

        if (len(self.loaded_files) > 0):
            self.clear_loaded_files_button.grid()

        if len(unique_files) < len(self.loaded_files):
            # If duplicates found, show the button
            self.remove_duplicates_button.grid()
        
        self.drop_label.configure(text=f"{len(self.loaded_files)} image(s) loaded")
        self.root.update_idletasks()
    
    def remove_duplicates(self):
        # Remove duplicates and hide the button
        self.loaded_files = list(set(self.loaded_files))
        self.remove_duplicates_button.grid_remove()
        self.drop_label.configure(text=f"{len(self.loaded_files)} image(s) loaded")
    
    def clear_loaded_files(self):
        self.loaded_files = []
        self.clear_loaded_files_button.grid_remove()
        self.drop_label.configure(text=f"{len(self.loaded_files)} image(s) loaded")
        self.remove_duplicates()

        self.clear_loaded_files_button.configure(text="Clear")
        self.go_to_explorer_button.grid_remove()
        self.progress_bar.grid_remove()
        self.status_label.grid_remove()
        self.convert_button.grid()

    def open_conversion_folder(self):
        try:
            if os.name == 'nt':  # Windows
                os.startfile(os.path.join(self.folder_path, "converted_images"))
            elif os.name == 'posix':  # macOS or Linux
                if sys.platform == "darwin":  # macOS
                    subprocess.Popen(["open", self.folder_path])
                else:  # Linux
                    subprocess.Popen(["xdg-open", self.folder_path])
        except Exception as e:
            print(f"Error opening folder: {e}")



    def convert_images(self):

        if(len(self.loaded_files) == 0):
            messagebox.showwarning("No Images", "No images were loaded to convert.")
            return

        total_files = len(self.loaded_files)
        self.progress_bar['maximum'] = total_files
        self.progress_bar.grid()
        
        output_dir = os.path.join(self.folder_path, "converted_images")
        os.makedirs(output_dir, exist_ok=True)
        self.status_label.grid()
        
        for i, file_path in enumerate(self.loaded_files, 1):
            try:
                # Update status
                self.status_label['text'] = f"Converting {os.path.basename(file_path)}..."
                
                self.root.update_idletasks()
                
                # Open and convert image
                with Image.open(file_path) as img:
                    # If converting from PNG to JPG, remove alpha channel
                    if img.mode in ('RGBA', 'LA') and self.target_format.get().lower() in ('.jpg', '.jpeg'):
                        bg = Image.new('RGB', img.size, (255, 255, 255))
                        bg.paste(img, mask=img.split()[-1])
                        img = bg
                    
                    # Generate output filename with correct extension
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    output_ext = self.target_format.get().lstrip('.')
                    output_path = os.path.join(output_dir, f"{base_name}.{output_ext}")
                    
                    # Save image in the desired format
                    img.save(output_path, output_ext.upper())
                
                # Update progress
                self.progress_bar['value'] = i
                self.root.update_idletasks()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error converting {os.path.basename(file_path)}: {str(e)}")
        
        self.status_label['text'] = f"Done! ({output_dir})"
       
        self.clear_loaded_files_button.configure(text="Restart Convertor")
        self.remove_duplicates()
        self.go_to_explorer_button.grid()
        self.convert_button.grid_remove()

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ImageConverterApp(root)
    root.mainloop()
