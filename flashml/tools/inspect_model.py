import tkinter as tk
from tkinter import ttk, font as tkFont
import inspect as py_inspect
import math

# --- Color Scheme Constants ---

# General
CURRENT_THEME = "dark"  # Can be 'light' or 'dark'

# --- Light Mode ---
LIGHT_BACKGROUND = "#F0F0F0"  # General background for root window
LIGHT_FRAME_BACKGROUND = "#F0F0F0"  # ttk.Frame background
LIGHT_LABELFRAME_BACKGROUND = "#EAEAEA"  # Content area of LabelFrame
LIGHT_LABELFRAME_LABEL_BACKGROUND = (
    LIGHT_FRAME_BACKGROUND  # Title background of LabelFrame
)
LIGHT_CANVAS_BACKGROUND = "#FFFFFF"  # Canvas where bars are drawn
LIGHT_TREEVIEW_BACKGROUND = "#FFFFFF"  # General background for Treeview
LIGHT_TREEVIEW_FIELD_BACKGROUND = "#FFFFFF"  # Background of the cells in Treeview
LIGHT_TREEVIEW_ROW_ODD = "#F0F0F0"
LIGHT_TREEVIEW_ROW_EVEN = "#FFFFFF"
LIGHT_TREEVIEW_SELECTED_BACKGROUND = "#B0D7FF"
LIGHT_TREEVIEW_SELECTED_FOREGROUND = "#000000"
LIGHT_BUTTON_BACKGROUND = "#E1E1E1"
LIGHT_BUTTON_FOREGROUND = "#000000"
LIGHT_SCROLLBAR_TROUGHCOLOR = "#E1E1E1"
LIGHT_SCROLLBAR_BACKGROUND = "#C0C0C0"  # The slider button itself

LIGHT_TEXT_PRIMARY = "#000000"
LIGHT_TEXT_SECONDARY = "gray50"
LIGHT_TEXT_HEADER = "#000000"  # For LabelFrame titles and main headers
LIGHT_TEXT_TREE_HEADING = "#000000"
LIGHT_TEXT_BUTTON = "#000000"
LIGHT_TEXT_CANVAS_LABEL = "#000000"  # Labels for legends below bars
LIGHT_TEXT_CANVAS_SEGMENT_DARK_BG = "#FFFFFF"  # Text on dark bar segments
LIGHT_TEXT_CANVAS_SEGMENT_LIGHT_BG = "#000000"  # Text on light bar segments
LIGHT_OUTLINE_COLOR = "black"

# --- Dark Mode ---
DARK_BACKGROUND = "#2E2E2E"
DARK_FRAME_BACKGROUND = "#2E2E2E"
DARK_LABELFRAME_BACKGROUND = "#2C2C2C"
DARK_LABELFRAME_LABEL_BACKGROUND = DARK_FRAME_BACKGROUND
DARK_CANVAS_BACKGROUND = (
    "#3A3A3A"  # Slightly different from frame for contrast if needed
)
DARK_TREEVIEW_BACKGROUND = "#3C3C3C"
DARK_TREEVIEW_FIELD_BACKGROUND = "#4A4A4A"
DARK_TREEVIEW_ROW_ODD = "#454545"
DARK_TREEVIEW_ROW_EVEN = "#3C3C3C"
DARK_TREEVIEW_SELECTED_BACKGROUND = "#005A9E"
DARK_TREEVIEW_SELECTED_FOREGROUND = "#FFFFFF"
DARK_BUTTON_BACKGROUND = "#555555"
DARK_BUTTON_FOREGROUND = "#E0E0E0"
DARK_SCROLLBAR_TROUGHCOLOR = "#444444"
DARK_SCROLLBAR_BACKGROUND = "#666666"

DARK_TEXT_PRIMARY = "#E0E0E0"
DARK_TEXT_SECONDARY = "gray75"
DARK_TEXT_HEADER = "#FFFFFF"
DARK_TEXT_TREE_HEADING = "#E0E0E0"
DARK_TEXT_BUTTON = "#E0E0E0"
DARK_TEXT_CANVAS_LABEL = "#E0E0E0"
DARK_TEXT_CANVAS_SEGMENT_DARK_BG = "#E0E0E0"
DARK_TEXT_CANVAS_SEGMENT_LIGHT_BG = "#121212"
DARK_OUTLINE_COLOR = "gray60"

# --- Data Visualization Colors ---
# These are kept distinct but could also be themed if desired.
# For now, they are general.
TRAINABLE_COLORS_VIZ = {"Trainable": "#e9524a", "Frozen": "#03abe3"}
DTYPE_BASE_COLORS_VIZ = [
    "#318FEE",
    "#96841A",
    "#CE36A5",
    "#C82C20",
    "#13BD13",
    "#2AAF33",
    "#F0E68C",
    "#FFA07A",
]
DEVICE_COLORS_VIZ_MAP = {
    "cpu": "#4682b4",
    "cuda:0": "#04BC04",
    "mps": "#FFD700",
    "rocm": "#bf7316",
    "other_gpu_shades": ["#00b050", "#32cd32", "#90ee90", "#98fb98", "#c1ffc1"],
    "default": "#A9A9A9",  # Darker gray for default if needed
}
# --- End Color Scheme Constants ---


def get_parameter_dtype(param):
    """Gets the dtype of a parameter."""
    return str(param.dtype)


def get_parameter_device(param):
    """Gets the device of a parameter."""
    return str(param.device)


def get_memory_size(num_params, dtype_str):
    """Estimates memory size in bytes based on num_params and dtype."""
    dtype_sizes = {
        "torch.float32": 4,
        "torch.float": 4,
        "torch.float64": 8,
        "torch.double": 8,
        "torch.float16": 2,
        "torch.half": 2,
        "torch.bfloat16": 2,
        "torch.int64": 8,
        "torch.long": 8,
        "torch.int32": 4,
        "torch.int": 4,
        "torch.int16": 2,
        "torch.short": 2,
        "torch.int8": 1,
        "torch.uint8": 1,
        "torch.bool": 1,
    }
    return num_params * dtype_sizes.get(dtype_str.lower(), 4)


def format_size(size_bytes):
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024))) if size_bytes > 0 else 0
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def draw_partition_bar(
    canvas, width, height, data_percentages, colors, label_prefix="", icon_map=None
):
    canvas.delete("all")
    current_x = 0
    bar_y0 = 5
    bar_y1 = bar_y0 + height
    text_y = bar_y1 + 15
    max_text_height = 0

    # Theme-dependent colors for drawing
    if CURRENT_THEME == "light":
        text_on_dark_segment = LIGHT_TEXT_CANVAS_SEGMENT_DARK_BG
        text_on_light_segment = LIGHT_TEXT_CANVAS_SEGMENT_LIGHT_BG
        segment_outline_color = LIGHT_OUTLINE_COLOR
        legend_text_color = LIGHT_TEXT_CANVAS_LABEL
        line_color = LIGHT_OUTLINE_COLOR
    else:  # dark
        text_on_dark_segment = DARK_TEXT_CANVAS_SEGMENT_DARK_BG
        text_on_light_segment = DARK_TEXT_CANVAS_SEGMENT_LIGHT_BG
        segment_outline_color = DARK_OUTLINE_COLOR
        legend_text_color = DARK_TEXT_CANVAS_LABEL
        line_color = DARK_OUTLINE_COLOR

    for i, (name, percentage, val_str) in enumerate(data_percentages):
        segment_width = (percentage / 100) * width
        color = colors.get(name, "#cccccc")
        canvas.create_rectangle(
            current_x,
            bar_y0,
            current_x + segment_width,
            bar_y1,
            fill=color,
            outline=segment_outline_color,
        )

        icon = icon_map.get(name, "") if icon_map else ""
        segment_text = f"{icon}{name}: {percentage:.1f}%"
        if percentage < 5:
            segment_text = f"{icon}" if icon else f"{percentage:.1f}%"

        temp_font = tkFont.Font(family="Helvetica", size=8)
        text_width = temp_font.measure(segment_text)
        text_height = temp_font.metrics("linespace")
        max_text_height = max(max_text_height, text_height)

        if segment_width > text_width + 4 and percentage > 1:
            r, g, b = canvas.winfo_rgb(color)
            brightness = (
                r / 256 * 0.299 + g / 256 * 0.2126 + b / 256 * 0.0722
            )  # Original calculation

            # Use theme-aware text colors for segments
            bar_text_color = (
                text_on_dark_segment if brightness < 140 else text_on_light_segment
            )

            canvas.create_text(
                current_x + segment_width / 2,
                bar_y0 + height / 2,
                text=segment_text,
                fill=bar_text_color,
                font=temp_font,
                anchor=tk.CENTER,
            )
        elif percentage > 0:  # Small segments, draw a line
            canvas.create_line(
                current_x + segment_width / 2,
                bar_y1,
                current_x + segment_width / 2,
                bar_y1 + 3,
                fill=line_color,
            )
        current_x += segment_width

    current_label_x = 10
    for i, (name, percentage, val_str) in enumerate(data_percentages):
        if percentage == 0:
            continue
        icon = icon_map.get(name, "") if icon_map else ""
        full_label_text = f"{icon}{name}: {val_str} ({percentage:.1f}%)"
        color = colors.get(name, "#cccccc")

        canvas.create_rectangle(  # Legend color box
            current_label_x,
            text_y - text_height / 2,
            current_label_x + 10,
            text_y + text_height / 2 - 2,
            fill=color,
            outline=segment_outline_color,
        )
        current_label_x += 15

        label_id = canvas.create_text(  # Legend text
            current_label_x,
            text_y,
            text=full_label_text,
            anchor=tk.W,
            font=("Helvetica", 9),
            fill=legend_text_color,
        )
        bbox = canvas.bbox(label_id)
        current_label_x = bbox[2] + 10
    final_canvas_height = bar_y1 + max_text_height + 20
    canvas.config(height=final_canvas_height)


def inspect_model(model):
    """Inspects a PyTorch model and displays information about its parameters and modules."""
    root = tk.Tk()
    root.title(f"Model Inspector: {model.__class__.__name__}")
    root.geometry("1200x800")

    style = ttk.Style()
    style.theme_use("clam")  # Base theme

    # --- Theme application logic ---
    global CURRENT_THEME  # Allow modification by toggle_theme

    # Define all widgets that need theme updates here, so apply_current_theme can access them
    main_frame = ttk.Frame(root, padding="10")
    general_info_frame = ttk.LabelFrame(
        main_frame, text="General Model Information", padding="10"
    )
    module_frame = ttk.LabelFrame(main_frame, text="Module Details", padding="10")
    footer_frame = ttk.Frame(main_frame, padding="5")

    # Canvases (will be configured in apply_current_theme)
    trainable_canvas = tk.Canvas(general_info_frame, height=60)
    dtype_canvas = tk.Canvas(general_info_frame, height=60)
    device_canvas = tk.Canvas(general_info_frame, height=60)

    # Treeview (will be configured in apply_current_theme)
    columns = (
        "name",
        "num_params",
        "params_perc",
        "dtypes",
        "device",
        "memory",
        "trainable",
    )
    tree = ttk.Treeview(module_frame, columns=columns, show="headings", height=15)
    vsb = ttk.Scrollbar(module_frame, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(module_frame, orient="horizontal", command=tree.xview)

    status_label = ttk.Label(
        footer_frame, text="Inspection Complete.", font=("Helvetica", 8)
    )
    close_button = ttk.Button(footer_frame, text="Close", command=root.destroy)
    theme_toggle_button = ttk.Button(
        footer_frame, text="Switch to Dark Mode", command=lambda: toggle_theme()
    )  # Placeholder text

    # Placeholder for model_name_source_frame, ensure it's a ttk.Frame if it needs styling
    # This frame is created later, so apply_current_theme needs to handle it or be called after its creation.
    # For simplicity, we'll assume its direct child labels will pick up the theme.
    # If model_name_source_frame itself needs a background, it needs to be ttk.Frame and styled.

    def apply_current_theme():
        is_light = CURRENT_THEME == "light"

        bg_color = LIGHT_BACKGROUND if is_light else DARK_BACKGROUND
        frame_bg = LIGHT_FRAME_BACKGROUND if is_light else DARK_FRAME_BACKGROUND
        labelframe_bg = (
            LIGHT_LABELFRAME_BACKGROUND if is_light else DARK_LABELFRAME_BACKGROUND
        )
        labelframe_label_bg = (
            LIGHT_LABELFRAME_LABEL_BACKGROUND
            if is_light
            else DARK_LABELFRAME_LABEL_BACKGROUND
        )
        canvas_bg = LIGHT_CANVAS_BACKGROUND if is_light else DARK_CANVAS_BACKGROUND

        tree_bg = LIGHT_TREEVIEW_BACKGROUND if is_light else DARK_TREEVIEW_BACKGROUND
        tree_field_bg = (
            LIGHT_TREEVIEW_FIELD_BACKGROUND
            if is_light
            else DARK_TREEVIEW_FIELD_BACKGROUND
        )
        tree_odd_bg = LIGHT_TREEVIEW_ROW_ODD if is_light else DARK_TREEVIEW_ROW_ODD
        tree_even_bg = LIGHT_TREEVIEW_ROW_EVEN if is_light else DARK_TREEVIEW_ROW_EVEN
        tree_selected_bg = (
            LIGHT_TREEVIEW_SELECTED_BACKGROUND
            if is_light
            else DARK_TREEVIEW_SELECTED_BACKGROUND
        )
        tree_selected_fg = (
            LIGHT_TREEVIEW_SELECTED_FOREGROUND
            if is_light
            else DARK_TREEVIEW_SELECTED_FOREGROUND
        )

        btn_bg = LIGHT_BUTTON_BACKGROUND if is_light else DARK_BUTTON_BACKGROUND
        btn_fg = LIGHT_BUTTON_FOREGROUND if is_light else DARK_BUTTON_FOREGROUND

        scrollbar_trough = (
            LIGHT_SCROLLBAR_TROUGHCOLOR if is_light else DARK_SCROLLBAR_TROUGHCOLOR
        )
        scrollbar_slider_bg = (
            LIGHT_SCROLLBAR_BACKGROUND if is_light else DARK_SCROLLBAR_BACKGROUND
        )  # slider button

        text_primary = LIGHT_TEXT_PRIMARY if is_light else DARK_TEXT_PRIMARY
        text_secondary = LIGHT_TEXT_SECONDARY if is_light else DARK_TEXT_SECONDARY
        text_header = LIGHT_TEXT_HEADER if is_light else DARK_TEXT_HEADER
        text_tree_heading = (
            LIGHT_TEXT_TREE_HEADING if is_light else DARK_TEXT_TREE_HEADING
        )
        outline_col = LIGHT_OUTLINE_COLOR if is_light else DARK_OUTLINE_COLOR

        root.configure(bg=bg_color)

        style.configure("TFrame", background=frame_bg)
        main_frame.configure(style="TFrame")
        footer_frame.configure(style="TFrame")
        # model_name_source_frame will be created later, ensure its children use themed labels

        style.configure(
            "TLabel",
            background=frame_bg,
            foreground=text_primary,
            font=("Helvetica", 10),
        )
        style.configure(
            "Header.TLabel",
            background=frame_bg,
            foreground=text_header,
            font=("Helvetica", 12, "bold"),
        )
        style.configure(
            "Gray.TLabel",
            background=frame_bg,
            foreground=text_secondary,
            font=("Helvetica", 10),
        )
        status_label.configure(style="TLabel")  # Re-apply style

        style.configure(
            "TLabelframe", background=labelframe_bg, bordercolor=outline_col
        )
        style.configure(
            "TLabelframe.Label",
            background=labelframe_label_bg,
            foreground=text_header,
            font=("Helvetica", 10, "bold"),
        )
        general_info_frame.configure(style="TLabelframe")
        module_frame.configure(style="TLabelframe")

        trainable_canvas.configure(bg=canvas_bg)
        dtype_canvas.configure(bg=canvas_bg)
        device_canvas.configure(bg=canvas_bg)

        style.configure(
            "Treeview.Heading",
            background=frame_bg,
            foreground=text_tree_heading,
            font=("Helvetica", 10, "bold"),
        )
        style.configure(
            "Treeview",
            background=tree_bg,
            foreground=text_primary,
            fieldbackground=tree_field_bg,
            font=("Helvetica", 9),
            rowheight=28,
        )
        style.map(
            "Treeview",
            background=[("selected", tree_selected_bg)],
            foreground=[("selected", tree_selected_fg)],
        )

        tree.tag_configure("oddrow", background=tree_odd_bg, foreground=text_primary)
        tree.tag_configure("evenrow", background=tree_even_bg, foreground=text_primary)

        style.configure(
            "TButton", background=btn_bg, foreground=btn_fg, font=("Helvetica", 9)
        )
        style.map(
            "TButton",
            background=[("active", LIGHT_BUTTON_BACKGROUND if is_light else "#6A6A6A")],
        )
        close_button.configure(style="TButton")
        theme_toggle_button.configure(
            style="TButton",
            text="Switch to Dark Mode" if is_light else "Switch to Light Mode",
        )

        style.configure(
            "Vertical.TScrollbar",
            troughcolor=scrollbar_trough,
            background=scrollbar_slider_bg,
            bordercolor=frame_bg,
            arrowcolor=text_primary,
        )
        style.configure(
            "Horizontal.TScrollbar",
            troughcolor=scrollbar_trough,
            background=scrollbar_slider_bg,
            bordercolor=frame_bg,
            arrowcolor=text_primary,
        )
        vsb.configure(style="Vertical.TScrollbar")
        hsb.configure(style="Horizontal.TScrollbar")

        # Re-draw canvases as their internal colors are theme-dependent
        for canvas_widget in [trainable_canvas, dtype_canvas, device_canvas]:
            if canvas_widget.winfo_ismapped():  # Check if widget exists and is mapped
                canvas_widget.event_generate("<Configure>")

    def toggle_theme():
        global CURRENT_THEME
        CURRENT_THEME = "dark" if CURRENT_THEME == "light" else "light"
        apply_current_theme()

    # --- End Theme application logic ---

    main_frame.pack(expand=True, fill=tk.BOTH)
    general_info_frame.pack(pady=10, fill=tk.X)
    general_info_frame.columnconfigure(1, weight=1)
    general_info_frame.columnconfigure(3, weight=1)

    total_params = 0
    param_dtypes_counts = {}
    param_devices_counts = {}
    param_devices_memory_bytes = {}
    total_memory_bytes = 0
    trainable_params = 0

    module_details = []
    all_params_list = []
    for _, param in model.named_parameters():
        all_params_list.append(param)
        num_p = param.numel()
        total_params += num_p

        dtype = get_parameter_dtype(param)
        param_dtypes_counts[dtype] = param_dtypes_counts.get(dtype, 0) + num_p

        device = get_parameter_device(param)
        param_devices_counts[device] = param_devices_counts.get(device, 0) + num_p

        param_mem_size = get_memory_size(num_p, dtype)
        total_memory_bytes += param_mem_size
        param_devices_memory_bytes[device] = (
            param_devices_memory_bytes.get(device, 0) + param_mem_size
        )

        if param.requires_grad:
            trainable_params += num_p
    non_trainable_params = total_params - trainable_params

    for name, module in model.named_modules():
        module_num_params, module_memory_bytes, module_param_trainable_count = 0, 0, 0
        module_dtypes, module_devices = {}, {}
        module_has_params = False
        current_params_list = list(module.parameters(recurse=False))

        if not current_params_list and not list(module.children()):
            module_details.append(
                {
                    "name": name if name else model.__class__.__name__,
                    "num_params": 0,
                    "params_percentage": 0.0,
                    "dtypes": "N/A",
                    "device": "N/A",
                    "memory": format_size(0),
                    "trainable_status": "N/A",
                }
            )
            continue

        for param in current_params_list:
            module_has_params = True
            num_p = param.numel()
            module_num_params += num_p
            dtype = get_parameter_dtype(param)
            module_dtypes[dtype] = module_dtypes.get(dtype, 0) + num_p
            device = get_parameter_device(param)
            module_devices[device] = module_devices.get(device, 0) + num_p
            module_memory_bytes += get_memory_size(num_p, dtype)
            if param.requires_grad:
                module_param_trainable_count += num_p

        if module_has_params:
            module_dtype_str = (
                ", ".join(
                    [
                        f"{d.replace('torch.', '')} ({count * 100 / module_num_params:.0f}%)"
                        for d, count in module_dtypes.items()
                    ]
                )
                if module_num_params > 0
                else "N/A"
            )
            module_device_str = (
                ", ".join(
                    [
                        f"{d} ({count * 100 / module_num_params:.0f}%)"
                        for d, count in module_devices.items()
                    ]
                )
                if module_num_params > 0
                else "N/A"
            )
            trainable_status = "N/A"
            if module_num_params > 0:
                if module_param_trainable_count == module_num_params:
                    trainable_status = "Yes"
                elif module_param_trainable_count == 0:
                    trainable_status = "No"
                else:
                    trainable_status = "Partial"

            module_details.append(
                {
                    "name": name if name else model.__class__.__name__,
                    "num_params": module_num_params,
                    "params_percentage": (module_num_params / total_params * 100)
                    if total_params > 0
                    else 0.0,
                    "dtypes": module_dtype_str,
                    "device": module_device_str,
                    "memory": format_size(module_memory_bytes),
                    "trainable_status": trainable_status,
                }
            )

    row_idx = 0
    ttk.Label(general_info_frame, text="Model Class:", style="Header.TLabel").grid(
        row=row_idx, column=0, sticky="w", padx=5, pady=3
    )
    model_class_name_str = model.__class__.__name__
    model_source_file_str = ""
    try:
        model_file = py_inspect.getfile(model.__class__)
        model_source_file_str = f" ({model_file})"
    except (TypeError, OSError):
        model_source_file_str = " (Source not found)"

    model_name_source_frame = ttk.Frame(general_info_frame)  # This is a ttk.Frame
    model_name_source_frame.grid(
        row=row_idx, column=1, columnspan=3, sticky="w", padx=5, pady=3
    )
    ttk.Label(model_name_source_frame, text=model_class_name_str).pack(
        side=tk.LEFT
    )  # Will pick up TLabel style
    ttk.Label(
        model_name_source_frame, text=model_source_file_str, style="Gray.TLabel"
    ).pack(side=tk.LEFT, padx=(5, 0))
    model_name_source_frame.configure(
        style="TFrame"
    )  # Ensure it gets the theme background
    row_idx += 1

    ttk.Label(general_info_frame, text="Total Parameters:", style="Header.TLabel").grid(
        row=row_idx, column=0, sticky="w", padx=5, pady=3
    )
    ttk.Label(general_info_frame, text=f"{total_params:,}").grid(
        row=row_idx, column=1, sticky="w", padx=5, pady=3
    )
    ttk.Label(
        general_info_frame, text="Estimated Total Size:", style="Header.TLabel"
    ).grid(row=row_idx, column=2, sticky="w", padx=(20, 5), pady=3)
    ttk.Label(general_info_frame, text=format_size(total_memory_bytes)).grid(
        row=row_idx, column=3, sticky="w", padx=5, pady=3
    )
    row_idx += 1

    ttk.Label(general_info_frame, text="Parameter Status:", style="Header.TLabel").grid(
        row=row_idx, column=0, sticky="nw", padx=5, pady=3
    )
    trainable_canvas.grid(
        row=row_idx, column=1, columnspan=3, sticky="ew", padx=5, pady=3
    )
    row_idx += 1
    if total_params > 0:
        trainable_data = [
            (
                "Trainable",
                (trainable_params / total_params) * 100,
                f"{trainable_params:,}",
            ),
            (
                "Frozen",
                (non_trainable_params / total_params) * 100,
                f"{non_trainable_params:,}",
            ),
        ]
        trainable_icons = {"Trainable": "🔥     ", "Frozen": "❄️"}
        trainable_canvas.bind(
            "<Configure>",
            lambda e,
            c=trainable_canvas,
            d=trainable_data,
            cl=TRAINABLE_COLORS_VIZ,
            i=trainable_icons: draw_partition_bar(c, e.width, 20, d, cl, icon_map=i),
        )

    ttk.Label(general_info_frame, text="Parameter Dtypes:", style="Header.TLabel").grid(
        row=row_idx, column=0, sticky="nw", padx=5, pady=3
    )
    dtype_canvas.grid(row=row_idx, column=1, columnspan=3, sticky="ew", padx=5, pady=3)
    row_idx += 1
    dtype_data = []
    if total_params > 0:
        for dt, count in sorted(
            param_dtypes_counts.items(), key=lambda item: item[1], reverse=True
        ):
            dtype_data.append(
                (dt.replace("torch.", ""), (count / total_params) * 100, f"{count:,}")
            )
    dtype_colors = {
        dt_data_item[0]: color
        for dt_data_item, color in zip(
            dtype_data,
            DTYPE_BASE_COLORS_VIZ * (len(dtype_data) // len(DTYPE_BASE_COLORS_VIZ) + 1),
        )
    }
    dtype_canvas.bind(
        "<Configure>",
        lambda e, c=dtype_canvas, d=dtype_data, cl=dtype_colors: draw_partition_bar(
            c, e.width, 20, d, cl
        ),
    )

    ttk.Label(
        general_info_frame, text="Parameter Devices:", style="Header.TLabel"
    ).grid(row=row_idx, column=0, sticky="nw", padx=5, pady=3)
    device_canvas.grid(row=row_idx, column=1, columnspan=3, sticky="ew", padx=5, pady=3)
    row_idx += 1
    device_data = []
    if total_params > 0:
        sorted_devices_by_count = sorted(
            param_devices_counts.items(), key=lambda item: item[1], reverse=True
        )
        for dev, count in sorted_devices_by_count:
            mem_on_device = param_devices_memory_bytes.get(dev, 0)
            formatted_mem_str = format_size(mem_on_device)
            percentage_of_params_on_device = (
                (count / total_params) * 100 if total_params > 0 else 0
            )
            device_data.append((dev, percentage_of_params_on_device, formatted_mem_str))

    device_colors = {}
    cuda_idx = 0
    for dev_name, _, _ in device_data:
        if dev_name == "cpu":
            device_colors[dev_name] = DEVICE_COLORS_VIZ_MAP["cpu"]
        elif "cuda" in dev_name:
            if dev_name == "cuda:0":
                device_colors[dev_name] = DEVICE_COLORS_VIZ_MAP["cuda:0"]
            else:
                device_colors[dev_name] = DEVICE_COLORS_VIZ_MAP["other_gpu_shades"][
                    cuda_idx % len(DEVICE_COLORS_VIZ_MAP["other_gpu_shades"])
                ]
                cuda_idx += 1
        elif "mps" in dev_name:
            device_colors[dev_name] = DEVICE_COLORS_VIZ_MAP["mps"]
        elif "rocm" in dev_name:
            device_colors[dev_name] = DEVICE_COLORS_VIZ_MAP["rocm"]
        else:
            device_colors[dev_name] = DEVICE_COLORS_VIZ_MAP["default"]
    device_canvas.bind(
        "<Configure>",
        lambda e, c=device_canvas, d=device_data, cl=device_colors: draw_partition_bar(
            c, e.width, 20, d, cl
        ),
    )

    module_frame.pack(expand=True, fill=tk.BOTH, pady=10)
    col_widths = {
        "name": 280,
        "num_params": 100,
        "params_perc": 80,
        "dtypes": 180,
        "device": 150,
        "memory": 100,
        "trainable": 80,
    }
    col_anchors = {
        "name": tk.W,
        "num_params": tk.E,
        "params_perc": tk.E,
        "dtypes": tk.W,
        "device": tk.W,
        "memory": tk.E,
        "trainable": tk.CENTER,
    }
    col_headings = {
        "name": "Module Name",
        "num_params": "Params",
        "params_perc": "% Total",
        "dtypes": "Dtypes (% module)",
        "device": "Device(s) (% module)",
        "memory": "Est. Size",
        "trainable": "Trainable?",
    }

    for col in columns:
        tree.heading(col, text=col_headings[col])
        tree.column(
            col,
            width=col_widths[col],
            minwidth=max(50, col_widths[col] - 40),
            anchor=col_anchors[col],
            stretch=tk.YES if col == "name" else tk.NO,
        )

    trainable_emojis = {
        "Yes": "✔️",
        "No": "❌",
        "Partial": "◑",
        "N/A": "",
    }  # Using ◑ for partial
    for i, md in enumerate(module_details):
        tag = "oddrow" if i % 2 == 0 else "evenrow"
        trainable_display = trainable_emojis.get(
            md["trainable_status"], md["trainable_status"]
        )
        tree.insert(
            "",
            tk.END,
            values=(
                md["name"],
                f"{md['num_params']:,}"
                if md["num_params"] > 0
                else ("0" if md["num_params"] == 0 else "N/A"),
                f"{md['params_percentage']:.2f}%" if md["num_params"] > 0 else "----",
                md["dtypes"],
                md["device"],
                md["memory"],
                trainable_display,
            ),
            tags=(tag,),
        )

    vsb.pack(side="right", fill="y")
    tree.configure(yscrollcommand=vsb.set)
    hsb.pack(side="bottom", fill="x")
    tree.configure(xscrollcommand=hsb.set)
    tree.pack(expand=True, fill=tk.BOTH)

    footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
    theme_toggle_button.pack(side=tk.LEFT, padx=5)  # Add before other footer items
    status_label.pack(side=tk.LEFT)
    close_button.pack(side=tk.RIGHT)

    apply_current_theme()  # Apply initial theme after all widgets are defined

    root.update_idletasks()  # Ensure layout is calculated
    # Initial draw for canvases if they are already mapped.
    # apply_current_theme now handles this.
    # for canvas_widget in [trainable_canvas, dtype_canvas, device_canvas]:
    #    canvas_widget.event_generate("<Configure>")

    root.mainloop()
