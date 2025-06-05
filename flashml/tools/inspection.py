import math
import random

### Global state for uint8 view ###
VIEW_UINT8_AS_INT4 = True
###################################

DARK_BACKGROUND = "#2E2E2E"
DARK_FRAME_BACKGROUND = "#2E2E2E"
DARK_LABELFRAME_BACKGROUND = "#2C2C2C"
DARK_LABELFRAME_LABEL_BACKGROUND = DARK_FRAME_BACKGROUND
DARK_CANVAS_BACKGROUND = "#3A3A3A"
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
DARK_TEXT_CANVAS_SEGMENT_DARK_BG = (
    "#E0E0E0"  # Text on dark bar segments (e.g. white text on dark blue bar)
)
DARK_TEXT_CANVAS_SEGMENT_LIGHT_BG = (
    "#121212"  # Text on light bar segments (e.g. black text on yellow bar)
)
DARK_OUTLINE_COLOR = "gray60"

# --- Data Visualization Colors ---
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
    "default": "#A9A9A9",
}

HAS_UINT8_PARAMS = False


def get_parameter_dtype(param):
    original_dtype_str = str(param.dtype)
    if VIEW_UINT8_AS_INT4 and original_dtype_str == "torch.uint8":
        return "torch.int4"
    return original_dtype_str


def get_parameter_device(param):
    return str(param.device)


def get_memory_size(num_params, dtype_str):
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
        "torch.int4": 0.5,
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


def to_bold_digits(text_string):
    """Converts numeric characters in a string to their Unicode bold equivalents."""
    bold_map = {
        "0": "𝟬",
        "1": "𝟭",
        "2": "𝟮",
        "3": "𝟯",
        "4": "𝟰",
        "5": "𝟱",
        "6": "𝟲",
        "7": "𝟳",
        "8": "𝟴",
        "9": "𝟵",
        # Add comma if you want it bold, or leave it to pass through
        # ',': '､', # Example: bold comma (Ideographic Comma) - might not be ideal
    }
    # Apply bold mapping only to digits, keep others (like comma) as is
    # unless explicitly mapped.
    return "".join(bold_map.get(char, char) for char in str(text_string))


def draw_partition_bar(
    canvas, width, height, data_percentages, colors, label_prefix="", icon_map=None
):
    import tkinter as tk
    from tkinter import ttk as tkFont

    canvas.delete("all")
    current_x = 0
    bar_y0 = 5
    bar_y1 = bar_y0 + height
    text_y = bar_y1 + 15
    max_text_height = 0

    text_on_dark_segment = DARK_TEXT_CANVAS_SEGMENT_DARK_BG
    text_on_light_segment = DARK_TEXT_CANVAS_SEGMENT_LIGHT_BG
    segment_outline_color = DARK_OUTLINE_COLOR
    legend_text_color = DARK_TEXT_CANVAS_LABEL
    line_color = DARK_OUTLINE_COLOR

    if (
        not data_percentages
    ):  # If no data, ensure canvas height is minimal but consistent
        canvas.config(height=bar_y1 + 5)  # bar height + padding
        return

    for i, (name, percentage, val_str) in enumerate(data_percentages):
        segment_width = (percentage / 100) * width
        # Ensure color is fetched correctly, using a dark default if name not in colors map
        color = colors.get(
            name, DARK_TREEVIEW_FIELD_BACKGROUND
        )  # Use a theme color for default

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
        if percentage < 5 and percentage > 0:
            segment_text = f"{icon}" if icon else f"{percentage:.1f}%"
        elif percentage == 0:
            current_x += segment_width
            continue

        temp_font = tkFont.Font(family="Helvetica", size=8)
        text_width = temp_font.measure(segment_text)
        text_height = temp_font.metrics("linespace")
        max_text_height = max(max_text_height, text_height)

        if segment_width > text_width + 4 and percentage > 1:
            try:
                r, g, b = canvas.winfo_rgb(color)
                r_norm, g_norm, b_norm = r / 256, g / 256, b / 256
                brightness = 0.2126 * r_norm + 0.7152 * g_norm + 0.0722 * b_norm
                bar_text_color = (
                    text_on_dark_segment if brightness < 100 else text_on_light_segment
                )
            except tk.TclError:
                bar_text_color = text_on_light_segment
            canvas.create_text(
                current_x + segment_width / 2,
                bar_y0 + height / 2,
                text=segment_text,
                fill=bar_text_color,
                font=temp_font,
                anchor=tk.CENTER,
            )
        elif percentage > 0:
            canvas.create_line(
                current_x + segment_width / 2,
                bar_y1,
                current_x + segment_width / 2,
                bar_y1 + 3,
                fill=line_color,
            )
        current_x += segment_width

    current_label_x = 10
    legend_items_on_canvas = 0
    for i, (name, percentage, val_str) in enumerate(data_percentages):
        if percentage == 0:
            continue
        legend_items_on_canvas += 1
        icon = icon_map.get(name, "") if icon_map else ""
        full_label_text = f"{icon}{name}: {val_str} ({percentage:.1f}%)"
        color_for_legend_box = colors.get(name, DARK_TREEVIEW_FIELD_BACKGROUND)

        canvas.create_rectangle(
            current_label_x,
            text_y - text_height / 2 if text_height > 0 else text_y - 5,
            current_label_x + 10,
            text_y + text_height / 2 - 2 if text_height > 0 else text_y + 5,
            fill=color_for_legend_box,
            outline=segment_outline_color,
        )
        current_label_x += 15
        label_id = canvas.create_text(
            current_label_x,
            text_y,
            text=full_label_text,
            anchor=tk.W,
            font=("Helvetica", 9),
            fill=legend_text_color,
        )
        bbox = canvas.bbox(label_id)
        current_label_x = (
            bbox[2] + 10
            if bbox
            else current_label_x + temp_font.measure(full_label_text) + 10
        )

    final_canvas_height = (
        (bar_y1 + max_text_height + 20) if legend_items_on_canvas > 0 else (bar_y1 + 5)
    )
    canvas.config(height=final_canvas_height)


def apply_dark_theme(
    root,
    style,
    main_frame,
    general_info_frame,
    module_frame,
    footer_frame,
    trainable_canvas,
    dtype_canvas,
    device_canvas,
    tree,
    vsb,
    hsb,
    status_label,
    close_button,
    model_name_source_frame,
    model_class_name_label,
    model_source_file_label,
    total_params_val_label,
    est_total_size_val_label,
    uint8_toggle_button_widget,
):
    root.configure(bg=DARK_BACKGROUND)
    style.configure("TFrame", background=DARK_FRAME_BACKGROUND)
    main_frame.configure(style="TFrame")
    footer_frame.configure(style="TFrame")
    model_name_source_frame.configure(style="TFrame")

    style.configure(
        "TLabel",
        background=DARK_FRAME_BACKGROUND,
        foreground=DARK_TEXT_PRIMARY,
        font=("Helvetica", 10),
    )
    style.configure(
        "Header.TLabel",
        background=DARK_FRAME_BACKGROUND,
        foreground=DARK_TEXT_HEADER,
        font=("Helvetica", 12, "bold"),
    )
    style.configure(
        "Gray.TLabel",
        background=DARK_FRAME_BACKGROUND,
        foreground=DARK_TEXT_SECONDARY,
        font=("Helvetica", 10),
    )

    status_label.configure(style="TLabel")
    model_class_name_label.configure(style="TLabel")
    model_source_file_label.configure(style="Gray.TLabel")
    total_params_val_label.configure(style="TLabel")
    est_total_size_val_label.configure(style="TLabel")

    style.configure(
        "TLabelframe",
        background=DARK_LABELFRAME_BACKGROUND,
        bordercolor=DARK_OUTLINE_COLOR,
    )
    style.configure(
        "TLabelframe.Label",
        background=DARK_LABELFRAME_LABEL_BACKGROUND,
        foreground=DARK_TEXT_HEADER,
        font=("Helvetica", 10, "bold"),
    )
    general_info_frame.configure(style="TLabelframe")
    module_frame.configure(style="TLabelframe")

    trainable_canvas.configure(bg=DARK_CANVAS_BACKGROUND)
    dtype_canvas.configure(bg=DARK_CANVAS_BACKGROUND)
    device_canvas.configure(bg=DARK_CANVAS_BACKGROUND)

    style.configure(
        "Treeview.Heading",
        background=DARK_FRAME_BACKGROUND,
        foreground=DARK_TEXT_TREE_HEADING,
        font=("Helvetica", 10, "bold"),
    )
    style.configure(
        "Treeview",
        background=DARK_TREEVIEW_BACKGROUND,
        foreground=DARK_TEXT_PRIMARY,
        fieldbackground=DARK_TREEVIEW_FIELD_BACKGROUND,
        font=("Helvetica", 9),
        rowheight=28,
    )
    style.map(
        "Treeview",
        background=[("selected", DARK_TREEVIEW_SELECTED_BACKGROUND)],
        foreground=[("selected", DARK_TREEVIEW_SELECTED_FOREGROUND)],
    )
    tree.tag_configure(
        "oddrow", background=DARK_TREEVIEW_ROW_ODD, foreground=DARK_TEXT_PRIMARY
    )
    tree.tag_configure(
        "evenrow", background=DARK_TREEVIEW_ROW_EVEN, foreground=DARK_TEXT_PRIMARY
    )

    style.configure(
        "TButton",
        background=DARK_BUTTON_BACKGROUND,
        foreground=DARK_BUTTON_FOREGROUND,
        font=("Helvetica", 9),
    )
    style.map("TButton", background=[("active", "#6A6A6A")])
    close_button.configure(style="TButton")
    if uint8_toggle_button_widget:
        uint8_toggle_button_widget.configure(style="TButton")

    style.configure(
        "Vertical.TScrollbar",
        troughcolor=DARK_SCROLLBAR_TROUGHCOLOR,
        background=DARK_SCROLLBAR_BACKGROUND,
        bordercolor=DARK_FRAME_BACKGROUND,
        arrowcolor=DARK_TEXT_PRIMARY,
    )
    style.configure(
        "Horizontal.TScrollbar",
        troughcolor=DARK_SCROLLBAR_TROUGHCOLOR,
        background=DARK_SCROLLBAR_BACKGROUND,
        bordercolor=DARK_FRAME_BACKGROUND,
        arrowcolor=DARK_TEXT_PRIMARY,
    )
    vsb.configure(style="Vertical.TScrollbar")
    hsb.configure(style="Horizontal.TScrollbar")


def _repopulate_ui_with_data(
    model,
    root,
    trainable_canvas,
    dtype_canvas,
    device_canvas,
    tree,
    model_class_name_label,
    model_source_file_label,
    total_params_val_label,
    est_total_size_val_label,
    status_label,
    uint8_toggle_button_widget,
):
    import tkinter as tk

    for item in tree.get_children():
        tree.delete(item)
    # It's important to ensure canvases are cleared before attempting to redraw.
    trainable_canvas.delete("all")
    dtype_canvas.delete("all")
    device_canvas.delete("all")

    total_params = 0
    param_dtypes_counts = {}
    param_devices_counts = {}
    param_devices_memory_bytes = {}
    total_memory_bytes = 0
    trainable_params = 0
    module_details = []
    all_params_list = list(model.named_parameters())

    for _, param in all_params_list:
        original_num_p = param.numel()
        original_dtype_str = str(param.dtype)
        display_dtype_str = get_parameter_dtype(param)
        display_num_p = (
            original_num_p * 2
            if VIEW_UINT8_AS_INT4 and original_dtype_str == "torch.uint8"
            else original_num_p
        )

        total_params += display_num_p
        param_dtypes_counts[display_dtype_str] = (
            param_dtypes_counts.get(display_dtype_str, 0) + display_num_p
        )
        device = get_parameter_device(param)
        param_devices_counts[device] = (
            param_devices_counts.get(device, 0) + display_num_p
        )
        param_mem_size = get_memory_size(display_num_p, display_dtype_str)
        total_memory_bytes += param_mem_size
        param_devices_memory_bytes[device] = (
            param_devices_memory_bytes.get(device, 0) + param_mem_size
        )
        if param.requires_grad:
            trainable_params += display_num_p
    non_trainable_params = total_params - trainable_params

    for name, module in model.named_modules():
        module_num_params, module_memory_bytes, module_param_trainable_count = 0, 0, 0
        module_dtypes, module_devices = {}, {}
        module_param_shapes = []
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
                    "shape_str_internal": "N/A",
                }
            )
            continue

        for param in current_params_list:
            module_has_params = True
            original_num_p_module = param.numel()
            original_dtype_str_module = str(param.dtype)
            try:
                module_param_shapes.append(str(list(param.shape)))
            except Exception:
                module_param_shapes.append("Error")

            display_dtype_str_module = get_parameter_dtype(param)
            display_num_p_module = (
                original_num_p_module * 2
                if VIEW_UINT8_AS_INT4 and original_dtype_str_module == "torch.uint8"
                else original_num_p_module
            )

            module_num_params += display_num_p_module
            module_dtypes[display_dtype_str_module] = (
                module_dtypes.get(display_dtype_str_module, 0) + display_num_p_module
            )
            device = get_parameter_device(param)
            module_devices[device] = (
                module_devices.get(device, 0) + display_num_p_module
            )
            module_memory_bytes += get_memory_size(
                display_num_p_module, display_dtype_str_module
            )
            if param.requires_grad:
                module_param_trainable_count += display_num_p_module

        module_shape_str = (
            ", ".join(module_param_shapes) if module_param_shapes else "N/A"
        )

        if module_has_params or name == "" or list(module.children()):
            module_dtype_str = (
                ", ".join(
                    [
                        f"{d.replace('torch.', '')} ({count * 100 / module_num_params:.0f}%)"
                        for d, count in module_dtypes.items()
                    ]
                )
                if module_num_params > 0
                else ("N/A" if module_has_params else "Container")
            )
            module_device_str = (
                ", ".join(
                    [
                        f"{d} ({count * 100 / module_num_params:.0f}%)"
                        for d, count in module_devices.items()
                    ]
                )
                if module_num_params > 0
                else ("N/A" if module_has_params else "Container")
            )
            trainable_status = "N/A"
            if module_num_params > 0:
                trainable_status = (
                    "Yes"
                    if module_param_trainable_count == module_num_params
                    else ("No" if module_param_trainable_count == 0 else "Partial")
                )
            elif not module_has_params and list(module.children()):
                trainable_status = "-"
                module_shape_str = "-"
            current_module_percentage = (
                (module_num_params / total_params * 100) if total_params > 0 else 0.0
            )
            module_details.append(
                {
                    "name": name if name else model.__class__.__name__,
                    "num_params": module_num_params,
                    "params_percentage": current_module_percentage,
                    "dtypes": module_dtype_str,
                    "device": module_device_str,
                    "memory": format_size(module_memory_bytes),
                    "trainable_status": trainable_status,
                    "shape_str_internal": module_shape_str,
                }
            )

    total_params_val_label.config(text=f"{total_params:,}")
    est_total_size_val_label.config(text=format_size(total_memory_bytes))

    trainable_emojis = {"Yes": "✔️", "No": "❌", "Partial": "◑", "N/A": "", "-": ""}
    for i, md in enumerate(module_details):
        tag = "oddrow" if i % 2 == 0 else "evenrow"
        trainable_display = trainable_emojis.get(
            md["trainable_status"], md["trainable_status"]
        )
        params_shape_display, params_perc_display = "-", "----"

        if md["trainable_status"] == "-":
            params_perc_display = "-"
        elif md["num_params"] > 0:
            bold_num_p_str = to_bold_digits(f"{md['num_params']:,}")
            shape_s = md["shape_str_internal"]
            params_shape_display = (
                f"{bold_num_p_str} = {shape_s}"
                if shape_s and shape_s != "N/A"
                else f"{bold_num_p_str} = (shape N/A)"
            )
            params_perc_display = (
                f"{md['params_percentage']:.2f}%" if total_params > 0 else "0.00%"
            )
        else:  # 0 params, not container
            params_shape_display = f"{to_bold_digits('0')} = (N/A)"
            params_perc_display = (
                "0.00%"
                if total_params > 0 and md["trainable_status"] != "-"
                else ("----" if md["trainable_status"] != "-" else "-")
            )
        if md["trainable_status"] != "-" and md["num_params"] == 0:
            params_perc_display = "----"
        tree.insert(
            "",
            tk.END,
            values=(
                md["name"],
                params_shape_display,
                params_perc_display,
                md["dtypes"],
                md["device"],
                md["memory"],
                trainable_display,
            ),
            tags=(tag,),
        )

    # --- Trainable Bar ---
    trainable_data = []
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
    trainable_icons = {"Trainable": "🔥 ", "Frozen": "❄️ "}
    trainable_canvas.unbind("<Configure>")
    trainable_canvas.bind(
        "<Configure>",
        lambda e,
        c=trainable_canvas,
        d=trainable_data,
        cl=TRAINABLE_COLORS_VIZ,
        i=trainable_icons: draw_partition_bar(c, e.width, 20, d, cl, icon_map=i),
    )
    if trainable_canvas.winfo_ismapped():
        trainable_canvas.event_generate("<Configure>")
    elif not trainable_data:
        draw_partition_bar(
            trainable_canvas,
            trainable_canvas.winfo_width()
            if trainable_canvas.winfo_width() > 1
            else 300,
            20,
            trainable_data,
            TRAINABLE_COLORS_VIZ,
            icon_map=trainable_icons,
        )

    # --- Dtype Bar ---
    dtype_data = []
    if total_params > 0:
        for dt, count in sorted(
            param_dtypes_counts.items(), key=lambda item: item[1], reverse=True
        ):
            dtype_data.append(
                (dt.replace("torch.", ""), (count / total_params) * 100, f"{count:,}")
            )
    dtype_colors = {
        item[0]: color
        for item, color in zip(
            dtype_data,
            DTYPE_BASE_COLORS_VIZ * (len(dtype_data) // len(DTYPE_BASE_COLORS_VIZ) + 1),
        )
    }
    dtype_canvas.unbind("<Configure>")
    dtype_canvas.bind(
        "<Configure>",
        lambda e, c=dtype_canvas, d=dtype_data, cl=dtype_colors: draw_partition_bar(
            c, e.width, 20, d, cl
        ),
    )
    if dtype_canvas.winfo_ismapped():
        dtype_canvas.event_generate("<Configure>")
    elif not dtype_data:
        draw_partition_bar(
            dtype_canvas,
            dtype_canvas.winfo_width() if dtype_canvas.winfo_width() > 1 else 300,
            20,
            dtype_data,
            dtype_colors,
        )

    # --- Device Bar ---
    device_data = []
    if total_params > 0:
        for dev, count in sorted(
            param_devices_counts.items(), key=lambda item: item[1], reverse=True
        ):
            device_data.append(
                (
                    dev,
                    (count / total_params) * 100,
                    format_size(param_devices_memory_bytes.get(dev, 0)),
                )
            )
    device_colors = {}
    cuda_idx = 0
    for dev_name, _, _ in device_data:
        if dev_name == "cpu":
            device_colors[dev_name] = DEVICE_COLORS_VIZ_MAP["cpu"]
        elif "cuda" in dev_name:
            device_colors[dev_name] = DEVICE_COLORS_VIZ_MAP.get(
                dev_name,
                DEVICE_COLORS_VIZ_MAP["other_gpu_shades"][
                    cuda_idx % len(DEVICE_COLORS_VIZ_MAP["other_gpu_shades"])
                ],
            )
            if dev_name not in DEVICE_COLORS_VIZ_MAP:
                cuda_idx += 1
        elif "mps" in dev_name:
            device_colors[dev_name] = DEVICE_COLORS_VIZ_MAP["mps"]
        elif "rocm" in dev_name:
            device_colors[dev_name] = DEVICE_COLORS_VIZ_MAP["rocm"]
        else:
            device_colors[dev_name] = DEVICE_COLORS_VIZ_MAP["default"]
    device_canvas.unbind("<Configure>")
    device_canvas.bind(
        "<Configure>",
        lambda e, c=device_canvas, d=device_data, cl=device_colors: draw_partition_bar(
            c, e.width, 20, d, cl
        ),
    )
    if device_canvas.winfo_ismapped():
        device_canvas.event_generate("<Configure>")
    elif not device_data:
        draw_partition_bar(
            device_canvas,
            device_canvas.winfo_width() if device_canvas.winfo_width() > 1 else 300,
            20,
            device_data,
            device_colors,
        )

    if HAS_UINT8_PARAMS and uint8_toggle_button_widget:
        btn_text = "uint8 as 2Xint4" if VIEW_UINT8_AS_INT4 else "uint8 quantization"
        uint8_toggle_button_widget.config(text=btn_text)

    status_label.config(text="Inspection data refreshed.")
    root.update_idletasks()


def inspect_model(model):
    import inspect as py_inspect
    import tkinter as tk
    from tkinter import ttk

    global VIEW_UINT8_AS_INT4, HAS_UINT8_PARAMS
    HAS_UINT8_PARAMS = any(
        str(param.dtype) == "torch.uint8" for _, param in model.named_parameters()
    )

    root = tk.Tk()
    root.title(f"Model Inspector: {model.__class__.__name__}")
    root.geometry("1250x850")  # Slightly increased height for button

    style = ttk.Style()
    style.theme_use("clam")

    main_frame = ttk.Frame(root, padding="10")
    general_info_frame = ttk.LabelFrame(main_frame)
    module_frame = ttk.LabelFrame(main_frame)
    footer_frame = ttk.Frame(main_frame, padding="5")

    trainable_canvas = tk.Canvas(
        general_info_frame, height=10, highlightthickness=0
    )  # Min height
    dtype_canvas = tk.Canvas(
        general_info_frame, height=10, highlightthickness=0
    )  # Min height
    device_canvas = tk.Canvas(
        general_info_frame, height=10, highlightthickness=0
    )  # Min height

    columns = (
        "name",
        "params_shape",
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
        footer_frame, text="Initializing...", font=("Helvetica", 8)
    )
    close_button = ttk.Button(footer_frame, text="Close", command=root.destroy)
    uint8_toggle_button_widget = None

    model_name_source_frame = ttk.Frame(general_info_frame)
    model_class_name_label = ttk.Label(model_name_source_frame, text="")
    model_source_file_label = ttk.Label(model_name_source_frame, text="")
    model_class_name_label.pack(side=tk.LEFT)
    model_source_file_label.pack(side=tk.LEFT, padx=(5, 0))

    try:
        model_class_name_str = model.__class__.__name__
    except:
        model_class_name_str = "N/A"
    try:
        model_source_file_str = f" ({py_inspect.getfile(model.__class__)})"
    except (TypeError, OSError):
        model_source_file_str = " (Source not found)"
    model_class_name_label.config(text=model_class_name_str)
    model_source_file_label.config(text=model_source_file_str)

    total_params_val_label = ttk.Label(general_info_frame, text="")
    est_total_size_val_label = ttk.Label(general_info_frame, text="")

    main_frame.pack(expand=True, fill=tk.BOTH)
    general_info_frame.pack(pady=10, fill=tk.X)
    for i in range(4):
        general_info_frame.columnconfigure(
            i, weight=1 if i > 0 else 0
        )  # Col 0 for labels, 1-3 for content

    current_gi_row = 0
    ttk.Label(general_info_frame, text="Model Class:", style="Header.TLabel").grid(
        row=current_gi_row, column=0, sticky="w", padx=5, pady=3
    )
    model_name_source_frame.grid(
        row=current_gi_row, column=1, columnspan=3, sticky="w", padx=5, pady=3
    )
    current_gi_row += 1
    ttk.Label(general_info_frame, text="Total Parameters:", style="Header.TLabel").grid(
        row=current_gi_row, column=0, sticky="w", padx=5, pady=3
    )
    total_params_val_label.grid(
        row=current_gi_row, column=1, sticky="w", padx=5, pady=3
    )
    ttk.Label(
        general_info_frame, text="Estimated Total Size:", style="Header.TLabel"
    ).grid(row=current_gi_row, column=2, sticky="w", padx=(10, 5), pady=3)
    est_total_size_val_label.grid(
        row=current_gi_row, column=3, sticky="w", padx=5, pady=3
    )
    current_gi_row += 1
    ttk.Label(general_info_frame, text="Parameter Status:", style="Header.TLabel").grid(
        row=current_gi_row, column=0, sticky="nw", padx=5, pady=(5, 0)
    )
    trainable_canvas.grid(
        row=current_gi_row, column=1, columnspan=3, sticky="ew", padx=5, pady=(5, 0)
    )
    current_gi_row += 1
    ttk.Label(general_info_frame, text="Parameter Dtypes:", style="Header.TLabel").grid(
        row=current_gi_row, column=0, sticky="nw", padx=5, pady=(5, 0)
    )
    dtype_canvas.grid(
        row=current_gi_row, column=1, columnspan=3, sticky="ew", padx=5, pady=(5, 0)
    )
    current_gi_row += 1

    if HAS_UINT8_PARAMS:

        def _toggle_and_refresh_command_local():
            nonlocal uint8_toggle_button_widget
            global VIEW_UINT8_AS_INT4
            VIEW_UINT8_AS_INT4 = not VIEW_UINT8_AS_INT4
            _repopulate_ui_with_data(
                model,
                root,
                trainable_canvas,
                dtype_canvas,
                device_canvas,
                tree,
                model_class_name_label,
                model_source_file_label,
                total_params_val_label,
                est_total_size_val_label,
                status_label,
                uint8_toggle_button_widget,
            )

        uint8_toggle_button_widget = ttk.Button(
            general_info_frame, command=_toggle_and_refresh_command_local
        )
        # Place button under the dtype_canvas content area (column 1, spanning available content columns)
        uint8_toggle_button_widget.grid(
            row=current_gi_row, column=1, columnspan=3, sticky="w", padx=10, pady=(2, 5)
        )  # pady top=2 to be close to canvas legend
        current_gi_row += 1

    ttk.Label(
        general_info_frame, text="Parameter Devices:", style="Header.TLabel"
    ).grid(row=current_gi_row, column=0, sticky="nw", padx=5, pady=(5, 0))
    device_canvas.grid(
        row=current_gi_row, column=1, columnspan=3, sticky="ew", padx=5, pady=(5, 0)
    )

    module_frame.pack(expand=True, fill=tk.BOTH, pady=10)
    col_widths = {
        "name": 280,
        "params_shape": 230,
        "params_perc": 80,
        "dtypes": 180,
        "device": 150,
        "memory": 100,
        "trainable": 80,
    }
    col_anchors = {
        "name": tk.W,
        "params_shape": tk.W,
        "params_perc": tk.E,
        "dtypes": tk.W,
        "device": tk.W,
        "memory": tk.E,
        "trainable": tk.CENTER,
    }
    col_headings = {
        "name": "Module Name",
        "params_shape": "Params (Shape)",
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
            stretch=(col == "name"),
        )

    vsb.pack(side="right", fill="y")
    tree.configure(yscrollcommand=vsb.set)
    hsb.pack(side="bottom", fill="x")
    tree.configure(xscrollcommand=hsb.set)
    tree.pack(expand=True, fill=tk.BOTH)

    footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
    status_label.pack(side=tk.LEFT, padx=5)
    close_button.pack(side=tk.RIGHT, padx=5)

    apply_dark_theme(
        root,
        style,
        main_frame,
        general_info_frame,
        module_frame,
        footer_frame,
        trainable_canvas,
        dtype_canvas,
        device_canvas,
        tree,
        vsb,
        hsb,
        status_label,
        close_button,
        model_name_source_frame,
        model_class_name_label,
        model_source_file_label,
        total_params_val_label,
        est_total_size_val_label,
        uint8_toggle_button_widget,
    )
    _repopulate_ui_with_data(
        model,
        root,
        trainable_canvas,
        dtype_canvas,
        device_canvas,
        tree,
        model_class_name_label,
        model_source_file_label,
        total_params_val_label,
        est_total_size_val_label,
        status_label,
        uint8_toggle_button_widget,
    )
    root.mainloop()


########################################################################################################

########################################################################################################

########################################################################################################

########################################################################################################

########################################################################################################


def inspect_tokenizer(tokenizer):
    import tkinter as tk
    from tkinter import ttk, font as scrolledtext

    # --- Dark Mode Style ---
    BG_COLOR = "#2E2E2E"
    FG_COLOR = "#FFFFFF"
    TEXT_BG_COLOR = "#3C3C3C"
    BUTTON_BG_COLOR = "#555555"
    BUTTON_FG_COLOR = "#FFFFFF"
    HIGHLIGHT_COLOR = "#4A4A4A"
    ENTRY_BORDER_COLOR = "#777777"

    # --- Create Main Window ---
    window = tk.Tk()
    window.title(f"Tokenizer Inspector: {tokenizer.__class__.__name__}")
    window.configure(bg=BG_COLOR)
    window.geometry("1300x800")

    # --- Style Configuration ---
    style = ttk.Style(window)
    style.theme_use("clam")

    # Basic widget styles
    style.configure(
        "TLabel",
        background=BG_COLOR,
        foreground=FG_COLOR,
        padding=5,
        font=("Segoe UI", 10),
    )
    style.configure("TFrame", background=BG_COLOR)
    style.configure(
        "TButton",
        background=BUTTON_BG_COLOR,
        foreground=BUTTON_FG_COLOR,
        font=("Segoe UI", 10, "bold"),
        borderwidth=1,
    )
    style.map("TButton", background=[("active", HIGHLIGHT_COLOR)])
    style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
    style.configure("TSeparator", background=HIGHLIGHT_COLOR)
    style.configure(
        "TEntry",
        fieldbackground=TEXT_BG_COLOR,
        foreground=FG_COLOR,
        insertcolor=FG_COLOR,
        bordercolor=ENTRY_BORDER_COLOR,
        lightcolor=TEXT_BG_COLOR,
        darkcolor=TEXT_BG_COLOR,
        padding=5,
    )
    style.configure(
        "Vertical.TScrollbar",
        background=BUTTON_BG_COLOR,
        troughcolor=BG_COLOR,
        arrowcolor=FG_COLOR,
        bordercolor=HIGHLIGHT_COLOR,
        lightcolor=BUTTON_BG_COLOR,
        darkcolor=BUTTON_BG_COLOR,
    )
    style.map("Vertical.TScrollbar", background=[("active", HIGHLIGHT_COLOR)])

    # Dark-mode Treeview for Special Tokens table
    style.configure(
        "Dark.Treeview",
        background=TEXT_BG_COLOR,
        fieldbackground=TEXT_BG_COLOR,
        foreground=FG_COLOR,
        bordercolor=HIGHLIGHT_COLOR,
        borderwidth=1,
    )
    style.configure(
        "Dark.Treeview.Heading",
        background=BUTTON_BG_COLOR,
        foreground=FG_COLOR,
        font=("Segoe UI", 10, "bold"),
    )
    style.map("Dark.Treeview.Heading", background=[("active", HIGHLIGHT_COLOR)])

    # --- Main Frame & Panes ---
    main_frame = ttk.Frame(window, padding="10")
    main_frame.pack(expand=True, fill=tk.BOTH)
    left_pane = ttk.Frame(main_frame, padding=(0, 0, 5, 0))
    left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    right_pane = ttk.Frame(main_frame, padding=(5, 0, 0, 0))
    right_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    for p in (left_pane, right_pane):
        for i in range(2):
            p.grid_rowconfigure(i, weight=1)
        p.grid_columnconfigure(0, weight=1)

    def add_section_heading(parent, title):
        ttk.Label(parent, text=title, style="Header.TLabel").pack(
            pady=(10, 5), anchor="w"
        )
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=(0, 10))

    # --- Quadrant 1: Basic Information ---
    basic_frame = ttk.Frame(left_pane)
    basic_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
    add_section_heading(basic_frame, "Basic Information")
    canvas = tk.Canvas(basic_frame, bg=BG_COLOR, highlightthickness=0)
    vsb = ttk.Scrollbar(
        basic_frame,
        orient="vertical",
        command=canvas.yview,
        style="Vertical.TScrollbar",
    )
    content = ttk.Frame(canvas)
    content.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=content, anchor="nw")
    canvas.configure(yscrollcommand=vsb.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)

    basic_info = {
        "Tokenizer Class": tokenizer.__class__.__name__,
        "Name or Path": getattr(tokenizer, "name_or_path", "N/A"),
        "Vocabulary Size": tokenizer.vocab_size,
        "Model Max Length": getattr(tokenizer, "model_max_length", "N/A"),
        "Padding Side": getattr(tokenizer, "padding_side", "N/A"),
        "Truncation Side": getattr(tokenizer, "truncation_side", "N/A"),
        "Model Input Names": getattr(tokenizer, "model_input_names", "N/A"),
    }
    for tk_name in ["bos", "eos", "unk", "sep", "pad", "cls", "mask"]:
        val = getattr(tokenizer, f"{tk_name}_token", None)
        vid = getattr(tokenizer, f"{tk_name}_token_id", None)
        if val is not None:
            basic_info[f"{tk_name.upper()} Token"] = (
                f"'{val}' (ID: {vid if vid is not None else 'N/A'})"
            )
    extras = getattr(tokenizer, "additional_special_tokens", [])
    if extras:
        basic_info["Additional Special Tokens"] = extras

    for key, val in basic_info.items():
        row = ttk.Frame(content)
        row.pack(fill=tk.X, pady=1, anchor="w")
        ttk.Label(
            row, text=f"{key}:", font=("Segoe UI", 9, "bold"), wraplength=160
        ).pack(side=tk.LEFT, padx=(0, 5), anchor="nw")
        ttk.Label(
            row, text=str(val), wraplength=320, justify=tk.LEFT, font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, anchor="nw")

    # --- Quadrant 2: Special Tokens Table ---
    special_frame = ttk.Frame(left_pane)
    special_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
    add_section_heading(special_frame, "Special Tokens")
    cols = ("Token", "ID")
    table = ttk.Treeview(
        special_frame, columns=cols, show="headings", height=10, style="Dark.Treeview"
    )
    for c in cols:
        table.heading(c, text=c, anchor="center")
        table.column(c, anchor="center")
    table.pack(fill=tk.BOTH, expand=True)

    uniq = {}
    if hasattr(tokenizer, "all_special_tokens"):
        for t, i in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids):
            uniq[t] = i
    for pfx in ["bos", "eos", "unk", "sep", "pad", "cls", "mask"]:
        t = getattr(tokenizer, f"{pfx}_token", None)
        i = getattr(tokenizer, f"{pfx}_token_id", None)
        if t and i is not None:
            uniq.setdefault(t, i)
    if hasattr(tokenizer, "added_tokens_decoder"):
        for i, obj in tokenizer.added_tokens_decoder.items():
            uniq.setdefault(obj.content, i)
    if not uniq:
        table.insert("", "end", values=("No special tokens found", ""))
    else:
        for t, i in sorted(uniq.items(), key=lambda x: (x[1], x[0])):
            table.insert("", "end", values=(t, i))

    # --- Quadrant 3: Token Finder ---
    finder_frame = ttk.Frame(right_pane)
    finder_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
    add_section_heading(finder_frame, "Find Token")
    fcont = ttk.Frame(finder_frame)
    fcont.pack(fill=tk.BOTH, expand=True)
    finput = ttk.Frame(fcont)
    finput.pack(fill=tk.X, pady=(0, 5))
    ttk.Label(finput, text="ID or Text:").pack(side=tk.LEFT, padx=(0, 5))
    tf_entry = ttk.Entry(finput, width=30, style="TEntry")
    tf_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
    tf_res = scrolledtext.ScrolledText(
        fcont,
        wrap=tk.WORD,
        height=8,
        bg=TEXT_BG_COLOR,
        fg=FG_COLOR,
        font=("Courier New", 9),
        relief=tk.FLAT,
        borderwidth=1,
        highlightbackground=HIGHLIGHT_COLOR,
        highlightthickness=1,
        insertbackground=FG_COLOR,
    )
    tf_res.pack(fill=tk.BOTH, expand=True)

    def perform_find(event=None):
        q = tf_entry.get().strip()
        out = []
        if not q:
            out.append("Please enter a token ID or text.")
        else:
            if q.lstrip("-").isdigit():
                tid = int(q)
                out.append(f"Input ID: {tid}")
                try:
                    out.append(f"  ↪ Decoded: '{tokenizer.decode([tid])}'")
                except:
                    out.append("  ↪ Error decoding")
                try:
                    tok = tokenizer.convert_ids_to_tokens([tid])[0]
                    out.append(f"  ↪ Token: '{tok}'")
                except:
                    out.append("  ↪ Error converting")
            else:
                out.append(f"Input Text: '{q}'")
                try:
                    out.append(f"  ↪ ID: {tokenizer.convert_tokens_to_ids(q)}")
                except:
                    out.append("  ↪ Not a single known token")
                if hasattr(tokenizer, "vocab") and q in tokenizer.vocab:
                    out.append(f"  ↪ Vocab ID: {tokenizer.vocab[q]}")
                try:
                    enc = tokenizer.encode(q, add_special_tokens=False)
                    toks = tokenizer.convert_ids_to_tokens(enc)
                    if not (len(enc) == 1 and toks[0] == q):
                        out.append("  ↪ As sequence:")
                        out.append(f"      Tokens: {toks}")
                        out.append(f"      IDs   : {enc}")
                except:
                    pass
        tf_res.configure(state="normal")
        tf_res.delete("1.0", tk.END)
        tf_res.insert("1.0", "\n".join(out))
        tf_res.configure(state="disabled")

    ttk.Button(finput, text="Find", command=perform_find, style="TButton").pack(
        side=tk.LEFT, padx=(5, 0)
    )
    tf_entry.bind("<Return>", perform_find)
    perform_find()

    # --- Quadrant 4: Live Tokenization ---
    live_frame = ttk.Frame(right_pane)
    live_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
    add_section_heading(live_frame, "Live Tokenization")
    live_cont = ttk.Frame(live_frame)
    live_cont.pack(fill=tk.BOTH, expand=True)
    input_hl_text = tk.Text(
        live_cont,
        wrap=tk.WORD,
        height=10,
        relief=tk.FLAT,
        bg=TEXT_BG_COLOR,
        fg=FG_COLOR,
        insertbackground=FG_COLOR,
        font=("Segoe UI", 11),
        borderwidth=1,
        highlightthickness=1,
        highlightbackground=HIGHLIGHT_COLOR,
    )
    input_hl_text.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
    ids_disp_text = tk.Text(
        live_cont,
        wrap=tk.WORD,
        height=5,
        relief=tk.FLAT,
        bg=TEXT_BG_COLOR,
        fg=FG_COLOR,
        insertbackground=FG_COLOR,
        font=("Courier New", 10),
        borderwidth=1,
        highlightthickness=1,
        highlightbackground=HIGHLIGHT_COLOR,
    )
    ids_disp_text.pack(fill=tk.BOTH, expand=True)
    ids_disp_text.configure(state="disabled")

    def update_live_hl_tokenize(event=None):
        text = input_hl_text.get("1.0", tk.END)
        # clear previous tags
        for tag in input_hl_text.tag_names():
            if tag.startswith("tok_"):
                input_hl_text.tag_delete(tag)
        ids_disp_text.configure(state="normal")
        ids_disp_text.delete("1.0", tk.END)

        if not text.strip():
            ids_disp_text.insert(tk.END, "Token IDs will appear here.")
        else:
            try:
                out = tokenizer(
                    text, return_offsets_mapping=True, add_special_tokens=False
                )
                tids = out.input_ids
                offsets = out.offset_mapping
                for idx, (start, end) in enumerate(offsets):
                    if start < end:
                        r = random.randint(200, 255)
                        g = random.randint(200, 255)
                        b = random.randint(200, 255)
                        bgc = f"#{r:02x}{g:02x}{b:02x}"
                        tag = f"tok_{idx}"
                        input_hl_text.tag_configure(
                            tag, background=bgc, foreground="black"
                        )
                        input_hl_text.tag_add(tag, f"1.0+{start}c", f"1.0+{end}c")
                for idx, tid in enumerate(tids):
                    r = random.randint(200, 255)
                    g = random.randint(200, 255)
                    b = random.randint(200, 255)
                    bgc = f"#{r:02x}{g:02x}{b:02x}"
                    tag = f"id_{idx}"
                    ids_disp_text.tag_configure(
                        tag,
                        background=bgc,
                        foreground="black",
                        relief="raised",
                        borderwidth=1,
                        lmargin1=2,
                        lmargin2=2,
                        rmargin=2,
                        spacing1=1,
                        spacing3=1,
                    )
                    ids_disp_text.insert(tk.END, f"{tid} ", tag)
            except Exception as e:
                ids_disp_text.insert(tk.END, f"Error: {str(e)[:100]}...")
        ids_disp_text.configure(state="disabled")

    input_hl_text.bind("<KeyRelease>", update_live_hl_tokenize)
    input_hl_text.insert("1.0", "Type text here to see live tokenization...")
    update_live_hl_tokenize()

    window.mainloop()
