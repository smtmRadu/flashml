TRAINABLE_COLORS_VIZ = {"Trainable": "#ff6b6b", "Frozen": "#57b7e0"}
DTYPE_BASE_COLORS_VIZ = [
    "#667eea",
    "#f093fb",
    "#4facfe",
    "#43e97b",
    "#fa709a",
    "#fee140",
    "#a8edea",
    "#ffecd2",
    "#ff9a9e",
    "#fecfef",
]
DEVICE_COLORS_VIZ_MAP = {
    "cpu": "#667eea",
    "cuda:0": "#43e97b",
    "mps": "#fee140",
    "rocm": "#fa709a",
    "other_gpu_shades": ["#4facfe", "#f093fb", "#a8edea", "#ffecd2", "#ff9a9e"],
    "default": "#95a5a6",
}


def get_parameter_dtype(param, view_uint8_as_int4):
    original_dtype_str = str(param.dtype)
    if view_uint8_as_int4 and original_dtype_str == "torch.uint8":
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
    import math

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
    }
    return "".join(bold_map.get(char, char) for char in str(text_string))


def create_pie_chart(data_percentages, colors, title, icon_map=None):
    import plotly.graph_objects as go

    """Create a pie chart with Plotly"""
    if not data_percentages:
        # Empty pie chart
        fig = go.Figure(
            data=[go.Pie(labels=["No Data"], values=[1], marker_colors=["#444444"])]
        )
        fig.update_layout(title=title, showlegend=False)
        return fig

    labels = []
    values = []
    chart_colors = []
    hover_texts = []
    text_labels = []

    for name, percentage, val_str in data_percentages:
        if percentage > 0:
            icon = icon_map.get(name, "") if icon_map else ""
            labels.append(f"{icon}{name}")
            values.append(percentage)
            chart_colors.append(colors.get(name, "#A9A9A9"))
            hover_texts.append(f"<b>{name}</b><br>{val_str}<br>{percentage:.1f}%")

            # For small segments, show percentage only; for larger ones, show name + percentage
            if percentage < 5:
                text_labels.append(f"{percentage:.1f}%")
            else:
                text_labels.append(f"{name}<br>{percentage:.1f}%")

    if not values:
        return create_pie_chart([], colors, title, icon_map)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker_colors=chart_colors,
                hovertemplate="<b>%{hovertext}</b><extra></extra>",
                hovertext=hover_texts,
                text=text_labels,
                textinfo="text",
                textposition="auto",
                textfont=dict(size=11, color="white"),
                hole=0.3,  # Donut chart for better aesthetics
                pull=[0.02] * len(values),  # Slight separation between segments
            )
        ]
    )

    fig.update_layout(title=title, showlegend=False)

    return fig


def inspect_model_notebook(model, input_data=None, renderer="vscode", view_uint8_as_int4=True):
    """
    Inspect PyTorch model with interactive Plotly visualizations

    Args:
        model: PyTorch model to inspect
        input_data: the input tensor/dict or object the model receives as arg
        renderer: Plotly renderer ('vscode', 'browser', 'notebook', etc.)
        view_uint8_as_int4: Whether to interpret uint8 parameters as packed int4
    """
    import inspect as py_inspect
    import math

    import plotly.io as pio

    TRAINABLE_COLORS_VIZ = {"Trainable": "#2ecc71", "Frozen": "#3498db"}
    DTYPE_BASE_COLORS_VIZ = ["#9b59b6", "#e74c3c", "#f1c40f", "#34495e", "#1abc9c"]
    DEVICE_COLORS_VIZ_MAP = {
        "cpu": "#f39c12",
        "cuda:0": "#16a085",
        "mps": "#8e44ad",
        "rocm": "#c0392b",
        "other_gpu_shades": ["#1abc9c", "#27ae60", "#2980b9"],
        "default": "#7f8c8d",
    }
    # --- End Helper Functions ---

    pio.templates.default = "plotly_dark"

    # --- Helper Functions (assuming these are defined elsewhere) ---
    def format_size(size_bytes):
        if size_bytes == 0:
            return "0 B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_name[i]}"

    def get_parameter_dtype(param, view_uint8_as_int4_flag):
        if view_uint8_as_int4_flag and str(param.dtype) == "torch.uint8":
            return "torch.int4"
        return str(param.dtype)

    def get_parameter_device(param):
        try:
            return str(param.device)
        except RuntimeError:
            return "N/A"

    def get_memory_size(num_params, dtype_str):
        dtype_to_bytes = {
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
            "torch.int4": 0.5,  # Custom handling
            "torch.bool": 1,  # Often stored as a byte
        }
        return num_params * dtype_to_bytes.get(dtype_str, 0)

    def to_bold_digits(s):
        # Placeholder for unicode bolding if needed, otherwise just return string
        return s

    def analyze_model_data():
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
            display_dtype_str = get_parameter_dtype(param, view_uint8_as_int4)
            display_num_p = (
                original_num_p * 2
                if view_uint8_as_int4 and original_dtype_str == "torch.uint8"
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

        # Analyze modules
        for name, module in model.named_modules():
            module_num_params, module_memory_bytes, module_param_trainable_count = (
                0,
                0,
                0,
            )
            module_dtypes, module_devices = {}, {}
            module_param_shapes = []
            module_has_params = False
            current_params_list = list(module.parameters(recurse=False))
            module_type = module.__class__.__name__

            if not current_params_list and not list(module.children()):
                module_details.append(
                    {
                        "name": name if name else model.__class__.__name__,
                        "type": module_type,
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

                display_dtype_str_module = get_parameter_dtype(
                    param, view_uint8_as_int4
                )
                display_num_p_module = (
                    original_num_p_module * 2
                    if view_uint8_as_int4 and original_dtype_str_module == "torch.uint8"
                    else original_num_p_module
                )

                module_num_params += display_num_p_module
                module_dtypes[display_dtype_str_module] = (
                    module_dtypes.get(display_dtype_str_module, 0)
                    + display_num_p_module
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
                    (module_num_params / total_params * 100)
                    if total_params > 0
                    else 0.0
                )
                module_details.append(
                    {
                        "name": name if name else model.__class__.__name__,
                        "type": module_type,
                        "num_params": module_num_params,
                        "params_percentage": current_module_percentage,
                        "dtypes": module_dtype_str,
                        "device": module_device_str,
                        "memory": format_size(module_memory_bytes),
                        "trainable_status": trainable_status,
                        "shape_str_internal": module_shape_str,
                    }
                )

        return {
            "total_params": total_params,
            "total_memory_bytes": total_memory_bytes,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
            "param_dtypes_counts": param_dtypes_counts,
            "param_devices_counts": param_devices_counts,
            "param_devices_memory_bytes": param_devices_memory_bytes,
            "module_details": module_details,
        }

    def create_dashboard():
        import plotly.graph_objects as go
        import polars as pl
        from plotly.subplots import make_subplots

        data = analyze_model_data()

        # Create subplots - all three pies on the same row
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=(
                "Parameter Status",
                "Parameter Data Types",
                "Parameter Devices",
                "",
                "",
                "",  # Empty titles for second row
            ),
            specs=[
                [{"type": "pie"}, {"type": "pie"}, {"type": "pie"}],
                [{"colspan": 3, "type": "table"}, None, None],
            ],
            vertical_spacing=0.0,  # Reduced vertical spacing
            horizontal_spacing=0.05,  # GAP between the pies
        )

        # Trainable pie chart
        trainable_data = []
        if data["total_params"] > 0:
            trainable_data = [
                (
                    "Trainable",
                    (data["trainable_params"] / data["total_params"]) * 100,
                    f"{data['trainable_params']:,}",
                ),
                (
                    "Frozen",
                    (data["non_trainable_params"] / data["total_params"]) * 100,
                    f"{data['non_trainable_params']:,}",
                ),
            ]

        if trainable_data:
            labels, values, colors_list, hover_texts, text_labels = [], [], [], [], []
            for name, perc, val_str in trainable_data:
                if perc > 0:
                    labels.append(name)
                    values.append(perc)
                    colors_list.append(TRAINABLE_COLORS_VIZ.get(name, "#95a5a6"))
                    hover_texts.append(f"<b>{name}</b><br>{val_str}<br>{perc:.1f}%")
                    text_labels.append(
                        f"{name}<br>{perc:.1f}%" if perc >= 5 else f"{perc:.1f}%"
                    )

            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=values,
                    marker_colors=colors_list,
                    hovertemplate="<b>%{hovertext}</b><extra></extra>",
                    hovertext=hover_texts,
                    text=text_labels,
                    textinfo="text",
                    textposition="auto",
                    textfont=dict(size=11, color="white"),
                    hole=0.3,
                    pull=[0.02] * len(values),
                    name="Trainable Status",
                ),
                row=1,
                col=1,
            )

        # Dtype pie chart
        dtype_data = []
        if data["total_params"] > 0:
            for dt, count in sorted(
                data["param_dtypes_counts"].items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                dtype_data.append(
                    (
                        dt.replace("torch.", ""),
                        (count / data["total_params"]) * 100,
                        f"{count:,}",
                    )
                )

        if dtype_data:
            dtype_colors = {
                item[0]: color
                for item, color in zip(
                    dtype_data,
                    DTYPE_BASE_COLORS_VIZ
                    * (len(dtype_data) // len(DTYPE_BASE_COLORS_VIZ) + 1),
                )
            }
            labels, values, colors_list, hover_texts, text_labels = [], [], [], [], []
            for name, perc, val_str in dtype_data:
                if perc > 0:
                    labels.append(name)
                    values.append(perc)
                    colors_list.append(dtype_colors.get(name, "#95a5a6"))
                    hover_texts.append(f"<b>{name}</b><br>{val_str}<br>{perc:.1f}%")
                    text_labels.append(
                        f"{name}<br>{perc:.1f}%" if perc >= 5 else f"{perc:.1f}%"
                    )

            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=values,
                    marker_colors=colors_list,
                    hovertemplate="<b>%{hovertext}</b><extra></extra>",
                    hovertext=hover_texts,
                    text=text_labels,
                    textinfo="text",
                    textposition="auto",
                    textfont=dict(size=11, color="white"),
                    hole=0.3,
                    pull=[0.02] * len(values),
                    name="Data Types",
                ),
                row=1,
                col=2,
            )

        # Device pie chart
        device_data = []
        if data["total_params"] > 0:
            for dev, count in sorted(
                data["param_devices_counts"].items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                device_data.append(
                    (
                        dev,
                        (count / data["total_params"]) * 100,
                        format_size(data["param_devices_memory_bytes"].get(dev, 0)),
                    )
                )

        if device_data:
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

            labels, values, colors_list, hover_texts, text_labels = [], [], [], [], []
            for name, perc, val_str in device_data:
                if perc > 0:
                    labels.append(name)
                    values.append(perc)
                    colors_list.append(device_colors.get(name, "#95a5a6"))
                    hover_texts.append(f"<b>{name}</b><br>{val_str}<br>{perc:.1f}%")
                    text_labels.append(
                        f"{name}<br>{perc:.1f}%" if perc >= 5 else f"{perc:.1f}%"
                    )

            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=values,
                    marker_colors=colors_list,
                    hovertemplate="<b>%{hovertext}</b><extra></extra>",
                    hovertext=hover_texts,
                    text=text_labels,
                    textinfo="text",
                    textposition="auto",
                    textfont=dict(size=11, color="white"),
                    hole=0.3,
                    pull=[0.02] * len(values),
                    name="Devices",
                ),
                row=1,
                col=3,
            )

        # Module details table
        module_df = pl.DataFrame(data["module_details"])
        if not module_df.is_empty():
            trainable_emojis = {
                "Yes": "✔️",
                "No": "❌",
                "Partial": "◑",
                "N/A": "",
                "-": "",
            }
            table_data = []
            for row in module_df.iter_rows(named=True):
                trainable_display = trainable_emojis.get(
                    row["trainable_status"], row["trainable_status"]
                )
                if row["trainable_status"] == "-":
                    params_shape_display = "-"
                    params_perc_display = "-"
                elif row["num_params"] > 0:
                    bold_num_p_str = to_bold_digits(f"{row['num_params']:,}")
                    shape_s = row["shape_str_internal"]
                    params_shape_display = (
                        f"{bold_num_p_str} = {shape_s}"
                        if shape_s and shape_s != "N/A"
                        else f"{bold_num_p_str} = (shape N/A)"
                    )
                    params_perc_display = (
                        f"{row['params_percentage']:.2f}%"
                        if data["total_params"] > 0
                        else "0.00%"
                    )
                else:
                    params_shape_display = "-"
                    params_perc_display = "----"

                table_data.append(
                    [
                        row["name"],
                        row["type"],
                        params_shape_display,
                        params_perc_display,
                        row["dtypes"],
                        row["device"],
                        row["memory"],
                        trainable_display,
                    ]
                )

            fig.add_trace(
                go.Table(
                    header=dict(
                        values=[
                            "Module Name",
                            "Type",
                            "Params (Shape)",
                            "% Total",
                            "Dtypes (% module)",
                            "Device(s) (% module)",
                            "Est. Size",
                            "Trainable",
                        ],
                        fill_color="#404040",
                        font=dict(color="white", size=12),
                        align="left",
                        line=dict(width=0),
                    ),
                    cells=dict(
                        values=list(zip(*table_data))
                        if table_data
                        else [[] for _ in range(8)],
                        fill_color="#2d2d2d",
                        font=dict(color="white", size=10),
                        align="left",
                        height=32,
                        line=dict(width=0),
                    ),
                    columnwidth=[
                        3.0,
                        1.25,
                        2.0,
                        0.75,
                        1.25,
                        1.15,
                        0.75,
                        0.6,
                    ],  # Adjusted widths
                ),
                row=2,
                col=1,
            )

        # Update layout
        model_class_name_str = model.__class__.__name__
        try:
            model_source_file_str = py_inspect.getfile(model.__class__)
        except (TypeError, OSError):
            model_source_file_str = "Source not found"

        title_text = (
            f"<b>{model_class_name_str}</b><br>"
            f"<sub>Source: {model_source_file_str}</sub><br>"
            f"<sub>Total Parameters: {data['total_params']:,} | "
            f"Estimated Size: {format_size(data['total_memory_bytes'])}</sub>"
        )

        fig.update_layout(
            title=dict(text=title_text, x=0.5, font=dict(size=16), y=0.98),
            height=1100,
            showlegend=False,
            margin=dict(t=25, b=25, l=25, r=25),  # pad (let it like this)
        )

        # Update subplot titles positioning
        fig.update_annotations(font_size=12, yshift=-80)  # Reduced yshift

        return fig

    fig = create_dashboard()

    try:
        from IPython import get_ipython

        if get_ipython() is None:
            fig.show(renderer=renderer)
    except ImportError:
        fig.show(renderer=renderer)

    if input_data is not None:
        from torchinfo import summary

        try:
            s = summary(model, input_data)
            return fig, s
        except:
            return fig
    else:
        return fig  # let this here