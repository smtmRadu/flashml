## begin of file


def plot_tensor(
    tensor,
    title: str = "",
    colorscale: str = "Viridis",
    show_values: bool = True,
    renderer: str = "vscode",
):
    """
    Creates interactive Plotly visualizations for 0D, 1D, 2D, and 3D tensors.

    Features:
    - 0D: Single value display with gauge
    - 1D: Interactive bar chart and heatmap
    - 2D: Interactive heatmap with hover details
    - 3D: Animated slice navigation + 3D surface plots + interactive slice selector

    Args:
        tensor: Input tensor (numpy array, torch tensor, or list)
        title: Plot title
        colorscale: Plotly colorscale ('Viridis', 'Plasma', 'Turbo', 'RdBu', etc.)
        show_values: Whether to show values on heatmaps (for small tensors)
        animation_frame_duration: Animation speed for 3D tensor slices (ms)
        width/height: Figure dimensions (auto-calculated if None)

    Returns:
        Plotly figure object
    """
    import plotly.graph_objects as go
    import plotly.io as pio
    import numpy as np

    pio.templates.default = "plotly_dark"
    # Convert input to numpy array
    original_device = "cpu"
    try:
        original_device = tensor.device
    except:
        pass

    if not isinstance(tensor, np.ndarray):
        try:
            if hasattr(tensor, "cpu"):  # PyTorch tensor
                tensor = np.array(tensor.cpu())
            else:
                tensor = np.array(tensor)
        except Exception as e:
            raise ValueError(f"Could not convert input to NumPy array: {e}")

    # Handle empty tensor
    if tensor.size == 0:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Empty Tensor<br>Shape: {tensor.shape}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="orange"),
        )
        fig.update_layout(
            title=f"{title} Empty Tensor",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=600,
            height=400,
        )
        return fig

    dim = tensor.ndim

    if dim <= 1:
        width, height = 800, 500
    elif dim == 2:
        aspect_ratio = tensor.shape[1] / tensor.shape[0]
        width = min(1000, max(600, int(600 * aspect_ratio)))
        height = min(800, max(400, int(600 / aspect_ratio)))
    else:  # 3D
        width, height = 1200, 800

    if dim == 0:
        return _plot_0d_tensor(
            tensor, title, colorscale, width, height, original_device, renderer
        )
    elif dim == 1:
        return _plot_1d_tensor(
            tensor,
            title,
            colorscale,
            show_values,
            width,
            height,
            original_device,
            renderer,
        )
    elif dim == 2:
        return _plot_2d_tensor(
            tensor,
            title,
            colorscale,
            show_values,
            width,
            height,
            original_device,
            renderer,
        )
    elif dim == 3:
        return _plot_3d_tensor(
            tensor,
            title,
            colorscale,
            show_values,
            original_device,
            renderer=renderer,
        )
    else:
        # High-dimensional tensor - show error
        fig = go.Figure()
        fig.add_annotation(
            text=f"Cannot visualize {dim}D tensor<br>Supports 0D-3D tensors only",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="red"),
        )
        fig.update_layout(
            title=f"{title} Unsupported Dimension ({dim}D)",
            width=width,
            height=height,
        )
        fig.show(renderer=renderer)


def _plot_0d_tensor(
    tensor, title, colorscale, width, height, original_device, renderer
):
    """Plot 0D tensor (scalar) with gauge and info card"""
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    value = tensor.item()

    # Create subplot with gauge and info
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.6, 0.4],
        specs=[[{"type": "indicator"}, {"type": "table"}]],
        subplot_titles=["Value Gauge", "Tensor Info"],
    )

    # Gauge plot
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Scalar Value"},
            gauge={
                "axis": {
                    "range": [value - abs(value), value + abs(value)]
                    if value != 0
                    else [-1, 1]
                },
                "bar": {"color": px.colors.sequential.Viridis[4]},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {
                        "range": [
                            value - abs(value) / 2 if value != 0 else -0.5,
                            value + abs(value) / 2 if value != 0 else 0.5,
                        ],
                        "color": "lightgray",
                    }
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": value,
                },
            },
        ),
        row=1,
        col=1,
    )

    # Info table
    fig.add_trace(
        go.Table(
            header=dict(values=["Property", "Value"], fill_color="paleturquoise"),
            cells=dict(
                values=[
                    ["Dimension", "Shape", "Type", "Value"],
                    [
                        "0D (Scalar)",
                        str(tensor.shape),
                        str(tensor.dtype),
                        f"{value:.6g}",
                    ],
                ]
            ),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"{title} 0D Tensor (Scalar) | Device: {original_device}",
        width=width,
        height=height,
    )

    fig.show(renderer=renderer)


def _plot_1d_tensor(
    tensor, title, colorscale, show_values, width, height, original_device, renderer
):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import numpy as np

    """Plot 1D tensor with bar chart and heatmap views"""

    # Create subplots: bar chart and heatmap
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.6, 0.4],
        subplot_titles=["Bar Chart View", "Heatmap View"],
        vertical_spacing=0.20,  # Increased spacing to prevent overlap
    )

    indices = np.arange(len(tensor))

    # Bar chart
    fig.add_trace(
        go.Bar(
            x=indices,
            y=tensor,
            marker=dict(
                color=tensor,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title="Value",
                    y=0.5,  # Center the colorbar vertically
                    len=1.0,  # Full height of the plot
                    yanchor="middle",
                ),
            ),
            hovertemplate="Index: %{x}<br>Value: %{y:.4f}<extra></extra>",
            name="Values",
        ),
        row=1,
        col=1,
    )

    # Heatmap view
    tensor_2d = tensor.reshape(1, -1)

    fig.add_trace(
        go.Heatmap(
            z=tensor_2d,
            colorscale=colorscale,
            showscale=False,
            hovertemplate="Index: %{x}<br>Value: %{z:.4f}<extra></extra>",
            name="Heatmap",
        ),
        row=2,
        col=1,
    )

    # Add value annotations if requested and tensor is small
    if show_values and len(tensor) <= 20:
        for i, val in enumerate(tensor):
            fig.add_annotation(
                x=i,
                y=0,
                text=f"{val:.2g}",
                showarrow=False,
                row=2,
                col=1,
                font=dict(
                    size=10,  # Smaller font size to reduce overlap
                    color="white"
                    if abs(val - tensor.mean()) > tensor.std()
                    else "black",
                ),
            )

    fig.update_xaxes(title="Index", row=1, col=1)
    fig.update_yaxes(title="Value", row=1, col=1)
    fig.update_xaxes(title="Index", row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)

    fig.update_layout(
        title=dict(
            text=f"{title} 1D Tensor ({len(tensor)} elements) | Device: {original_device}",
            x=0.5,  # Center the title horizontally
            xanchor="center",
        ),
        width=width,
        height=height,
        showlegend=False,
    )

    fig.show(renderer=renderer)


def _plot_2d_tensor(
    tensor, title, colorscale, show_values, width, height, original_device, renderer
):
    """Plot 2D tensor with interactive heatmap"""
    import plotly.graph_objects as go

    rows, cols = tensor.shape

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=tensor,
            colorscale=colorscale,
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z:.4f}<extra></extra>",
            colorbar=dict(title="Value"),
        )
    )

    # Add value annotations if requested and tensor is small enough
    if (
        show_values and tensor.shape[-1] <= 15 and tensor.shape[-2] <= 20
    ):  # Only for small tensors
        annotations = []
        for i in range(rows):
            for j in range(cols):
                # Determine text color based on value relative to colorscale
                val = tensor[i, j]
                normalized_val = (
                    (val - tensor.min()) / (tensor.max() - tensor.min())
                    if tensor.max() != tensor.min()
                    else 0.5
                )
                text_color = "white" if normalized_val < 0.5 else "black"

                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f"{val:.2g}",
                        showarrow=False,
                        font=dict(color=text_color, size=10),
                    )
                )
        fig.update_layout(annotations=annotations)

    fig.update_layout(
        title=dict(
            text=f"{title} 2D Tensor ({rows}×{cols}) | Device: {original_device}",
            x=0.5,  # Center the title horizontally
            xanchor="center",
        ),
        xaxis_title="Column Index",
        yaxis_title="Row Index",
        width=width,
        height=height,
        yaxis=dict(autorange="reversed"),  # Match matrix convention
    )

    fig.show(renderer=renderer)


def _plot_3d_tensor(
    tensor,
    title,
    colorscale,
    show_values,
    original_device,
    renderer,
):
    """Plot 3D tensor with slice slider"""
    import plotly.graph_objects as go

    depth, rows, cols = tensor.shape

    # Auto-threshold show_values based on slice dimensions
    if show_values is None:
        show_values = rows < 15 and cols < 15

    # Create initial heatmap for first slice
    fig = go.Figure(
        data=go.Heatmap(
            z=tensor[0],
            colorscale=colorscale,
            hovertemplate="Slice: 0<br>Row: %{y}<br>Col: %{x}<br>Value: %{z:.4f}<extra></extra>",
            colorbar=dict(title="Value"),
        )
    )

    # Add value annotations if requested and tensor is small enough
    if show_values:
        annotations = []
        for i in range(rows):
            for j in range(cols):
                val = tensor[0, i, j]
                normalized_val = (
                    (val - tensor.min()) / (tensor.max() - tensor.min())
                    if tensor.max() != tensor.min()
                    else 0.5
                )
                text_color = "white" if normalized_val < 0.5 else "black"

                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f"{val:.2g}",
                        showarrow=False,
                        font=dict(color=text_color, size=10),
                    )
                )
        fig.update_layout(annotations=annotations)

    # Create slider steps
    steps = []
    for i in range(depth):
        # Prepare annotations for this slice if show_values is enabled
        slice_annotations = []
        if show_values:
            for row in range(rows):
                for col in range(cols):
                    val = tensor[i, row, col]
                    normalized_val = (
                        (val - tensor.min()) / (tensor.max() - tensor.min())
                        if tensor.max() != tensor.min()
                        else 0.5
                    )
                    text_color = "white" if normalized_val < 0.5 else "black"

                    slice_annotations.append(
                        dict(
                            x=col,
                            y=row,
                            text=f"{val:.2g}",
                            showarrow=False,
                            font=dict(color=text_color, size=10),
                        )
                    )

        step = dict(
            method="update",
            args=[
                {"z": [tensor[i]]},
                {
                    "title": f"{title} (Slice {i}) 3D Tensor ({depth}×{rows}×{cols})",
                    "annotations": slice_annotations,
                },
            ],
            label=f"{i}",
        )
        steps.append(step)

    # Create slider
    sliders = [
        dict(active=0, currentvalue={"prefix": "Slice: "}, pad={"t": 50}, steps=steps)
    ]

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{title} (Slice 0) 3D Tensor ({depth}×{rows}×{cols}) | Device: {original_device}",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Column",
        yaxis_title="Row",
        yaxis=dict(autorange="reversed"),
        sliders=sliders,
    )

    fig.show(renderer=renderer)

    return fig
