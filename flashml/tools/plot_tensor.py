from typing import Optional


def plot_tensor(
    tensor,
    title: str = "Interactive Tensor Visualization",
    colorscale: str = "Viridis",
    show_values: bool = True,
    animation_frame_duration: int = 800,
    width: Optional[int] = None,
    height: Optional[int] = None,
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
            title=f"{title} - Empty Tensor",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=600,
            height=400,
        )
        return fig

    dim = tensor.ndim

    # Auto-calculate dimensions if not provided
    if width is None or height is None:
        if dim <= 1:
            width, height = 800, 500
        elif dim == 2:
            aspect_ratio = tensor.shape[1] / tensor.shape[0]
            width = min(1000, max(600, int(600 * aspect_ratio)))
            height = min(800, max(400, int(600 / aspect_ratio)))
        else:  # 3D
            width, height = 1200, 800

    if dim == 0:
        return _plot_0d_tensor(tensor, title, colorscale, width, height)
    elif dim == 1:
        return _plot_1d_tensor(tensor, title, colorscale, show_values, width, height)
    elif dim == 2:
        return _plot_2d_tensor(tensor, title, colorscale, show_values, width, height)
    elif dim == 3:
        return _plot_3d_tensor(
            tensor,
            title,
            colorscale,
            show_values,
            animation_frame_duration,
            width,
            height,
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
            title=f"{title} - Unsupported Dimension ({dim}D)",
            width=width,
            height=height,
        )
        fig.show(renderer="browser")


def _plot_0d_tensor(tensor, title, colorscale, width, height):
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

    fig.update_layout(title=f"{title} - 0D Tensor (Scalar)", width=width, height=height)

    fig.show(renderer="browser")


def _plot_1d_tensor(tensor, title, colorscale, show_values, width, height):
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
        vertical_spacing=0.15,
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
                colorbar=dict(title="Value", y=0.8, len=0.6),
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
                    color="white"
                    if abs(val - tensor.mean()) > tensor.std()
                    else "black"
                ),
            )

    fig.update_xaxes(title="Index", row=1, col=1)
    fig.update_yaxes(title="Value", row=1, col=1)
    fig.update_xaxes(title="Index", row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)

    fig.update_layout(
        title=f"{title} - 1D Tensor ({len(tensor)} elements)",
        width=width,
        height=height,
        showlegend=False,
    )

    fig.show(renderer="browser")


def _plot_2d_tensor(tensor, title, colorscale, show_values, width, height):
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
    if show_values and tensor.size <= 400:  # Only for small tensors
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
        title=f"{title} - 2D Tensor ({rows}×{cols})",
        xaxis_title="Column Index",
        yaxis_title="Row Index",
        width=width,
        height=height,
        yaxis=dict(autorange="reversed"),  # Match matrix convention
    )

    fig.show(renderer="browser")


def _plot_3d_tensor(
    tensor, title, colorscale, show_values, animation_frame_duration, width, height
):
    """Plot 3D tensor with multiple interactive views"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    depth, rows, cols = tensor.shape

    # Create subplot layout:
    # Top row: Animated slice viewer, 3D surface
    # Bottom row: Slice selector heatmaps, statistics
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "heatmap"}, {"type": "surface"}],
            [{"type": "heatmap"}, {"type": "bar"}],
        ],
        subplot_titles=[
            "Animated Slice Viewer",
            "3D Surface (First Slice)",
            "All Slices Overview",
            "Slice Statistics",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # 1. Animated slice viewer (top-left)
    frames = []
    for i in range(depth):
        frame_data = [
            go.Heatmap(
                z=tensor[i],
                colorscale=colorscale,
                showscale=False,
                hovertemplate=f"Slice {i}<br>Row: %{{y}}<br>Col: %{{x}}<br>Value: %{{z:.4f}}<extra></extra>",
            )
        ]
        frames.append(go.Frame(data=frame_data, name=str(i)))

    # Initial slice
    fig.add_trace(
        go.Heatmap(
            z=tensor[0],
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title="Value", x=0.48, len=0.4, y=0.78),
            hovertemplate="Slice 0<br>Row: %{y}<br>Col: %{x}<br>Value: %{z:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # 2. 3D Surface plot (top-right)
    x_surf = np.arange(cols)
    y_surf = np.arange(rows)
    X_surf, Y_surf = np.meshgrid(x_surf, y_surf)

    fig.add_trace(
        go.Surface(
            z=tensor[0],
            x=X_surf,
            y=Y_surf,
            colorscale=colorscale,
            showscale=False,
            name="Surface",
        ),
        row=1,
        col=2,
    )

    # 3. All slices overview (bottom-left) - create a mosaic
    if depth <= 16:  # Only if not too many slices
        grid_size = int(np.ceil(np.sqrt(depth)))
        mosaic = np.zeros((grid_size * rows, grid_size * cols))

        for i in range(depth):
            row_start = (i // grid_size) * rows
            col_start = (i % grid_size) * cols
            mosaic[row_start : row_start + rows, col_start : col_start + cols] = tensor[
                i
            ]

        fig.add_trace(
            go.Heatmap(
                z=mosaic,
                colorscale=colorscale,
                showscale=False,
                hovertemplate="Mosaic View<br>Value: %{z:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    else:
        # For many slices, show a representative sample
        sample_indices = np.linspace(0, depth - 1, min(9, depth), dtype=int)
        grid_size = 3
        mosaic = np.zeros((grid_size * rows, grid_size * cols))

        for idx, i in enumerate(sample_indices):
            row_start = (idx // grid_size) * rows
            col_start = (idx % grid_size) * cols
            if row_start < grid_size * rows and col_start < grid_size * cols:
                mosaic[row_start : row_start + rows, col_start : col_start + cols] = (
                    tensor[i]
                )

        fig.add_trace(
            go.Heatmap(
                z=mosaic,
                colorscale=colorscale,
                showscale=False,
                hovertemplate="Sample Slices<br>Value: %{z:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # 4. Slice statistics (bottom-right)
    slice_means = [tensor[i].mean() for i in range(depth)]
    # slice_stds = [tensor[i].std() for i in range(depth)]
    # slice_mins = [tensor[i].min() for i in range(depth)]
    # slice_maxs = [tensor[i].max() for i in range(depth)]

    fig.add_trace(
        go.Bar(
            x=list(range(depth)),
            y=slice_means,
            name="Mean",
            marker_color="blue",
            opacity=0.7,
            hovertemplate="Slice: %{x}<br>Mean: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    # Add animation controls
    fig.frames = frames

    sliders = [
        dict(
            steps=[
                dict(
                    args=[
                        ["{}".format(i)],
                        {"frame": {"duration": animation_frame_duration}},
                    ],
                    label="{}".format(i),
                    method="animate",
                )
                for i in range(depth)
            ],
            active=0,
            x=0.1,
            y=0.02,
            len=0.8,
            currentvalue={"prefix": "Slice: "},
        )
    ]

    play_button = dict(
        type="buttons",
        showactive=False,
        x=0.05,
        y=0.02,
        buttons=[
            dict(
                label="▶",
                method="animate",
                args=[None, {"frame": {"duration": animation_frame_duration}}],
            ),
            dict(
                label="⏸", method="animate", args=[[None], {"frame": {"duration": 0}}]
            ),
        ],
    )

    fig.update_layout(
        title=f"{title} - 3D Tensor Interactive Explorer ({depth}×{rows}×{cols})",
        width=width,
        height=height,
        sliders=sliders,
        updatemenus=[play_button],
        scene=dict(xaxis_title="Column", yaxis_title="Row", zaxis_title="Value"),
    )

    # Update subplot titles and axes
    fig.update_xaxes(title="Column", row=1, col=1)
    fig.update_yaxes(title="Row", row=1, col=1, autorange="reversed")
    fig.update_xaxes(title="Slice", row=2, col=2)
    fig.update_yaxes(title="Mean Value", row=2, col=2)

    fig.show(renderer="browser")
