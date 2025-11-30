## begin of file


def plot_tensor(
    tensor,
    title: str = "",
    colorscale: str = "Viridis",
    show_values: bool = True,
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
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.templates.default = "plotly_dark"
    # Convert input to numpy array
    original_device = "cpu"
    try:
        original_device = tensor.device
    except:
        pass

    original_dtype = tensor.dtype

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
            tensor, title, colorscale, width, height, original_device, original_dtype
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
            original_dtype,
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
            original_dtype,
        )
    elif dim == 3:
        return _plot_3d_tensor(
            tensor, title, colorscale, show_values, original_device, original_dtype
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
            title=f"{title} Unsupported Dimension ({dim}D) | device: {original_device} | dtype: {original_dtype}",
            width=width,
            height=height,
        )
        return fig


def _plot_0d_tensor(
    tensor, title, colorscale, width, height, original_device, original_dtype
):
    """Plot 0D tensor (scalar) with gauge and info card"""
    import plotly.express as px
    import plotly.graph_objects as go
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
        title=f"{title} 0D Tensor (Scalar) | device: {original_device} | dtype: {original_dtype}",
        width=width,
        height=height,
    )
    return fig


def _plot_1d_tensor(
    tensor,
    title,
    colorscale,
    show_values,
    width,
    height,
    original_device,
    original_dtype,
):
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

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
            text=f"{title} 1D Tensor ({len(tensor)} elements) | device: {original_device} | dtype: {original_dtype}",
            x=0.5,  # Center the title horizontally
            xanchor="center",
        ),
        width=width,
        height=height,
        showlegend=False,
    )
    return fig


def _plot_2d_tensor(
    tensor,
    title,
    colorscale,
    show_values,
    width,
    height,
    original_device,
    original_dtype,
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
            text=f"{title} 2D Tensor ({rows}×{cols}) | device: {original_device} | dtype: {original_dtype}",
            x=0.5,  # Center the title horizontally
            xanchor="center",
        ),
        xaxis_title="Column Index",
        yaxis_title="Row Index",
        width=width,
        height=height,
        yaxis=dict(autorange="reversed"),  # Match matrix convention
    )
    return fig


def _plot_3d_tensor(
    tensor, title, colorscale, show_values, original_device, original_dtype
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
            text=f"{title} (Slice 0) 3D Tensor ({depth}×{rows}×{cols}) | device: {original_device} | dtype: {original_dtype}",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Column",
        yaxis_title="Row",
        yaxis=dict(autorange="reversed"),
        sliders=sliders,
    )

    return fig


def plot_image_tensor(
    tensors, 
    max_figures: int = 32,
    titles: list = None,
    image_size: int = 200
):
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import torch
    import numpy as np
    import math

    """
    Plot image tensors in a grid layout using Plotly.
    
    Args:
        tensors: Single tensor of shape (B, C, H, W) or list of tensors
        N: Maximum number of samples from batch to plot (default: 32)
        titles: Optional list of titles for each tensor group (e.g., ['Input', 'Target', 'Output'])
        image_size: Size of each individual image in pixels (default: 200)
    
    Returns:
        plotly.graph_objects.Figure: The created figure
    """
    # Normalize tensors to be a list
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    
    # Ensure all tensors are on CPU and detached
    tensors = [t.detach().cpu() for t in tensors]
    
    # Get batch size and limit to N
    batch_size = min(tensors[0].shape[0], max_figures)
    num_tensor_groups = len(tensors)
    
    # Helper function to convert tensor to displayable format
    def tensor_to_image(tensor):
        """Convert tensor (C, H, W) to numpy array for display."""
        if tensor.shape[0] == 1:  # Grayscale
            img = tensor.squeeze(0).numpy()
            # Normalize to [0, 255] for display
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            return img
        elif tensor.shape[0] == 3:  # RGB
            img = tensor.permute(1, 2, 0).numpy()
            # Clip to [0, 1] range and convert to [0, 255]
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            return img
        else:
            # For other channel counts, just show first channel
            img = tensor[0].numpy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            return img
    
    if num_tensor_groups == 1:
        # Single tensor: arrange in nearly-square grid
        cols = math.ceil(math.sqrt(batch_size))
        rows = math.ceil(batch_size / cols)
        
        fig = make_subplots(
            rows=rows, cols=cols,
            horizontal_spacing=0.01,
            vertical_spacing=0.01
        )
        
        for idx in range(batch_size):
            row = idx // cols + 1
            col = idx % cols + 1
            
            img = tensor_to_image(tensors[0][idx])
            
            fig.add_trace(
                go.Image(z=img),
                row=row, col=col
            )
        
        # Update layout
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        
        fig.update_layout(
            height=rows * image_size,
            width=cols * image_size,
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10)
        )
            
    else:
        # Multiple tensors: each row shows corresponding images from all tensors
        rows = batch_size
        cols = num_tensor_groups
        
        # Create subplot titles if provided
        subplot_titles = []
        if titles:
            # Only add titles to the first row
            for row_idx in range(rows):
                for col_idx in range(cols):
                    if row_idx == 0 and col_idx < len(titles):
                        subplot_titles.append(titles[col_idx])
                    else:
                        subplot_titles.append("")
        else:
            subplot_titles = None
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.01,
            vertical_spacing=0.01
        )
        
        for row_idx in range(batch_size):
            for col_idx, tensor in enumerate(tensors):
                img = tensor_to_image(tensor[row_idx])
                
                fig.add_trace(
                    go.Image(z=img),
                    row=row_idx + 1, col=col_idx + 1
                )
        
        # Update layout
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        
        fig.update_layout(
            height=rows * image_size,
            width=cols * image_size,
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10)
        )
    
    return fig
