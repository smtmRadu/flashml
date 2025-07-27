def plot_graph(
    values: list | tuple[list] | list[list],
    steps: list = None,
    x_label: str = "x",
    y_label: str = "y",
    color: str | list[str] = None,
    linestyle: str | list[str] = "solid",
    marker: str | list[str] = "",
):
    from collections.abc import Iterable

    import plotly.graph_objects as go
    from plotly.colors import qualitative

    """
    Plot single or multiple line graphs using Plotly with dark theme.

    Args:
        values (list|tuple[list]|list[list]): Single list for one line or list of lists for multiple lines
        steps (list): Values for x-axis
        x_label (str): Label for x-axis
        y_label (str): Label for y-axis
        color (str|list[str]): Single color or list of colors for multiple lines. If None, uses automatic colors
        linestyle (str|list[str]): Single line style or list of styles for multiple lines
        marker (str|list[str]): Marker style(s)
        blocking (bool): Ignored (kept for compatibility)
    
    Returns:
        plotly.graph_objects.Figure: The plotly figure object
    """

    # Validate steps length
    if steps is not None:
        assert (
            len(steps) == len(values)
            if not isinstance(values[0], Iterable)
            else len(steps) == len(values[0])
        ), "Length of steps must be equal to length of values"

    # Create figure with dark theme
    fig = go.Figure()

    fig.update_yaxes(
        tickformat=".2e",         # 2 decimals in scientific notation
        exponentformat="e"        # Forces e-notation, not SI prefixes
    )
    
    # Map matplotlib linestyles to plotly dash styles
    linestyle_map = {
        "-": "solid",
        "--": "dash",
        "-.": "dashdot",
        ":": "dot",
        "solid": "solid",
        "dash": "dash",
        "dashdot": "dashdot",
        "dot": "dot",
    }

    # Map matplotlib markers to plotly symbols
    marker_map = {
        "o": "circle",
        "s": "square",
        "^": "triangle-up",
        "v": "triangle-down",
        "D": "diamond",
        "+": "cross",
        "x": "x",
        "*": "star",
        "": None,
        None: None,
    }

    # Check if we have multiple series
    if isinstance(values[0], Iterable):
        if steps is None:
            steps = list(range(len(values[0])))

        n_lines = len(values)

        # Handle colors
        if color is None:
            colors = qualitative.Plotly[:n_lines]
            if n_lines > len(qualitative.Plotly):
                colors = qualitative.Set3[:n_lines]
        else:
            colors = [color] * n_lines if isinstance(color, str) else color
            colors = colors[:n_lines] + qualitative.Plotly[: (n_lines - len(colors))]

        # Handle line styles
        styles = [linestyle] * n_lines if isinstance(linestyle, str) else linestyle
        styles = styles[:n_lines] + ["solid"] * (n_lines - len(styles))

        # Handle markers
        markers = [marker] * n_lines if isinstance(marker, str) else marker
        markers = markers[:n_lines] + [""] * (n_lines - len(markers))

        # Add traces for each series
        for i, each in enumerate(values):
            dash_style = linestyle_map.get(styles[i], "solid")
            marker_symbol = marker_map.get(markers[i], None)

            # line_width = min(100 / len(values), 3)

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=each,
                    mode="lines+markers" if marker_symbol else "lines",
                    line=dict(color=colors[i], dash=dash_style, width=2),
                    marker=dict(symbol=marker_symbol, size=6)
                    if marker_symbol
                    else None,
                    name=f"Series {i + 1}",
                )
            )
    else:
        if steps is None:
            steps = list(range(len(values)))

        dash_style = linestyle_map.get(linestyle, "solid")
        marker_symbol = marker_map.get(marker, None)
        line_color = "blue" if color is None else color

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=values,
                mode="lines+markers" if marker_symbol else "lines",
                line=dict(color=line_color, dash=dash_style, width=2),
                marker=dict(symbol=marker_symbol, size=6) if marker_symbol else None,
                showlegend=False,
            )
        )
    fig.update_layout(
        template="plotly_dark",
        title=f"{y_label} vs {x_label}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=isinstance(values[0], Iterable),
        hovermode="x unified",
        width=800,
        height=400,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.3)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.3)")

    return fig
