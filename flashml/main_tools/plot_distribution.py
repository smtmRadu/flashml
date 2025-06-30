from typing import Literal


def plot_dist(
    freq_dict: dict,
    sort_values: Literal["ascending", "descending"] | None = None,
    top_n: int = None,
    title: str = "Distribution",
    x_label: str = "Item",
    y_label: str = "Frequency",
    bar_color: str = "skyblue",
    rotation: int = 90,
    show_values_on_top_of_the_bar: bool = False,
    grid: bool = False,
) -> None:
    """
    Plots a bar chart distribution from a frequency dictionary using Plotly.

    Args:
        freq_dict: Dictionary with items as keys and frequencies as values. (e.g. dict = {'apple': 5, 'banana': 3})
        sort_values: How to sort the items. Can be 'ascending', 'descending', or None (default).
        top_n: Display only the top N items.
        title: Title of the plot.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        bar_color: Color of the bars.
        rotation: Rotation angle for x-axis tick labels.
        max_xticks: Maximum number of x-axis ticks to display labels for.
                     If exceeded, labels are hidden.
        show_values_on_top_of_the_bar: If True, displays the frequency value on top of each bar.
        grid: If True, adds a grid to the plot.
        renderer: Renderer for displaying the plot (default: "vscode").
    """

    import plotly.graph_objects as go
    import plotly.io as pio

    MAX_XTICKS: int = 100
    pio.templates.default = "plotly_dark"

    if not isinstance(freq_dict, dict):
        raise TypeError("freq_dict must be a dictionary.")
    if not freq_dict:
        print("Warning: freq_dict is empty. Nothing to plot.")
        return

    items = list(freq_dict.items())

    if sort_values:
        if sort_values.lower() == "descending":
            items.sort(key=lambda x: x[1], reverse=True)
        elif sort_values.lower() == "ascending":
            items.sort(key=lambda x: x[1])
        else:
            raise ValueError("sort_values must be 'ascending', 'descending', or None.")

    if top_n is not None:
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer.")
        items = items[:top_n]

    if not items:
        print(
            "Warning: No items to plot after filtering (e.g., empty original dict or top_n applied to empty list)."
        )
        return

    keys, values = zip(*items)
    str_keys = [str(k) for k in keys]  # Ensure keys are strings for plotting

    # Create the bar chart
    fig = go.Figure(
        data=go.Bar(
            x=str_keys,
            y=values,
            marker_color=bar_color,
            text=values if show_values_on_top_of_the_bar else None,
            textposition="outside" if show_values_on_top_of_the_bar else None,
            texttemplate="%{text:.2f}"
            if show_values_on_top_of_the_bar
            and any(isinstance(v, float) for v in values)
            else "%{text}"
            if show_values_on_top_of_the_bar
            else None,
        )
    )

    # Auto-adjust figure size based on number of items
    num_items = len(str_keys)
    width = max(1100, min(1200, num_items * 40))  # Scale width with number of items
    height = max(500, min(800, 300 + num_items * 5))  # Scale height modestly

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=width,
        height=height,
        showlegend=False,
        xaxis=dict(
            tickangle=-rotation if rotation > 0 else 0,
            showticklabels=len(str_keys) <= MAX_XTICKS,
            showgrid=grid,
        ),
        yaxis=dict(
            showgrid=grid,
        ),
        margin=dict(
            l=50,
            r=50,
            t=80,
            b=10
            if len(str_keys) <= MAX_XTICKS
            else 50,  # Adjust bottom margin for rotated labels
        ),
    )
    return fig
