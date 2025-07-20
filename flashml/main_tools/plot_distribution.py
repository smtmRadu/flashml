from typing import Literal, Union, Collection

def plot_dist(
    data: Union[dict, Collection[float], Collection[int], Collection[bool], Collection[str]],
    sort: Literal["ascending", "descending"] | None = None,
    top_n: int = None,
    title: str = "Distribution",
    x_label: str = "Item",
    y_label: str = "Frequency",
    bar_color: str = "skyblue",
    rotation: int = 90,
    draw_details: bool = True,
    grid: bool = True,
    bins: int = 10,
    renderer='vscode'
):
    import plotly.graph_objects as go
    import plotly.io as pio
    import numpy as np

    MAX_XTICKS: int = 100
    pio.templates.default = "plotly_dark"

    def add_percentage_annotations(fig, x_vals, y_vals):
        total = sum(y_vals)
        y_max = max(y_vals)
        min_height_for_percent = y_max * 0.15  # Only annotate if bar is at least 15% of max height (let it 15, it is the best)

        for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            percent = y / total * 100 if total > 0 else 0
            if y >= min_height_for_percent:
                fig.add_annotation(
                    x=i,  # Use index position instead of string value
                    y=y / 2,
                    text=f"{percent:.1f}%",
                    showarrow=False,
                    font=dict(size=13, color='white'),
                    textangle=-90,
                    xanchor="center",
                    yanchor="middle",
                    align="center",
                )
    
    # If data is dict
    if isinstance(data, dict):
        freq_dict = data
        if not freq_dict:
            print("Warning: freq_dict is empty. Nothing to plot.")
            return

        items = list(freq_dict.items())

        if sort:
            if sort.lower() == "descending":
                items.sort(key=lambda x: x[1], reverse=True)
            elif sort.lower() == "ascending":
                items.sort(key=lambda x: x[1])
            else:
                raise ValueError("sort_values must be 'ascending', 'descending', or None.")

        if top_n is not None:
            if not isinstance(top_n, int) or top_n <= 0:
                raise ValueError("top_n must be a positive integer.")
            items = items[:top_n]

        if not items:
            print("Warning: No items to plot after filtering.")
            return

        keys, values = zip(*items)
        str_keys = [str(k) for k in keys]
        total = sum(values)
        percentages = [v / total * 100 if total > 0 else 0 for v in values]
        customdata = np.array(percentages).reshape(-1, 1)

        fig = go.Figure(
            data=go.Bar(
                x=str_keys,
                y=values,
                marker_color=bar_color,
                text=values if draw_details else None,
                textposition="outside" if draw_details else None,
                texttemplate="%{text:.2f}" if draw_details and any(isinstance(v, float) for v in values)
                            else "%{text}" if draw_details else None,
                customdata=customdata,
                hovertemplate=(
                    "Item: %{x}<br>"
                    "No. samples: %{y}<br>"
                    "Percentage: %{customdata[0]:.1f}%<extra></extra>"
                )
            )
        )
        num_items = len(str_keys)
        width = max(1100, min(1200, num_items * 40))
        height = max(500, min(800, 300 + num_items * 5))
        fig.update_layout(
            title=f"{title} ({len(freq_dict)} elements{f", displayed {top_n}" if top_n else ""}{f", sorted {sort}" if sort else ""})",
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
                b=10 if len(str_keys) <= MAX_XTICKS else 50,
            ),
        )
        if draw_details:
            add_percentage_annotations(fig, str_keys, values)
        return fig

    # If data is a collection
    data_list = list(data)
    
    if len(data_list) == 0:
        print("Warning: Data is empty. Nothing to plot.")
        return

    # Count None values separately
    none_count = sum(1 for x in data_list if x is None)
    non_none_data = [x for x in data_list if x is not None]
    
    # If all data is None
    if len(non_none_data) == 0:
        keys = ["None"]
        values = [none_count]
        total = none_count
        percentages = [100.0]
        customdata = np.array(percentages).reshape(-1, 1)

        fig = go.Figure(
            data=go.Bar(
                x=keys,
                y=values,
                marker_color=bar_color,
                text=values if draw_details else None,
                textposition="outside" if draw_details else None,
                texttemplate="%{text}",
                customdata=customdata,
                hovertemplate=(
                    "Item: %{x}<br>"
                    "No. samples: %{y}<br>"
                    "Percentage: %{customdata[0]:.1f}%<extra></extra>"
                )
            )
        )
        fig.update_layout(
            title=f"{title} ({len(data_list)} elements, 1 unique)",
            xaxis_title=x_label,
            yaxis_title=y_label,
            width=800,
            height=500,
            showlegend=False,
            xaxis=dict(showgrid=grid),
            yaxis=dict(showgrid=grid),
            margin=dict(l=50, r=50, t=80, b=50),
        )
        if draw_details:
            add_percentage_annotations(fig, keys, values)
        return fig

    # Process non-None data
    try:
        arr = np.array(non_none_data)
    except Exception as e:
        raise TypeError("If not a dict, data must be a collection of numbers.") from e

    unique_vals = np.unique(arr)
    
    # Check if data should be treated as discrete values
    # Use discrete mode if:
    # 1. Number of unique values is reasonable (< 50 by default)
    # 2. OR data consists of integers/booleans/strings (not continuous floats)
    # 3. OR we have None values (mixed discrete case)
    is_discrete = (len(unique_vals) <= 50 or 
                   all(isinstance(x, (int, bool, str, np.integer)) for x in arr) or
                   all(float(x).is_integer() for x in arr if isinstance(x, (int, float))) or
                   none_count > 0)
    
    if is_discrete:
        # Discrete mode - treat each unique value as a separate bar
        unique_counts = {val: int(np.sum(arr == val)) for val in unique_vals}
        
        # Add None count if present
        if none_count > 0:
            unique_counts[None] = none_count
        
        def sort_key(x):
            if x[0] is None:
                return (-1, 0)  # None comes first
            elif isinstance(x[0], bool):
                return (0, x[0])
            else:
                return (1, x[0])
        
        sorted_items = sorted(unique_counts.items(), key=sort_key)
        keys = [str(k) if k is not None else "None" for k, _ in sorted_items]
        values = [v for _, v in sorted_items]
        total = sum(values)
        percentages = [v / total * 100 if total > 0 else 0 for v in values]
        customdata = np.array(percentages).reshape(-1, 1)

        fig = go.Figure(
            data=go.Bar(
                x=keys,
                y=values,
                marker_color=bar_color,
                text=values if draw_details else None,
                textposition="outside" if draw_details else None,
                texttemplate="%{text:.2f}" if draw_details and any(isinstance(v, float) for v in values)
                            else "%{text}" if draw_details else None,
                customdata=customdata,
                hovertemplate=(
                    "Item: %{x}<br>"
                    "No. samples: %{y}<br>"
                    "Percentage: %{customdata[0]:.1f}%<extra></extra>"
                )
            )
        )
        num_items = len(keys)
        width = max(1100, min(1200, num_items * 40))
        height = max(500, min(800, 300 + num_items * 5))
        unique_count = len(unique_vals) + (1 if none_count > 0 else 0)
        fig.update_layout(
            title=f"{title} ({len(data_list)} elements, {unique_count} unique)",
            xaxis_title=x_label,
            yaxis_title=y_label,
            width=width,
            height=height,
            showlegend=False,
            xaxis=dict(
                tickangle=-rotation if rotation > 0 else 0,
                showticklabels=len(keys) <= MAX_XTICKS,
                showgrid=grid,
            ),
            yaxis=dict(
                showgrid=grid,
            ),
            margin=dict(
                l=50,
                r=50,
                t=80,
                b=10 if len(keys) <= MAX_XTICKS else 50,
            ),
        )
        if draw_details:
            add_percentage_annotations(fig, keys, values)
        return fig

    # Histogram mode for continuous data (no None values here since we filtered them out)
    import math
    bins_to_use = bins if bins and bins > 0 else min(10, math.ceil(len(arr)/5))
    hist, edges = np.histogram(arr, bins=bins_to_use)
    x_labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(edges) - 1)]
    values = list(hist)
    
    # Add None as a separate bar if present
    if none_count > 0:
        x_labels.append("None")
        values.append(none_count)
    
    total = sum(values)
    percentages = [v / total * 100 if total > 0 else 0 for v in values]
    customdata = np.array(percentages).reshape(-1, 1)

    fig = go.Figure(
        data=go.Bar(
            x=x_labels,
            y=values,
            marker_color=bar_color,
            text=values if draw_details else None,
            textposition="outside" if draw_details else None,
            texttemplate="%{text:.2f}" if draw_details and any(isinstance(v, float) for v in values)
                        else "%{text}" if draw_details else None,
            customdata=customdata,
            hovertemplate=(
                "Range: %{x}<br>"
                "No. samples: %{y}" + f" (out of {total})<br>"
                "Percentage: %{customdata[0]:.1f}%<extra></extra>"
            )
        )
    )
    num_items = len(x_labels)
    width = max(1100, min(1200, num_items * 40))
    height = max(500, min(800, 300 + num_items * 5))
    fig.update_layout(
        title=f"{title} ({len(data_list)} elements)",
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=width,
        height=height,
        showlegend=False,
        xaxis=dict(
            tickangle=-rotation if rotation > 0 else 0,
            showticklabels=len(x_labels) <= MAX_XTICKS,
            showgrid=grid,
        ),
        yaxis=dict(
            showgrid=grid,
        ),
        margin=dict(
            l=50,
            r=50,
            t=80,
            b=10 if len(x_labels) <= MAX_XTICKS else 50,
        ),
    )
    if draw_details:
        add_percentage_annotations(fig, x_labels, values)
        
    fig.show(renderer=renderer)
    # return fig
    # we plot it because in notebooks when we want to plot twice we see only the last one