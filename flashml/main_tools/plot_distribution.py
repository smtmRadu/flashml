from typing import Literal, Union, Collection

def plot_dist(
    data: Union[dict, Collection[float], Collection[int], Collection[bool], Collection[str]],
    sort: Literal["ascending", "descending"] | None = None,
    top_n: int = None,
    bins: int = None,
    title: str = "Distribution",
    x_label: str = "Item",
    y_label: str = "Frequency",
    bar_color: str = "skyblue",
    rotation: int = 60,
    draw_details: bool = True,
    renderer='notebook'
):
    import plotly.graph_objects as go
    import plotly.io as pio
    import numpy as np

    MAX_XTICKS: int = 100
    pio.templates.default = "plotly_dark"

    def add_percentage_annotations(fig, x_vals, y_vals):
        total = sum(y_vals)
        y_max = max(y_vals)
        min_height_for_percent = y_max * 0.15
        for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            percent = y / total * 100 if total > 0 else 0
            if y >= min_height_for_percent:
                fig.add_annotation(
                    x=i,
                    y=y / 2,
                    text=f"{percent:.1f}%",
                    showarrow=False,
                    font=dict(size=13, color='white'),
                    textangle=-90,
                    xanchor="center",
                    yanchor="middle",
                    align="center",
                )

    def is_number(x):
        try:
            float(x)
            return True
        except Exception:
            return False

    # --- Handle dict input ---
    if isinstance(data, dict):
        freq_dict = data
        if not freq_dict:
            print("Warning: freq_dict is empty. Nothing to plot.")
            return
        items = list(freq_dict.items())
        if sort:
            if sort.lower() == "descending":
                items.sort(key=lambda x: (-x[1], str(x[0])))
            elif sort.lower() == "ascending":
                items.sort(key=lambda x: (x[1], str(x[0])))
            else:
                raise ValueError("sort must be 'ascending', 'descending', or None.")
        if top_n is not None:
            if not isinstance(top_n, int) or top_n <= 0:
                raise ValueError("top_n must be a positive integer.")
            items = items[:top_n]
        if not items:
            print("Warning: No items to plot after filtering.")
            return
        keys, values = zip(*items)
        str_keys = [str(k) for k in keys]
        valid_values = []
        none_count_dict = 0
        for v in values:
            if v is None:
                none_count_dict += 1
                valid_values.append(0)
            else:
                valid_values.append(v)
        total = sum(v for v in values if v is not None) + none_count_dict
        percentages = [v / total * 100 if total > 0 and v is not None else (1 / total * 100 if v is None else 0) for v in values]
        customdata = np.array(percentages).reshape(-1, 1)
        colors = []
        display_values = []
        for i, v in enumerate(values):
            if v is None:
                colors.append("gray")
                display_values.append(1)
            else:
                colors.append(bar_color)
                display_values.append(v)
        fig = go.Figure(
            data=go.Bar(
                x=str_keys,
                y=display_values,
                marker_color=colors,
                text=display_values if draw_details else None,
                textposition="outside" if draw_details else None,
                texttemplate="%{text:.2f}" if draw_details and any(isinstance(v, float) for v in display_values)
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
            title=f"{title} ({len(freq_dict)} elements{f', displayed {top_n}' if top_n else ''}{f', sorted {sort}' if sort else ''})",
            xaxis_title=x_label,
            yaxis_title=y_label,
            width=width,
            height=height,
            showlegend=False,
            xaxis=dict(
                tickangle=-rotation if rotation > 0 else 0,
                showticklabels=len(str_keys) <= MAX_XTICKS,
                showgrid=draw_details,
            ),
            yaxis=dict(
                showgrid=draw_details,
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
        fig.show(renderer=renderer)
        return

    # --- Handle sequence input (list, set, etc) ---
    data_list = list(data)
    if len(data_list) == 0:
        print("Warning: Data is empty. Nothing to plot.")
        return
    none_count = sum(1 for x in data_list if x is None)
    non_none_data = [x for x in data_list if x is not None]

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
                marker_color="gray",
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
            xaxis=dict(showgrid=draw_details),
            yaxis=dict(showgrid=draw_details),
            margin=dict(l=50, r=50, t=80, b=50),
        )
        if draw_details:
            add_percentage_annotations(fig, keys, values)
        fig.show(renderer=renderer)
        return

    all_numeric = all(is_number(x) for x in non_none_data)

    # --- Discrete/categorical plot (for non-numeric or numeric with few uniques) ---
    if not all_numeric or (bins is None and len(set(non_none_data)) <= 50):
        if not all_numeric and bins is not None:
            print("Warning: bins parameter ignored for non-numeric (categorical) data.")

        # Build frequency counts
        unique_vals, counts = np.unique(non_none_data, return_counts=True)
        unique_counts = dict(zip(unique_vals, counts))
        if none_count > 0:
            unique_counts[None] = none_count

        # SORTING BY BAR COUNT!
        if sort:
            if sort.lower() == "descending":
                sorted_items = sorted(unique_counts.items(), key=lambda x: (-x[1], str(x[0])))
            elif sort.lower() == "ascending":
                sorted_items = sorted(unique_counts.items(), key=lambda x: (x[1], str(x[0])))
            else:
                raise ValueError("sort must be 'ascending', 'descending', or None.")
        else:
            sorted_items = sorted(unique_counts.items(), key=lambda x: str(x[0]))  # fallback: label sort

        keys = [str(k) if k is not None else "None" for k, _ in sorted_items]
        values = [v for _, v in sorted_items]
        total = sum(values)
        percentages = [v / total * 100 if total > 0 else 0 for v in values]
        customdata = np.array(percentages).reshape(-1, 1)
        colors = ["gray" if k == "None" else bar_color for k in keys]
        fig = go.Figure(
            data=go.Bar(
                x=keys,
                y=values,
                marker_color=colors,
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
                showgrid=draw_details,
            ),
            yaxis=dict(
                showgrid=draw_details,
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
        fig.show(renderer=renderer)
        return

    # --- Numeric histogram mode (with bins) ---
    arr = np.array(non_none_data)
    bins_to_use = bins if bins and bins > 0 else 10
    hist, edges = np.histogram(arr, bins=bins_to_use)
    x_labels = []
    for i in range(len(edges) - 1):
        if i == len(edges) - 2:
            x_labels.append(f"[{edges[i]:.2f}, {edges[i+1]:.2f}]")
        else:
            x_labels.append(f"[{edges[i]:.2f}, {edges[i+1]:.2f})")
    values = list(hist)
    if none_count > 0:
        x_labels.insert(0, "None")
        values.insert(0, none_count)
    total = sum(values)
    percentages = [v / total * 100 if total > 0 else 0 for v in values]
    customdata = np.array(percentages).reshape(-1, 1)
    colors = []
    for label in x_labels:
        if label == "None":
            colors.append("gray")
        else:
            colors.append(bar_color)
    fig = go.Figure(
        data=go.Bar(
            x=x_labels,
            y=values,
            marker_color=colors,
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
            showgrid=draw_details,
        ),
        yaxis=dict(
            showgrid=draw_details,
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
    return
