from typing import Literal, Union, Collection

def plot_dist(
    data: Union[dict, Collection[float], Collection[int], Collection[bool], Collection[str]],
    title: str = "Distribution",
    sort: Literal["ascending", "descending"] | None = None,
    top_n: int | None = None,
    bins: int | None = None,
    xlabel: str = "Item",
    ylabel: str = "Frequency",
    bar_color: str = "skyblue",
    xlabel_rotation: int | Literal["auto"] = "auto", # degrees
    draw_details: bool = True,
    size: Literal["small", "big"] = "big",
    renderer='notebook',
):
    import plotly.graph_objects as go
    import plotly.io as pio
    import numpy as np
    MAX_XTICKS: int = 100
    pio.templates.default = "plotly_dark"
    
    scale = 1.0 if size == "big" else 0.66

    def add_quantile_boxes(fig, x_vals, y_vals, y_max):
        """
        Add transparent quantile boxes behind the bars to show data distribution.
        
        Args:
            fig: Plotly figure object
            x_vals: List of x-axis labels/positions
            y_vals: List of y-axis values (frequencies)
            y_max: Maximum y value for box height
        """
        if not y_vals or len(y_vals) <= 1:
            return
            
        total_samples = sum(y_vals)
        if total_samples == 0:
            return
            
        # Calculate cumulative sample counts
        cumulative_counts = []
        running_sum = 0
        for val in y_vals:
            running_sum += val
            cumulative_counts.append(running_sum)
        
        # Define all possible quantile thresholds
        all_quantiles = [
            (total_samples * 0.25, '25%', 'rgba(255, 100, 100, 0.15)'),
            (total_samples * 0.50, '50%', 'rgba(100, 150, 255, 0.15)'),
            (total_samples * 0.75, '75%', 'rgba(100, 255, 150, 0.15)'),
            (total_samples, '100%', 'rgba(200, 200, 0, 0.2)')
        ]
        
        # Find which bars contain each quantile
        quantile_bars = []
        for threshold, name, color in all_quantiles:
            for j, cum_count in enumerate(cumulative_counts):
                if cum_count >= threshold:
                    # Calculate precise position within the bar
                    if j == 0:
                        bar_start_count = 0
                    else:
                        bar_start_count = cumulative_counts[j-1]
                    
                    bar_total_samples = y_vals[j]
                    samples_needed_from_bar = threshold - bar_start_count
                    
                    if bar_total_samples > 0:
                        position_in_bar = samples_needed_from_bar / bar_total_samples
                        position = j - 0.5 + position_in_bar
                    else:
                        position = j - 0.5
                    
                    quantile_bars.append((position, name, color, threshold))
                    break
        
        # Group quantiles that fall in the same or adjacent bars
        if not quantile_bars:
            return
            
        # Sort by position
        quantile_bars.sort(key=lambda x: x[0])
        
        # Merge quantiles that are too close together
        merged_boxes = []
        current_start = -0.5
        current_end = quantile_bars[0][0]
        current_names = [quantile_bars[0][1]]
        current_color = quantile_bars[0][2]
        
        for i in range(1, len(quantile_bars)):
            next_pos = quantile_bars[i][0]
            next_name = quantile_bars[i][1]
            next_color = quantile_bars[i][2]
            
            # If the next quantile is very close (less than 0.8 bar widths away), merge them
            if next_pos - current_end < 0.8:
                current_end = next_pos
                current_names.append(next_name)
                # Use the color of the highest quantile in the group
                current_color = next_color
            else:
                # Finalize current box
                if current_end > current_start + 0.05:
                    # Calculate the percentage this box represents
                    if len(current_names) == 1:
                        label = "25%"  # Each quartile represents 25%
                    else:
                        # Multiple quartiles merged - calculate total percentage
                        quartile_count = len(current_names)
                        total_percentage = quartile_count * 25
                        label = f"{total_percentage}%"
                    
                    merged_boxes.append((current_start, current_end, label, current_color))
                
                # Start new box
                current_start = current_end
                current_end = next_pos
                current_names = [next_name]
                current_color = next_color
        
        # Add the final box
        if current_end > current_start + 0.05:
            # Calculate the percentage this box represents
            if len(current_names) == 1:
                label = "25%"  # Each quartile represents 25%
            else:
                # Multiple quartiles merged - calculate total percentage
                quartile_count = len(current_names)
                total_percentage = quartile_count * 25
                label = f"{total_percentage}%"
            
            merged_boxes.append((current_start, current_end, label, current_color))
        
        # Draw the merged boxes
        for start_pos, end_pos, label, color in merged_boxes:
            # Ensure we don't go beyond the data
            end_pos = min(end_pos, len(x_vals) - 0.5)
            
            if end_pos > start_pos:
                # Add transparent rectangle
                fig.add_shape(
                    type="rect",
                    x0=start_pos,
                    y0=0,
                    x1=end_pos,
                    y1=y_max * 1.1,
                    fillcolor=color,
                    line=dict(width=0),
                    layer="below"
                )
                
                # Add quantile label at the top
                fig.add_annotation(
                    x=(start_pos + end_pos) / 2,
                    y=y_max * 1.05,
                    text=label,
                    showarrow=False,
                    font=dict(size=11, color='rgba(255, 255, 255, 0.8)'),
                    xanchor="center",
                    yanchor="middle",
                    bgcolor='rgba(0, 0, 0, 0.4)',
                    bordercolor='rgba(255, 255, 255, 0.3)',
                    borderwidth=1
                )

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
        if xlabel_rotation == "auto":
            xlabel_rotation = min(len(data) * 2, 90)
            if xlabel_rotation <= 10:
                xlabel_rotation = 0
        freq_dict = data
        if not freq_dict:
            print("Warning: freq_dict is empty. Nothing to plot.")
            return
        
        # Calculate statistics if all values are numeric
        dict_mean = None
        dict_std = None
        if all(isinstance(v, (int, float)) and v is not None for v in freq_dict.values()):
            values_for_stats = [v for v in freq_dict.values() if v is not None]
            if values_for_stats:
                dict_mean = np.mean(values_for_stats)
                dict_std = np.std(values_for_stats)
        
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
        width = int(max(1100, min(1200, num_items * 40)) * scale)
        height = int(max(500, min(800, 300 + num_items * 5)) * scale)
        
        title_suffix = f", mean={dict_mean:.2f}, std={dict_std:.2f}" if dict_mean is not None else ""
        fig.update_layout(
            title=f"{title} ({len(freq_dict)} elements{f', displayed {top_n}' if top_n else ''}{f', sorted {sort}' if sort else ''}{title_suffix})",
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=width,
            height=height,
            showlegend=False,
            xaxis=dict(
                tickangle=-xlabel_rotation if xlabel_rotation > 0 else 0,
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
            add_quantile_boxes(fig, str_keys, values, max(display_values) if display_values else 1)
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
    
    # Calculate statistics for numeric data
    mean_val = None
    std_val = None
    if non_none_data and all(is_number(x) for x in non_none_data):
        numeric_data = [float(x) for x in non_none_data]
        mean_val = np.mean(numeric_data)
        std_val = np.std(numeric_data)

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
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=int(800 * scale),
            height=int(500 * scale),
            showlegend=False,
            xaxis=dict(showgrid=draw_details),
            yaxis=dict(showgrid=draw_details),
            margin=dict(l=50, r=50, t=80, b=50),
        )
        if draw_details:
            # No quantile boxes for single value
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

        # SORTING BY BAR COUNT OR LABEL!
        if sort:
            if sort.lower() == "descending":
                sorted_items = sorted(unique_counts.items(), key=lambda x: (-x[1], str(x[0])))
            elif sort.lower() == "ascending":
                sorted_items = sorted(unique_counts.items(), key=lambda x: (x[1], str(x[0])))
            else:
                raise ValueError("sort must be 'ascending', 'descending', or None.")
        else:
            # Fixed fallback sorting: sort numerically if all non-None values are numeric
            non_none_keys = [k for k in unique_counts.keys() if k is not None]
            if non_none_keys and all(is_number(k) for k in non_none_keys):
                # Sort numerically for numeric keys
                sorted_items = sorted(unique_counts.items(), key=lambda x: (x[0] is None, x[0] if x[0] is not None else float('inf')))
            else:
                # Sort as strings for non-numeric keys
                sorted_items = sorted(unique_counts.items(), key=lambda x: str(x[0]))

        keys = [str(k) if k is not None else "None" for k, _ in sorted_items]
        values = [v for _, v in sorted_items]
        total = sum(values)
        if xlabel_rotation == "auto":
            xlabel_rotation = min(len(keys) * 2, 90)
            if xlabel_rotation <= 10:
                xlabel_rotation = 0
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
        width = int(max(1100, min(1200, num_items * 40)) * scale)
        height = int(max(500, min(800, 300 + num_items * 5)) * scale)
        unique_count = len(unique_vals) + (1 if none_count > 0 else 0)
        
        title_suffix = f", mean={mean_val:.2f}, std={std_val:.2f}" if mean_val is not None else ""
        fig.update_layout(
            title=f"{title} ({len(data_list)} elements, {unique_count} unique{title_suffix})",
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=width,
            height=height,
            showlegend=False,
            xaxis=dict(
                tickangle=-xlabel_rotation if xlabel_rotation > 0 else 0,
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
            add_quantile_boxes(fig, keys, values, max(values) if values else 1)
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
    width = int(max(1100, min(1200, num_items * 40)) * scale)
    height = int(max(500, min(800, 300 + num_items * 5)) * scale)
    if xlabel_rotation == "auto":
        xlabel_rotation = min(len(x_labels) * 2, 90)
        if xlabel_rotation <= 10:
            xlabel_rotation = 0
    
    title_suffix = f", mean={mean_val:.2f}, std={std_val:.2f}" if mean_val is not None else ""
    fig.update_layout(
        title=f"{title} ({len(data_list)} elements{title_suffix})",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=width,
        height=height,
        showlegend=False,
        xaxis=dict(
            tickangle=-xlabel_rotation if xlabel_rotation > 0 else 0,
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
        add_quantile_boxes(fig, x_labels, values, max(values) if values else 1)
        add_percentage_annotations(fig, x_labels, values)
    fig.show(renderer=renderer)
    return