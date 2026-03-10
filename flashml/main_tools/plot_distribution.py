from typing import Literal, Union, Collection


def _detect_plot_renderer() -> Literal["notebook", "console"]:
    try:
        from IPython import get_ipython
    except Exception:
        return "console"

    shell = get_ipython()
    if shell is None:
        return "console"

    shell_name = shell.__class__.__name__
    if shell_name != "ZMQInteractiveShell":
        return "console"

    try:
        import ipykernel  # noqa: F401
    except Exception:
        return "console"

    return "notebook"


def _plot_dist_console(
    labels: Collection[str],
    values: Collection[float | int],
    percentages: Collection[float],
    title: str,
    xlabel: str,
    ylabel: str,
    bar_color: str = "skyblue",
    size: Literal["small", "big"] = "big",
):
    import numbers
    import os
    import re
    import shutil
    import sys
    import textwrap

    labels = [str(label) for label in labels]
    values = list(values)
    percentages = list(percentages)

    def format_number(value):
        if isinstance(value, bool):
            return str(int(value))
        if isinstance(value, numbers.Integral):
            return str(int(value))
        if isinstance(value, numbers.Real):
            numeric_value = float(value)
            if numeric_value.is_integer():
                return str(int(numeric_value))
            return f"{numeric_value:.2f}"
        return str(value)

    def wrap_label(label: str, width: int, max_lines: int = 3):
        wrapped = textwrap.wrap(label, width=width) or [label]
        if len(wrapped) <= max_lines:
            return wrapped

        head = wrapped[: max_lines - 1]
        remainder = "".join(wrapped[max_lines - 1 :])
        tail = remainder[: max(1, width - 3)] + "..."
        return [*head, tail]

    def ansi_color_prefix(color_name: str):
        if os.getenv("NO_COLOR") or not sys.stdout.isatty():
            return ""

        if isinstance(color_name, str) and len(color_name) == 7 and color_name.startswith("#"):
            try:
                red = int(color_name[1:3], 16)
                green = int(color_name[3:5], 16)
                blue = int(color_name[5:7], 16)
                return f"\033[38;2;{red};{green};{blue}m"
            except ValueError:
                return ""

        color_lookup = {
            "black": "\033[30m",
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "white": "\033[37m",
            "gray": "\033[90m",
            "grey": "\033[90m",
            "orange": "\033[38;5;214m",
            "pink": "\033[38;5;212m",
            "purple": "\033[38;5;141m",
            "skyblue": "\033[38;5;117m",
        }
        return color_lookup.get(str(color_name).lower(), "")

    def colorize_with(text: str, color_name: str):
        prefix = ansi_color_prefix(color_name)
        if not prefix or not text:
            return text
        return f"{prefix}{text}\033[0m"

    def colorize(text: str):
        return colorize_with(text, bar_color)

    def marginize(text: str):
        return colorize_with(text, "gray")

    ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")

    def visible_len(text: str):
        return len(ansi_pattern.sub("", text))

    def visible_pad(text: str, width: int, align: Literal["left", "center", "right"] = "left"):
        padding = max(0, width - visible_len(text))
        if align == "center":
            left_padding = padding // 2
            right_padding = padding - left_padding
            return (" " * left_padding) + text + (" " * right_padding)
        if align == "right":
            return (" " * padding) + text
        return text + (" " * padding)

    def add_total_to_title_text(text: str, total):
        total_text = f"total={format_number(total)}"
        if "total=" in text:
            return text
        last_open = text.rfind("(")
        if last_open != -1 and text.endswith(")"):
            inner = text[last_open + 1 : -1].strip()
            separator = ", " if inner else ""
            return f"{text[:last_open + 1]}{inner}{separator}{total_text})"
        return f"{text} ({total_text})"

    def split_title_metadata(text: str):
        stripped = text.strip()
        last_open = stripped.rfind("(")
        if last_open == -1 or not stripped.endswith(")"):
            return stripped, []

        title_text = stripped[:last_open].rstrip()
        metadata_text = stripped[last_open + 1 : -1].strip()
        metadata_parts = [part.strip() for part in metadata_text.split(",") if part.strip()]
        return title_text or stripped, metadata_parts

    def render_metadata_lines(parts: list[str], width: int):
        if not parts:
            return []

        palette = ["skyblue", "yellow", "green", "magenta", "cyan", "orange", "pink", "purple"]
        token_groups: list[list[str]] = []
        current_group: list[str] = []
        current_len = 1  # opening parenthesis on the first line
        tokens = [f"{part}," if index < len(parts) - 1 else part for index, part in enumerate(parts)]

        for index, token in enumerate(tokens):
            extra_space = 1 if current_group else 0
            closing_paren = 1 if index == len(tokens) - 1 else 0
            proposed_len = current_len + extra_space + len(token) + closing_paren
            if current_group and proposed_len > width:
                token_groups.append(current_group)
                current_group = [token]
                current_len = len(token)
            else:
                current_group.append(token)
                current_len += extra_space + len(token)

        if current_group:
            token_groups.append(current_group)

        rendered_lines = []
        color_index = 0
        for line_index, group in enumerate(token_groups):
            rendered_line = marginize("(") if line_index == 0 else ""
            for token_index, token in enumerate(group):
                if token_index > 0:
                    rendered_line += " "
                token_has_comma = token.endswith(",")
                part_text = token[:-1] if token_has_comma else token
                rendered_line += colorize_with(part_text, palette[color_index % len(palette)])
                if token_has_comma:
                    rendered_line += marginize(",")
                color_index += 1
            if line_index == len(token_groups) - 1:
                rendered_line += marginize(")")
            rendered_lines.append(rendered_line)

        return rendered_lines

    filled_bar_char = "▓"
    partial_bar_char = "▓" # keep same as filled
    empty_bar_char = "░"
    frame_vertical_char = "│"
    frame_horizontal_char = "─"
    frame_top_left_char = "╭"
    frame_top_right_char = "╮"
    frame_middle_left_char = "├"
    frame_middle_right_char = "┤"
    frame_bottom_left_char = "╰"
    frame_bottom_right_char = "╯"
    frame_top_join_char = "┬"
    frame_bottom_join_char = "┴"
    frame_cross_char = "┼"
    column_separator = marginize("│")

    def build_positive_track(length: int, width: int, has_value: bool):
        if length > 0:
            return colorize(filled_bar_char * length) + (empty_bar_char * (width - length))
        if has_value and width > 0:
            return colorize(partial_bar_char) + (empty_bar_char * (width - 1))
        return empty_bar_char * width

    def build_negative_track(length: int, width: int, has_value: bool):
        if length > 0:
            return (empty_bar_char * (width - length)) + colorize(filled_bar_char * length)
        if has_value and width > 0:
            return (empty_bar_char * (width - 1)) + colorize(partial_bar_char)
        return empty_bar_char * width

    default_width = 110 if size == "big" else 88
    min_width = 70 if size == "big" else 58
    terminal_width = shutil.get_terminal_size((default_width, 24)).columns
    total_width = max(min_width, min(terminal_width, default_width))
    left_margin_plain = f"{frame_vertical_char}  "
    right_margin_plain = f"  {frame_vertical_char}"
    canvas_width = max(24, total_width - len(left_margin_plain) - len(right_margin_plain))
    left_margin = marginize(left_margin_plain)
    right_margin = marginize(right_margin_plain)
    border_left = marginize(frame_vertical_char)
    border_right = marginize(frame_vertical_char)

    def frame_line(text: str = "", align: Literal["left", "center"] = "left"):
        return f"{left_margin}{visible_pad(text, canvas_width, align=align)}{right_margin}"

    def frame_rule_line(
        kind: Literal["top", "middle", "bottom"],
        connector_positions: Collection[int] | None = None,
    ):
        rule_width = max(1, total_width - 2)
        if kind == "top":
            left_char = frame_top_left_char
            right_char = frame_top_right_char
            join_char = frame_top_join_char
        elif kind == "bottom":
            left_char = frame_bottom_left_char
            right_char = frame_bottom_right_char
            join_char = frame_bottom_join_char
        else:
            left_char = frame_middle_left_char
            right_char = frame_middle_right_char
            join_char = frame_top_join_char

        rule_chars = [frame_horizontal_char] * rule_width
        for position in connector_positions or []:
            if 0 <= position < rule_width:
                rule_chars[position] = join_char

        return (
            f"{marginize(left_char)}"
            f"{marginize(''.join(rule_chars))}"
            f"{marginize(right_char)}"
        )

    def get_rule_connector_positions():
        left_inner_padding = len(left_margin_plain) - 1
        positions: list[int] = []

        if stacked_labels:
            bar_start = left_inner_padding + 2
            value_separator = bar_start + bar_width + 1
            positions.append(value_separator)
            count_percent_separator = value_separator + 2 + count_width + 1
            positions.append(count_percent_separator)
            if has_negative_values:
                left_width = max(4, (bar_width - 1) // 2)
                positions.append(bar_start + left_width)
            return positions

        first_separator = left_inner_padding + label_width + 1
        bar_start = first_separator + 2
        second_separator = bar_start + bar_width + 1
        positions.extend([first_separator, second_separator])
        count_percent_separator = second_separator + 2 + count_width + 1
        positions.append(count_percent_separator)
        if has_negative_values:
            left_width = max(4, (bar_width - 1) // 2)
            positions.append(bar_start + left_width)
        return positions

    longest_label = max((len(label) for label in labels), default=0)
    label_width_cap = 30 if size == "big" else 22
    label_width = max(8, min(longest_label, label_width_cap))
    count_texts = [format_number(value) for value in values]
    percent_texts = [f"{percentage:4.1f}%" for percentage in percentages]
    count_width = max((len(text) for text in count_texts), default=1)
    percent_width = max((len(text) for text in percent_texts), default=4)
    value_plain_texts = [
        f"{count_text:>{count_width}} {frame_vertical_char} {percent_text:>{percent_width}}"
        for count_text, percent_text in zip(count_texts, percent_texts)
    ]
    value_width = max((len(text) for text in value_plain_texts), default=12)

    def format_value_text(count_text: str, percent_text: str):
        count_field = visible_pad(count_text, count_width, align="right")
        percent_field = colorize(visible_pad(percent_text, percent_width, align="right"))
        return f"{count_field} {column_separator} {percent_field}"

    stacked_labels = longest_label > label_width_cap or len(labels) > (12 if size == "big" else 9)

    if stacked_labels:
        bar_width = max(
            16 if size == "big" else 12,
            min(56 if size == "big" else 34, canvas_width - value_width - 6),
        )
    else:
        bar_width = max(
            16 if size == "big" else 12,
            min(
                56 if size == "big" else 34,
                canvas_width - label_width - value_width - 6,
            ),
        )

    max_positive = max((max(float(value), 0.0) for value in values), default=0.0)
    min_negative = min((min(float(value), 0.0) for value in values), default=0.0)
    has_negative_values = min_negative < 0
    total_value = sum(values)
    title = add_total_to_title_text(title, total_value)
    title_text, metadata_parts = split_title_metadata(title)

    rule_connector_positions = get_rule_connector_positions()

    lines = [frame_rule_line("top")]
    for title_line in textwrap.wrap(title_text, width=canvas_width) or [title_text]:
        lines.append(frame_line(title_line, align="center"))
    for metadata_line in render_metadata_lines(metadata_parts, canvas_width):
        lines.append(frame_line(metadata_line, align="center"))
    lines.append(frame_rule_line("middle", connector_positions=rule_connector_positions))

    for label, value, count_text, percent_text in zip(labels, values, count_texts, percent_texts):
        numeric_value = float(value)
        value_text = format_value_text(count_text, percent_text)
        if has_negative_values:
            max_abs = max(abs(min_negative), abs(max_positive), 1.0)
            left_width = max(4, (bar_width - 1) // 2)
            right_width = max(4, bar_width - left_width - 1)
            negative_value = abs(min(numeric_value, 0.0))
            positive_value = max(numeric_value, 0.0)
            negative_length = int(round((negative_value / max_abs) * left_width))
            positive_length = int(round((positive_value / max_abs) * right_width))
            negative_track = build_negative_track(
                negative_length,
                left_width,
                has_value=negative_value > 0,
            )
            positive_track = build_positive_track(
                positive_length,
                right_width,
                has_value=positive_value > 0,
            )
            bar_track = f"{negative_track}{marginize(frame_vertical_char)}{positive_track}"
        else:
            scale_base = max(max_positive, 1.0)
            bar_length = int(round((max(numeric_value, 0.0) / scale_base) * bar_width))
            bar_track = build_positive_track(
                bar_length,
                bar_width,
                has_value=numeric_value > 0,
            )

        if stacked_labels:
            wrapped_label = wrap_label(label, width=min(label_width_cap, canvas_width))
            lines.extend(frame_line(label_line) for label_line in wrapped_label)
            lines.append(frame_line(f"  {bar_track} {column_separator} {value_text}"))
        else:
            wrapped_label = wrap_label(label, width=label_width, max_lines=2)
            for extra_label_line in wrapped_label[:-1]:
                lines.append(frame_line(f"{extra_label_line:<{label_width}} {column_separator}"))
            lines.append(
                frame_line(
                    f"{wrapped_label[-1]:<{label_width}} {column_separator} {bar_track} {column_separator} {value_text}"
                )
            )

    lines.append(frame_rule_line("bottom", connector_positions=rule_connector_positions))
    rendered = "\n".join(lines).rstrip()
    print(rendered)
    return None


def plot_dist(
    data: Union[dict, Collection[float], Collection[int], Collection[bool], Collection[str]],
    title: str = "Distribution",
    sort: Literal["ascending", "descending"] | None = None,
    top_n: int | None = None,
    bins: int | None = None,
    xlabel: str = "Item",
    ylabel: str = "Frequency",
    bar_color: Literal[
        "skyblue",
        "deepskyblue",
        "dodgerblue",
        "steelblue",
        "royalblue",
        "cadetblue",
        "teal",
        "turquoise",
        "mediumturquoise",
        "seagreen",
        "mediumseagreen",
        "limegreen",
        "olivedrab",
        "goldenrod",
        "darkorange",
        "coral",
        "tomato",
        "orangered",
        "indianred",
        "crimson",
        "hotpink",
        "deeppink",
        "orchid",
        "mediumorchid",
        "mediumpurple",
        "slateblue",
        "plum",
        "lavender",
    ] | str = "skyblue",
    xlabel_rotation: int | Literal["auto"] = "auto", # degrees
    draw_details: bool = True,
    size: Literal["small", "big"] = "big",
):
    import numpy as np
    MAX_XTICKS: int = 100
    GRID_COLOR = 'rgba(150, 150, 150, 0.3)'
    renderer = _detect_plot_renderer()

    if renderer == "notebook":
        import plotly.graph_objects as go
        import plotly.io as pio

        pio.templates.default = "plotly_dark"
    
    scale = 1.0 if size == "big" else 0.66
    title_font_size = 18 if size == "big" else 12
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
        
        # Plotly bar default width for categorical bars (centered on integer slots).
        bar_width = 0.8
            
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
        
        # Find where each quantile threshold is crossed.
        # If a threshold falls inside a bin, place the separator proportionally
        # inside that bin (instead of snapping to the bin boundary).
        quantile_bars = []
        for threshold, name, color in all_quantiles:
            for j, cum_count in enumerate(cumulative_counts):
                if cum_count >= threshold:
                    if j == 0:
                        bar_start_count = 0
                    else:
                        bar_start_count = cumulative_counts[j - 1]

                    bar_total_samples = y_vals[j]
                    samples_needed_from_bar = threshold - bar_start_count

                    if bar_total_samples > 0:
                        position_in_bar = samples_needed_from_bar / bar_total_samples
                        # Keep floating-point artifacts from pushing outside bin edges.
                        position_in_bar = max(0.0, min(1.0, position_in_bar))
                        if position_in_bar <= 0.0:
                            position = j - 0.5
                        elif position_in_bar >= 1.0:
                            position = j + 0.5
                        else:
                            # Place separator inside the visible bar body, not the slot.
                            position = (j - (bar_width / 2)) + (position_in_bar * bar_width)
                    else:
                        position = j - 0.5

                    quantile_bars.append((position, name, color, threshold))
                    break
        
        # Group quantiles into readable zones.
        # When quartile slices are cramped, choose where the 50% zone goes
        # based on high-importance bins (largest frequencies), not only distance.
        if not quantile_bars:
            return

        quantile_bars.sort(key=lambda x: x[0])
        if len(quantile_bars) < 4:
            return

        start_pos = -0.5
        end_pos = len(x_vals) - 0.5
        merge_threshold = 0.35
        q_positions = [q[0] for q in quantile_bars]
        q_colors = [q[2] for q in quantile_bars]

        def overlap_fraction(zone_start, zone_end, bar_index):
            if zone_end <= zone_start:
                return 0.0
            bar_left = bar_index - (bar_width / 2)
            bar_right = bar_index + (bar_width / 2)
            overlap = min(zone_end, bar_right) - max(zone_start, bar_left)
            if overlap <= 0:
                return 0.0
            return overlap / bar_width

        def candidate_score(fifty_start, fifty_end):
            top_bar_index = int(np.argmax(y_vals))
            weighted_importance = 0.0
            for i, val in enumerate(y_vals):
                if val <= 0:
                    continue
                frac = overlap_fraction(fifty_start, fifty_end, i)
                if frac > 0:
                    # Square to bias toward high-frequency bars.
                    weighted_importance += (val ** 2) * frac

            top_overlap = overlap_fraction(fifty_start, fifty_end, top_bar_index)
            center_distance = abs(((fifty_start + fifty_end) / 2) - top_bar_index)
            return (weighted_importance, top_overlap, -center_distance)

        zone_widths = [
            q_positions[0] - start_pos,
            q_positions[1] - q_positions[0],
            q_positions[2] - q_positions[1],
            end_pos - q_positions[2],
        ]
        should_use_three_zones = any(width < merge_threshold for width in zone_widths)

        if should_use_three_zones:
            # 3-zone candidates:
            # A) 50-25-25, B) 25-50-25, C) 25-25-50
            candidates = [
                (
                    [
                        (start_pos, q_positions[1], "50%", q_colors[1]),
                        (q_positions[1], q_positions[2], "25%", q_colors[2]),
                        (q_positions[2], end_pos, "25%", q_colors[3]),
                    ],
                    (start_pos, q_positions[1]),
                ),
                (
                    [
                        (start_pos, q_positions[0], "25%", q_colors[0]),
                        (q_positions[0], q_positions[2], "50%", q_colors[2]),
                        (q_positions[2], end_pos, "25%", q_colors[3]),
                    ],
                    (q_positions[0], q_positions[2]),
                ),
                (
                    [
                        (start_pos, q_positions[0], "25%", q_colors[0]),
                        (q_positions[0], q_positions[1], "25%", q_colors[1]),
                        (q_positions[1], end_pos, "50%", q_colors[3]),
                    ],
                    (q_positions[1], end_pos),
                ),
            ]

            best_boxes = None
            best_score = None
            for boxes, fifty_zone in candidates:
                score = candidate_score(fifty_zone[0], fifty_zone[1])
                if best_score is None or score > best_score:
                    best_score = score
                    best_boxes = boxes

            merged_boxes = best_boxes if best_boxes is not None else []
        else:
            merged_boxes = [
                (start_pos, q_positions[0], "25%", q_colors[0]),
                (q_positions[0], q_positions[1], "25%", q_colors[1]),
                (q_positions[1], q_positions[2], "25%", q_colors[2]),
                (q_positions[2], end_pos, "25%", q_colors[3]),
            ]
        
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
        dict_median = None
        if all(isinstance(v, (int, float)) and v is not None for v in freq_dict.values()):
            values_for_stats = [v for v in freq_dict.values() if v is not None]
            if values_for_stats:
                dict_mean = np.mean(values_for_stats)
                dict_std = np.std(values_for_stats)
                dict_median = np.median(values_for_stats)
        
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
        title_suffix = f", mean={dict_mean:.2f}, std={dict_std:.2f}, median={dict_median:.2f}" if dict_mean is not None else ""
        plot_title = f"{title} ({len(freq_dict)} elements{f', displayed {top_n}' if top_n else ''}{f', sorted {sort}' if sort else ''}{title_suffix})"
        if renderer == "console":
            return _plot_dist_console(
                labels=str_keys,
                values=display_values,
                percentages=percentages,
                title=plot_title,
                xlabel=xlabel,
                ylabel=ylabel,
                bar_color=bar_color,
                size=size,
            )
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

        fig.update_layout(
            title=plot_title,
            title_font_size=title_font_size,  # ADD THIS LINE
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=width,
            height=height,
            showlegend=False,
            xaxis=dict(
                tickangle=-xlabel_rotation if xlabel_rotation > 0 else 0,
                showticklabels=len(str_keys) <= MAX_XTICKS,
                showgrid=draw_details,
                gridcolor=GRID_COLOR
            ),
            yaxis=dict(
                showgrid=draw_details,
                gridcolor=GRID_COLOR
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
    median_val = None
    if non_none_data and all(is_number(x) for x in non_none_data):
        numeric_data = [float(x) for x in non_none_data]
        mean_val = np.mean(numeric_data)
        std_val = np.std(numeric_data)
        median_val = np.median(numeric_data)

    if len(non_none_data) == 0:
        keys = ["None"]
        values = [none_count]
        total = none_count
        percentages = [100.0]
        customdata = np.array(percentages).reshape(-1, 1)
        plot_title = f"{title} ({len(data_list)} elements, 1 unique)"
        if renderer == "console":
            return _plot_dist_console(
                labels=keys,
                values=values,
                percentages=percentages,
                title=plot_title,
                xlabel=xlabel,
                ylabel=ylabel,
                bar_color="gray",
                size=size,
            )
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
            title=plot_title,
            title_font_size=title_font_size,  # ADD THIS LINE
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=int(800 * scale),
            height=int(500 * scale),
            showlegend=False,
            xaxis=dict(showgrid=draw_details,gridcolor=GRID_COLOR),
            yaxis=dict(showgrid=draw_details, gridcolor=GRID_COLOR),
            
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
        unique_count = len(unique_vals) + (1 if none_count > 0 else 0)
        title_suffix = f", mean={mean_val:.2f}, std={std_val:.2f}, median={median_val:.2f}" if mean_val is not None else ""
        plot_title = f"{title} ({len(data_list)} elements, {unique_count} unique{title_suffix})"
        if renderer == "console":
            return _plot_dist_console(
                labels=keys,
                values=values,
                percentages=percentages,
                title=plot_title,
                xlabel=xlabel,
                ylabel=ylabel,
                bar_color=bar_color,
                size=size,
            )
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

        fig.update_layout(
            title=plot_title,
            title_font_size=title_font_size,  # ADD THIS LINE
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=width,
            height=height,
            showlegend=False,
            xaxis=dict(
                tickangle=-xlabel_rotation if xlabel_rotation > 0 else 0,
                showticklabels=len(keys) <= MAX_XTICKS,
                showgrid=draw_details,
                gridcolor=GRID_COLOR
            ),
            yaxis=dict(
                showgrid=draw_details,
                gridcolor=GRID_COLOR
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
    title_suffix = f", mean={mean_val:.2f}, std={std_val:.2f}, median={median_val:.2f}" if mean_val is not None else ""
    plot_title = f"{title} ({len(data_list)} elements{title_suffix})"
    if renderer == "console":
        return _plot_dist_console(
            labels=x_labels,
            values=values,
            percentages=percentages,
            title=plot_title,
            xlabel=xlabel,
            ylabel=ylabel,
            bar_color=bar_color,
            size=size,
        )
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

    fig.update_layout(
        title=plot_title,
        title_font_size=title_font_size,  # ADD THIS LINE
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=width,
        height=height,
        showlegend=False,
        xaxis=dict(
            tickangle=-xlabel_rotation if xlabel_rotation > 0 else 0,
            showticklabels=len(x_labels) <= MAX_XTICKS,
            showgrid=draw_details,
            gridcolor=GRID_COLOR
        ),
        yaxis=dict(
            showgrid=draw_details,
            gridcolor=GRID_COLOR
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
