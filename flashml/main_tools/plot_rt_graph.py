from typing import Union

_GRAPHS = {}


def _init_graph(
    name,
    initial_values,
    initial_steps,
    x_label="X",
    y_label="Y",
    linestyles="-",
    markers="",
    colors=["lightblue"],
):
    """Initialize a real-time graph with one or more lines."""
    import matplotlib.pyplot as plt

    if not isinstance(initial_values, (list, tuple)):
        initial_values = [initial_values]

    if initial_steps is None:
        initial_steps = [0] * len(initial_values)
    elif not isinstance(initial_steps, (list, tuple)):
        initial_steps = [initial_steps] * len(initial_values)

    if not isinstance(colors, (list, tuple)):
        colors = [colors] * len(initial_values)

    if not isinstance(linestyles, (list, tuple)):
        linestyles = [linestyles] * len(initial_values)

    if not isinstance(markers, (list, tuple)):
        markers = [markers] * len(initial_values)

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("flashml")

    fig.patch.set_facecolor("#212121")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="both", colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.set_title(name, color="white")
    ax.set_facecolor("#2E2E2E")

    lines = []
    x_data = []
    y_data = []

    for i in range(len(initial_values)):
        (line,) = ax.plot(
            [initial_steps[i]],
            [initial_values[i]],
            color=colors[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=0.75,
        )
        lines.append(line)
        x_data.append([initial_steps[i]])
        y_data.append([initial_values[i]])

    min_x = min(initial_steps)
    max_x = max(initial_steps) + 10  # Reduced initial padding from 50 to 10
    min_y = min(initial_values) - 1
    max_y = max(initial_values) + 1
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    plt.ion()
    plt.show()

    _GRAPHS[name] = {
        "fig": fig,
        "ax": ax,
        "lines": lines,
        "x_data": x_data,
        "y_data": y_data,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
    }


def _update_graph(name, new_values, new_steps):
    """Update the graph with new data and resize axes using stored min/max."""
    graph = _GRAPHS[name]

    if not isinstance(new_values, (list, tuple)):
        new_values = [new_values]

    if new_steps is None:
        if graph["x_data"] and graph["x_data"][0]:
            last_step = max(line_x[-1] for line_x in graph["x_data"])
            new_steps = [last_step + 1] * len(new_values)
        else:
            new_steps = [0] * len(new_values)
    elif not isinstance(new_steps, (list, tuple)):
        new_steps = [new_steps] * len(new_values)

    if len(new_steps) != len(new_values):
        raise ValueError("new_steps and new_values must have the same length")

    min_length = min(len(graph["lines"]), len(new_steps), len(new_values))

    for i, (line, x_list, y_list) in enumerate(
        zip(graph["lines"], graph["x_data"], graph["y_data"])
    ):
        if i >= min_length:
            break
        x_list.append(new_steps[i])
        y_list.append(new_values[i])
        line.set_xdata(x_list)
        line.set_ydata(y_list)
        line.set_linewidth(min(100 / len(x_list), 1))

    # Update stored min/max values
    new_min_x = min(new_steps)
    new_max_x = max(new_steps)
    new_min_y = min(new_values)
    new_max_y = max(new_values)

    graph["min_x"] = min(graph["min_x"], new_min_x)
    graph["max_x"] = max(graph["max_x"], new_max_x)
    graph["min_y"] = min(graph["min_y"], new_min_y)
    graph["max_y"] = max(graph["max_y"], new_max_y)

    # Adjust axis limits with configurable margin
    x_range = graph["max_x"] - graph["min_x"]
    y_range = graph["max_y"] - graph["min_y"]
    x_margin = 0
    y_margin = 1

    graph["ax"].set_xlim(graph["min_x"] - x_margin, graph["max_x"] + x_margin)
    graph["ax"].set_ylim(graph["min_y"] - y_margin, graph["max_y"] + y_margin)

    graph["fig"].canvas.draw_idle()
    graph["fig"].canvas.flush_events()


def plot_rt_graph(
    name,
    value: Union[int, float] | list | tuple,
    step: int | list | tuple = None,
    x_label: str | list | tuple = "X",
    y_label: str | list | tuple = "Y",
    color: str | list | tuple = None,
    linestyle: str | list | tuple = "-",
    marker: str | list | tuple = "",
):
    """
    Plot or update a real-time graph.

    Args:
        name (str): The name of window to plot in. You can run multiple graph windows in the same time, specifying different names.
        value (Union[int, float] | list | tuple): The value(s) to plot.
        step (int | list | tuple, optional): The step(s) to plot. Defaults to None.
        x_label, y_label, color, linestyle, marker: Plot styling parameters.
        x_margin_factor (float): Factor for x-axis margin (default 0.05 = 5%).

    Returns:
        dict: All created/updated graphs.
    """
    if name not in _GRAPHS:
        _init_graph(
            name,
            value,
            step,
            x_label=x_label,
            y_label=y_label,
            linestyles=linestyle,
            markers=marker,
            colors=color,
        )
    else:
        _update_graph(name, value, step)

    return _GRAPHS
