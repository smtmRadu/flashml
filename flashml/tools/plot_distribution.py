from typing import Literal
import matplotlib.pyplot as plt


def plot_distribution(
    freq_dict: dict,
    sort_values: Literal["ascending", "descending"] | None = None,
    top_n: int = None,
    title: str = "Distribution",
    x_label: str = "Item",
    y_label: str = "Frequency",
    figsize: tuple = (12, 6),
    bar_color: str = "skyblue",
    rotation: int = 90,
    max_xticks: int = 100,
    show_values: bool = False,
    grid: bool = False,
) -> None:
    """
    Plots a bar chart distribution from a frequency dictionary.

    Args:
        freq_dict: Dictionary with items as keys and frequencies as values.
        sort_values: How to sort the items. Can be 'ascending', 'descending', or None (default).
        top_n: Display only the top N items.
        title: Title of the plot.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        figsize: Tuple specifying the figure size.
        bar_color: Color of the bars.
        rotation: Rotation angle for x-axis tick labels.
        max_xticks: Maximum number of x-axis ticks to display labels for.
                     If exceeded, labels are hidden.
        show_values: If True, displays the frequency value on top of each bar.
        grid: If True, adds a grid to the plot.
    """
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

    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(str_keys)), values, color=bar_color)

    if len(str_keys) <= max_xticks:
        plt.xticks(
            range(len(str_keys)),
            str_keys,
            rotation=rotation,
            ha="right" if rotation > 0 else "center",
        )
    else:
        plt.xticks([])  # Hide x-axis ticks if too many

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if grid:
        plt.grid(axis="y", linestyle="--")

    if show_values:
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 0.05 * max(values),  # Adjust offset based on max value
                f"{yval:.2f}"
                if isinstance(yval, float)
                else str(yval),  # Format float values
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()
