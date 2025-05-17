import matplotlib.pyplot as plt


def plot_distribution(
    freq_dict,
    sort_descending: bool = None,
    top_n: int = None,
    title="Distribution",
    x_label="Item",
    y_label="Frequency",
):
    items = list(freq_dict.items())

    if sort_descending is True:
        items.sort(key=lambda x: x[1], reverse=True)
    elif sort_descending is False:
        items.sort(key=lambda x: x[1])

    if top_n:
        items = items[:top_n]

    keys, values = zip(*items)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(keys)), values)

    if len(keys) <= 100:
        plt.xticks(range(len(keys)), [str(k) for k in keys], rotation=90)
    else:
        plt.xticks([])  # Hide x-axis ticks if too many

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.show()
