## begin of file


def plot_confusion_matrix(
    predicted,
    target,
    average: str = "macro",
    classes: list = None,
    title: str = None,
    cmap: str = "Blues",
    renderer: str = "vscode",
):
    """
    Computes metrics and plots a square, auto-adjusting confusion matrix using Plotly.
    Each cell displays both the unnormalized (bold) and normalized count.

    Args:
        predicted (list | np.ndarray): The predicted values.
        target (list | np.ndarray): The target values.
        classes (list, optional): List of class names for display. If None, inferred from data.
        average (str, optional): Averaging method for metrics. Defaults to "macro".
        title (str, optional): Title for the plot.
        cmap (str, optional): Plotly colorscale for the heatmap. Defaults to "Blues".
        renderer (str, optional): Plotly renderer to use. Defaults to "vscode".

    Returns:
        tuple: A tuple containing (accuracy, precision, recall, f1_score).
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
    import plotly.graph_objects as go
    import plotly.io as pio

    def _compute_confusion_matrix(pred, targ, labels):
        """Computes the confusion matrix from predictions and targets."""
        n_labels = len(labels)
        label_to_int = {label: i for i, label in enumerate(labels)}

        mapped_pred = np.array([label_to_int.get(p, -1) for p in pred])
        mapped_targ = np.array([label_to_int.get(t, -1) for t in targ])

        valid_indices = (mapped_pred >= 0) & (mapped_targ >= 0)
        mapped_pred = mapped_pred[valid_indices]
        mapped_targ = mapped_targ[valid_indices]

        indices = mapped_targ * n_labels + mapped_pred
        cm = np.bincount(indices.astype(np.int64), minlength=n_labels * n_labels)
        return cm.reshape(n_labels, n_labels)

    pio.templates.default = "plotly_dark"

    def _flat(x):
        if isinstance(x, (list, tuple)):
            return np.array(x).flatten()
        elif isinstance(x, np.ndarray):
            return x.flatten()
        else:
            raise ValueError(f"Unsupported input type: {type(x)}")

    pred_array = _flat(predicted)
    target_array = _flat(target)

    unique_labels = sorted(np.unique(np.concatenate([target_array, pred_array])))

    if classes is None:
        display_classes = [str(x) for x in unique_labels]
        labels_for_matrix = unique_labels
    else:
        assert len(classes) >= len(unique_labels), (
            "Provided `classes` list is smaller than the number of unique labels found in data."
        )
        display_classes = [str(c) for c in classes]
        labels_for_matrix = unique_labels

    cm = _compute_confusion_matrix(pred_array, target_array, labels_for_matrix)
    n_classes = len(display_classes)

    # --- Calculate Metrics ---
    accuracy = accuracy_score(y_true=target_array, y_pred=pred_array)
    precision = precision_score(
        y_true=target_array, y_pred=pred_array, average=average, zero_division=0
    )
    recall = recall_score(
        y_true=target_array, y_pred=pred_array, average=average, zero_division=0
    )
    f1 = f1_score(
        y_true=target_array, y_pred=pred_array, average=average, zero_division=0
    )

    # --- Prepare text and normalized values for plotting ---
    cm_unnormalized = cm.astype(int)
    row_sum = cm.sum(axis=1, keepdims=True)
    # The colormap (`z` value) is based on the normalized numbers
    cm_normalized = np.divide(
        cm.astype(float),
        row_sum,
        out=np.zeros_like(cm, dtype=float),
        where=row_sum != 0,
    )

    # Create the text to be displayed in each cell
    text_values = []
    for i in range(n_classes):
        row = []
        for j in range(n_classes):
            # Format: Bold unnormalized value, newline, normalized value
            cell_text = f"<b>{cm_unnormalized[i, j]}</b><br>{cm_normalized[i, j]:.2f}"
            row.append(cell_text)
        text_values.append(row)

    # --- Plotting ---
    heatmap = go.Heatmap(
        z=cm_normalized,  # Colors are based on normalized values
        x=display_classes,
        y=display_classes,
        text=text_values,
        texttemplate="%{text}",  # Use the pre-formatted text
        colorscale=cmap,
        hovertemplate="Prediction: %{x}<br>Target: %{y}<br>Count: %{customdata}<extra></extra>",
        customdata=cm_unnormalized,  # Pass unnormalized data for hover
    )

    fig = go.Figure(data=heatmap)

    metrics_text = (
        f"<b>Accuracy</b>: {accuracy:.4f} | "
        f"<b>Precision ({average})</b>: {precision:.4f} | "
        f"<b>Recall ({average})</b>: {recall:.4f} | "
        f"<b>F1-Score ({average})</b>: {f1:.4f}"
    )

    fig.update_layout(
        title_text=None if title is None else f"<i><b>{title}</b></i>",
        title_x=0.5,
        # xaxis_title="Predicted Label",
        # yaxis_title="True Label",
        yaxis=dict(autorange="reversed"),
        annotations=[
            dict(
                text=metrics_text,
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.2 if n_classes < 5 else -0.15,
                xanchor="center",
                yanchor="top",
                font=dict(size=12),
            )
        ],
        margin=dict(t=100, b=150),
    )

    # --- This is the key part to enforce square cells ---
    # It forces the y-axis to have the same scale as the x-axis.
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    # Let plotly auto-adjust the width and height while maintaining the square ratio
    fig.update_layout(width=None, height=None)

    fig.show(renderer=renderer)

    return accuracy, precision, recall, f1
