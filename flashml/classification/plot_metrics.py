def plot_confusion_matrix(
    predicted,
    target,
    classes: list = None,
    cmap: str = "Blues",
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

    Returns:
        tuple: A tuple containing (accuracy, precision, recall, f1_score).
    """
    import numpy as np
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
        texttemplate="%{text}",
        colorscale=cmap,
        hovertemplate="Prediction: %{x}<br>Target: %{y}<br>Count: %{customdata}<extra></extra>",
        customdata=cm_unnormalized,
    )

    fig = go.Figure(data=heatmap)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(width=None, height=None)

    return fig


def plot_roc_curve(scores, target):
    """
    Plot ROC curve using Plotly with dark theme

    Parameters:
    -----------
    scores : array-like
        Prediction scores or probabilities
    target : array-like
        True binary labels (0 or 1)

    Returns:
    --------
    plotly.graph_objects.Figure
        The ROC curve figure
    """
    import plotly.graph_objects as go
    import plotly.io as pio
    from sklearn.metrics import auc, roc_curve

    # Set the default renderer for VS Code
    pio.renderers.default = "vscode"

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(target, scores)
    roc_auc = auc(fpr, tpr)

    # Create the figure
    fig = go.Figure()

    # Add ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC Curve (AUC = {roc_auc:.3f})",
            line=dict(color="#00D4FF", width=3),
            hovertemplate="<b>False Positive Rate:</b> %{x:.3f}<br>"
            + "<b>True Positive Rate:</b> %{y:.3f}<br>"
            + "<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(color="#FF6B6B", width=2, dash="dash"),
            hovertemplate="<b>Random Classifier</b><br>"
            + "FPR: %{x:.3f}<br>"
            + "TPR: %{y:.3f}<br>"
            + "<extra></extra>",
        )
    )

    # Update layout with dark theme
    fig.update_layout(
        xaxis=dict(
            title="False Positive Rate",
            # titlefont=dict(size=14, color="white"),
            tickfont=dict(color="white"),
            gridcolor="#444444",
            zerolinecolor="#666666",
        ),
        yaxis=dict(
            title="True Positive Rate",
            # titlefont=dict(size=14, color="white"),
            tickfont=dict(color="white"),
            gridcolor="#444444",
            zerolinecolor="#666666",
        ),
        plot_bgcolor="#1E1E1E",
        paper_bgcolor="#1E1E1E",
        font=dict(color="white"),
        legend=dict(
            x=0.6, y=0.2, bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1
        ),
        width=700,
        height=600,
        margin=dict(l=80, r=50, t=80, b=80),
    )

    # Add AUC annotation
    fig.add_annotation(
        x=0.6,
        y=0.4,
        text=f"AUC = {roc_auc:.3f}",
        showarrow=False,
        font=dict(size=16, color="#00D4FF"),
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="#00D4FF",
        borderwidth=1,
    )
    return fig


def plot_pr_curve(scores, target):
    """
    Plot Precision-Recall curve using Plotly with dark theme

    Parameters:
    -----------
    scores : array-like
        Prediction scores or probabilities
    target : array-like
        True binary labels (0 or 1)

    Returns:
    --------
    plotly.graph_objects.Figure
        The PR curve figure
    """
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    from sklearn.metrics import auc, precision_recall_curve

    # Set the default renderer for VS Code
    pio.renderers.default = "vscode"

    # Calculate PR curve
    precision, recall, thresholds = precision_recall_curve(target, scores)
    pr_auc = auc(recall, precision)
    # Alternative: pr_auc = average_precision_score(target, scores)

    # Create the figure
    fig = go.Figure()

    # Add PR curve
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"PR Curve (AUC = {pr_auc:.3f})",
            line=dict(color="#00D4FF", width=3),
            hovertemplate="<b>Recall:</b> %{x:.3f}<br>"
            + "<b>Precision:</b> %{y:.3f}<br>"
            + "<extra></extra>",
        )
    )

    # Add baseline (random classifier for imbalanced data)
    baseline = np.sum(target) / len(target)  # Positive class proportion
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[baseline, baseline],
            mode="lines",
            name=f"Random Classifier (AP = {baseline:.3f})",
            line=dict(color="#FF6B6B", width=2, dash="dash"),
            hovertemplate="<b>Random Classifier</b><br>"
            + "Recall: %{x:.3f}<br>"
            + f"Precision: {baseline:.3f}<br>"
            + "<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis=dict(
            title="Recall",
            # titlefont=dict(size=14, color="white"),
            tickfont=dict(color="white"),
            gridcolor="#444444",
            zerolinecolor="#666666",
        ),
        yaxis=dict(
            title="Precision",
            # titlefont=dict(size=14, color="white"),
            tickfont=dict(color="white"),
            gridcolor="#444444",
            zerolinecolor="#666666",
        ),
        plot_bgcolor="#1E1E1E",
        paper_bgcolor="#1E1E1E",
        font=dict(color="white"),
        legend=dict(
            x=0.6, y=0.2, bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1
        ),
        width=700,
        height=600,
        margin=dict(l=80, r=50, t=80, b=80),
    )

    fig.add_annotation(
        x=0.6,
        y=0.4,
        text=f"AUC = {pr_auc:.3f}",
        showarrow=False,
        font=dict(size=16, color="#00D4FF"),
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="#00D4FF",
        borderwidth=1,
    )
    return fig
