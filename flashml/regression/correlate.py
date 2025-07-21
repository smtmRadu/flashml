def compute_correlation(list1, list2, renderer='notebook'):
    """
    Computes the correlation of the two distributions of numbers and plots a scatter plot.

    Args:
        list1, list2 (torch.Tensor, Collection, ndarray): Input arrays or tensors.
        renderer (str): Plotly renderer (default 'notebook'). Options include 'notebook', 'browser', 'png', etc.
    """
    import numpy as np
    import plotly.graph_objects as go
    from scipy.stats import pearsonr
    import plotly.io as pio

    pio.templates.default = "plotly_dark"

    # Ensure that inputs are numpy arrays (convert from tensors or lists if needed)
    def to_numpy(arr):
        if isinstance(arr, np.ndarray):
            return arr
        elif hasattr(arr, 'numpy'):  # PyTorch tensor
            return arr.numpy()
        else:
            return np.array(arr)

    list1 = to_numpy(list1)
    list2 = to_numpy(list2)

    # Calculate Pearson correlation
    pearson_corr, _ = pearsonr(list1, list2)
    # Calculate the correlation coefficient using numpy
    correlation_matrix = np.corrcoef(list1, list2)

    # Create a Plotly scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list1, y=list2, mode='markers', name='Data Points'))

    # Calculate R-squared value
    r_squared = pearson_corr ** 2

    # Create text for the annotation
    stats_text = (
        f"Pearson correlation: {pearson_corr:.2f}<br>"
        f"R-squared: {r_squared:.2f}<br>"
        f"Covariance: {np.cov(list1, list2)[0][1]:.2f}<br>"
        f"Correlation matrix: {correlation_matrix[0][1]:.2f}"
    )

    # Add annotation with correlation statistics - adjusted positioning
    fig.add_annotation(
        x=0.75, y=-0.25,  # Moved higher to fit within margins
        xref="paper", yref="paper",
        showarrow=False,
        text=stats_text,
        font=dict(size=12),
        align="center",
        bgcolor="rgba(255, 255, 255, 0.7)",
        borderpad=10
    )

    # Update layout with increased bottom margin
    fig.update_layout(
        title=f'Correlation between List 1 and List 2 (Pearson: {pearson_corr:.2f})',
        xaxis_title='List 1',
        yaxis_title='List 2',
        height=600,
        margin={"b": 130, "l": 60, "r": 60, "t": 80},  # Increased bottom margin and adjusted others
    )

    # Show the plot using the selected renderer
    fig.show(renderer=renderer)