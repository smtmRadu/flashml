### begin of file


def plot_tsne(
    data,
    labels=None,
    mode="3d",
    verbose=1,
    perplexity=30,
    n_iter=300,
    title="t-SNE Visualization",
    point_size=5,
):
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
    from sklearn.manifold import TSNE

    """
    Plot in browser the t-SNE 2D/3D of the given data using Plotly.

    Args:
        data (numpy array): A numpy array of shape (n_samples, n_features) containing the data to be plotted.
        labels (array/list of length data.shape[0], optional): Labels of the samples to see if clustering was correct. If None, uses continuous coloring.

        mode (str): Either '2d' or '3d' for the type of visualization.
        verbose (int): Verbosity level for t-SNE.
        perplexity (float): The perplexity parameter for t-SNE.
        n_iter (int): Maximum number of iterations for t-SNE.
        title (str): Title for the plot.
        point_size (int): Size of the scatter points.

    Returns:
        tsne_results (numpy array): A numpy array of shape (n_samples, 2|3) containing the t-SNE mapped data.
    """
    mode = mode.lower()
    assert mode in ["2d", "3d"], "Mode should be either '2d' or '3d'!"

    is_3d = mode == "3d"
    tsne = TSNE(
        n_components=3 if is_3d else 2,
        verbose=verbose,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=42,
    )
    tsne_results = tsne.fit_transform(data)

    if labels is not None:
        unique_labels = np.unique(labels)
        label_to_num = {label: i for i, label in enumerate(unique_labels)}
        color_data = [label_to_num[label] for label in labels]
        color_scale = px.colors.qualitative.Set3
        showscale = True
        colorbar_title = "Categories"
        hover_labels = labels
    else:
        if is_3d:
            color_data = tsne_results[:, 2]
        else:
            center_x, center_y = (
                np.mean(tsne_results[:, 0]),
                np.mean(tsne_results[:, 1]),
            )
            color_data = np.sqrt(
                (tsne_results[:, 0] - center_x) ** 2
                + (tsne_results[:, 1] - center_y) ** 2
            )
        color_scale = "Viridis"
        showscale = True
        colorbar_title = "Color Scale"
        hover_labels = [f"Point {i}" for i in range(len(tsne_results))]

    if is_3d:
        # Create 3D scatter plot
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=tsne_results[:, 0],
                    y=tsne_results[:, 1],
                    z=tsne_results[:, 2],
                    mode="markers",
                    marker=dict(
                        size=point_size,
                        color=color_data,
                        colorscale=color_scale,
                        opacity=0.8,
                        showscale=showscale,
                        colorbar=dict(
                            title=colorbar_title,
                            tickmode="linear",
                            tickvals=list(range(len(unique_labels)))
                            if labels is not None
                            else None,
                            ticktext=list(unique_labels)
                            if labels is not None
                            else None,
                        )
                        if showscale
                        else None,
                        line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
                    ),
                    text=hover_labels,
                    hovertemplate="<b>%{text}</b><br>"
                    + "X: %{x:.2f}<br>"
                    + "Y: %{y:.2f}<br>"
                    + "Z: %{z:.2f}<br>"
                    + "<extra></extra>",
                )
            ]
        )

        # Update 3D layout
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20, color="white")),
            scene=dict(
                xaxis=dict(
                    title="t-SNE Dimension 1",
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.2)",
                    showbackground=True,
                    zerolinecolor="rgba(255,255,255,0.4)",
                ),
                yaxis=dict(
                    title="t-SNE Dimension 2",
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.2)",
                    showbackground=True,
                    zerolinecolor="rgba(255,255,255,0.4)",
                ),
                zaxis=dict(
                    title="t-SNE Dimension 3",
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.2)",
                    showbackground=True,
                    zerolinecolor="rgba(255,255,255,0.4)",
                ),
                bgcolor="rgba(0,0,0,0.9)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            paper_bgcolor="rgba(0,0,0,0.95)",
            plot_bgcolor="rgba(0,0,0,0.95)",
            font=dict(color="white"),
            width=900,
            height=700,
        )

    else:
        # Create 2D scatter plot
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=tsne_results[:, 0],
                    y=tsne_results[:, 1],
                    mode="markers",
                    marker=dict(
                        size=point_size,
                        color=color_data,
                        colorscale=color_scale,
                        opacity=0.8,
                        showscale=showscale,
                        colorbar=dict(
                            title=colorbar_title,
                            tickvals=list(range(len(unique_labels)))
                            if labels is not None
                            else None,
                            ticktext=list(unique_labels)
                            if labels is not None
                            else None,
                        )
                        if showscale
                        else None,
                        line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
                    ),
                    text=hover_labels,
                    hovertemplate="<b>%{text}</b><br>"
                    + "X: %{x:.2f}<br>"
                    + "Y: %{y:.2f}<br>"
                    + "<extra></extra>",
                )
            ]
        )

        # Update 2D layout
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20, color="white")),
            xaxis=dict(
                title="t-SNE Dimension 1",
                gridcolor="rgba(255,255,255,0.2)",
                zerolinecolor="rgba(255,255,255,0.4)",
                color="white",
            ),
            yaxis=dict(
                title="t-SNE Dimension 2",
                gridcolor="rgba(255,255,255,0.2)",
                zerolinecolor="rgba(255,255,255,0.4)",
                color="white",
            ),
            paper_bgcolor="rgba(0,0,0,0.95)",
            plot_bgcolor="rgba(0,0,0,0.95)",
            font=dict(color="white"),
            width=900,
            height=700,
        )

    fig.show(renderer="browser")

    return tsne_results
