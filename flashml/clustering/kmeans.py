from typing import Literal, Collection

def run_kmeans(x, n_clusters :int|Collection|range = range(2, 21), max_iter=300, weights= None, init:Literal['k-means++', 'random'] = 'k-means++', renderer='vscode'):
    """Run k means (+Elbow method)

    Args:
        x (_type_): _description_
        n_clusters (int | Collection | range, optional): _description_. Defaults to range(2, 20).
        weights (_type_, optional): _description_. Defaults to None.
        init (Literal[&#39;k, optional): _description_. Defaults to 'k-means++'.
        
    Returns:
        The best kmeans model
    """
    from sklearn.cluster import KMeans
    import numpy as np
    from tqdm import tqdm
    

    x = np.asarray(x)
    
    if type(n_clusters) is int:
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter,init=init)
        return kmeans.fit(x, sample_weight=weights)
    else:
        from kneed import KneeLocator
        import plotly.graph_objects as go
        import plotly.io as pio
        pio.templates.default = "plotly_dark"
        inertias = []
        kmns_results = []
        k_values = list(n_clusters) 
        
        
        for k in tqdm(k_values):
            kmeans = KMeans(n_clusters=k,max_iter=max_iter, init=init)
            kmeans.fit(x, sample_weight=weights)
            inertias.append(kmeans.inertia_)
            kmns_results.append(kmeans)
            
        kl = KneeLocator(k_values, inertias, curve="convex", direction="decreasing")
        elbow_k = kl.elbow
        elbow_inertia = inertias[k_values.index(elbow_k)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_values, y=inertias, mode='lines+markers', name='Inertia',showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=[kl.elbow],
            y=[inertias[n_clusters.index(kl.elbow)]],
            mode='markers+text',
            marker=dict(color='red', size=8, symbol='circle'),  # smaller size for just a bullet
            name=f"Elbow Point (k={kl.elbow})",
            text=[f"k={kl.elbow}"],
            textposition="top center",
            showlegend=False  # optional: hide from legend if you don't want it there
        ))
        fig.add_shape(
            type='circle',
            xref='x', yref='y',
            x0=elbow_k - 0.5, x1=elbow_k + 0.5,
            y0=elbow_inertia - (elbow_inertia * 0.05),
            y1=elbow_inertia + (elbow_inertia * 0.05),
            line_color='red'
        )

        fig.update_layout(
            title='K-Means (Elbow Method)',
            xaxis_title='Number of clusters',
            yaxis_title='Inertia',
            showlegend=True
        )

        fig.show(renderer=renderer)

        print(f"\033[34mK-Means (Elbow Method) optimal number of clusters: {elbow_k}\033[37m")

        return kmns_results[k_values.index(elbow_k)]
    