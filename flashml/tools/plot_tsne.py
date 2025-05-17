import matplotlib.pyplot as plt


def plot_tsne(data, mode='3d',verbose=1, perplexity=30, n_iter=300):
    '''
    Plot the t-SNE 2D/3D visualization of the given data.
    Args:
        data (numpy array): A numpy array of shape (n_samples, n_features) containing the data to be plotted.
    Returns:
        tsne_results (numpy array): A numpy array of shape (n_samples, 2|3) containing the t-SNE 2D/3D mapped data.
    '''
    mode=mode.lower()
    assert mode in ['2d', '3d'], "Mode should be either '2d' or '3d'!"
    from sklearn.manifold import TSNE
    is_3d = (mode == '3d')
    tsne = TSNE(n_components=3 if is_3d else 2, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(
        tsne_results[:, 0], 
        tsne_results[:, 1], 
        c=tsne_results[:, 2] if is_3d else None,  
        cmap='plasma',   
        alpha=0.7,
        s=30,                
        edgecolors='none'     
    )
    
    if is_3d:
        plt.colorbar(scatter, label=f't-SNE dimension 3')   
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.grid(True, linestyle='--', alpha=0.5)  
    plt.tight_layout()
    plt.show()

    return tsne_results