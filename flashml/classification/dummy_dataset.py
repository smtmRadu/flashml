def make_dummy_classification_dataset(
    num_samples=1024,
    num_features=16,
    num_classes=2,
    bias_std=(0.0, 1.0),
    test_ratio=0.2,
    weights=None,
    seed=42,
):
    """Makes a dummy classification dataset.

    Args:
        num_samples (int, optional): _description_. Defaults to 1024.
        num_features (int, optional): _description_. Defaults to 16.
        num_classes (int, optional): _description_. Defaults to 2.
        bias_std (tuple, optional): _description_. Defaults to (0.0, 1.0).
        test_ratio (float, optional): _description_. Defaults to 0.2.
        weights (_type_, optional): _description_. Defaults to None.
        seed (int, optional): _description_. Defaults to 42.

    Returns: X_train, Y_train, X_test, Y_test
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, Y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        random_state=seed,
        shift=bias_std[0],
        scale=bias_std[1],
        n_classes=num_classes,
        weights=weights,
    )
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_ratio, random_state=seed
    )

    return X_train, Y_train, X_test, Y_test
