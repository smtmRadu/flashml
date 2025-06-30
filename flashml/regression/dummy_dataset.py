def make_dummy_regression_dataset(
    num_samples=1024,
    num_features=16,
    num_targets=1,
    test_ratio=0.2,
    noise=0.0,
    seed=42,
):
    """Makes a dummy regression dataset.

    Args:
        num_samples (int, optional): _description_. Defaults to 1024.
        num_features (int, optional): _description_. Defaults to 16.
        num_targets (int, optional): _description_. Defaults to 1.
        noise (float, optional): _description_. Defaults to 0.0.
        test_ratio (float, optional): _description_. Defaults to 0.2.
        seed (int, optional): _description_. Defaults to 42.

    Returns: X_train, Y_train, X_test, Y_test
    """
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X, Y = make_regression(
        n_samples=num_samples,
        n_features=num_features,
        noise=noise,
        n_targets=num_targets,
        random_state=seed,
    )

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_ratio, random_state=seed
    )

    return X_train, Y_train, X_test, Y_test
