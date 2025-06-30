def compute_regression_metrics(predicted, target):
    """
    Compute standard regression metrics: MSE, MAE, RMSE, R².

    Args:
        predicted (array-like): Predicted values.
        target (array-like): Ground truth values.

    Returns:
        dict: Dictionary containing regression metrics.
    """
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    predicted = np.asarray(predicted)
    target = np.asarray(target)

    metrics = {}
    metrics["mse"] = mean_squared_error(target, predicted)
    metrics["mae"] = mean_absolute_error(target, predicted)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["r2"] = r2_score(target, predicted)

    return metrics
